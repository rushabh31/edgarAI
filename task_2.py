import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, IntegerType
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Outlines
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel
import json
from pathlib import Path
from datetime import datetime
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')

@dataclass
class ChunkingConfig:
    """Configuration settings for text chunking."""
    model_token_limit: int
    overlap_ratio: float

class EDGARAnalysisSystem:
    """
    A comprehensive system for analyzing EDGAR financial reports using RAG (Retrieval Augmented Generation).
    
    This class provides functionality to load, process, and analyze SEC EDGAR filings
    using PySpark for data processing and LangChain for RAG-based question answering.
    
    Attributes:
        spark (SparkSession): The active Spark session
        logger (logging.Logger): Logger instance for the class
        chunking_config (ChunkingConfig): Configuration for text chunking
        chunks_per_query (dict): Storage for chunks used in each query
        vectordb (Chroma): Vector database for document storage
        llm (Outlines): Language model instance
    """

    def __init__(
        self,
        app_name: str = "EDGAR Analysis System",
        model_token_limit: int = 512,
        overlap_ratio: float = 0.1,
        log_level: int = logging.INFO
    ):
        """
        Initialize the EDGAR Analysis System.
        
        Args:
            app_name: Name of the Spark application
            model_token_limit: Maximum token limit for the embedding model
            overlap_ratio: Overlap ratio for text chunking
            log_level: Logging level to use
        """
        self._setup_logging(log_level)
        self.logger.info("Initializing EDGAR Analysis System")
        self.spark = self._create_spark_session(app_name)
        self.chunking_config = ChunkingConfig(model_token_limit, overlap_ratio)
        self.chunks_per_query = {}
        self.vectordb = None
        self.llm = None

    def _setup_logging(self, log_level: int) -> None:
        """Configure logging for the system."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create and configure Spark session."""
        return SparkSession.builder.appName(app_name).getOrCreate()

    def load_filtered_datasets(
        self,
        years: List[str],
        company_cik: str
    ) -> List[Dict[str, str]]:
        """
        Load and filter EDGAR datasets for specific years and company.
        
        Args:
            years: List of years to load data for
            company_cik: Company CIK number to filter by
        
        Returns:
            List of dictionaries containing filtered dataset entries
        """
        self.logger.info(f"Loading datasets for years: {years} and company CIK: {company_cik}")
        datasets = []
        
        for year in years:
            self.logger.info(f"Loading dataset for year: {year}")
            dataset = load_dataset(
                "eloukas/edgar-corpus",
                f"year_{year}",
                split="train",
                trust_remote_code=True
            )
            
            filtered_dataset = dataset.filter(lambda x: x['cik'] == company_cik)
            
            for filing in filtered_dataset:
                for section in dataset.column_names:
                    if section.startswith('section_') and filing[section] is not None:
                        section_number = section.replace('section_', '', 1)
                        datasets.append({
                            'year': filing['year'],
                            'section_number': section_number,
                            'section_text': filing[section]
                        })
        
        return datasets

    def create_spark_dataframe(self, data: List[Dict[str, str]]) -> DataFrame:
        """
        Convert processed dataset into a Spark DataFrame.
        
        Args:
            data: List of dictionaries containing dataset entries
        
        Returns:
            Spark DataFrame with the processed data
        """
        schema = StructType([
            StructField("year", StringType(), True),
            StructField("section_number", StringType(), True),
            StructField("section_text", StringType(), True),
        ])
        df = self.spark.createDataFrame(data, schema)
        self.logger.info(f"Created Spark DataFrame with {df.count()} rows")
        return df

    def calculate_chunking_requirements(
        self,
        spark_df: DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate text chunking requirements for each section.
        
        Args:
            spark_df: Spark DataFrame containing the document sections
        
        Returns:
            Dictionary containing chunking requirements for each section
        """
        self.logger.info("Calculating chunking requirements with PySpark")
        
        def count_tokens(text: str) -> int:
            return len(text.split())
        
        count_tokens_udf = F.udf(count_tokens, IntegerType())
        
        token_stats_df = spark_df.withColumn(
            "token_count",
            count_tokens_udf(F.col("section_text"))
        )
        
        token_stats = token_stats_df.groupBy("section_number").agg(
            F.max("token_count").alias("max_tokens"),
            F.avg("token_count").alias("mean_tokens"),
            F.stddev("token_count").alias("std_dev_tokens")
        )
        
        token_stats_pd = token_stats.toPandas()
        chunking_requirements = {}
        
        for _, row in token_stats_pd.iterrows():
            section = f"section_{row['section_number']}"
            max_tokens = row['max_tokens']
            mean_tokens = row['mean_tokens']
            std_dev_tokens = row['std_dev_tokens'] or 0
            
            if max_tokens > self.chunking_config.model_token_limit:
                chunks_needed = (
                    max_tokens // 
                    (self.chunking_config.model_token_limit * 
                     (1 - self.chunking_config.overlap_ratio))
                ) + 1
                
                chunking_requirements[section] = {
                    "requires_chunking": True,
                    "max_tokens": max_tokens,
                    "mean_tokens": mean_tokens,
                    "suggested_chunk_size": self.chunking_config.model_token_limit,
                    "overlap": int(self.chunking_config.model_token_limit * 0.1),
                    "chunks_needed": chunks_needed
                }
            else:
                chunking_requirements[section] = {
                    "requires_chunking": False,
                    "max_tokens": max_tokens,
                    "mean_tokens": mean_tokens,
                    "suggested_chunk_size": None,
                    "overlap": None,
                    "chunks_needed": None
                }
        
        return chunking_requirements

    def process_and_split_data(
        self,
        spark_df: DataFrame,
        chunking_requirements: Dict[str, Dict[str, Any]]
    ) -> DataFrame:
        """
        Process and split the data according to chunking requirements.
        
        Args:
            spark_df: Spark DataFrame containing the document sections
            chunking_requirements: Dictionary of chunking requirements per section
        
        Returns:
            Processed Spark DataFrame with split text chunks
        """
        def split_or_full_text(
            section_text: str,
            year: str,
            section_number: str
        ) -> List[Dict[str, Any]]:
            section_name = f"section_{section_number}"
            if section_name in chunking_requirements:
                req = chunking_requirements[section_name]
                if req["requires_chunking"]:
                    chunk_size = req["suggested_chunk_size"]
                    overlap = req["overlap"]
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    chunks = text_splitter.split_text(section_text)
                    return [
                        {
                            "text": chunk,
                            "metadata": {
                                "year": year,
                                "section_number": section_number,
                                "chunk_number": idx + 1
                            }
                        }
                        for idx, chunk in enumerate(chunks)
                    ]
            return [{
                "text": section_text,
                "metadata": {
                    "year": year,
                    "section_number": section_number,
                    "chunk_number": 1
                }
            }]

        split_schema = ArrayType(
            StructType([
                StructField("text", StringType(), True),
                StructField("metadata", StructType([
                    StructField("year", StringType(), True),
                    StructField("section_number", StringType(), True),
                    StructField("chunk_number", IntegerType(), True)
                ]))
            ])
        )

        split_or_full_text_udf = F.udf(split_or_full_text, split_schema)
        
        processed_df = spark_df.withColumn(
            "split_data",
            split_or_full_text_udf(
                F.col("section_text"),
                F.col("year"),
                F.col("section_number")
            )
        )
        
        exploded_df = processed_df.select(
            F.explode(F.col("split_data")).alias("split")
        )
        
        final_df = exploded_df.select(
            F.col("split.text").alias("text"),
            F.col("split.metadata.year").alias("year"),
            F.col("split.metadata.section_number").alias("section_number"),
            F.col("split.metadata.chunk_number").alias("chunk_number")
        )
        
        self.logger.info(f"Processed and split data into {final_df.count()} chunks")
        return final_df

    def create_vector_store(
        self,
        df: DataFrame,
        model_name: str,
        persist_directory: str
    ) -> None:
        """
        Create and initialize the vector store from processed data.
        
        Args:
            df: Processed Spark DataFrame
            model_name: Name of the embedding model to use
            persist_directory: Directory to persist the vector store
        """
        self.logger.info("Creating vector store from Spark DataFrame")
        texts = df.select("text").rdd.flatMap(lambda x: x).collect()
        metadata = df.select(
            "year",
            "section_number",
            "chunk_number"
        ).rdd.map(lambda row: row.asDict()).collect()
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"}
        )
        
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadata)
        ]
        
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    def setup_llm(self, model_name: str) -> None:
        """
        Initialize the language model pipeline.
        
        Args:
            model_name: Name of the LLM model to use
        """
        class Response(BaseModel):
            answer: str

        self.llm = Outlines(
            model=model_name,
            max_new_tokens=100,
            json_schema=Response
        )

    def create_year_specific_pipeline(self, year: str):
        """
        Create a query pipeline specific to a given year.
        
        Args:
            year: Year to filter results for
        
        Returns:
            Chain object for processing year-specific queries
        """
        def retrieve_with_year_filter(query: str) -> str:
            retriever = self.vectordb.as_retriever(search_kwargs={"k": 200})
            raw_results = retriever.get_relevant_documents(query)
            
            filtered_results = [
                {"metadata": doc.metadata, "content": doc.page_content}
                for doc in raw_results if doc.metadata.get("year") == str(year)
            ]
            
            unique_chunks = []
            seen_contents = set()
            for chunk in filtered_results:
                content = chunk["content"]
                if content not in seen_contents:
                    unique_chunks.append(chunk)
                    seen_contents.add(content)
            
            combined_context = "\n".join(
                f"Chunk Metadata: {chunk['metadata']} \nContent: {chunk['content']}"
                for chunk in unique_chunks[:1]
            )
            
            self.chunks_per_query[query] = unique_chunks
            return combined_context

        year_specific_template = """
            You are an assistant specialized in answering questions from 10-K SEC financial reports using a Retrieval Augmented Generation (RAG) system.
            Your role is to focus exclusively on financial reports from {year}, leveraging the provided context.
            Base your response solely on the given context, and refrain from using external information.
            
            **Task Instructions:**
            - Extract the precise answer based on the provided context.
            - Ensure that the response is concise and accurate.
            - Format the response as JSON and include only the requested fields.

            **Question:** {query}
            **Context:** {context}

            **Response (in JSON format):**
        """

        year_specific_prompt = PromptTemplate.from_template(year_specific_template)
        return (
            {
                "context": retrieve_with_year_filter,
                "query": RunnablePassthrough()
            }
            | year_specific_prompt.partial(year=year)
            | self.llm
        )

    def run_queries(
        self,
        year_specific_chain: Any,
        queries: List[str]
    ) -> Dict[str, Any]:
        """
        Run multiple queries through the year-specific pipeline.
        
        Args:
            year_specific_chain: Chain object for processing queries
            queries: List of queries to process
        
        Returns:
            Dictionary containing query results and used chunks
        """
        results = {}
        for query in queries:
            try:
                output = year_specific_chain.invoke(
                    query,
                    config={'temperature': 0.3, 'max_new_tokens': 200}
                )
                chunks_used = self.chunks_per_query.get(query, [])
                results[query] = {'answer': output, 'chunks': chunks_used}
            except Exception as e:
                self.logger.error(
                    f"Error during query execution for '{query}': {str(e)}")
                results[query] = f"Error: {str(e)}"
        return results

    def cleanup(self):
        """Clean up resources and stop the Spark session."""
        if self.spark:
            self.logger.info("Stopping Spark session")
            self.spark.stop()
            
    def process_company_data(
        self,
        years: List[str],
        company_cik: str,
        embedding_model: str,
        llm_model: str,
        persist_dir: str
    ) -> None:
        """
        Process company data end-to-end from loading to creating the query pipeline.
        
        Args:
            years: List of years to analyze
            company_cik: Company CIK number
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model
            persist_dir: Directory to persist vector store
        """
        try:
            # Load and process data
            filtered_data = self.load_filtered_datasets(years, company_cik)
            spark_df = self.create_spark_dataframe(filtered_data)
            
            # Calculate chunking requirements and process data
            chunking_requirements = self.calculate_chunking_requirements(spark_df)
            processed_df = self.process_and_split_data(spark_df, chunking_requirements)
            
            # Create vector store and setup LLM
            self.create_vector_store(processed_df, embedding_model, persist_dir)
            self.setup_llm(llm_model)
            
            self.logger.info("Company data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error processing company data: {str(e)}", exc_info=True)
            raise

    def save_query_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        company_cik: str
    ) -> None:
        """
        Save query results to a JSON file, preserving the original structure with chunks.
        
        Args:
            results: Dictionary containing query results
            output_dir: Directory to save results
            company_cik: Company CIK number for identification
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process and structure the results
            structured_output = {}
            
            for query, result in results.items():
                # Parse the answer JSON string
                answer = result.get('answer', '{}')
                
                # Structure chunks with metadata and content
                chunks = []
                for chunk in result.get('chunks', []):
                    chunk_data = {
                        'metadata': {
                            'chunk_number': chunk['metadata']['chunk_number'],
                            'section_number': chunk['metadata']['section_number'],
                            'year': chunk['metadata']['year']
                        },
                        'content': chunk['content']
                    }
                    chunks.append(chunk_data)
                
                # Add to final structure
                structured_output[query] = {
                    'answer': answer,
                    'chunks': chunks
                }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edgar_analysis_{company_cik}_{timestamp}.json"
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_output, f, indent=2)
            
            self.logger.info(f"Saved query results to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving query results: {str(e)}", exc_info=True)
            raise

def main():
    """
    Main execution function demonstrating the usage of EDGARAnalysisSystem.
    """
    # Configuration
    CONFIG = {
        'years': ['2018', '2019', '2020'],
        'company_cik': '320193',  # Apple Inc.
        'embedding_model': "sentence-transformers/all-mpnet-base-v2",
        'llm_model': "meta-llama/Llama-3.2-3B-Instruct",
        'persist_dir': "chroma_db",
        'analysis_year': '2020',
        'results_dir': "analysis_results"
    }
    
    # Sample queries
    SAMPLE_QUERIES = [
        "How many total employees does Apple have?",
        "How much Total net sales increased from previous year in percent and in dollars?",
        "How did overall iPhone sales compare to the previous year? Did they increase or decrease?",
        "What are the exact values of the company's net deferred tax assets and deferred tax liabilities?",
        "What was the amount of the company's quarterly cash dividend per share?"
    ]
    
    try:
        # Initialize the system
        edgar_system = EDGARAnalysisSystem()
        
        # Process company data
        edgar_system.process_company_data(
            years=CONFIG['years'],
            company_cik=CONFIG['company_cik'],
            embedding_model=CONFIG['embedding_model'],
            llm_model=CONFIG['llm_model'],
            persist_dir=CONFIG['persist_dir']
        )
        
        # Create year-specific pipeline
        year_specific_chain = edgar_system.create_year_specific_pipeline(
            CONFIG['analysis_year']
        )
        
        # Run queries
        results = edgar_system.run_queries(year_specific_chain, SAMPLE_QUERIES)
        
        # Print results
        for query, result in results.items():
            print(f"\nQuery: {query}")
            print(f"Result: {result}")

        # Save results
        edgar_system.save_query_results(
            results,
            CONFIG['results_dir'],
            CONFIG['company_cik']
        )
            
    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}", exc_info=True)
    finally:
        edgar_system.cleanup()

if __name__ == "__main__":
    main()
