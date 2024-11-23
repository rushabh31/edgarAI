import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, struct, udf
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, IntegerType
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import torch
from pyspark.sql import functions as F

import sys
sys.modules['sqlite3'] = __import__('pysqlite3')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Spark Session
logger.info("Initializing Spark Session.")
spark = SparkSession.builder \
    .appName("EDGAR PySpark Task 2") \
    .getOrCreate()

# Step 1: Load the Dataset
def load_filtered_datasets(years, company_cik):
    logger.info(f"Loading datasets for years: {years} and company CIK: {company_cik}.")
    datasets = []
    for year in years:
        logger.info(f"Loading dataset for year: {year}.")
        dataset = load_dataset("eloukas/edgar-corpus", f"year_{year}", split="train", trust_remote_code=True)
        logger.info(f"Filtering dataset for company CIK: {company_cik}.")
        
        # Filter dataset by company CIK
        filtered_dataset = dataset.filter(lambda x: x['cik'] == company_cik)
        
        # Extract sections and create dictionaries
        for filing in filtered_dataset:
            for section in dataset.column_names:
                if section.startswith('section_') and filing[section] is not None:
                    # Extract section number by removing the "section_" prefix
                    section_number = section.replace('section_', '', 1)
                    datasets.append({
                        'year': filing['year'],  # Ensure correct year from dataset
                        'section_number': section_number,  # Truncated section number
                        'section_text': filing[section]  # Corresponding text
                    })
    return datasets

# Step 2: Convert to Spark DataFrame
def create_spark_dataframe(data, spark):
    logger.info("Creating Spark DataFrame.")
    schema = StructType([
        StructField("year", StringType(), True),
        StructField("section_number", StringType(), True),
        StructField("section_text", StringType(), True),
    ])
    df = spark.createDataFrame(data, schema)
    logger.info(f"Created Spark DataFrame with {df.count()} rows.")
    return df

# Step 3: Process and Split Data
# Step 1: Calculate chunking requirements
def calculate_chunking_requirements(spark_df, model_token_limit):
    logger.info("Calculating chunking requirements with PySpark.")
    
    # Token count UDF
    def count_tokens(text):
        return len(text.split())
    
    count_tokens_udf = F.udf(count_tokens, IntegerType())

    # Add token counts to Spark DataFrame
    token_stats_df = spark_df.withColumn("token_count", count_tokens_udf(F.col("section_text")))

    # Aggregate token stats by section_number
    token_stats = token_stats_df.groupBy("section_number").agg(
        F.max("token_count").alias("max_tokens"),
        F.avg("token_count").alias("mean_tokens"),
        F.stddev("token_count").alias("std_dev_tokens")
    )

    # Convert to Pandas for easier computation
    token_stats_pd = token_stats.toPandas()

    # Determine chunking requirements
    chunking_requirements = {}
    for _, row in token_stats_pd.iterrows():
        section = f"section_{row['section_number']}"
        max_tokens = row['max_tokens']
        mean_tokens = row['mean_tokens']
        std_dev_tokens = row['std_dev_tokens'] or 0  # Handle NaN for stddev
        
        if max_tokens > model_token_limit:
            chunks_needed = (max_tokens // (model_token_limit * (1 - OVERLAP_RATIO))) + 1
            chunking_requirements[section] = {
                "requires_chunking": True,
                "max_tokens": max_tokens,
                "mean_tokens": mean_tokens,
                "suggested_chunk_size": model_token_limit,
                "overlap": int(model_token_limit * 0.1),
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

# Step 2: Process and split data with dynamic chunking
# Step 2: Process and split data with dynamic chunking
def process_and_split_data(spark_df, chunking_requirements):
    logger.info("Processing data with chunking requirements using PySpark.")

    # UDF to split or retain full text
    def split_or_full_text(section_text, year, section_number):
        section_name = f"section_{section_number}"
        if section_name in chunking_requirements:
            req = chunking_requirements[section_name]
            if req["requires_chunking"]:
                chunk_size = req["suggested_chunk_size"]
                overlap = req["overlap"]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
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
            else:
                return [{
                    "text": section_text,
                    "metadata": {
                        "year": year,
                        "section_number": section_number,
                        "chunk_number": 1
                    }
                }]
        else:
            return [{
                "text": section_text,
                "metadata": {
                    "year": year,
                    "section_number": section_number,
                    "chunk_number": 1
                }
            }]

    split_or_full_text_udf = F.udf(split_or_full_text, ArrayType(
        StructType([
            StructField("text", StringType(), True),
            StructField("metadata", StructType([
                StructField("year", StringType(), True),
                StructField("section_number", StringType(), True),
                StructField("chunk_number", IntegerType(), True)
            ]))
        ])
    ))

    # Apply the UDF
    processed_df = spark_df.withColumn(
        "split_data", split_or_full_text_udf(F.col("section_text"), F.col("year"), F.col("section_number"))
    )
    
    # Explode the split data
    exploded_df = processed_df.select(F.explode(F.col("split_data")).alias("split"))
    
    # Final DataFrame
    final_df = exploded_df.select(
        F.col("split.text").alias("text"),
        F.col("split.metadata.year").alias("year"),
        F.col("split.metadata.section_number").alias("section_number"),
        F.col("split.metadata.chunk_number").alias("chunk_number")
    )
    logger.info(f"Processed and split data into {final_df.count()} chunks.")
    return final_df


# Step 4: Create Vector Store
def create_vector_store_from_spark(df, model_name, persist_directory):
    logger.info("Creating vector store from Spark DataFrame.")
    texts = df.select("text").rdd.flatMap(lambda x: x).collect()
    metadata = df.select("year", "section_number", "chunk_number").rdd.map(lambda row: row.asDict()).collect()
    
    logger.info("Initializing embeddings model.")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]
    logger.info("Creating and persisting vector database.")
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    return vectordb

# Step 5: Setup LLM Pipeline
def setup_llm_pipeline(model_name):
    logger.info(f"Setting up LLM pipeline with model: {model_name}.")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    query_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info("LLM pipeline setup complete.")
    return HuggingFacePipeline(pipeline=query_pipeline)

# Step 6: Create Year-Specific Query Pipeline
def create_year_specific_pipeline(year, vectordb, llm):
    logger.info(f"Creating year-specific query pipeline for year: {year}.")
    def retrieve_with_year_filter(query):
        logger.info(f"Retrieving documents for year: {year}.")
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        raw_results = retriever.get_relevant_documents(query)
        
        # Filter results by year
        filtered_results = [
            {"metadata": doc.metadata, "content": doc.page_content}
            for doc in raw_results if doc.metadata.get("year") == str(year)
        ]
        
        # Deduplicate chunks based on content
        unique_chunks = []
        seen_contents = set()
        for chunk in filtered_results:
            content = chunk["content"]
            if content not in seen_contents:
                unique_chunks.append(chunk)
                seen_contents.add(content)
        
        # Combine unique chunks into a single context string
        combined_context = "\n".join(
            f"Chunk Metadata: {chunk['metadata']} \nContent: {chunk['content']}"
            for chunk in unique_chunks[:1]
        )
        
        logger.info(f"Combined {len(unique_chunks)} unique chunks into context.")
        return combined_context  # Return combined context and metadata mapping


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
            {{
                "answer": "<Your Answer Here>"
            }}
    """

    year_specific_prompt = PromptTemplate.from_template(year_specific_template)
    return (
        {"context": retrieve_with_year_filter, "query": RunnablePassthrough()}
        | year_specific_prompt.partial(year=year)
        | llm
        # | StrOutputParser()
    )

def run_queries(year_specific_chain, queries):
    logger.info(f"Running multiple queries: {queries}")
    results = {}
    for query in queries:
        try:
            output = year_specific_chain.invoke(
                query,
                config={'temperature': 0.3, 'max_new_tokens': 200}
            )
            results[query] = output.strip()
        except Exception as e:
            logger.error(f"Error during query execution for '{query}': {str(e)}")
            results[query] = f"Error: {str(e)}"
    return results


# Main Execution
if __name__ == "__main__":
    logger.info("Main execution started.")
    YEARS = ['2018', '2019', '2020']
    COMPANY_CIK = '320193'  # Apple Inc.
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    PERSIST_DIR = "chroma_db"
    MODEL_TOKEN_LIMIT = 512  # Embedding model max token limit
    OVERLAP_RATIO = 0.1  # Overlap for splitting
    YEAR = 2020
    
    try:
        logger.info("Loading datasets.")
        filtered_data = load_filtered_datasets(YEARS, COMPANY_CIK)
        
        spark_df = create_spark_dataframe(filtered_data, spark)
        
        logger.info("Calculating chunking requirements.")
        chunking_requirements = calculate_chunking_requirements(spark_df, MODEL_TOKEN_LIMIT)
        
        logger.info("Processing and splitting data with chunking requirements.")
        processed_df = process_and_split_data(spark_df, chunking_requirements)
        
        logger.info("Creating vector store.")
        vectordb = create_vector_store_from_spark(processed_df, MODEL_NAME, PERSIST_DIR)
        
        logger.info("Setting up LLM pipeline.")
        llm = setup_llm_pipeline(LLM_MODEL_NAME)
        
        logger.info(f"Creating year-specific pipeline for year: {YEAR}.")
        year_specific_chain = create_year_specific_pipeline(YEAR, vectordb, llm)
        
        queries = [
            "How many total employees does Apple have?",
            "How much Total net sales increased from previous year in percent and in dollars?",
            "How did overall iPhone sales compare to the previous year? Did they increase or decrease?",
            "What are the exact values of the company's net deferred tax assets and deferred tax liabilities?",
            "What was the amount of the company's quarterly cash dividend per share?"
        ]

        # Run multiple queries
        output = run_queries(year_specific_chain, queries)

        # # Output JSON results
        # import json
        # results_json = json.dumps(output, indent=2)
        # print(results_json)

        logger.info(f"Query result: {output}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
