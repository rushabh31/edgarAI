import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import os
import time
import datasets
from datasets import DatasetDict


# DocumentAnalyzer class
class DocumentAnalyzer:
    def __init__(self, spark_session=None, logger=None):
        """Initialize the DocumentAnalyzer."""
        self.spark = spark_session or SparkSession.builder \
            .appName("Document_Analysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
        self.logger = logger or logging.getLogger('DocumentAnalyzer')
        self.logger.info("Initialized DocumentAnalyzer.")

    def preprocess_data(self, dataset: DatasetDict) -> pd.DataFrame:
        """
        Preprocess the input Hugging Face DatasetDict:
        - Convert the 'train' split to a pandas DataFrame.
        - Filter rows where 'cik' is not null.
        - Limit to the first 10 rows.

        Args:
        dataset (DatasetDict): The Hugging Face DatasetDict object.

        Returns:
        pd.DataFrame: Preprocessed pandas DataFrame.
        """
        self.logger.info("Starting data preprocessing...")
        # Filter and limit rows using Dataset API
        filtered_dataset = dataset['train'].filter(lambda x: x['cik'] is not None)
        limited_dataset = filtered_dataset.select(range(10))
        self.logger.info("Filtered dataset to first 10 rows with non-null 'cik'.")

        # Convert to pandas DataFrame
        df = limited_dataset.to_pandas()
        self.logger.info(f"Converted dataset to pandas DataFrame with shape {df.shape}.")
        return df

    def chunk_documents(self, data, model_token_limit: int = 512, overlap_ratio: float = 0.1):
        """Dynamically split documents into chunks with overlap based on token limits."""
        self.logger.info("Starting dynamic document chunking...")

        # Identify section columns
        section_cols = [col for col in data.columns if col.startswith('section_')]
        self.logger.info(f"Found {len(section_cols)} section columns for chunking.")

        def calculate_chunking(text: str, token_limit: int, overlap_ratio: float) -> List[str]:
            """Calculate chunks dynamically based on token limit and overlap ratio."""
            words = text.split()
            token_count = len(words)
            chunks = []

            if token_count <= token_limit:
                # No chunking required
                return [text]
            
            chunk_size = token_limit
            overlap = int(chunk_size * overlap_ratio)
            start = 0

            while start < len(words):
                end = start + chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
                start = end - overlap  # Move forward with overlap

            return chunks

        schema = StructType([
            StructField("cik", StringType(), True),
            StructField("year", StringType(), True),
            StructField("section", StringType(), True),
            StructField("chunk_text", StringType(), True),
            StructField("chunk_id", IntegerType(), True)
        ])

        chunk_rows = []
        for row in data.collect():
            self.logger.info(f"Processing row with cik: {row['cik']}, year: {row['year']}")
            for section in section_cols:
                if row[section] and len(str(row[section])) > 0:
                    text = str(row[section])
                    chunks = calculate_chunking(text, model_token_limit, overlap_ratio)
                    self.logger.info(f"Created {len(chunks)} chunks for section {section}.")
                    for i, chunk in enumerate(chunks):
                        chunk_rows.append((row['cik'], row['year'], section, chunk, i))

        self.logger.info(f"Total chunks created: {len(chunk_rows)}")
        return self.spark.createDataFrame(chunk_rows, schema)


    def create_embeddings(self, chunk_df, batch_size: int = 32):
        """Generate embeddings for chunks."""
        self.logger.info("Starting embedding generation...")
        chunks_data = chunk_df.collect()
        chunks = [row['chunk_text'] for row in chunks_data]
        self.logger.info(f"Total number of chunks: {len(chunks)}")

        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Encoding batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")
            batch_embeddings = self.model.encode(batch)
            all_embeddings.extend(batch_embeddings)

        rows = []
        for i, row in enumerate(chunks_data):
            embedding_vector = Vectors.dense(all_embeddings[i])
            rows.append((row['cik'], row['year'], row['section'], row['chunk_text'], row['chunk_id'], embedding_vector))

        schema = StructType([
            StructField("cik", StringType(), True),
            StructField("year", StringType(), True),
            StructField("section", StringType(), True),
            StructField("chunk_text", StringType(), True),
            StructField("chunk_id", IntegerType(), True),
            StructField("embeddings", VectorUDT(), True)
        ])

        self.logger.info("Completed embedding generation.")
        return self.spark.createDataFrame(rows, schema)

    def process_embeddings(self, embedding_df, n_components: int = 2, n_clusters: int = 5):
        """Process embeddings: scale, reduce dimensionality, cluster, and detect outliers."""
        self.logger.info("Starting processing of embeddings...")
        # Standard scaling
        self.logger.info("Performing standard scaling of embeddings...")
        scaler = StandardScaler(inputCol="embeddings", outputCol="scaled_embeddings",
                                withStd=True, withMean=True)
        scaler_model = scaler.fit(embedding_df)
        scaled_df = scaler_model.transform(embedding_df)
        self.logger.info("Scaling complete.")

        # PCA for dimensionality reduction
        self.logger.info(f"Performing PCA to reduce dimensions to {n_components}...")
        pca = PCA(k=n_components, inputCol="scaled_embeddings", outputCol="pca_features")
        pca_model = pca.fit(scaled_df)
        pca_df = pca_model.transform(scaled_df)
        self.logger.info("PCA complete.")

        # KMeans Clustering
        self.logger.info(f"Clustering data into {n_clusters} clusters using KMeans...")
        kmeans = KMeans(k=n_clusters, featuresCol="pca_features", predictionCol="cluster")
        kmeans_model = kmeans.fit(pca_df)
        clustered_df = kmeans_model.transform(pca_df)
        self.logger.info("Clustering complete.")

        # Calculate distances and identify outliers
        self.logger.info("Calculating distances to cluster centers and identifying outliers...")
        pca_features = np.array([row.pca_features.toArray() for row in clustered_df.select("pca_features").collect()])
        clusters = np.array([row.cluster for row in clustered_df.select("cluster").collect()])
        centers = np.array(kmeans_model.clusterCenters())

        distances = [np.linalg.norm(pca_features[i] - centers[clusters[i]]) for i in range(len(pca_features))]
        threshold = np.mean(distances) + 2 * np.std(distances)
        outliers = [1 if d > threshold else 0 for d in distances]
        num_outliers = sum(outliers)
        self.logger.info(f"Identified {num_outliers} outliers out of {len(distances)} data points.")

        # Add distances and outliers to DataFrame
        result_df = clustered_df.withColumn("distance_to_center", F.lit(distances))
        result_df = result_df.withColumn("is_outlier", F.lit(outliers))

        self.logger.info("Processing of embeddings complete.")
        return result_df, pca_features, distances, outliers

    def visualize_and_save_results(self, pca_features, clusters, outliers, section_labels, output_dir):
        """Visualize and save results."""
        self.logger.info("Starting visualization and saving results...")
        # Create directories for saving results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        self.logger.info(f"Created output directories at {output_dir}")

        # Create a DataFrame for visualization and saving
        viz_data = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
        viz_data['Cluster'] = clusters
        viz_data['Is_Outlier'] = outliers
        viz_data['Section'] = section_labels

        # Define plot settings
        plot_settings = [
            {
                'title': "Embeddings in 2 Dimensions",
                'filename': "Embeddings_in_2_Dimensions.png",
                'hue': None,
                'palette': None
            },
            {
                'title': "Embeddings in 2D: Colored by Cluster",
                'filename': "Embeddings_in_2D_Colored_by_Cluster.png",
                'hue': 'Cluster',
                'palette': 'viridis'
            },
            {
                'title': "Embeddings in 2D: Colored by Outlier Flag",
                'filename': "Embeddings_in_2D_Colored_by_Outlier_Flag.png",
                'hue': 'Is_Outlier',
                'palette': {0: 'blue', 1: 'red'}
            },
            {
                'title': "Embeddings in 2D: Colored by Section",
                'filename': "Embeddings_in_2D_Colored_by_Section.png",
                'hue': 'Section',
                'palette': 'tab20'
            }
        ]

        for plot in plot_settings:
            plt.figure(figsize=(15, 10))
            sns.scatterplot(x='PC1', y='PC2', hue=plot['hue'], data=viz_data,
                            palette=plot['palette'], alpha=0.7)
            plt.title(plot['title'])
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            if plot['hue'] is not None:
                plt.legend(title=plot['hue'])
            else:
                plt.legend().remove()
            save_path = os.path.join(plots_dir, plot['filename'])
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved plot: {save_path}")

        # Save embeddings and metadata as CSV
        embeddings_path = os.path.join(output_dir, "embeddings_metadata.csv")
        viz_data.to_csv(embeddings_path, index=False)
        self.logger.info(f"Saved embeddings and metadata to: {embeddings_path}")

        self.logger.info("Visualization and saving results completed.")
        return viz_data

# Main Execution
if __name__ == "__main__":
    # Define output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Added %(name)s to include logger name
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('Main')
    logger.info("Starting the document analysis process.")

    # Load the 2020 dataset
    logger.info("Loading the 2020 dataset...")
    dataset = datasets.load_dataset("eloukas/edgar-corpus", "year_2020", trust_remote_code=True)

    # Initialize Spark and DocumentAnalyzer
    logger.info("Initializing Spark session and DocumentAnalyzer...")
    spark = SparkSession.builder.appName("Document_Analysis").getOrCreate()
    analyzer = DocumentAnalyzer(spark_session=spark, logger=logging.getLogger('DocumentAnalyzer'))
    logger.info("Spark session and DocumentAnalyzer initialized.")

    # Preprocess Data
    df = analyzer.preprocess_data(dataset)
    logger.info("Dataset loaded and preprocessed.")

    # Convert to PySpark DataFrame
    logger.info("Converting pandas DataFrame to PySpark DataFrame...")
    spark_df = spark.createDataFrame(df)
    logger.info("Conversion complete.")

    # Analyze Documents
    logger.info("Starting document analysis...")
    chunk_df = analyzer.chunk_documents(spark_df, model_token_limit=512, overlap_ratio=0.1)
    embedding_df = analyzer.create_embeddings(chunk_df)
    processed_df, pca_features, distances, outliers = analyzer.process_embeddings(embedding_df, n_components=2, n_clusters=5)
    logger.info("Document analysis complete.")

    # Visualization
    logger.info("Starting visualization of results...")
    sections = [row.section.replace("section_", "") for row in chunk_df.collect()]
    clusters = [row.cluster for row in processed_df.select("cluster").collect()]
    viz_data = analyzer.visualize_and_save_results(pca_features, clusters, outliers, sections, output_dir)
    logger.info("Visualization complete.")

    logger.info("Document analysis process finished.")
