# Document Analysis Pipeline for SEC 10-K Filings

This project implements a document analysis pipeline designed to help users understand SEC 10-K filings from the year 2020 in a two-dimensional space and identify outliers within the data. The pipeline processes the documents by chunking, embedding, scaling, dimensionality reduction, clustering, and visualization.

## Table of Contents

- [Document Analysis Pipeline for SEC 10-K Filings](#document-analysis-pipeline-for-sec-10-k-filings)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset Description](#dataset-description)
  - [Pipeline Overview](#pipeline-overview)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Detailed Explanation](#detailed-explanation)
    - [1. Preprocessing Data](#1-preprocessing-data)
    - [2. Chunking Documents](#2-chunking-documents)
    - [3. Generating Embeddings](#3-generating-embeddings)
    - [4. Scaling Embeddings](#4-scaling-embeddings)
    - [5. Dimensionality Reduction](#5-dimensionality-reduction)
    - [6. Clustering](#6-clustering)
    - [7. Outlier Detection](#7-outlier-detection)
    - [8. Visualization](#8-visualization)
  - [Improvements and Enhancements](#improvements-and-enhancements)
  - [Results](#results)

## Introduction

Analyzing large textual datasets like SEC filings requires robust data processing pipelines. This project provides a solution that processes 10-K filings from 2020, limited to 10 companies, and visualizes the documents in two dimensions, allowing users to identify clusters and outliers effectively.

## Dataset Description

- **Year:** 2020
- **Filing Type:** 10-K
- **Sections:** All sections of the filings
- **Companies:** Limited to 10 companies

The dataset is sourced from the Hugging Face `eloukas/edgar-corpus` dataset, specifically the 2020 filings.

## Pipeline Overview

The pipeline follows these steps:

1. **Preprocessing Data**
2. **Converting Documents to Chunks**
3. **Generating Embeddings for Chunks**
4. **Standard Scaling of Embeddings**
5. **Principal Component Analysis (PCA)**
6. **Dimensionality Reduction**
7. **KMeans Clustering**
8. **Outlier Detection**
9. **Visualization**

Each step is designed to process and analyze the data efficiently, resulting in visualizations that help users understand the underlying patterns in the documents.

## Prerequisites

- Python 3.10 or higher
- check requirments.txt

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/document-analysis-pipeline.git
   cd document-analysis-pipeline
   ```

2. **Create a virtual environment and activate the venv:**

   ```bash
   make venv
   source edgar-venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

## Usage

1. **Run the Script:**

   ```bash
   python task_1.py
   ```

2. **Monitor the Output:**

   - Logs will be displayed in the console and saved in the `experiment_<timestamp>/experiment.log` file.
   - Progress and any issues can be traced through the log messages.

3. **Review the Results:**

   - Visualizations are saved in the `plots` directory within the output folder.
   - The embeddings and metadata are saved as a CSV file for further analysis.

## Detailed Explanation

### 1. Preprocessing Data

**Objective:** Load the dataset, filter relevant filings, and limit to 10 companies.

**Process:**

- **Loading the Dataset:**
  - Use the Hugging Face `datasets` library to load the `eloukas/edgar-corpus` dataset for the year 2020.
  
    ```python
    dataset = datasets.load_dataset("eloukas/edgar-corpus", "year_2020", trust_remote_code=True)
    ```

- **Filtering Data:**
  - Filter out rows where 'cik' (Central Index Key) is null.
  - Limit the dataset to the first 10 companies.

    ```python
    filtered_dataset = dataset['train'].filter(lambda x: x['cik'] is not None)
    limited_dataset = filtered_dataset.select(range(10))
    ```

- **Conversion to pandas DataFrame:**
  - Convert the filtered dataset to a pandas DataFrame for easier manipulation.

    ```python
    df = limited_dataset.to_pandas()
    ```

**Logging Statements:**

- Start of data preprocessing.
- Dataset filtered to first 10 companies with non-null 'cik'.
- Conversion to pandas DataFrame with its shape.

### 2. Chunking Documents

**Objective:** Split each document into smaller chunks to handle large text data efficiently.

**Process:**

- **Identifying Sections:**
  - Extract all columns starting with 'section_' to process all sections of the filings.

    ```python
    section_cols = [col for col in data.columns if col.startswith('section_')]
    ```

- **Creating Chunks:**
  - For each document and section, split the text into chunks of 500 words with an overlap of 50 words.
  - The overlap ensures continuity between chunks.

    ```python
    def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
        # Function to split text into chunks
    ```

- **Collecting Chunks:**
  - Collect all chunks into a PySpark DataFrame with relevant metadata (CIK, year, section, chunk text, chunk ID).

    ```python
    chunk_df = self.spark.createDataFrame(chunk_rows, schema)
    ```

**Logging Statements:**

- Start of document chunking.
- Number of section columns found.
- Processing of each row with 'cik' and 'year'.
- Number of chunks created for each section.
- Total chunks created.

### 3. Generating Embeddings

**Objective:** Convert text chunks into numerical embeddings using a pre-trained model.

**Process:**

- **Loading the Embedding Model:**
  - Use the `all-MiniLM-L6-v2` model from SentenceTransformers for generating embeddings.

    ```python
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    ```

- **Encoding Chunks:**
  - Encode the chunk texts in batches to generate embeddings.
  - Batching helps manage memory usage and processing time.

    ```python
    for i in range(0, len(chunks), batch_size):
        batch_embeddings = self.model.encode(batch)
    ```

- **Creating Embedding DataFrame:**
  - Create a PySpark DataFrame that includes the embeddings along with the original metadata.

    ```python
    embedding_df = self.spark.createDataFrame(rows, schema)
    ```

**Logging Statements:**

- Start of embedding generation.
- Total number of chunks.
- Encoding progress for each batch.
- Completion of embedding generation.

### 4. Scaling Embeddings

**Objective:** Standardize the embeddings to have a mean of zero and unit variance.

**Process:**

- **Standard Scaling:**
  - Use `StandardScaler` from PySpark ML to scale the embeddings.

    ```python
    scaler = StandardScaler(inputCol="embeddings", outputCol="scaled_embeddings", withStd=True, withMean=True)
    scaled_df = scaler_model.transform(embedding_df)
    ```

**Logging Statements:**

- Performing standard scaling of embeddings.
- Scaling complete.

### 5. Dimensionality Reduction

**Objective:** Reduce the dimensionality of the embeddings to two principal components for visualization.

**Process:**

- **Principal Component Analysis (PCA):**
  - Apply PCA to reduce the high-dimensional embeddings to 2 dimensions.

    ```python
    pca = PCA(k=n_components, inputCol="scaled_embeddings", outputCol="pca_features")
    pca_df = pca_model.transform(scaled_df)
    ```

**Logging Statements:**

- Performing PCA to reduce dimensions to 2.
- PCA complete.

### 6. Clustering

**Objective:** Group similar chunks together using KMeans clustering.

**Process:**

- **KMeans Clustering:**
  - Cluster the data into 5 clusters based on the principal components.

    ```python
    kmeans = KMeans(k=n_clusters, featuresCol="pca_features", predictionCol="cluster")
    clustered_df = kmeans_model.transform(pca_df)
    ```

**Logging Statements:**

- Clustering data into 5 clusters using KMeans.
- Clustering complete.

### 7. Outlier Detection

**Objective:** Identify chunks that are outliers based on their distance from cluster centers.

**Process:**

- **Calculating Distances:**
  - Compute the Euclidean distance of each point to its cluster center.

    ```python
    distances = [np.linalg.norm(pca_features[i] - centers[clusters[i]]) for i in range(len(pca_features))]
    ```

- **Determining Threshold:**
  - Set a threshold based on the mean and standard deviation of distances.

    ```python
    threshold = np.mean(distances) + 2 * np.std(distances)
    ```

- **Flagging Outliers:**
  - Label points with distances exceeding the threshold as outliers.

    ```python
    outliers = [1 if d > threshold else 0 for d in distances]
    ```

**Logging Statements:**

- Calculating distances to cluster centers and identifying outliers.
- Number of outliers identified.

### 8. Visualization

**Objective:** Generate plots to visualize the data in two dimensions.

**Process:**

- **Creating a Visualization DataFrame:**
  - Combine principal components, cluster assignments, outlier flags, and section labels into a DataFrame.

    ```python
    viz_data = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
    viz_data['Cluster'] = clusters
    viz_data['Is_Outlier'] = outliers
    viz_data['Section'] = section_labels
    ```

- **Generating Plots:**
  - **Plot 1:** Embeddings in 2 dimensions.
  - **Plot 2:** Embeddings colored by assigned clusters.
  - **Plot 3:** Embeddings colored by outlier flag.
  - **Plot 4:** Embeddings colored by section number.

    ```python
    for plot in plot_settings:
        sns.scatterplot(...)
        plt.savefig(save_path)
    ```

- **Saving Results:**
  - Save plots to the `plots` directory.
  - Save embeddings and metadata as a CSV file.

**Logging Statements:**

- Starting visualization and saving results.
- Created output directories.
- Saved each plot with its path.
- Saved embeddings and metadata.
- Visualization and saving results completed.

## Improvements and Enhancements

- **Detailed Logging:**
  - Added granular logging at each step to facilitate debugging and traceability.
  - Included the logger name in log messages for clarity.

- **Class Encapsulation:**
  - Encapsulated the preprocessing function within the `DocumentAnalyzer` class for better organization.

- **Parameterization:**
  - Made parameters like chunk size, overlap, number of PCA components, and number of clusters configurable.

- **Output Organization:**
  - Organized outputs into timestamped directories for easy tracking and comparison between runs.

- **Visualization Enhancements:**
  - Improved plot aesthetics with clear titles, labels, and legends.

- **Error Handling:**
  - Ensured that exceptions are logged, and the program exits gracefully if errors occur.

## Results

After running the script, an output directory named `experiment_<timestamp>/` is created, containing:

- **Log File (`experiment.log`):**
  - Comprehensive logs detailing the execution process.

- **Plots (`plots/`):**
  - `Embeddings_in_2_Dimensions.png`
  - `Embeddings_in_2D_Colored_by_Cluster.png`
  - `Embeddings_in_2D_Colored_by_Outlier_Flag.png`
  - `Embeddings_in_2D_Colored_by_Section.png`

- **Embeddings Metadata (`embeddings_metadata.csv`):**
  - CSV file containing the principal components, cluster assignments, outlier flags, and section labels.

These results provide valuable insights into the structure and patterns within the 10-K filings of the selected companies, facilitating better understanding and analysis.
