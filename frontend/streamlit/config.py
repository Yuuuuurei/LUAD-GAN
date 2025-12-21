# Frontend configuration for Streamlit app

# Help texts for different sections
HELP_TEXTS = {
    'upload': """
    **Data Upload Instructions:**
    - Upload a TSV file containing gene expression data
    - The file should have genes as rows and samples as columns
    - First column should contain gene names/IDs
    - Data will be validated and preprocessed automatically
    """,
    'clustering': """
    **Baseline Clustering:**
    - Select a clustering algorithm from the available options
    - Choose the number of clusters (K) to identify
    - The algorithm will group similar samples based on gene expression patterns
    - Results will be visualized and evaluated
    """,
    'gan_training': """
    **GAN Training:**
    - Train a Wasserstein GAN with Gradient Penalty (WGAN-GP)
    - The GAN will learn to generate synthetic gene expression data
    - Training parameters can be adjusted for optimal performance
    - Monitor training progress and loss curves
    """
}

# Error messages
ERROR_MESSAGES = {
    'no_data': "No data available. Please upload data first.",
    'invalid_file': "Invalid file format. Please upload a valid TSV file.",
    'training_failed': "Training failed. Please check the logs for details.",
    'clustering_failed': "Clustering failed. Please check the parameters."
}

# Success messages
SUCCESS_MESSAGES = {
    'data_uploaded': "Data uploaded and processed successfully!",
    'training_completed': "GAN training completed successfully!",
    'clustering_done': "Clustering completed successfully!"
}

# Clustering algorithms
CLUSTERING_ALGORITHMS = ["kmeans", "hierarchical", "spectral"]

# Default values for clustering
DEFAULT_K = 5
DEFAULT_K_RANGE = list(range(2, 11))  # 2 to 10

# Default values for data upload
DEFAULT_TOP_N_GENES = 2000
DEFAULT_PCA_COMPONENTS = 500

# Default values for GAN training
DEFAULT_LATENT_DIM = 128
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 500
DEFAULT_N_CRITIC = 5
DEFAULT_GP_WEIGHT = 10

def get_algorithm_description(algorithm: str) -> str:
    """Get description for a clustering algorithm."""
    descriptions = {
        "kmeans": """
        **K-Means Clustering:**
        - Partitions data into K clusters
        - Each sample belongs to the cluster with the nearest mean
        - Fast and scalable for large datasets
        - Assumes spherical clusters of similar size
        """,
        "hierarchical": """
        **Hierarchical Clustering:**
        - Builds a hierarchy of clusters
        - Can be agglomerative (bottom-up) or divisive (top-down)
        - Produces a dendrogram showing cluster relationships
        - Good for understanding cluster structure
        """,
        "spectral": """
        **Spectral Clustering:**
        - Uses eigenvalues of similarity matrix
        - Can find non-convex clusters
        - Effective for complex cluster shapes
        - Computationally more expensive
        """
    }
    return descriptions.get(algorithm, "No description available.")

def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file format and content."""
    if uploaded_file is None:
        return False, "No file uploaded."

    # Check file extension
    if not uploaded_file.name.endswith('.tsv'):
        return False, "File must be a TSV file."

    # Check file size (basic check)
    if uploaded_file.size == 0:
        return False, "File is empty."

    # Additional validation can be added here
    return True, "File is valid."