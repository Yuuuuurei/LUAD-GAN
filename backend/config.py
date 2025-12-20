"""
Configuration module for GAN-LUAD Clustering project.
Loads environment variables and defines project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"

# Model directories
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BASELINE_DIR = MODEL_DIR / "baseline"
GAN_ASSISTED_DIR = MODEL_DIR / "gan_assisted"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
GAN_RESULTS_DIR = RESULTS_DIR / "gan_assisted"
COMPARISON_DIR = RESULTS_DIR / "comparison"
VALIDATION_DIR = RESULTS_DIR / "validation"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
TRAINING_LOGS_DIR = LOGS_DIR / "training"
CLUSTERING_LOGS_DIR = LOGS_DIR / "clustering"

# Create directories if they don't exist
for directory in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, SAMPLE_DATA_DIR,
    CHECKPOINT_DIR, BASELINE_DIR, GAN_ASSISTED_DIR,
    BASELINE_RESULTS_DIR, GAN_RESULTS_DIR, COMPARISON_DIR, VALIDATION_DIR,
    TRAINING_LOGS_DIR, CLUSTERING_LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seeds (for reproducibility)
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
NUMPY_SEED = int(os.getenv("NUMPY_SEED", 42))
TORCH_SEED = int(os.getenv("TORCH_SEED", 42))

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Data URLs
TCGA_LUAD_URL = os.getenv(
    "TCGA_LUAD_URL",
    "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.star_tpm.tsv.gz"
)
TCGA_LUAD_CLINICAL_URL = os.getenv(
    "TCGA_LUAD_CLINICAL_URL",
    "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LUAD.clinical.tsv.gz"
)

# Data Configuration
TUMOR_SAMPLE_SUFFIX = "01A"  # Primary tumor samples
NORMAL_SAMPLE_SUFFIX = "11A"  # Normal tissue samples (to exclude)

# Feature Selection
N_TOP_VARIABLE_GENES = 2000  # Top N most variable genes to keep
MIN_VARIANCE_THRESHOLD = 0.01  # Minimum variance threshold

# PCA Configuration
N_PCA_COMPONENTS = 500  # Number of PCA components
PCA_VARIANCE_RATIO = 0.85  # Preserve 85% of variance

# GAN Training Configuration
LATENT_DIM = 128
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
LEARNING_RATE_G = float(os.getenv("LEARNING_RATE", 0.0001))
LEARNING_RATE_C = float(os.getenv("LEARNING_RATE", 0.0001))
N_CRITIC = 5  # Train critic 5 times per generator update
GRADIENT_PENALTY_WEIGHT = 10
MAX_EPOCHS = int(os.getenv("NUM_EPOCHS", 500))
EARLY_STOPPING_PATIENCE = 50

# Adam Optimizer Betas
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.9

# Augmentation
AUGMENTATION_RATIO = 2.0  # Generate 2x original data size

# Clustering Configuration
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
DEFAULT_N_CLUSTERS = 5

# Evaluation Metrics
CLUSTERING_ALGORITHMS = ["kmeans", "hierarchical", "spectral"]
DIMENSIONALITY_REDUCTION_METHODS = ["pca", "tsne", "umap"]

# Visualization
FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)

# Logging
LOG_INTERVAL = 10  # Log every N epochs
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N epochs

# File Naming Patterns
PROCESSED_DATA_FILE = "luad_processed.pt"
FEATURE_NAMES_FILE = "feature_names.txt"
SAMPLE_IDS_FILE = "sample_ids.txt"
PCA_TRANSFORMER_FILE = "pca_transformer.pkl"
METADATA_FILE = "metadata.json"
SYNTHETIC_DATA_FILE = "gan_generated_samples.pt"
AUGMENTED_DATA_FILE = "augmented_data.pt"

# Model Naming
BEST_MODEL_NAME = "wgan_gp_best.pt"
FINAL_MODEL_NAME = "wgan_gp_final.pt"

def get_checkpoint_path(epoch: int) -> Path:
    """Get checkpoint path for a specific epoch."""
    return CHECKPOINT_DIR / f"wgan_gp_epoch_{epoch}.pt"

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("GAN-LUAD Clustering Configuration")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()