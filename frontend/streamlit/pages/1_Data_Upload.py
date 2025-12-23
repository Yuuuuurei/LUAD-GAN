"""
Page 1: Data Upload & Preprocessing
Streamlit MVP - Phase 10

Upload TCGA-LUAD data and apply preprocessing steps.

Author: GAN-LUAD Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.config import (
    HELP_TEXTS, ERROR_MESSAGES, SUCCESS_MESSAGES,
    DEFAULT_TOP_N_GENES, DEFAULT_PCA_COMPONENTS,
    validate_uploaded_file
)

st.title("📁 Data Upload & Preprocessing")
st.markdown("---")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'gene_names' not in st.session_state:
    st.session_state.gene_names = None
if 'sample_ids' not in st.session_state:
    st.session_state.sample_ids = None
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

# Sidebar
with st.sidebar:
    st.header("📖 Help")
    with st.expander("How to use this page"):
        st.markdown(HELP_TEXTS['upload'])
    
    st.markdown("---")
    
    # Quick stats if data loaded
    if st.session_state.data_ready:
        st.success("✅ Data Ready")
        st.metric("Samples", st.session_state.get('n_samples', 'N/A'))
        st.metric("Features", st.session_state.get('n_features', 'N/A'))

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload", "⚙️ Preprocess", "👁️ Preview"])

# ============================================================================
# TAB 1: Upload
# ============================================================================
with tab1:
    st.header("Upload Gene Expression Data")
    
    st.info("""
    **Expected format:**
    - TSV or CSV file
    - Rows: Genes (Ensembl IDs or gene names)
    - Columns: Samples (TCGA barcodes)
    - Values: Expression levels (already log-transformed)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['tsv', 'txt', 'csv'],
        help="Upload your TCGA-LUAD gene expression data"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_msg = validate_uploaded_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            try:
                with st.spinner("Loading data..."):
                    # Determine separator
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    sep = '\t' if file_ext in ['.tsv', '.txt'] else ','
                    
                    # Read data
                    df = pd.read_csv(uploaded_file, sep=sep, index_col=0)
                    
                    st.success(SUCCESS_MESSAGES['data_uploaded'])
                    
                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Genes (rows)", df.shape[0])
                    with col2:
                        st.metric("Samples (columns)", df.shape[1])
                    with col3:
                        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                        st.metric("Missing Values", f"{missing_pct:.2f}%")
                    
                    # Data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Store raw data in session state
                    st.session_state.raw_data = df
                    st.session_state.raw_gene_names = df.index.tolist()
                    st.session_state.raw_sample_ids = df.columns.tolist()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Demo data option
    st.markdown("---")
    if st.button("📦 Load Demo Data (Small Sample)", help="Load a small sample dataset for testing"):
        st.info("Demo data feature - to be implemented with sample dataset")

# ============================================================================
# TAB 2: Preprocessing
# ============================================================================
with tab2:
    st.header("Preprocessing Options")
    
    if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
        st.warning(ERROR_MESSAGES['no_data'])
    else:
        df_raw = st.session_state.raw_data
        
        st.info(f"""
        **Current data:**
        - Genes: {df_raw.shape[0]:,}
        - Samples: {df_raw.shape[1]:,}
        """)
        
        # Preprocessing options
        st.subheader("1. Sample Filtering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            filter_samples = st.checkbox(
                "Filter samples by barcode pattern",
                value=True,
                help="Keep only primary tumor samples (ending in -01A)"
            )
        
        with col2:
            if filter_samples:
                sample_pattern = st.text_input(
                    "Pattern to keep",
                    value="-01A",
                    help="Keep samples containing this pattern"
                )
        
        st.markdown("---")
        st.subheader("2. Feature Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_selection_method = st.selectbox(
                "Method",
                ["Variance-based", "All genes"],
                help="How to select genes for analysis"
            )
        
        with col2:
            if feature_selection_method == "Variance-based":
                top_n_genes = st.number_input(
                    "Number of top variable genes",
                    min_value=100,
                    max_value=df_raw.shape[0],
                    value=min(DEFAULT_TOP_N_GENES, df_raw.shape[0]),
                    step=100,
                    help="Keep this many most variable genes"
                )
            else:
                top_n_genes = df_raw.shape[0]
        
        st.markdown("---")
        st.subheader("3. Dimensionality Reduction (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            apply_pca = st.checkbox(
                "Apply PCA",
                value=False,
                help="Reduce dimensions using PCA before clustering"
            )
        
        with col2:
            if apply_pca:
                max_components = min(500, df_raw.shape[1])
                pca_components = st.number_input(
                    "Number of components",
                    min_value=10,
                    max_value=max_components,
                    value=min(DEFAULT_PCA_COMPONENTS, max_components),
                    step=10,
                    help="Number of principal components to keep"
                )
        
        st.markdown("---")
        
        # Process button
        if st.button("🔄 Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    # Start with raw data
                    df_processed = df_raw.copy()
                    
                    # 1. Sample filtering
                    if filter_samples:
                        original_samples = len(df_processed.columns)
                        mask = df_processed.columns.str.contains(sample_pattern)
                        df_processed = df_processed.loc[:, mask]
                        filtered_samples = len(df_processed.columns)
                        st.info(f"Sample filtering: {original_samples} → {filtered_samples} samples")
                    
                    # 2. Feature selection
                    if feature_selection_method == "Variance-based":
                        original_genes = len(df_processed)
                        
                        # Calculate variance
                        variances = df_processed.var(axis=1)
                        
                        # Sort by variance and keep top N
                        top_genes_idx = variances.nlargest(top_n_genes).index
                        df_processed = df_processed.loc[top_genes_idx]
                        
                        st.info(f"Feature selection: {original_genes:,} → {len(df_processed):,} genes")
                    
                    # 3. Convert to numpy and prepare for storage
                    data_array = df_processed.values.T  # Transpose to (samples, genes)
                    gene_names = df_processed.index.tolist()
                    sample_ids = df_processed.columns.tolist()
                    
                    # 4. Optional PCA
                    if apply_pca:
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        
                        # Standardize
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data_array)
                        
                        # Apply PCA
                        pca = PCA(n_components=pca_components)
                        data_pca = pca.fit_transform(data_scaled)
                        
                        explained_var = pca.explained_variance_ratio_.sum()
                        st.info(f"PCA: {data_array.shape[1]} → {pca_components} components ({explained_var*100:.1f}% variance)")
                        
                        # Use PCA data
                        data_array = data_pca
                    
                    # Store in session state
                    st.session_state.processed_data = torch.FloatTensor(data_array)
                    st.session_state.gene_names = gene_names
                    st.session_state.sample_ids = sample_ids
                    st.session_state.n_samples = data_array.shape[0]
                    st.session_state.n_features = data_array.shape[1]
                    st.session_state.data_ready = True
                    st.session_state.pca_applied = apply_pca
                    
                    st.success(SUCCESS_MESSAGES['data_processed'])
                    
                    # Show final statistics
                    st.subheader("✅ Processing Complete")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Samples", st.session_state.n_samples)
                    with col2:
                        st.metric("Final Features", st.session_state.n_features)
                    with col3:
                        st.metric("Status", "✅ Ready")
                    
                    # Show data range
                    data_min = data_array.min()
                    data_max = data_array.max()
                    data_mean = data_array.mean()
                    
                    st.info(f"""
                    **Data Statistics:**
                    - Range: [{data_min:.3f}, {data_max:.3f}]
                    - Mean: {data_mean:.3f}
                    - Shape: {data_array.shape}
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# TAB 3: Preview
# ============================================================================
with tab3:
    st.header("Processed Data Preview")
    
    if not st.session_state.data_ready:
        st.warning("⏳ No processed data available. Please preprocess data first.")
    else:
        st.success("✅ Data ready for analysis")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", st.session_state.n_samples)
        with col2:
            st.metric("Features", st.session_state.n_features)
        with col3:
            data_array = st.session_state.processed_data.numpy()
            st.metric("Mean", f"{data_array.mean():.3f}")
        with col4:
            st.metric("Std Dev", f"{data_array.std():.3f}")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("Sample Preview (First 10 samples, First 10 features)")
        
        # Convert to DataFrame for display
        preview_data = data_array[:10, :10]
        preview_df = pd.DataFrame(
            preview_data,
            index=[f"Sample {i+1}" for i in range(len(preview_data))],
            columns=[f"Feature {i+1}" for i in range(preview_data.shape[1])]
        )
        
        st.dataframe(preview_df, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution plot
        st.subheader("Expression Distribution")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data_array.flatten(),
            nbinsx=50,
            name="Expression Values"
        ))
        
        fig.update_layout(
            title="Distribution of Expression Values",
            xaxis_title="Expression Level",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Sample IDs
        with st.expander("📋 View Sample IDs"):
            st.write(st.session_state.sample_ids)
        
        # Gene names
        if not st.session_state.get('pca_applied', False):
            with st.expander("🧬 View Gene Names"):
                st.write(st.session_state.gene_names[:100])  # Show first 100
                if len(st.session_state.gene_names) > 100:
                    st.info(f"Showing first 100 of {len(st.session_state.gene_names)} genes")

# Navigation
st.markdown("---")
if st.session_state.data_ready:
    st.success("✅ Data is ready! Proceed to the next page to run baseline clustering.")
    st.info("👉 Use the sidebar to navigate to **📊 Baseline Clustering**")
else:
    st.warning("⏳ Please upload and preprocess data before proceeding.")