import streamlit as st
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.config import *
from backend.clustering.algorithms import ClusteringPipeline
from backend.clustering.evaluation import ClusteringEvaluator

st.title("🎨 GAN-Assisted Clustering")
st.markdown("---")

# Check prerequisites
if not st.session_state.get('synthetic_data'):
    st.error("⚠️ No synthetic data. Please complete GAN training first.")
    st.stop()

# Configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("Augmentation Strategy")
    strategy = st.selectbox(
        "Strategy",
        list(AUGMENTATION_STRATEGIES.keys()),
        format_func=lambda x: AUGMENTATION_STRATEGIES[x]
    )

with col2:
    st.subheader("Clustering")
    algorithm = st.selectbox("Algorithm", CLUSTERING_ALGORITHMS)
    k = st.slider("Number of clusters (k)", 2, 10, 3)

# Run clustering
if st.button("🚀 Run GAN-Assisted Clustering", type="primary"):
    with st.spinner("Clustering augmented data..."):
        # Combine data based on strategy
        real = st.session_state.processed_data.numpy()
        synthetic = st.session_state.synthetic_data.numpy()
        
        if strategy == 'add':
            data = np.vstack([real, synthetic])
            labels = np.array([0]*len(real) + [1]*len(synthetic))
        elif strategy == 'mixed':
            n_half = len(real) // 2
            data = np.vstack([real[:n_half], synthetic[:n_half]])
            labels = np.array([0]*n_half + [1]*n_half)
        else:  # replace
            data = synthetic
            labels = np.array([1]*len(synthetic))
        
        # Run clustering
        pipeline = ClusteringPipeline(data, labels, scale_data=True)
        
        if algorithm == 'kmeans':
            result = pipeline.kmeans_clustering(k)
        elif algorithm == 'hierarchical':
            result = pipeline.hierarchical_clustering(k)
        else:
            result = pipeline.spectral_clustering(k)
        
        # Evaluate
        evaluator = ClusteringEvaluator(pipeline.data_scaled, result['labels'])
        metrics = evaluator.compute_all_metrics()
        
        result['metrics'] = metrics
        
        # Extract real sample clusters
        real_labels = pipeline.extract_real_sample_clusters(result['labels'])
        
        # Store
        st.session_state.gan_results = {k: result}
        st.session_state.gan_algorithm = algorithm
        st.session_state.gan_real_labels = real_labels
        
        st.success("✅ GAN-assisted clustering complete!")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette", f"{metrics['silhouette_score']:.4f}")
        with col2:
            st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_index']:.4f}")
        with col3:
            st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.2f}")

# Show results if available
if 'gan_results' in st.session_state:
    st.markdown("---")
    st.success("✅ Results ready! Go to **📈 Results & Validation**")