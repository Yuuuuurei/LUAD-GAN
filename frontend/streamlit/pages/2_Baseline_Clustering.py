"""
Page 2: Baseline Clustering
Streamlit MVP - Phase 10

Run clustering on original data to establish baseline.

Author: GAN-LUAD Team
Date: 2025
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.config import (
    HELP_TEXTS, ERROR_MESSAGES, SUCCESS_MESSAGES,
    CLUSTERING_ALGORITHMS, DEFAULT_K, DEFAULT_K_RANGE,
    get_algorithm_description
)
from backend.clustering.algorithms import ClusteringPipeline
from backend.clustering.evaluation import ClusteringEvaluator
from backend.clustering.visualization import ClusterVisualizer

st.title("📊 Baseline Clustering")
st.markdown("---")

# Check if data is ready
if not st.session_state.get('data_ready', False):
    st.error(ERROR_MESSAGES['no_data'])
    st.info("👈 Please go to **📁 Data Upload** page first to upload and preprocess data.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📖 Help")
    with st.expander("About Baseline Clustering"):
        st.markdown(HELP_TEXTS['clustering'])
    
    st.markdown("---")
    
    # Current data info
    st.info(f"""
    **Current Data:**
    - Samples: {st.session_state.n_samples}
    - Features: {st.session_state.n_features}
    """)

# Main content
tab1, tab2 = st.tabs(["⚙️ Configure & Run", "📈 Results"])

# ============================================================================
# TAB 1: Configure & Run
# ============================================================================
with tab1:
    st.header("Clustering Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Algorithm")
        
        algorithm = st.selectbox(
            "Select clustering algorithm",
            options=CLUSTERING_ALGORITHMS,
            format_func=lambda x: x.upper(),
            help="Choose clustering algorithm"
        )
        
        st.info(get_algorithm_description(algorithm))
        
    with col2:
        st.subheader("Number of Clusters")
        
        k_mode = st.radio(
            "Selection mode",
            ["Single k", "Multiple k values"],
            help="Test a single k or multiple k values"
        )
        
        if k_mode == "Single k":
            k_value = st.slider(
                "k (number of clusters)",
                min_value=2,
                max_value=10,
                value=DEFAULT_K,
                help="Number of clusters to identify"
            )
            k_values = [k_value]
        else:
            k_min = st.number_input("Minimum k", min_value=2, max_value=10, value=2)
            k_max = st.number_input("Maximum k", min_value=k_min, max_value=10, value=10)
            k_values = list(range(k_min, k_max + 1))
            st.info(f"Will test k values: {k_values}")
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        scale_data = st.checkbox(
            "Standardize features",
            value=True,
            help="Scale features to mean=0, std=1"
        )
        
        if algorithm == 'hierarchical':
            linkage = st.selectbox(
                "Linkage method",
                ['ward', 'complete', 'average', 'single'],
                help="Method for calculating cluster distances"
            )
        elif algorithm == 'spectral':
            affinity = st.selectbox(
                "Affinity",
                ['rbf', 'nearest_neighbors'],
                help="Kernel function for spectral clustering"
            )
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 Run Baseline Clustering", type="primary", use_container_width=True):
        
        with st.spinner(f"Running {algorithm.upper()} clustering..."):
            try:
                # Get data
                data = st.session_state.processed_data.numpy()
                
                # Initialize pipeline
                pipeline = ClusteringPipeline(
                    data=data,
                    sample_labels=None,
                    scale_data=scale_data,
                    random_state=42
                )
                
                # Run clustering for all k values
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, k in enumerate(k_values):
                    status_text.text(f"Clustering with k={k}...")
                    
                    # Run clustering
                    if algorithm == 'kmeans':
                        result = pipeline.kmeans_clustering(n_clusters=k)
                    elif algorithm == 'hierarchical':
                        result = pipeline.hierarchical_clustering(
                            n_clusters=k,
                            linkage=linkage if 'linkage' in locals() else 'ward'
                        )
                    elif algorithm == 'spectral':
                        result = pipeline.spectral_clustering(
                            n_clusters=k,
                            affinity=affinity if 'affinity' in locals() else 'rbf'
                        )
                    
                    # Evaluate
                    evaluator = ClusteringEvaluator(pipeline.data_scaled, result['labels'])
                    metrics = evaluator.compute_all_metrics()
                    
                    result['metrics'] = metrics
                    results[k] = result
                    
                    progress_bar.progress((i + 1) / len(k_values))
                
                status_text.empty()
                progress_bar.empty()
                
                # Store results
                st.session_state.baseline_results = results
                st.session_state.baseline_algorithm = algorithm
                st.session_state.baseline_pipeline = pipeline
                
                st.success(SUCCESS_MESSAGES['baseline_complete'])
                
                # Show quick summary
                st.subheader("✅ Clustering Complete")
                
                # Find best k (by silhouette score)
                best_k = max(
                    results.keys(),
                    key=lambda k: results[k]['metrics'].get('silhouette_score', -1)
                )
                
                st.info(f"""
                **Best k (by Silhouette Score):** {best_k}
                - Silhouette: {results[best_k]['metrics']['silhouette_score']:.4f}
                - Davies-Bouldin: {results[best_k]['metrics']['davies_bouldin_index']:.4f}
                - Calinski-Harabasz: {results[best_k]['metrics']['calinski_harabasz_score']:.2f}
                """)
                
                st.info("👉 Switch to **📈 Results** tab to view detailed results")
                
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# ============================================================================
# TAB 2: Results
# ============================================================================
with tab2:
    st.header("Baseline Clustering Results")
    
    if 'baseline_results' not in st.session_state:
        st.warning("⏳ No results yet. Please run clustering first.")
    else:
        results = st.session_state.baseline_results
        algorithm = st.session_state.baseline_algorithm
        pipeline = st.session_state.baseline_pipeline
        
        st.success(f"✅ Results available for {len(results)} k values")
        
        # Metrics comparison table
        st.subheader("📊 Metrics Comparison")
        
        import pandas as pd
        
        metrics_data = []
        for k, result in results.items():
            metrics_data.append({
                'k': k,
                'Silhouette': result['metrics'].get('silhouette_score', None),
                'Davies-Bouldin': result['metrics'].get('davies_bouldin_index', None),
                'Calinski-Harabasz': result['metrics'].get('calinski_harabasz_score', None),
                'WCSS': result['metrics'].get('wcss', None)
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Highlight best values
        st.dataframe(
            df_metrics.style.highlight_max(
                subset=['Silhouette', 'Calinski-Harabasz'],
                color='lightgreen'
            ).highlight_min(
                subset=['Davies-Bouldin', 'WCSS'],
                color='lightgreen'
            ),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Select k for detailed view
        st.subheader("🔍 Detailed View")
        
        selected_k = st.selectbox(
            "Select k to visualize",
            options=list(results.keys()),
            format_func=lambda x: f"k = {x}"
        )
        
        selected_result = results[selected_k]
        selected_metrics = selected_result['metrics']
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Silhouette Score",
                f"{selected_metrics.get('silhouette_score', 0):.4f}",
                help="Higher is better"
            )
        
        with col2:
            st.metric(
                "Davies-Bouldin",
                f"{selected_metrics.get('davies_bouldin_index', 0):.4f}",
                help="Lower is better"
            )
        
        with col3:
            st.metric(
                "Calinski-Harabasz",
                f"{selected_metrics.get('calinski_harabasz_score', 0):.2f}",
                help="Higher is better"
            )
        
        with col4:
            st.metric(
                "WCSS",
                f"{selected_metrics.get('wcss', 0):.2f}",
                help="Within-cluster sum of squares"
            )
        
        st.markdown("---")
        
        # Cluster sizes
        st.subheader("📊 Cluster Distribution")
        
        cluster_sizes = selected_result['cluster_sizes']
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(cluster_sizes.keys()),
                y=list(cluster_sizes.values()),
                text=list(cluster_sizes.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Cluster Sizes (k={selected_k})",
            xaxis_title="Cluster",
            yaxis_title="Number of Samples",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization
        st.subheader("📈 Cluster Visualization")
        
        viz_type = st.radio(
            "Visualization method",
            ["PCA", "t-SNE"],
            horizontal=True,
            help="Method for dimensionality reduction"
        )
        
        with st.spinner(f"Creating {viz_type} visualization..."):
            try:
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE
                import plotly.express as px
                
                data = pipeline.data_scaled
                labels = selected_result['labels']
                
                # Reduce to 2D
                if viz_type == "PCA":
                    reducer = PCA(n_components=2, random_state=42)
                    data_2d = reducer.fit_transform(data)
                    var_exp = reducer.explained_variance_ratio_
                    title = f"PCA Visualization (k={selected_k}, {algorithm.upper()})<br>PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%"
                else:  # t-SNE
                    # Subsample if too many samples
                    if len(data) > 1000:
                        indices = np.random.choice(len(data), 1000, replace=False)
                        data_subset = data[indices]
                        labels_subset = labels[indices]
                    else:
                        data_subset = data
                        labels_subset = labels
                    
                    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                    data_2d = reducer.fit_transform(data_subset)
                    labels = labels_subset
                    title = f"t-SNE Visualization (k={selected_k}, {algorithm.upper()})"
                
                # Create plot
                fig = px.scatter(
                    x=data_2d[:, 0],
                    y=data_2d[:, 1],
                    color=labels.astype(str),
                    title=title,
                    labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Cluster'},
                    height=600
                )
                
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
        
        st.markdown("---")
        
        # Export results
        st.subheader("💾 Export Results")
        
        if st.button("📥 Export Baseline Results (CSV)"):
            # Export metrics
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name="baseline_clustering_metrics.csv",
                mime="text/csv"
            )
            
            # Export cluster assignments
            cluster_assignments = pd.DataFrame({
                'sample_id': st.session_state.sample_ids,
                f'cluster_k{selected_k}': selected_result['labels']
            })
            
            csv2 = cluster_assignments.to_csv(index=False)
            st.download_button(
                label=f"Download Cluster Assignments (k={selected_k})",
                data=csv2,
                file_name=f"baseline_clusters_k{selected_k}.csv",
                mime="text/csv"
            )

# Navigation
st.markdown("---")
if 'baseline_results' in st.session_state:
    st.success("✅ Baseline clustering complete! Proceed to GAN training.")
    st.info("👉 Use the sidebar to navigate to **🤖 GAN Training**")
else:
    st.warning("⏳ Please run baseline clustering before proceeding.")