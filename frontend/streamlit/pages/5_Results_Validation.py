import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.config import *

st.title("📈 Results & Validation")
st.markdown("---")

# Check if both results available
if 'baseline_results' not in st.session_state:
    st.error("⚠️ No baseline results")
    st.stop()

if 'gan_results' not in st.session_state:
    st.error("⚠️ No GAN-assisted results")
    st.stop()

if 'baseline_algorithm' not in st.session_state:
    st.error("⚠️ No baseline algorithm information")
    st.stop()

if 'gan_algorithm' not in st.session_state:
    st.error("⚠️ No GAN-assisted algorithm information")
    st.stop()

if 'gan_real_labels' not in st.session_state:
    st.error("⚠️ No GAN-assisted cluster labels")
    st.stop()

st.success("✅ Both baseline and GAN-assisted results available!")

# Get results (use first k for each)
baseline_k = list(st.session_state.baseline_results.keys())[0]
gan_k = list(st.session_state.gan_results.keys())[0]

baseline = st.session_state.baseline_results[baseline_k]['metrics']
gan = st.session_state.gan_results[gan_k]['metrics']

# ============================================================================
# Metrics Comparison
# ============================================================================
st.header("📊 Metrics Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Silhouette Score")
    b_sil = baseline.get('silhouette_score', 0)
    g_sil = gan.get('silhouette_score', 0)
    improvement = ((g_sil - b_sil) / abs(b_sil)) * 100 if b_sil != 0 else 0
    
    st.metric("Baseline", f"{b_sil:.4f}")
    st.metric("GAN-Assisted", f"{g_sil:.4f}", delta=f"{improvement:+.2f}%")

with col2:
    st.subheader("Davies-Bouldin")
    b_db = baseline.get('davies_bouldin_index', 0)
    g_db = gan.get('davies_bouldin_index', 0)
    improvement = ((b_db - g_db) / abs(b_db)) * 100 if b_db != 0 else 0
    
    st.metric("Baseline", f"{b_db:.4f}")
    st.metric("GAN-Assisted", f"{g_db:.4f}", delta=f"{improvement:+.2f}%")

with col3:
    st.subheader("Calinski-Harabasz")
    b_ch = baseline.get('calinski_harabasz_score', 0)
    g_ch = gan.get('calinski_harabasz_score', 0)
    improvement = ((g_ch - b_ch) / abs(b_ch)) * 100 if b_ch != 0 else 0
    
    st.metric("Baseline", f"{b_ch:.2f}")
    st.metric("GAN-Assisted", f"{g_ch:.2f}", delta=f"{improvement:+.2f}%")

st.markdown("---")

# ============================================================================
# Improvement Summary
# ============================================================================
st.header("📈 Improvement Summary")

# Calculate all improvements
improvements = {
    'Silhouette Score': ((g_sil - b_sil) / abs(b_sil)) * 100 if b_sil != 0 else 0,
    'Davies-Bouldin Index': ((b_db - g_db) / abs(b_db)) * 100 if b_db != 0 else 0,
    'Calinski-Harabasz': ((g_ch - b_ch) / abs(b_ch)) * 100 if b_ch != 0 else 0
}

# Bar chart
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(
        x=list(improvements.keys()),
        y=list(improvements.values()),
        text=[f"{v:+.2f}%" for v in improvements.values()],
        textposition='auto',
        marker_color=['green' if v > 0 else 'red' for v in improvements.values()]
    )
])

fig.update_layout(
    title="Improvement Percentages (GAN vs Baseline)",
    yaxis_title="Improvement (%)",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Overall assessment
avg_improvement = np.mean(list(improvements.values()))

if avg_improvement >= 15:
    st.success(f"🎉 **Excellent!** Average improvement: {avg_improvement:.2f}%")
elif avg_improvement >= 10:
    st.success(f"✅ **Good!** Average improvement: {avg_improvement:.2f}%")
elif avg_improvement >= 5:
    st.info(f"✓ **Moderate** improvement: {avg_improvement:.2f}%")
else:
    st.warning(f"⚠️ **Minimal** improvement: {avg_improvement:.2f}%")

st.markdown("---")

# ============================================================================
# Side-by-Side Visualization
# ============================================================================
st.header("📊 Side-by-Side Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Clustering")
    st.info(f"Algorithm: {st.session_state.baseline_algorithm.upper()}, k={baseline_k}")
    
    # Cluster sizes
    baseline_sizes = st.session_state.baseline_results[baseline_k]['cluster_sizes']
    
    fig1 = go.Figure(data=[go.Bar(
        x=list(baseline_sizes.keys()),
        y=list(baseline_sizes.values()),
        marker_color='steelblue'
    )])
    fig1.update_layout(title="Cluster Sizes", height=300)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("GAN-Assisted Clustering")
    st.info(f"Algorithm: {st.session_state.gan_algorithm.upper()}, k={gan_k}")
    
    # Cluster sizes
    gan_sizes = st.session_state.gan_results[gan_k]['cluster_sizes']
    
    fig2 = go.Figure(data=[go.Bar(
        x=list(gan_sizes.keys()),
        y=list(gan_sizes.values()),
        marker_color='coral'
    )])
    fig2.update_layout(title="Cluster Sizes", height=300)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================================================
# Saved Visualizations
# ============================================================================
st.header("🖼️ Detailed Visualizations")

st.info("📊 These visualizations were generated during the clustering analysis and saved to disk.")

# Display baseline visualizations
st.subheader("Baseline Clustering Visualizations")
baseline_viz_path = Path(project_root) / "results" / "baseline" / "baseline_visualizations"

if baseline_viz_path.exists():
    viz_files = list(baseline_viz_path.glob("*.png"))
    if viz_files:
        cols = st.columns(2)
        for i, viz_file in enumerate(viz_files[:4]):  # Show first 4
            with cols[i % 2]:
                st.image(str(viz_file), caption=viz_file.stem.replace("_", " ").title(), use_column_width=True)
    else:
        st.warning("No baseline visualization files found.")
else:
    st.warning("Baseline visualizations directory not found.")

# Display GAN-assisted visualizations
st.subheader("GAN-Assisted Clustering Visualizations")
gan_viz_path = Path(project_root) / "results" / "gan_assisted" / "visualizations" / st.session_state.gan_algorithm

if gan_viz_path.exists():
    # Show visualizations for the selected k value
    selected_k = st.selectbox("Select k for GAN visualizations", [2, 3, 4, 5], index=1)
    viz_files = list(gan_viz_path.glob(f"*k{selected_k}.png"))
    if viz_files:
        cols = st.columns(2)
        for i, viz_file in enumerate(viz_files):
            with cols[i % 2]:
                st.image(str(viz_file), caption=f"GAN-Assisted {viz_file.stem.replace('_', ' ').title()}", use_column_width=True)
    else:
        st.warning(f"No GAN-assisted visualization files found for k={selected_k}.")
else:
    st.warning("GAN-assisted visualizations directory not found.")

st.markdown("---")

# ============================================================================
# Export Results
# ============================================================================
st.header("💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Download Comparison Report (CSV)"):
        comparison_df = pd.DataFrame({
            'Metric': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'],
            'Baseline': [b_sil, b_db, b_ch],
            'GAN-Assisted': [g_sil, g_db, g_ch],
            'Improvement (%)': list(improvements.values())
        })
        
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="clustering_comparison.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📥 Download Cluster Assignments"):
        real_baseline = st.session_state.baseline_results[baseline_k]['labels']
        real_gan = st.session_state.gan_real_labels
        
        assignments_df = pd.DataFrame({
            'sample_id': st.session_state.sample_ids,
            'baseline_cluster': real_baseline,
            'gan_cluster': real_gan
        })
        
        csv = assignments_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="cluster_assignments.csv",
            mime="text/csv"
        )

# ============================================================================
# Summary & Conclusions
# ============================================================================
st.markdown("---")
st.header("📝 Summary & Conclusions")

st.success("""
**Analysis Complete!**

You have successfully:
1. ✅ Uploaded and preprocessed TCGA-LUAD data
2. ✅ Established baseline clustering performance
3. ✅ Trained a WGAN-GP to generate synthetic samples
4. ✅ Performed GAN-assisted clustering on augmented data
5. ✅ Compared and quantified improvements

""")

if avg_improvement > 10:
    st.info("""
    **Conclusion:**
    The GAN-assisted approach showed meaningful improvements in clustering quality.
    Synthetic data augmentation helped identify better-separated cluster structures.
    """)
else:
    st.warning("""
    **Conclusion:**
    The GAN-assisted approach showed modest improvements.
    Consider:
    - Improving GAN quality (retrain with better hyperparameters)
    - Testing different augmentation ratios
    - Trying alternative clustering algorithms
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h3>🎉 Project Complete!</h3>
    <p>Thank you for using the GAN-LUAD Clustering System</p>
</div>
""", unsafe_allow_html=True)