"""
Cluster Visualization Component
Create cluster visualizations (PCA, t-SNE, etc.)
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_clusters_2d(data, labels, method='PCA', title=None):
    """
    Plot clusters in 2D using PCA or t-SNE.
    
    Args:
        data: Feature matrix (n_samples, n_features)
        labels: Cluster labels
        method: 'PCA' or 't-SNE'
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Reduce to 2D
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        data_2d = reducer.fit_transform(data)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]*100:.1f}%)"
        ylabel = f"PC2 ({explained_var[1]*100:.1f}%)"
        default_title = f"PCA Visualization"
    else:  # t-SNE
        # Subsample if too large
        if len(data) > 1000:
            indices = np.random.choice(len(data), 1000, replace=False)
            data = data[indices]
            labels = labels[indices]
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        data_2d = reducer.fit_transform(data)
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"
        default_title = "t-SNE Visualization"
    
    # Create plot
    fig = px.scatter(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        color=labels.astype(str),
        title=title or default_title,
        labels={'x': xlabel, 'y': ylabel, 'color': 'Cluster'},
        height=600
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(legend_title_text='Cluster')
    
    return fig


def plot_cluster_sizes(cluster_sizes, title="Cluster Distribution"):
    """
    Plot cluster size distribution.
    
    Args:
        cluster_sizes: Dict mapping cluster ID to size
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(cluster_sizes.keys()),
            y=list(cluster_sizes.values()),
            text=list(cluster_sizes.values()),
            textposition='auto',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Number of Samples",
        height=400
    )
    
    return fig


def plot_silhouette_analysis(silhouette_values, cluster_labels, title="Silhouette Analysis"):
    """
    Create silhouette plot.
    
    Args:
        silhouette_values: Silhouette score per sample
        cluster_labels: Cluster assignments
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_clusters = len(np.unique(cluster_labels))
    
    fig = go.Figure()
    
    y_lower = 10
    
    for i in range(n_clusters):
        # Get silhouette values for this cluster
        cluster_silhouette_values = silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            fill='tozerox',
            name=f'Cluster {i}',
            showlegend=True
        ))
        
        y_lower = y_upper + 10
    
    # Average line
    avg_silhouette = np.mean(silhouette_values)
    fig.add_vline(x=avg_silhouette, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_silhouette:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Cluster",
        height=600
    )
    
    return fig
