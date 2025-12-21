"""
Data Preview Component
Reusable component for displaying data previews.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def show_data_preview(data, max_rows=10, max_cols=10, title="Data Preview"):
    """
    Display data preview with statistics.
    
    Args:
        data: numpy array or pandas DataFrame
        max_rows: Maximum rows to display
        max_cols: Maximum columns to display
        title: Preview title
    """
    st.subheader(title)
    
    # Convert to DataFrame if numpy
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(
            data[:max_rows, :max_cols],
            columns=[f"Feature {i+1}" for i in range(min(max_cols, data.shape[1]))],
            index=[f"Sample {i+1}" for i in range(min(max_rows, data.shape[0]))]
        )
    else:
        df = data.head(max_rows).iloc[:, :max_cols]
    
    # Display dataframe
    st.dataframe(df, use_container_width=True)
    
    # Statistics
    if isinstance(data, np.ndarray):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape", f"{data.shape}")
        with col2:
            st.metric("Mean", f"{data.mean():.3f}")
        with col3:
            st.metric("Std", f"{data.std():.3f}")
        with col4:
            st.metric("Range", f"[{data.min():.2f}, {data.max():.2f}]")


def show_distribution_plot(data, title="Distribution", bins=50):
    """
    Show distribution histogram.
    
    Args:
        data: numpy array or pandas Series
        title: Plot title
        bins: Number of bins
    """
    fig = go.Figure(data=[
        go.Histogram(x=data.flatten() if isinstance(data, np.ndarray) else data, nbinsx=bins)
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_summary_stats(data, title="Summary Statistics"):
    """
    Display summary statistics table.
    
    Args:
        data: numpy array
        title: Table title
    """
    st.subheader(title)
    
    if isinstance(data, np.ndarray):
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [
                data.size,
                data.mean(),
                data.std(),
                data.min(),
                np.percentile(data, 25),
                np.percentile(data, 50),
                np.percentile(data, 75),
                data.max()
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)