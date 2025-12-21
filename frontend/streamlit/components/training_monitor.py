"""
Training Monitor Component
Display real-time training progress and metrics.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def create_loss_plot(loss_history, title="Training Loss"):
    """
    Create interactive loss plot.
    
    Args:
        loss_history: Dict with 'epoch', 'g_loss', 'c_loss', 'gp' keys
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Generator & Critic Loss", "Gradient Penalty"),
        vertical_spacing=0.15
    )
    
    # Generator and Critic loss
    fig.add_trace(
        go.Scatter(
            x=loss_history['epoch'],
            y=loss_history['g_loss'],
            name='Generator',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=loss_history['epoch'],
            y=loss_history['c_loss'],
            name='Critic',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Gradient penalty
    fig.add_trace(
        go.Scatter(
            x=loss_history['epoch'],
            y=loss_history['gp'],
            name='Gradient Penalty',
            line=dict(color='green', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="GP", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, title_text=title)
    
    return fig


def show_training_progress(current_epoch, total_epochs, metrics):
    """
    Display training progress bar and current metrics.
    
    Args:
        current_epoch: Current epoch number
        total_epochs: Total epochs
        metrics: Dict with current metrics
    """
    progress = current_epoch / total_epochs
    
    st.progress(progress)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Epoch", f"{current_epoch}/{total_epochs}")
    
    with col2:
        st.metric("Generator Loss", f"{metrics.get('g_loss', 0):.4f}")
    
    with col3:
        st.metric("Critic Loss", f"{metrics.get('c_loss', 0):.4f}")


def show_training_status(status_message, status_type="info"):
    """
    Display training status message.
    
    Args:
        status_message: Message to display
        status_type: 'info', 'success', 'warning', 'error'
    """
    if status_type == "info":
        st.info(status_message)
    elif status_type == "success":
        st.success(status_message)
    elif status_type == "warning":
        st.warning(status_message)
    elif status_type == "error":
        st.error(status_message)