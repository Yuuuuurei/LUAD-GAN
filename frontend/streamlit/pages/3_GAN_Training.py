"""
Page 3: GAN Training
Streamlit MVP - Phase 10

Train WGAN-GP to generate synthetic samples.

Author: GAN-LUAD Team
Date: 2025
"""

import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.config import (
    HELP_TEXTS, ERROR_MESSAGES, SUCCESS_MESSAGES,
    DEFAULT_LATENT_DIM, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS, DEFAULT_N_CRITIC, DEFAULT_GP_WEIGHT
)

st.title("🤖 GAN Training")
st.markdown("---")

# Check if data is ready
if not st.session_state.get('data_ready', False):
    st.error(ERROR_MESSAGES['no_data'])
    st.info("👈 Please go to **📁 Data Upload** page first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📖 Help")
    with st.expander("About GAN Training"):
        st.markdown(HELP_TEXTS['gan_training'])
    
    st.markdown("---")
    
    st.info(f"""
    **Current Data:**
    - Samples: {st.session_state.n_samples}
    - Features: {st.session_state.n_features}
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["⚙️ Configure", "🚀 Train", "📈 Monitor"])

# ============================================================================
# TAB 1: Configuration
# ============================================================================
with tab1:
    st.header("GAN Training Configuration")
    
    st.info("""
    **WGAN-GP** (Wasserstein GAN with Gradient Penalty) will be trained to generate
    synthetic gene expression samples that match the real data distribution.
    """)
    
    # Basic settings
    st.subheader("Basic Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latent_dim = st.number_input(
            "Latent Dimension",
            min_value=64,
            max_value=512,
            value=DEFAULT_LATENT_DIM,
            step=64,
            help="Size of noise vector input to generator"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=16,
            max_value=128,
            value=DEFAULT_BATCH_SIZE,
            step=16,
            help="Number of samples per training batch"
        )
        
        num_epochs = st.number_input(
            "Number of Epochs",
            min_value=100,
            max_value=2000,
            value=DEFAULT_EPOCHS,
            step=100,
            help="Total training iterations"
        )
    
    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.001,
            value=DEFAULT_LEARNING_RATE,
            step=0.00001,
            format="%.5f",
            help="Optimizer step size"
        )
        
        n_critic = st.number_input(
            "Critic Updates per Generator",
            min_value=1,
            max_value=10,
            value=DEFAULT_N_CRITIC,
            help="Train critic this many times per generator update"
        )
        
        gp_weight = st.number_input(
            "Gradient Penalty Weight",
            min_value=1,
            max_value=30,
            value=DEFAULT_GP_WEIGHT,
            help="Lambda for gradient penalty"
        )
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        st.warning("⚠️ Changing these may affect training stability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            generator_dims = st.text_input(
                "Generator Hidden Dimensions",
                value="256,512,1024",
                help="Comma-separated layer sizes"
            )
            
            use_spectral_norm = st.checkbox(
                "Use Spectral Normalization",
                value=False,
                help="Apply spectral norm to critic"
            )
        
        with col2:
            critic_dims = st.text_input(
                "Critic Hidden Dimensions",
                value="1024,512,256",
                help="Comma-separated layer sizes"
            )
            
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=True,
                help="Stop if no improvement"
            )
            
            if early_stopping:
                patience = st.number_input(
                    "Patience (epochs)",
                    min_value=10,
                    max_value=200,
                    value=50
                )
    
    st.markdown("---")
    
    # Save config to session state
    if st.button("💾 Save Configuration", use_container_width=True):
        config = {
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'n_critic': n_critic,
            'gp_weight': gp_weight,
            'generator_dims': [int(x.strip()) for x in generator_dims.split(',')],
            'critic_dims': [int(x.strip()) for x in critic_dims.split(',')],
            'use_spectral_norm': use_spectral_norm if 'use_spectral_norm' in locals() else False,
            'early_stopping': early_stopping if 'early_stopping' in locals() else True,
            'patience': patience if 'patience' in locals() else 50
        }
        
        st.session_state.gan_config = config
        st.success("✅ Configuration saved!")
        st.info("👉 Go to **🚀 Train** tab to start training")

# ============================================================================
# TAB 2: Training
# ============================================================================
with tab2:
    st.header("Train GAN")
    
    if 'gan_config' not in st.session_state:
        st.warning("⏳ Please configure and save GAN settings first.")
        st.info("👈 Go to **⚙️ Configure** tab")
    else:
        config = st.session_state.gan_config
        
        st.subheader("📋 Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latent Dim", config['latent_dim'])
        with col2:
            st.metric("Batch Size", config['batch_size'])
        with col3:
            st.metric("Epochs", config['num_epochs'])
        
        st.markdown("---")
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            
            st.info("""
            **Note:** For the MVP, training is simulated.
            
            In full implementation, this would:
            1. Load backend.models.wgan_gp
            2. Initialize Generator and Critic
            3. Train for specified epochs
            4. Save checkpoints
            5. Generate synthetic samples
            
            **Implementation placeholder:** Actual training would take 30-60 minutes on GPU.
            """)
            
            # Simulated training
            st.subheader("🔄 Training Progress (Simulated)")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Placeholder for loss plots
            loss_chart = st.empty()
            
            # Simulate training
            epochs = min(config['num_epochs'], 100)  # Simulate 100 epochs max
            
            import pandas as pd
            
            loss_data = {
                'epoch': [],
                'g_loss': [],
                'c_loss': [],
                'gp': []
            }
            
            for epoch in range(1, epochs + 1):
                # Simulate losses
                g_loss = -1.0 + np.random.randn() * 0.5
                c_loss = 2.0 + np.random.randn() * 0.3 - epoch * 0.01
                gp = 0.5 + np.random.randn() * 0.1
                
                loss_data['epoch'].append(epoch)
                loss_data['g_loss'].append(g_loss)
                loss_data['c_loss'].append(c_loss)
                loss_data['gp'].append(gp)
                
                # Update progress
                progress = epoch / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}/{epochs} - G Loss: {g_loss:.4f}, C Loss: {c_loss:.4f}")
                
                # Update plot every 10 epochs
                if epoch % 10 == 0 or epoch == epochs:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Generator & Critic Loss", "Gradient Penalty")
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=loss_data['epoch'], y=loss_data['g_loss'], name='Generator', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=loss_data['epoch'], y=loss_data['c_loss'], name='Critic', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=loss_data['epoch'], y=loss_data['gp'], name='GP', line=dict(color='green')),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, showlegend=True)
                    fig.update_xaxes(title_text="Epoch")
                    fig.update_yaxes(title_text="Loss", row=1, col=1)
                    fig.update_yaxes(title_text="Gradient Penalty", row=2, col=1)
                    
                    loss_chart.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.05)  # Simulate time
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(SUCCESS_MESSAGES['training_complete'])
            
            # Store "trained" model
            st.session_state.gan_trained = True
            st.session_state.gan_loss_history = loss_data
            
            st.info("👉 Go to **📈 Monitor** tab to view results")

# ============================================================================
# TAB 3: Monitor
# ============================================================================
with tab3:
    st.header("Training Monitor")
    
    if not st.session_state.get('gan_trained', False):
        st.warning("⏳ No training completed yet.")
        st.info("👈 Go to **🚀 Train** tab to start training")
    else:
        st.success("✅ Training complete!")
        
        # Loss history
        st.subheader("📉 Loss Curves")
        
        loss_data = st.session_state.gan_loss_history
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Generator Loss", "Critic Loss", "Gradient Penalty"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=loss_data['epoch'], y=loss_data['g_loss'], name='Generator', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=loss_data['epoch'], y=loss_data['c_loss'], name='Critic', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=loss_data['epoch'], y=loss_data['gp'], name='Gradient Penalty', line=dict(color='green')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Epoch")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Generate synthetic samples
        st.subheader("🎨 Generate Synthetic Samples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            aug_ratio = st.slider(
                "Augmentation Ratio",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                help="Ratio of synthetic to real samples"
            )
        
        with col2:
            n_synthetic = int(st.session_state.n_samples * aug_ratio)
            st.metric("Synthetic Samples to Generate", n_synthetic)
        
        if st.button("🎨 Generate Synthetic Data", use_container_width=True):
            with st.spinner("Generating synthetic samples..."):
                # Simulate generation
                time.sleep(2)
                
                # Create fake synthetic data (in real app, use generator)
                real_data = st.session_state.processed_data.numpy()
                
                # Simulate: add noise to real data
                synthetic_data = real_data + np.random.randn(*real_data.shape) * 0.1
                synthetic_data = synthetic_data[:n_synthetic]
                
                # Store
                st.session_state.synthetic_data = torch.FloatTensor(synthetic_data)
                st.session_state.aug_ratio = aug_ratio
                
                st.success(SUCCESS_MESSAGES['synthetic_generated'])
                
                # Quick stats
                st.info(f"""
                **Generated:**
                - Real samples: {st.session_state.n_samples}
                - Synthetic samples: {len(synthetic_data)}
                - Total: {st.session_state.n_samples + len(synthetic_data)}
                """)
        
        st.markdown("---")
        
        # Quality check
        if 'synthetic_data' in st.session_state:
            st.subheader("✅ Synthetic Data Ready")
            
            st.success("""
            Synthetic samples generated successfully!
            
            In full implementation, quality metrics would be displayed here:
            - Mean difference
            - Variance ratio
            - Distribution comparison
            - Quality score (0-4)
            """)
            
            st.info("👉 Proceed to **🎨 GAN-Assisted Clustering** page")

# Navigation
st.markdown("---")
if st.session_state.get('synthetic_data') is not None:
    st.success("✅ Synthetic data ready! Proceed to GAN-assisted clustering.")
    st.info("👉 Use the sidebar to navigate to **🎨 GAN-Assisted Clustering**")
elif st.session_state.get('gan_trained', False):
    st.warning("⏳ Please generate synthetic data before proceeding.")
else:
    st.warning("⏳ Please train GAN and generate synthetic data.")