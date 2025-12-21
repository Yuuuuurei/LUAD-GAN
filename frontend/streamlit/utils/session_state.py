"""
Session State Management
Handle Streamlit session state initialization and persistence.
"""

import streamlit as st
import json
from pathlib import Path

def initialize_session_state():
    """
    Initialize session state with default values.
    Call this at the start of your app.
    """
    defaults = {
        # Data
        'raw_data': None,
        'processed_data': None,
        'gene_names': None,
        'sample_ids': None,
        'n_samples': 0,
        'n_features': 0,
        'data_ready': False,
        
        # Preprocessing
        'pca_applied': False,
        
        # Baseline clustering
        'baseline_results': None,
        'baseline_algorithm': None,
        'baseline_pipeline': None,
        
        # GAN
        'gan_config': None,
        'gan_trained': False,
        'gan_model': None,
        'gan_loss_history': None,
        'synthetic_data': None,
        'aug_ratio': 1.0,
        
        # GAN-assisted clustering
        'gan_results': None,
        'gan_algorithm': None,
        'gan_pipeline': None,
        'gan_real_labels': None,
        
        # Comparison
        'comparison_results': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_session_state():
    """Clear all session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()


def save_session_state(filepath='session_state.json'):
    """
    Save session state to file (excluding non-serializable objects).
    
    Args:
        filepath: Path to save session state
    """
    # Select serializable items
    serializable_state = {}
    
    for key, value in st.session_state.items():
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            serializable_state[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_state, f, indent=2)


def load_session_state(filepath='session_state.json'):
    """
    Load session state from file.
    
    Args:
        filepath: Path to load session state from
    """
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            loaded_state = json.load(f)
        
        for key, value in loaded_state.items():
            st.session_state[key] = value


def get_session_summary():
    """
    Get a summary of current session state.
    
    Returns:
        Dict with session state summary
    """
    return {
        'data_loaded': st.session_state.get('data_ready', False),
        'n_samples': st.session_state.get('n_samples', 0),
        'n_features': st.session_state.get('n_features', 0),
        'baseline_complete': st.session_state.get('baseline_results') is not None,
        'gan_trained': st.session_state.get('gan_trained', False),
        'synthetic_generated': st.session_state.get('synthetic_data') is not None,
        'gan_clustering_complete': st.session_state.get('gan_results') is not None
    }


def check_prerequisites(required_states):
    """
    Check if required session states are ready.
    
    Args:
        required_states: List of required session state keys
        
    Returns:
        tuple: (all_ready: bool, missing: list)
    """
    missing = []
    
    for state in required_states:
        if not st.session_state.get(state):
            missing.append(state)
    
    return len(missing) == 0, missing