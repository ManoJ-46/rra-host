import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.abstract_model import AbstractModel
from models.ground_model import GroundModel

def render():
    """Render the abstract model page"""
    st.title("🔄 Abstract Model")
    st.markdown("### Create and compare abstract MDP representations")
    
    if 'ground_model' not in st.session_state or st.session_state.ground_model is None:
        st.warning("⚠️ Please create and solve a Ground Model first from the 'Ground MDP Model' page.")
        return
    
    ground_model = st.session_state.ground_model
    
    st.sidebar.header("Abstraction Configuration")
    st.sidebar.subheader("Grouping Parameters")
    number_groups = st.sidebar.slider("Number of Groups", 1, ground_model.number_UEs.item(), 3,
                                     help="Number of UE groups for abstraction")
    
    st.sidebar.subheader("Weight Distribution")
    coef_owners = st.sidebar.radio("Weight Owners", ["UEs", "Groups"])
    coef_distribution = st.sidebar.selectbox("Distribution Criterion", 
                                            ["Uniform", "Similarity", "Dissimilarity"])
    
    variant = None
    if coef_distribution == "Dissimilarity" and coef_owners == "Groups":
        variant = st.sidebar.selectbox("Dissimilarity Variant", 
                                      ["Standard Deviation", "Cross Entropy", "Gini Index"])
    
    st.sidebar.subheader("Selection Mode")
    selection_mode = st.sidebar.radio("Representative Selection", 
                                     ["One (Random)", "Top (All Max)", "All"])
    
    selection_mode_map = {
        "One (Random)": "one",
        "Top (All Max)": "top",
        "All": "all"
    }
    
    if st.button("🔄 Create Abstract Model", type="primary"):
        create_abstract_model(
            ground_model,
            number_groups,
            coef_owners.lower(),
            coef_distribution.lower(),
            variant.lower().replace(" ", "_") if variant else None,
            selection_mode_map[selection_mode]
        )
    
    if 'abstract_model' in st.session_state and st.session_state.abstract_model is not None:
        display_abstract_results()

def create_abstract_model(ground_model, number_groups, coef_owners, 
                         coef_distribution, variant, selection_mode):
    """Create and solve abstract model"""
    try:
        with st.spinner("Creating abstract model..."):
            abstract_model = AbstractModel(
                ground_model=ground_model,
                number_groups=number_groups,
                coef_owners=coef_owners,
                coef_distribution_criterion=coef_distribution,
                variant=variant,
                selection_mode=selection_mode
            )
            abstract_model.solve()
        
        st.success("✅ Abstract model created and solved successfully!")
        st.session_state.abstract_model = abstract_model
        
    except Exception as e:
        st.error(f"❌ Error creating abstract model: {str(e)}")
        st.exception(e)

def display_abstract_results():
    """Display abstract model results"""
    abstract_model = st.session_state.abstract_model
    ground_model = st.session_state.ground_model
    
    st.markdown("---")
    st.subheader("📊 Abstraction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Abstract States", int(abstract_model.number_states))
    with col2:
        st.metric("Abstraction Time", f"{abstract_model.abstraction_elapse_time:.2f}s")
    with col3:
        st.metric("Solve Time", f"{abstract_model.resolution_elapse_time:.2f}s")
    
    st.subheader("Abstraction Error")
    ground_values = ground_model.solution[1]
    abstract_values = abstract_model.solution[1]
    value_diffs = torch.abs(ground_values - abstract_values)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Value Error", f"{value_diffs.max().item():.4f}")
    with col2:
        st.metric("Avg Value Error", f"{value_diffs.mean().item():.4f}")
    
    st.subheader("Value Function Comparison")
    num_samples = min(50, int(ground_model.number_states))
    sample_indices = np.random.choice(int(ground_model.number_states), num_samples, replace=False)
    ground_sample = ground_values[sample_indices].cpu().numpy()
    abstract_sample = abstract_values[sample_indices].cpu().numpy()
    
    fig = px.scatter(
        x=ground_sample,
        y=abstract_sample,
        labels={'x': 'Ground Value', 'y': 'Abstract Value'},
        title="Value Function Comparison"
    )
    fig.add_trace(px.line(x=[float(np.min(ground_sample)), float(np.max(ground_sample))], 
                          y=[float(np.min(ground_sample)), float(np.max(ground_sample))]).data[0])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="abstract_value_compare")
