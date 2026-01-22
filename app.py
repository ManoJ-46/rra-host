import streamlit as st
import torch
import sys
import os


# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pages import (
    ground_model_page,
    abstract_model_page,
    hddpg_page,
    dqn_page,
    comparison_page
)

# Configure Streamlit page
st.set_page_config(
    page_title="Radio Resource Allocation RL System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Add this to top of app.py after imports ---
MOBILE_CSS = """
<style>
/* Make Streamlit container use full width on small screens and reduce side padding */
@media (max-width: 600px) {
  .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
  .stButton>button, .stDownloadButton>button { font-size: 14px; padding: 6px 8px; }
  .element-container { margin-bottom: 10px; }
  h1 { font-size: 1.4rem !important; }
  h2 { font-size: 1.15rem !important; }
  h3 { font-size: 1rem !important; }
}
/* Slightly reduce spacing for desktop too for a cleaner look */
.main .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 1rem; }
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)
# --- end CSS ---

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.session_state.ground_model = None
    st.session_state.abstract_model = None
    st.session_state.hddpg_agent = None
    st.session_state.dqn_agent = None

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("📡 Radio Resource Allocation")
    st.sidebar.markdown("---")
    
    # Device information
    device_status = "🟢 CUDA Available" if torch.cuda.is_available() else "🔴 CPU Only"
    st.sidebar.markdown(f"**Device Status:** {device_status}")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Select Page",
        [
            " Home",
            " Ground MDP Model",
            " Abstract Model",
            " HDDPG Training",
            " DQN Training",
            " Model Comparison"
        ]
    )
    
    # Route to appropriate page
    if page == " Home":
        show_home_page()
    elif page == " Ground MDP Model":
        ground_model_page.render()
    elif page == " Abstract Model":
        abstract_model_page.render()
    elif page == " HDDPG Training":
        hddpg_page.render()
    elif page == " DQN Training":
        dqn_page.render()
    elif page == " Model Comparison":
        comparison_page.render()

def show_home_page():
    """Display the home page"""
    st.title("Radio Resource Allocation System")
    st.markdown("### Welcome to the Comprehensive RL-based Resource Management Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Ground MDP Model
        - **Purpose**: Solve the exact MDP for radio resource allocation
        - **Method**: Value iteration algorithm
        - **Features**: 
          - Configurable UEs, RBs, buffer sizes
          - Poisson arrival processes
          - Cost function optimization
        """)
    
    with col2:
        st.markdown("""
        #### Abstract Model
        - **Purpose**: Create simplified versions of the ground model
        - **Method**: State aggregation techniques
        - **Features**:
          - Multiple abstraction strategies
          - Weight distribution methods
          - Performance comparison
        """)
    
    with col3:
        st.markdown("""
        #### Deep RL Agents
        - **HDDPG**: Continuous action space control
        - **DQN**: Discrete action space control
        - **Features**:
          - Real-time training visualization
          - Performance metrics
          - Model saving/loading
        """)
    
    st.markdown("---")
    st.markdown("### Getting Started")
    st.markdown("""
    1. **Ground MDP Model**: Configure and solve the exact MDP problem
    2. **Abstract Model**: Create and compare different abstraction strategies
    3. **HDDPG Training**: Train continuous control agents
    4. **DQN Training**: Train discrete control agents
    5. **Model Comparison**: Compare performance across different approaches
    """)
    
    # System information
    st.markdown("---")
    st.markdown("### System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Device", "CUDA" if torch.cuda.is_available() else "CPU")
    
    with col2:
        if torch.cuda.is_available():
            st.metric("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            st.metric("CPU Cores", f"{os.cpu_count()}")
    
    with col3:
        st.metric("PyTorch Version", torch.__version__)

if __name__ == "__main__":
    main()

