import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.ground_model import GroundModel

def render():
    """Render the ground model page"""
    st.title("🎯 Ground MDP Model")
    st.markdown("### Configure and solve the exact MDP for radio resource allocation")
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # Basic parameters
    st.sidebar.subheader("Basic Parameters")
    max_size = st.sidebar.slider("Max Buffer Size", 1, 5, 3, 
                                help="Maximum number of bits in each UE's buffer")
    number_UEs = st.sidebar.slider("Number of UEs", 2, 8, 6,
                                  help="Number of User Equipment devices")
    number_RBs = st.sidebar.slider("Number of RBs", 1, 4, 2,
                                  help="Number of Resource Blocks")
    
    # Arrival rates
    st.sidebar.subheader("Arrival Rates")
    use_default_rates = st.sidebar.checkbox("Use Default Rates", True,
                                          help="Use default Poisson arrival rates")
    
    if not use_default_rates:
        arrival_rates = []
        for i in range(number_UEs):
            rate = st.sidebar.number_input(f"UE {i} Arrival Rate", 0.1, 5.0, 1.0, 0.1,
                                         key=f"arrival_rate_{i}")
            arrival_rates.append(rate)
    else:
        arrival_rates = None
    
    # CQI parameters
    st.sidebar.subheader("Channel Quality")
    CQI_base = st.sidebar.number_input("Base CQI Value", 0.5, 3.0, 1.0, 0.1,
                                      help="Base Channel Quality Indicator value")
    CQIs_equal = st.sidebar.checkbox("Equal CQIs", True,
                                   help="Use equal CQI values for all UE-RB pairs")
    
    # Cost function parameters
    st.sidebar.subheader("Cost Function")
    coef_drop = st.sidebar.number_input("Drop Coefficient (α)", 0.1, 5.0, 1.0, 0.1,
                                       help="Coefficient for packet drop cost")
    coef_latency = st.sidebar.number_input("Latency Coefficient (β)", 0.1, 5.0, 1.0, 0.1,
                                          help="Coefficient for latency cost")
    power_drop = st.sidebar.number_input("Drop Power (x)", 1.0, 3.0, 1.0, 0.1,
                                        help="Power for packet drop cost")
    power_latency = st.sidebar.number_input("Latency Power (y)", 1.0, 3.0, 1.0, 0.1,
                                           help="Power for latency cost")
    
    # MDP parameters
    st.sidebar.subheader("MDP Parameters")
    discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, 0.01,
                                       help="MDP discount factor")
    precision = st.sidebar.selectbox("Precision", [1e-16, 1e-12, 1e-8, 1e-6],
                                    help="Convergence precision for value iteration")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Overview")
        
        # Display model information
        st.info(f"""
        **Model Configuration:**
        - State Space Size: {(max_size + 1) ** number_UEs:,} states
        - Action Space Size: {number_UEs ** number_RBs:,} actions
        - Total State-Action Pairs: {(max_size + 1) ** number_UEs * number_UEs ** number_RBs:,}
        
        **Cost Function:** 
        Cost = α × (drop_cost)^x + β × (latency_cost)^y
        
        **Device:** {'🟢 CUDA' if torch.cuda.is_available() else '🔴 CPU'}
        """)
        
        # Create and solve model button
        if st.button("🚀 Create and Solve Model", type="primary"):
            solve_model(max_size, number_UEs, number_RBs, arrival_rates, 
                       CQI_base, CQIs_equal, coef_drop, coef_latency, 
                       power_drop, power_latency, discount_factor, precision)
    
    with col2:
        st.subheader("State Space Preview")
        
        # Show some example states
        if number_UEs <= 3:  # Only show for small state spaces
            example_states = []
            for i in range(min(10, (max_size + 1) ** number_UEs)):
                state = []
                temp = i
                for j in range(number_UEs):
                    state.append(temp % (max_size + 1))
                    temp //= (max_size + 1)
                example_states.append(state)
            
            df = pd.DataFrame(example_states, columns=[f"UE{i}" for i in range(number_UEs)])
            st.dataframe(df)
        else:
            st.warning("State space too large to display examples")
    
    # Display results if model is solved
    if 'ground_model' in st.session_state and st.session_state.ground_model is not None:
        display_results()

def solve_model(max_size, number_UEs, number_RBs, arrival_rates, 
                CQI_base, CQIs_equal, coef_drop, coef_latency, 
                power_drop, power_latency, discount_factor, precision):
    """Solve the ground model with given parameters"""
    
    try:
        # Create model
        with st.spinner("Creating ground model..."):
            model = GroundModel(
                max_size=max_size,
                number_UEs=number_UEs,
                number_RBs=number_RBs,
                arrival_rates=arrival_rates,
                CQI_base=CQI_base,
                CQIs_are_equal=CQIs_equal,
                coef_of_drop=coef_drop,
                coef_of_latency=coef_latency,
                power_of_drop=power_drop,
                power_of_latency=power_latency,
                discount_factor=discount_factor,
                precision=precision
            )
        
        st.success("✅ Ground model created successfully!")
        
        # Solve model with progress bar
        st.subheader("Solving MDP...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(progress, iteration, error):
            progress_bar.progress(progress)
            status_text.text(f"Iteration {iteration}, Error: {error:.2e}")
        
        start_time = time.time()
        
        with st.spinner("Solving MDP using value iteration..."):
            policy, value_function, error, iterations = model.solve(progress_callback)
        
        solve_time = time.time() - start_time
        
        # Store results in session state
        st.session_state.ground_model = model
        st.session_state.ground_policy = policy
        st.session_state.ground_value_function = value_function
        st.session_state.ground_error = error
        st.session_state.ground_iterations = iterations
        st.session_state.ground_solve_time = solve_time
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Solved in {iterations} iterations, Final error: {error:.2e}")
        
        st.success(f"🎉 Model solved successfully in {solve_time:.2f} seconds!")
        
    except Exception as e:
        st.error(f"❌ Error solving model: {str(e)}")
        st.exception(e)

def display_results():
    """Display the results of the solved model"""
    
    model = st.session_state.ground_model
    policy = st.session_state.ground_policy
    value_function = st.session_state.ground_value_function
    error = st.session_state.ground_error
    iterations = st.session_state.ground_iterations
    solve_time = st.session_state.ground_solve_time
    
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Iterations", iterations)
    
    with col2:
        st.metric("Final Error", f"{error:.2e}")
    
    with col3:
        st.metric("Solve Time", f"{solve_time:.2f}s")
    
    with col4:
        st.metric("Avg Value", f"{value_function.mean().item():.3f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Value Function", "🎯 Policy", "📋 State-Action Table", "🔍 Query State"])
    
    with tab1:
        display_value_function(model, value_function)
    
    with tab2:
        display_policy(model, policy)
    
    with tab3:
        display_state_action_table(model, policy, value_function)
    
    with tab4:
        query_state_interface(model)

def display_value_function(model, value_function):
    """Display value function visualization"""
    
    st.subheader("Value Function Visualization")
    
    # Convert to numpy for plotting
    values = value_function.cpu().numpy()
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Min Value", f"{values.min():.3f}")
    
    with col2:
        st.metric("Max Value", f"{values.max():.3f}")
    
    with col3:
        st.metric("Std Dev", f"{values.std():.3f}")
    
    # Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=30, name="Value Distribution"))
    fig.update_layout(
        title="Value Function Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D heatmap if state space is 2D or 3D
    if model.number_UEs.item() == 2:
        display_2d_value_heatmap(model, values)
    elif model.number_UEs.item() == 3:
        display_3d_value_heatmap(model, values)

def display_2d_value_heatmap(model, values):
    """Display 2D heatmap for 2-UE case"""
    
    max_size = model.max_size.item()
    
    # Reshape values into 2D grid
    value_grid = values.reshape(max_size + 1, max_size + 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=value_grid,
        x=list(range(max_size + 1)),
        y=list(range(max_size + 1)),
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Value Function Heatmap (2 UEs)",
        xaxis_title="UE 0 Buffer Level",
        yaxis_title="UE 1 Buffer Level"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_3d_value_heatmap(model, values):
    """Display 3D heatmap for 3-UE case"""
    
    max_size = model.max_size.item()
    
    # Create 3D scatter plot
    states = model.states_matrix.cpu().numpy()
    
    fig = go.Figure(data=go.Scatter3d(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Value")
        )
    ))
    
    fig.update_layout(
        title="Value Function 3D Visualization (3 UEs)",
        scene=dict(
            xaxis_title="UE 0 Buffer Level",
            yaxis_title="UE 1 Buffer Level",
            zaxis_title="UE 2 Buffer Level"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_policy(model, policy):
    """Display policy visualization"""
    
    st.subheader("Optimal Policy")
    
    # Policy statistics
    policy_np = policy.cpu().numpy()
    actions = model.actions_matrix.cpu().numpy()
    
    # Action distribution
    action_counts = np.bincount(policy_np)
    action_labels = [f"Action {i}" for i in range(len(action_counts))]
    
    fig = go.Figure(data=go.Bar(x=action_labels, y=action_counts))
    fig.update_layout(
        title="Action Distribution in Optimal Policy",
        xaxis_title="Action",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Policy heatmap for 2-UE case
    if model.number_UEs.item() == 2:
        display_2d_policy_heatmap(model, policy_np)

def display_2d_policy_heatmap(model, policy):
    """Display 2D policy heatmap for 2-UE case"""
    
    max_size = model.max_size.item()
    
    # Reshape policy into 2D grid
    policy_grid = policy.reshape(max_size + 1, max_size + 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=policy_grid,
        x=list(range(max_size + 1)),
        y=list(range(max_size + 1)),
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Optimal Policy Heatmap (2 UEs)",
        xaxis_title="UE 0 Buffer Level",
        yaxis_title="UE 1 Buffer Level"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_state_action_table(model, policy, value_function):
    """Display state-action table"""
    
    st.subheader("State-Action Table")
    
    # Create DataFrame
    states = model.states_matrix.cpu().numpy()
    actions = model.actions_matrix.cpu().numpy()
    policy_np = policy.cpu().numpy()
    values = value_function.cpu().numpy()
    
    data = []
    for i, state in enumerate(states):
        action_idx = policy_np[i]
        action = actions[action_idx]
        value = values[i]
        
        row = {
            'State Index': i,
            'State': str(state.tolist()),
            'Action Index': action_idx,
            'Action': str(action.tolist()),
            'Value': f"{value:.4f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Display with pagination
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download State-Action Table",
        data=csv,
        file_name="ground_model_state_action_table.csv",
        mime="text/csv"
    )

def query_state_interface(model):
    """Interface for querying specific states"""
    
    st.subheader("Query Specific States")
    
    # Input for state
    st.write("Enter buffer levels for each UE:")
    
    cols = st.columns(model.number_UEs.item())
    state_input = []
    
    for i, col in enumerate(cols):
        with col:
            level = st.number_input(
                f"UE {i}",
                min_value=0,
                max_value=model.max_size.item(),
                value=0,
                key=f"query_ue_{i}"
            )
            state_input.append(level)
    
    if st.button("🔍 Query State"):
        try:
            state_tensor = torch.tensor(state_input, dtype=torch.int).to(model.device)
            
            # Get optimal action and value
            optimal_action = model.get_optimal_action(state_tensor)
            state_value = model.get_state_value(state_tensor)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Optimal Action:** {optimal_action.cpu().numpy().tolist()}")
            
            with col2:
                st.success(f"**State Value:** {state_value:.4f}")
            
        except Exception as e:
            st.error(f"Error querying state: {str(e)}")
