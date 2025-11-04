import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_value_function_heatmap(values, title="Value Function Heatmap"):
    """Plot value function as heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=values,
        colorscale='Viridis'
    ))
    fig.update_layout(title=title)
    return fig

def plot_training_progress(rewards, avg_rewards, title="Training Progress"):
    """Plot training rewards over episodes"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rewards, 
        mode='lines', 
        name='Episode Reward'
    ))
    fig.add_trace(go.Scatter(
        y=avg_rewards, 
        mode='lines', 
        name='Avg Reward (10 episodes)'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Reward"
    )
    return fig

def plot_policy_comparison(ground_policy, learned_policy, title="Policy Comparison"):
    """Compare ground truth policy with learned policy"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(ground_policy))),
        y=ground_policy,
        mode='markers',
        name='Ground Truth'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(learned_policy))),
        y=learned_policy,
        mode='markers',
        name='Learned Policy'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="State Index",
        yaxis_title="Action"
    )
    return fig