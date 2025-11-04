# src/pages/dqn_page.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import time
import sys
import os

# ensure parent path contains models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.dqn_agent import DQNAgent
from models.environments import RadioResourceEnv

def render():
    """Render the DQN training page"""
    st.title("🎮 DQN Training")
    st.markdown("### Train a Deep Q-Network agent")
    
    # Check if ground model exists
    if 'ground_model' not in st.session_state or st.session_state.ground_model is None:
        st.warning("⚠️ Please create and solve a Ground Model first from the 'Ground MDP Model' page.")
        return
    
    ground_model = st.session_state.ground_model
    
    # Sidebar for configuration
    st.sidebar.header("DQN Configuration")
    
    # Agent parameters
    st.sidebar.subheader("Agent Parameters")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.95, 0.01)
    epsilon_start = st.sidebar.slider("Initial Epsilon", 0.1, 1.0, 1.0, 0.05)
    epsilon_end = st.sidebar.slider("Final Epsilon", 0.01, 0.2, 0.01, 0.01)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.995, 0.001)
    lr = st.sidebar.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, 1e-5)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_episodes = st.sidebar.number_input("Number of Episodes", 10, 1000, 100, 10)
    max_steps = st.sidebar.number_input("Max Steps per Episode", 10, 1000, 200, 10)
    batch_size = st.sidebar.number_input("Batch Size", 16, 256, 64, 16)
    buffer_size = st.sidebar.number_input("Replay Buffer Size", 1000, 100000, 10000, 1000)
    target_update = st.sidebar.number_input("Target Update Frequency", 1, 100, 10, 1)
    plot_every = st.sidebar.number_input("Update plots every N episodes", 1, 50, 1, 1)
    
    # Initialize DQN agent and environment if they don't exist
    if 'dqn_agent' not in st.session_state or 'dqn_env' not in st.session_state:
        state_dim = int(ground_model.number_UEs.item())
        action_dim = int(ground_model.number_actions)
        
        with st.spinner("Initializing DQN agent and environment..."):
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                lr=lr,
                batch_size=batch_size,
                buffer_size=buffer_size,
                target_update=target_update
            )
            st.session_state.dqn_agent = agent
            st.session_state.dqn_env = RadioResourceEnv(ground_model)
        st.success("DQN agent initialized")
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        # Run training (synchronous)
        train_dqn(int(num_episodes), int(max_steps), int(plot_every))
    
    # Display results if training data exists
    if 'dqn_rewards' in st.session_state:
        display_dqn_results()

def train_dqn(num_episodes, max_steps, plot_every=1):
    """Train the DQN agent with safe, live updates via placeholders."""
    if 'dqn_env' not in st.session_state:
        st.session_state.dqn_env = RadioResourceEnv(st.session_state.ground_model)
    
    agent = st.session_state.dqn_agent
    env = st.session_state.dqn_env
    
    rewards = []
    avg_rewards = []
    loss_history = []
    epsilons = []
    
    # placeholders for live updates (unique objects, no keys needed)
    progress_bar = st.progress(0)
    status_text = st.empty()
    reward_placeholder = st.empty()
    loss_placeholder = st.empty()
    epsilon_placeholder = st.empty()
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_losses = []
        
        for step in range(max_steps):
            try:
                action = agent.select_action(state)
            except Exception as e:
                st.error(f"select_action error at episode {episode}, step {step}: {e}")
                action = 0
            
            try:
                next_state, reward, done, _ = env.step(action)
            except Exception as e:
                st.error(f"env.step error at episode {episode}, step {step}: {e}")
                break
            
            try:
                agent.store_transition(state, action, reward, next_state, done)
            except Exception:
                # non-fatal for training UI
                pass
            
            try:
                loss = agent.train()
            except Exception as e:
                st.error(f"agent.train() error at episode {episode}, step {step}: {e}")
                loss = None
            
            if loss:
                try:
                    episode_losses.append(float(loss))
                except Exception:
                    episode_losses.append(loss)
            
            state = next_state
            episode_reward += float(reward)
            
            if done:
                break
        
        # end of episode bookkeeping
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]) if len(rewards) >= 1 else episode_reward)
        epsilons.append(getattr(agent, 'epsilon', np.nan))
        if episode_losses:
            loss_history.append(np.mean(episode_losses))
        
        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"Episode {episode+1}/{num_episodes} — Reward: {episode_reward:.4f} — Epsilon: {getattr(agent, 'epsilon', np.nan):.4f}")
        
        # update plots via placeholders (no keys)
        if (episode + 1) % max(1, plot_every) == 0:
            update_plots(rewards, avg_rewards, loss_history, epsilons,
                         reward_placeholder, loss_placeholder, epsilon_placeholder)
        
        # yield briefly so Streamlit can repaint
        time.sleep(0.01)
    
    # Save results and training time in session state
    training_time = time.time() - start_time
    st.session_state.dqn_rewards = rewards
    st.session_state.dqn_avg_rewards = avg_rewards
    st.session_state.dqn_loss_history = loss_history
    st.session_state.dqn_epsilons = epsilons
    st.session_state.dqn_training_time = training_time
    
    # Final stable charts (use unique keys)
    try:
        final_reward_fig = px.line(
            pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards, "Avg Reward": avg_rewards}),
            x="Episode", y=["Reward", "Avg Reward"], title="Final Training Rewards", labels={"value": "Reward"}
        )
        st.plotly_chart(final_reward_fig, use_container_width=True, key="dqn_reward_chart_final")
    except Exception as e:
        st.warning(f"Could not render final reward chart: {e}")
    
    if loss_history:
        try:
            final_loss_fig = px.line(
                pd.DataFrame({"Episode": range(len(loss_history)), "Loss": loss_history}),
                x="Episode", y="Loss", title="Final Training Loss", log_y=True
            )
            st.plotly_chart(final_loss_fig, use_container_width=True, key="dqn_loss_chart_final")
        except Exception as e:
            st.warning(f"Could not render final loss chart: {e}")
    
    try:
        final_epsilon_fig = px.line(
            pd.DataFrame({"Episode": range(len(epsilons)), "Epsilon": epsilons}),
            x="Episode", y="Epsilon", title="Final Epsilon Decay"
        )
        st.plotly_chart(final_epsilon_fig, use_container_width=True, key="dqn_epsilon_chart_final")
    except Exception as e:
        st.warning(f"Could not render final epsilon chart: {e}")
    
    st.success(f"✅ DQN training completed in {training_time:.2f} seconds!")

def update_plots(rewards, avg_rewards, loss_history, epsilons, 
                reward_placeholder, loss_placeholder, epsilon_placeholder):
    """Update training plots via placeholders (safe for repeated updates)."""
    # Reward plot
    try:
        reward_df = pd.DataFrame({
            "Episode": range(len(rewards)),
            "Reward": rewards,
            "Avg Reward": avg_rewards
        })
        fig_reward = px.line(reward_df, x="Episode", y=["Reward", "Avg Reward"],
                             title="Training Rewards (live)", labels={"value": "Reward"})
        reward_placeholder.plotly_chart(fig_reward, use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to update live reward plot: {e}")
    
    # Loss plot
    try:
        if loss_history:
            loss_df = pd.DataFrame({
                "Episode": range(len(loss_history)),
                "Loss": loss_history
            })
            fig_loss = px.line(loss_df, x="Episode", y="Loss", title="Training Loss (live)", log_y=True)
            loss_placeholder.plotly_chart(fig_loss, use_container_width=True)
        else:
            loss_placeholder.empty()
    except Exception as e:
        st.warning(f"Failed to update live loss plot: {e}")
    
    # Epsilon plot
    try:
        epsilon_df = pd.DataFrame({
            "Episode": range(len(epsilons)),
            "Epsilon": epsilons
        })
        fig_epsilon = px.line(epsilon_df, x="Episode", y="Epsilon", title="Epsilon Decay (live)")
        epsilon_placeholder.plotly_chart(fig_epsilon, use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to update live epsilon plot: {e}")

def display_dqn_results():
    """Display DQN training results"""
    rewards = st.session_state.dqn_rewards
    avg_rewards = st.session_state.dqn_avg_rewards
    loss_history = st.session_state.dqn_loss_history
    epsilons = st.session_state.dqn_epsilons
    
    st.subheader("Training Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Reward", f"{rewards[-1]:.2f}")
    with col2:
        st.metric("Avg Last 10 Rewards", f"{np.mean(rewards[-10:]):.2f}")
    with col3:
        st.metric("Final Epsilon", f"{epsilons[-1]:.4f}")
    
    st.subheader("Reward Progression")
    reward_df = pd.DataFrame({
        "Episode": range(len(rewards)),
        "Reward": rewards,
        "Avg Reward": avg_rewards
    })
    st.plotly_chart(
        px.line(reward_df, x="Episode", y=["Reward", "Avg Reward"],
                title="Training Rewards", labels={"value": "Reward"}),
        use_container_width=True,
        key="dqn_reward_chart"
    )
    
    if loss_history:
        st.subheader("Loss")
        loss_df = pd.DataFrame({
            "Episode": range(len(loss_history)),
            "Loss": loss_history
        })
        st.plotly_chart(
            px.line(loss_df, x="Episode", y="Loss", title="Training Loss", log_y=True),
            use_container_width=True,
            key="dqn_loss_chart"
        )
    
    st.subheader("Epsilon Decay")
    epsilon_df = pd.DataFrame({
        "Episode": range(len(epsilons)),
        "Epsilon": epsilons
    })
    st.plotly_chart(
        px.line(epsilon_df, x="Episode", y="Epsilon", title="Epsilon Decay"),
        use_container_width=True,
        key="dqn_epsilon_chart"
    )
