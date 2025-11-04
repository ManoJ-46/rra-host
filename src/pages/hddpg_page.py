import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import os, sys, time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.hddpg_agent import HDDPGAgent
from models.environments import DimReductionEnv

def render():
    st.title("🤖 HDDPG Training")
    st.markdown("Train an agent for resource allocation using HDDPG")

    # Check if the ground model exists
    if 'ground_model' not in st.session_state or st.session_state.ground_model is None:
        st.warning("Please create and solve a Ground Model first.")
        return

    g = st.session_state.ground_model

    st.sidebar.header("Hyperparameters")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.95, 0.01)
    tau = st.sidebar.slider("Soft Update Rate (τ)", 0.001, 0.1, 0.005, 0.001)
    actor_lr = st.sidebar.number_input("Actor Learning Rate", 1e-5, 1e-2, 1e-3, 1e-5)
    critic_lr = st.sidebar.number_input("Critic Learning Rate", 1e-5, 1e-2, 1e-3, 1e-5)
    episodes = st.sidebar.number_input("Episodes", 10, 1000, 100, 10)
    max_steps = st.sidebar.number_input("Max Steps", 10, 1000, 200, 10)
    batch_size = st.sidebar.number_input("Batch Size", 16, 256, 64, 16)
    buffer_size = st.sidebar.number_input("Replay Buffer Size", 1000, 100000, 10000, 1000)
    plot_every = st.sidebar.number_input("Update plots every N episodes", 1, 50, 1, 1)

    if 'hddpg_agent' not in st.session_state or 'hddpg_env' not in st.session_state:
        with st.spinner("Initializing agent and environment..."):
            state_dim = int(g.number_UEs.item())
            action_dim = int(g.number_RBs.item()) * state_dim
            st.session_state.hddpg_agent = HDDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                gamma=gamma,
                tau=tau,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                batch_size=batch_size,
                buffer_size=buffer_size
            )
            st.session_state.hddpg_env = DimReductionEnv(g)
        st.success("Agent and environment initialized")

    if st.button("🚀 Start Training", type="primary"):
        train_hddpg(int(episodes), int(max_steps), int(plot_every))

    if 'hddpg_rewards' in st.session_state:
        display_hddpg_results()

def train_hddpg(episodes, max_steps, plot_every=1):
    """Train HDDPG with live updates using placeholders and error handling."""
    agent = st.session_state.hddpg_agent
    env = st.session_state.hddpg_env

    rewards = []
    avg_rewards = []
    losses = []
    bar = st.progress(0)
    status = st.empty()
    # placeholders
    reward_placeholder = st.empty()
    loss_placeholder = st.empty()

    start_time = time.time()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_losses = []

        for step in range(max_steps):
            try:
                action = agent.select_action(state)
            except Exception as e:
                st.error(f"HDDPG select_action error at ep {ep}, step {step}: {e}")
                # fallback to zeros
                action = np.zeros(agent.action_dim, dtype=float)

            try:
                next_state, reward, done, _ = env.step(action)
            except Exception as e:
                st.error(f"HDDPG env.step error at ep {ep}, step {step}: {e}")
                break

            try:
                agent.store_transition(state, action, reward, next_state, done)
            except Exception:
                pass

            try:
                loss = agent.train()
            except Exception as e:
                st.error(f"HDDPG agent.train() error at ep {ep}, step {step}: {e}")
                loss = None

            if loss:
                try:
                    ep_losses.append(float(loss))
                except Exception:
                    ep_losses.append(loss)

            state = next_state
            total_reward += float(reward)
            if done:
                break

        rewards.append(total_reward)
        avg_rewards.append(np.mean(rewards[-10:]) if len(rewards) >= 1 else total_reward)
        if ep_losses:
            losses.append(np.mean(ep_losses))

        bar.progress((ep + 1) / episodes)
        status.text(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.4f}")

        # update plots via placeholders
        if (ep + 1) % max(1, plot_every) == 0:
            try:
                df = pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards, "Avg": avg_rewards})
                fig = px.line(df, x="Episode", y=["Reward", "Avg"], title="HDDPG Rewards (live)")
                reward_placeholder.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to update HDDPG reward plot: {e}")

            if losses:
                try:
                    df_loss = pd.DataFrame({"Episode": range(len(losses)), "Loss": losses})
                    fig_l = px.line(df_loss, x="Episode", y="Loss", title="HDDPG Loss (live)", log_y=True)
                    loss_placeholder.plotly_chart(fig_l, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to update HDDPG loss plot: {e}")
            else:
                loss_placeholder.empty()

        # tiny yield for UI
        time.sleep(0.01)

    training_time = time.time() - start_time
    st.session_state.hddpg_rewards = rewards
    st.session_state.hddpg_avg_rewards = avg_rewards
    st.session_state.hddpg_loss_history = losses
    st.session_state.hddpg_training_time = training_time

    # final stable charts with unique keys
    try:
        final_fig = px.line(pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards, "Avg": avg_rewards}),
                             x="Episode", y=["Reward", "Avg"], title="HDDPG Final Rewards")
        st.plotly_chart(final_fig, use_container_width=True, key="hddpg_reward_chart_final")
    except Exception as e:
        st.warning(f"Could not render final HDDPG reward chart: {e}")

    if losses:
        try:
            final_loss = px.line(pd.DataFrame({"Episode": range(len(losses)), "Loss": losses}), x="Episode", y="Loss", title="HDDPG Final Loss", log_y=True)
            st.plotly_chart(final_loss, use_container_width=True, key="hddpg_loss_chart_final")
        except Exception as e:
            st.warning(f"Could not render final HDDPG loss chart: {e}")

    st.success(f"Training complete in {training_time:.2f} seconds")
    
def display_hddpg_results():
    rew = st.session_state.hddpg_rewards
    avg = st.session_state.hddpg_avg_rewards
    loss = st.session_state.hddpg_loss_history

    st.subheader("Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Reward", f"{rew[-1]:.2f}")
    col2.metric("Avg Last 10", f"{np.mean(rew[-10:]):.2f}")
    if loss:
        col3.metric("Final Loss", f"{loss[-1]:.4f}")

    # Use stable keys for display page charts (different from final training keys)
    st.plotly_chart(
        px.line(pd.DataFrame({"Episode": range(len(rew)), "Reward": rew, "Avg": avg}),
                x="Episode", y=["Reward", "Avg"], title="Rewards"),
        use_container_width=True,
        key="hddpg_reward_chart"
    )

    if loss:
        st.plotly_chart(
            px.line(pd.DataFrame({"Episode": range(len(loss)), "Loss": loss}),
                    x="Episode", y="Loss", title="Loss", log_y=True),
            use_container_width=True,
            key="hddpg_loss_chart"
        )
