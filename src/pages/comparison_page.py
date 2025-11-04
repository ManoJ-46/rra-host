import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import io

def render():
    """Render the model comparison page"""
    st.title("📊 Model Comparison")
    st.markdown("### Compare Ground (exact), Abstract (approx), and RL approaches (HDDPG / DQN)")

    # Required models in session state
    required_models = ['ground_model', 'abstract_model', 'hddpg_agent', 'dqn_agent']
    missing_models = [name for name in required_models if name not in st.session_state or st.session_state[name] is None]

    if missing_models:
        st.warning(f"⚠️ Please create/compute the following first: {', '.join(missing_models)}")
        return

    # Access models and results
    gm = st.session_state.ground_model
    am = st.session_state.abstract_model
    hddpg = st.session_state.hddpg_agent
    dqn = st.session_state.dqn_agent

    # RL reward arrays and training times (may not exist until after training)
    hddpg_rewards = st.session_state.get('hddpg_rewards', [])
    dqn_rewards = st.session_state.get('dqn_rewards', [])
    hddpg_training_time = st.session_state.get('hddpg_training_time', None)
    dqn_training_time = st.session_state.get('dqn_training_time', None)

    # Build the comparison table with columns relevant to each model group
    rows = []

    # --- Ground MDP (exact) ---
    try:
        ground_states = int(gm.number_states)
    except Exception:
        ground_states = "N/A"

    ground_avg_value = "N/A"
    ground_solve_time = st.session_state.get('ground_solve_time', "N/A")
    try:
        if hasattr(gm, 'solution') and gm.solution is not None:
            ground_avg_value = float(gm.solution[1].mean().item())
    except Exception:
        ground_avg_value = "N/A"

    rows.append({
        "Model": "Ground (MDP)",
        "Type": "Exact",
        "States": ground_states,
        "Solve Time (s)": ground_solve_time,
        "Avg Value": ground_avg_value,
        "Training Time (s)": "N/A",
        "Avg Reward": "N/A",
        "Final Reward": "N/A"
    })

    # --- Abstract MDP (approx) ---
    try:
        abstract_states = int(getattr(am, 'number_states', "N/A"))
    except Exception:
        abstract_states = "N/A"
    abstract_avg_value = "N/A"
    abstract_solve_time = getattr(am, 'resolution_elapse_time', "N/A")
    try:
        if hasattr(am, 'solution') and am.solution is not None:
            abstract_avg_value = float(am.solution[1].mean().item())
    except Exception:
        abstract_avg_value = "N/A"

    rows.append({
        "Model": "Abstract (MDP)",
        "Type": "Approximate",
        "States": abstract_states,
        "Solve Time (s)": abstract_solve_time,
        "Avg Value": abstract_avg_value,
        "Training Time (s)": "N/A",
        "Avg Reward": "N/A",
        "Final Reward": "N/A"
    })

    # --- HDDPG (RL) ---
    rows.append({
        "Model": "HDDPG",
        "Type": "RL",
        "States": "N/A",
        "Solve Time (s)": "N/A",
        "Avg Value": "N/A",
        "Training Time (s)": float(hddpg_training_time) if hddpg_training_time is not None else "N/A",
        "Avg Reward": float(np.mean(hddpg_rewards)) if len(hddpg_rewards) > 0 else "N/A",
        "Final Reward": float(hddpg_rewards[-1]) if len(hddpg_rewards) > 0 else "N/A"
    })

    # --- DQN (RL) ---
    rows.append({
        "Model": "DQN",
        "Type": "RL",
        "States": "N/A",
        "Solve Time (s)": "N/A",
        "Avg Value": "N/A",
        "Training Time (s)": float(dqn_training_time) if dqn_training_time is not None else "N/A",
        "Avg Reward": float(np.mean(dqn_rewards)) if len(dqn_rewards) > 0 else "N/A",
        "Final Reward": float(dqn_rewards[-1]) if len(dqn_rewards) > 0 else "N/A"
    })

    comp_df = pd.DataFrame(rows)
    st.subheader("Summary table — metrics by model")
    st.dataframe(comp_df, use_container_width=True, key="comparison_summary_table")

    st.markdown("---")
    st.subheader("Value function comparison — sampled states")

    # Number of ground states (safe)
    try:
        N_states = int(gm.number_states)
    except Exception:
        st.error("Error reading ground model state count.")
        return

    # Slider to choose sample size
    sample_count = st.slider("Number of states to sample for comparison", 20, min(500, max(20, N_states)), min(100, N_states), 10)

    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(N_states, size=min(sample_count, N_states), replace=False)

    # Ground and abstract values
    ground_values = gm.solution[1][sample_indices].cpu().numpy()
    abstract_values = am.solution[1][sample_indices].cpu().numpy()

    # RL value approximations (HDDPG critic(actor(state)) and DQN max Q)
    hddpg_vals = []
    dqn_vals = []
    hover_texts = []

    for idx in sample_indices:
        state_np = gm.states_matrix[int(idx)].cpu().numpy()
        hover_texts.append(f"idx={int(idx)} • state={state_np.tolist()}")

        # HDDPG estimate
        try:
            s_t = torch.FloatTensor(state_np).unsqueeze(0)
            with torch.no_grad():
                a_h = hddpg.actor(s_t)
                v_h = hddpg.critic(s_t, a_h).cpu().item()
            hddpg_vals.append(v_h)
        except Exception:
            hddpg_vals.append(np.nan)

        # DQN estimate
        try:
            s_t = torch.FloatTensor(state_np).unsqueeze(0)
            with torch.no_grad():
                qvals = dqn.model(s_t)
                v_d = qvals.max().cpu().item()
            dqn_vals.append(v_d)
        except Exception:
            dqn_vals.append(np.nan)

    ground_vals = np.array(ground_values, dtype=float)
    abstract_vals = np.array(abstract_values, dtype=float)
    hddpg_vals = np.array(hddpg_vals, dtype=float)
    dqn_vals = np.array(dqn_vals, dtype=float)

    # 2D subplots: Ground vs Abstract & Ground vs RL
    fig2d = make_subplots(rows=1, cols=2, subplot_titles=("Ground vs Abstract", "Ground vs RL (HDDPG+DQN)"))

    # Ground vs Abstract
    fig2d.add_trace(
        go.Scatter(x=ground_vals, y=abstract_vals, mode='markers', marker=dict(size=6),
                   text=hover_texts, hoverinfo='text+x+y', name="Abstract"),
        row=1, col=1
    )
    minv, maxv = float(np.nanmin(ground_vals)), float(np.nanmax(ground_vals))
    fig2d.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', line=dict(dash='dash', color='black'),
                               showlegend=False), row=1, col=1)

    # Ground vs RL
    fig2d.add_trace(
        go.Scatter(x=ground_vals, y=hddpg_vals, mode='markers', marker=dict(size=6),
                   text=hover_texts, hoverinfo='text+x+y', name="HDDPG"),
        row=1, col=2
    )
    fig2d.add_trace(
        go.Scatter(x=ground_vals, y=dqn_vals, mode='markers', marker=dict(size=6),
                   text=hover_texts, hoverinfo='text+x+y', name="DQN"),
        row=1, col=2
    )
    fig2d.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', line=dict(dash='dash', color='black'),
                               showlegend=False), row=1, col=2)

    fig2d.update_xaxes(title_text="Ground Value", row=1, col=1)
    fig2d.update_yaxes(title_text="Abstract Value", row=1, col=1)
    fig2d.update_xaxes(title_text="Ground Value", row=1, col=2)
    fig2d.update_yaxes(title_text="Predicted Value", row=1, col=2)
    fig2d.update_layout(height=480, showlegend=True, title="2D: Value Comparisons")
    st.plotly_chart(fig2d, use_container_width=True, key="comparison_2d_scatter_v2")

    st.markdown("---")
    st.subheader("3D visualization — Ground vs Abstract vs RL")

    # 3D for HDDPG
    fig3_h = go.Figure()
    fig3_h.add_trace(go.Scatter3d(
        x=ground_vals, y=abstract_vals, z=hddpg_vals,
        mode='markers',
        marker=dict(size=5, color=hddpg_vals, colorscale='Viridis', showscale=True, colorbar=dict(title='HDDPG')),
        text=hover_texts,
        hovertemplate="%{text}<br>Ground=%{x:.3f}<br>Abstract=%{y:.3f}<br>HDDPG=%{z:.3f}<extra></extra>"
    ))
    fig3_h.update_layout(scene=dict(xaxis_title='Ground', yaxis_title='Abstract', zaxis_title='HDDPG'),
                         height=600, title="3D: Ground vs Abstract vs HDDPG")
    st.plotly_chart(fig3_h, use_container_width=True, key="comparison_3d_hddpg_v2")

    # 3D for DQN
    fig3_d = go.Figure()
    fig3_d.add_trace(go.Scatter3d(
        x=ground_vals, y=abstract_vals, z=dqn_vals,
        mode='markers',
        marker=dict(size=5, color=dqn_vals, colorscale='Plasma', showscale=True, colorbar=dict(title='DQN')),
        text=hover_texts,
        hovertemplate="%{text}<br>Ground=%{x:.3f}<br>Abstract=%{y:.3f}<br>DQN=%{z:.3f}<extra></extra>"
    ))
    fig3_d.update_layout(scene=dict(xaxis_title='Ground', yaxis_title='Abstract', zaxis_title='DQN'),
                         height=600, title="3D: Ground vs Abstract vs DQN")
    st.plotly_chart(fig3_d, use_container_width=True, key="comparison_3d_dqn_v2")

    st.markdown("---")
    st.subheader("Numeric comparison & correlations")

    def safe_corr(a, b):
        try:
            if np.all(np.isnan(a)) or np.all(np.isnan(b)):
                return np.nan
            # align lengths and remove NaNs pairs
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() == 0:
                return np.nan
            return float(np.corrcoef(a[mask], b[mask])[0, 1])
        except Exception:
            return np.nan

    corr_data = [
        {
            "Comparison": "Ground vs Abstract",
            "Pearson r": safe_corr(ground_vals, abstract_vals),
            "MAE": float(np.nanmean(np.abs(ground_vals - abstract_vals)))
        },
        {
            "Comparison": "Ground vs HDDPG",
            "Pearson r": safe_corr(ground_vals, hddpg_vals),
            "MAE": float(np.nanmean(np.abs(ground_vals - hddpg_vals)))
        },
        {
            "Comparison": "Ground vs DQN",
            "Pearson r": safe_corr(ground_vals, dqn_vals),
            "MAE": float(np.nanmean(np.abs(ground_vals - dqn_vals)))
        }
    ]
    corr_df = pd.DataFrame(corr_data)
    st.table(corr_df)

    st.markdown("---")
    st.subheader("Policy comparison (sampled states)")

    # Build policy comparison rows
    policy_rows = []
    sample_states = [gm.states_matrix[int(idx)].cpu().numpy() for idx in sample_indices]

    for idx, state in zip(sample_indices, sample_states):
        row = {"state_index": int(idx), "state": state.tolist()}

        # Ground action
        try:
            a_ground = gm.get_optimal_action(state)
            row["ground_action"] = np.array(a_ground).tolist()
        except Exception:
            try:
                a_ground = gm.get_optimal_action(torch.tensor(state))
                row["ground_action"] = np.array(a_ground).tolist()
            except Exception:
                row["ground_action"] = None

        # Abstract action
        try:
            a_abstract = am.get_optimal_action(state)
            row["abstract_action"] = np.array(a_abstract).tolist()
        except Exception:
            try:
                a_abstract = am.get_optimal_action(torch.tensor(state))
                row["abstract_action"] = np.array(a_abstract).tolist()
            except Exception:
                row["abstract_action"] = None

        # HDDPG action -> discrete approximation
        try:
            a_cont = hddpg.select_action(state, explore=False)
            a_flat = np.asarray(a_cont).flatten()
            num_RBs = int(gm.number_RBs.item())
            num_UEs = int(gm.number_UEs.item())
            if a_flat.size == num_RBs * num_UEs:
                try:
                    a_2d = a_flat.reshape(num_RBs, num_UEs)
                    assignment = np.argmax(a_2d, axis=1).tolist()
                except Exception:
                    chunks = np.array_split(a_flat, num_RBs)
                    assignment = [int(np.argmax(c)) for c in chunks]
            elif a_flat.size == num_RBs:
                assignment = [int(x) for x in a_flat]
            else:
                assignment = [int(i % num_UEs) for i in range(num_RBs)]
            row["hddpg_action"] = assignment
        except Exception:
            row["hddpg_action"] = None

        # DQN action
        try:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                qvals = dqn.model(s_t)
                action_idx = int(torch.argmax(qvals, dim=1).cpu().item())
            try:
                assignment = gm.actions_matrix[action_idx].cpu().numpy().tolist()
            except Exception:
                assignment = int(action_idx)
            row["dqn_action"] = assignment
        except Exception:
            row["dqn_action"] = None

        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)
    st.dataframe(policy_df, use_container_width=True, key="comparison_policy_table_v2")

    # Downloadable CSV
    csv_buf = io.StringIO()
    policy_df.to_csv(csv_buf, index=False)
    st.download_button("Download policy comparison (CSV)", data=csv_buf.getvalue(),
                       file_name="policy_comparison.csv", mime="text/csv", key="download_policy_csv_v2")

    st.markdown("### Notes")
    st.markdown("""
    - Ground/Abstract rows report **value-function** stats (Avg Value) — not per-episode rewards.
    - HDDPG/DQN rows report **training** metrics (avg reward over episodes, final episode reward, training time).
    - If you want per-episode 'rewards' for the MDP solutions, you would need to simulate rollouts using the ground/abstract policy and collect episode rewards — not computed automatically by `solve()`.
    - To have `Training Time (s)` populated for RL agents, record training_time in the training functions (examples below).
    """)

    st.markdown("#### How to record RL training time (1-line change)")
    st.code("""
# In dqn_page.py -> after training completes:
training_time = time.time() - start_time
st.session_state.dqn_training_time = training_time

# In hddpg_page.py -> after training completes:
training_time = time.time() - start_time
st.session_state.hddpg_training_time = training_time
    """)
