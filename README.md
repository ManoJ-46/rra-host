# Radio Resource Allocation using Reinforcement Learning

A **Streamlit-powered web application** for simulating and analyzing **radio resource allocation** using **Markov Decision Processes (MDPs)** and **Reinforcement Learning (RL)**.  
This platform integrates exact MDP solvers, abstraction techniques, and deep RL agents — all visualized interactively with 2D/3D plots.

---

## Key Features

### Ground MDP Model
- Solves the **exact radio resource allocation** problem via **Value Iteration**.
- Fully configurable:
  - Number of UEs (User Equipments)
  - Number of RBs (Resource Blocks)
  - Buffer sizes and arrival rates
  - Channel quality (CQI)
- Produces optimal value functions and policies.

---

### Abstract Model
- Generates **simplified abstractions** of the ground model for scalability.
- Supported strategies:
  - **Uniform abstraction**
  - **Similarity-based abstraction**
  - **Dissimilarity-based abstraction**
- Includes state-aggregation visualization and performance comparison.

---

### Reinforcement Learning Agents
- Implements:
  - **HDDPG (Hybrid Deep Deterministic Policy Gradient)** — continuous control.
  - **DQN (Deep Q-Network)** — discrete control.
- Features:
  - Live reward graphs updated during training.
  - Performance tracking (Avg/Final reward, training time).
  - Automatic CUDA/CPU device selection.

---

### 📊 Comparison Dashboard
- Compare results across **Ground**, **Abstract**, and **RL Agents**.
- Includes:
  - Interactive 2D & 3D visualizations (Plotly)
  - Pearson correlation & MAE statistics
  - Policy-action comparison tables
  - Downloadable CSV / PNG outputs

---

### Responsive Interface
- Optimized for **desktop and mobile** with adaptive CSS.
- Ready for **Streamlit Cloud** deployment or local execution.

---

## Architecture Overview
📂 app/
 ┣ 📜 app.py                    # Main Streamlit app (navigation & layout)
 ┣ 📂 src/
 ┃ ┣ 📂 pages/
 ┃ ┃ ┣ ground_model_page.py     # Ground MDP setup & computation
 ┃ ┃ ┣ abstract_model_page.py   # Abstract model creation & visualization
 ┃ ┃ ┣ hddpg_page.py            # HDDPG training interface
 ┃ ┃ ┣ dqn_page.py              # DQN training interface
 ┃ ┃ ┗ comparison_page.py       # 3D comparison dashboard & metrics
 ┃ ┣ 📂 models/
 ┃ ┃ ┣ ground_model.py          # Value-iteration solver for exact MDP
 ┃ ┃ ┣ abstract_model.py        # State aggregation & abstraction logic
 ┃ ┃ ┣ hddpg_agent.py           # HDDPG reinforcement learning agent
 ┃ ┃ ┣ dqn_agent.py             # DQN reinforcement learning agent
 ┃ ┃ ┗ environments.py          # Simulation environment for RL agents
 ┃ ┗ 📂 utils/                   # Optional helper utilities (if any)



## Installation

### Requirements
- Python ≥ 3.9  
- PyTorch  
- Streamlit  
- Plotly  
- NumPy, Pandas  
- *(Optional)* Kaleido for PNG export

### Steps
```bash
# Clone the repository
git clone https://github.com/ManoJ-46/rra-host
cd rra-host

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
