import gym
import numpy as np
import torch

class RadioResourceEnv(gym.Env):
    def __init__(self, ground_model):
        super().__init__()
        self.ground_model = ground_model

        self.num_UEs = int(ground_model.number_UEs.item())  # Ensure this is a tensor
        self.max_size = int(ground_model.max_size.item())  # Ensure this is a tensor
        self.num_actions = ground_model.number_actions  # Assuming this is already an integer

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_size, shape=(self.num_UEs,), dtype=np.int32
        )

        self.current_state = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        idx = np.random.randint(0, int(self.ground_model.number_states))
        self.current_state = self.ground_model.states_matrix[idx].cpu().numpy()
        self.current_step = 0
        return self.current_state.copy()

    def step(self, action_idx):
        action_tensor = self.ground_model.actions_matrix[action_idx]

        remainder = self.ground_model.remainder_fn(
            torch.tensor(self.current_state), action_tensor
        )

        next_state = np.zeros_like(self.current_state)
        for i in range(self.num_UEs):
            lam = self.ground_model.arrival_rates_vector[i].item()
            arrivals = np.random.poisson(lam)
            next_state[i] = min(remainder[i].item() + arrivals, self.max_size)

        cost = self.ground_model.cost_fn(
            torch.tensor(self.current_state), action_tensor
        ).item()
        reward = -cost

        self.current_state = next_state.copy()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}

class DimReductionEnv(gym.Env):
    def __init__(self, ground_model):
        super().__init__()
        self.ground_model = ground_model

        self.num_UEs = int(ground_model.number_UEs.item())
        self.num_RBs = int(ground_model.number_RBs.item())
        self.max_size = int(ground_model.max_size.item())

        self.flat_action_dim = self.num_UEs * self.num_RBs

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.flat_action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_size, shape=(self.num_UEs,), dtype=np.int32
        )

        self.current_state = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        idx = np.random.randint(0, int(self.ground_model.number_states))
        self.current_state = self.ground_model.states_matrix[idx].cpu().numpy()
        self.current_step = 0
        return self.current_state.copy()

    def step(self, action):
        action = np.asarray(action).flatten()
        if action.size != self.flat_action_dim:
            raise ValueError(
                f"[DimReductionEnv] Invalid action size: expected {self.flat_action_dim}, got {action.size}"
            )

        try:
            action_2d = action.reshape(self.num_RBs, self.num_UEs)
        except Exception as e:
            raise ValueError(f"[DimReductionEnv] Reshape failed: {e}")

        assignment = np.argmax(action_2d, axis=1)

        remainder = self.ground_model.remainder_fn(
            torch.tensor(self.current_state),
            torch.tensor(assignment, dtype=torch.int) 
        )

        next_state = np.zeros_like(self.current_state)
        for i in range(self.num_UEs):
            lam = self.ground_model.arrival_rates_vector[i].item()
            arrivals = np.random.poisson(lam)
            next_state[i] = min(remainder[i].item() + arrivals, self.max_size)

        cost = self.ground_model.cost_fn(
            torch.tensor(self.current_state),
            torch.tensor(assignment, dtype=torch.int)
        ).item()
        reward = -cost

        self.current_state = next_state.copy()
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return next_state, reward, done, {}
