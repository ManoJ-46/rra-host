import numpy as np
import torch

def soft_shift_poisson(rate, left_shift=0):
    """
    Returns the mathematical expectation of max(0, X - left_shift)
    where X follows a Poisson distribution.
    """
    if left_shift == 0:
        return rate
    
    # Calculate the tail expectation
    k = np.arange(left_shift)
    probabilities = np.exp(-rate) * (rate**k) / np.math.factorial(k)
    tail_expectation = rate * (1 - np.sum(probabilities))
    
    # Subtract the cumulative probability of values below left_shift
    cumulative_prob = np.sum(probabilities)
    return tail_expectation - left_shift * (1 - cumulative_prob)

def small_mdp_solution(num_states, num_actions, transition_matrix, cost_matrix,
                      discount_factor=0.9, precision=1e-6, device='cpu'):
    """
    Solve a small MDP using value iteration.
    Returns: (policy, value_function)
    """
    # Initialize
    V = torch.zeros(num_states, device=device)
    policy = torch.zeros(num_states, dtype=torch.long, device=device)
    gamma = discount_factor
    threshold = precision * (1 - gamma) / gamma
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s].item()
            # Compute Q-values
            Q = torch.zeros(num_actions, device=device)
            for a in range(num_actions):
                # Immediate cost
                cost = cost_matrix[a, s]
                # Expected future value
                next_value = torch.dot(transition_matrix[a, s], V)
                Q[a] = cost + gamma * next_value
            # Update value and policy
            V[s] = torch.min(Q)
            policy[s] = torch.argmin(Q)
            delta = max(delta, abs(v - V[s].item()))
        # Check convergence
        if delta < threshold:
            break
    return policy, V