import torch
import numpy as np
import os
import time
import shutil
from scipy.stats import poisson
from itertools import product
import streamlit as st

class GroundModel:
    """
    Complete implementation of the radio resource allocation ground model.
    
    This class implements the exact MDP formulation for radio resource allocation
    with configurable parameters for UEs, RBs, buffer sizes, and cost functions.
    """
    
    def __init__(self, max_size=3, number_UEs=6, number_RBs=2,
                 arrival_rates=None, CQI_base=1.0, CQIs_are_equal=True,
                 coef_of_drop=1.0, coef_of_latency=1.0, 
                 power_of_drop=1.0, power_of_latency=1.0,
                 discount_factor=0.9, precision=1e-16):
        """
        Initialize the ground model with all parameters.
        
        Args:
            max_size (int): Maximum buffer size for each UE
            number_UEs (int): Number of User Equipment devices
            number_RBs (int): Number of Resource Blocks
            arrival_rates (list): Arrival rates for each UE (if None, uses default)
            CQI_base (float): Base Channel Quality Indicator value
            CQIs_are_equal (bool): Whether all CQIs are equal
            coef_of_drop (float): Coefficient for drop cost (α)
            coef_of_latency (float): Coefficient for latency cost (β)
            power_of_drop (float): Power for drop cost (x)
            power_of_latency (float): Power for latency cost (y)
            discount_factor (float): MDP discount factor (γ)
            precision (float): Convergence precision for value iteration
        """
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Basic parameters
        self.max_size = torch.tensor([max_size], dtype=torch.int).to(self.device)
        self.number_UEs = torch.tensor([number_UEs], dtype=torch.int).to(self.device)
        self.number_RBs = torch.tensor([number_RBs], dtype=torch.int).to(self.device)
        
        # Cost function parameters
        self.coef_of_drop = torch.tensor([coef_of_drop], dtype=torch.float).to(self.device)
        self.coef_of_latency = torch.tensor([coef_of_latency], dtype=torch.float).to(self.device)
        self.power_of_drop = torch.tensor([power_of_drop], dtype=torch.float).to(self.device)
        self.power_of_latency = torch.tensor([power_of_latency], dtype=torch.float).to(self.device)
        
        # MDP parameters
        self.discount_factor = torch.tensor([discount_factor], dtype=torch.float).to(self.device)
        self.precision = torch.tensor([precision], dtype=torch.float).to(self.device)
        
        # Calculate state and action spaces
        self.number_states = (max_size + 1) ** number_UEs
        self.number_actions = number_UEs ** number_RBs
        
        # Create state and action spaces
        self._create_state_action_spaces()
        
        # Set arrival rates
        self._set_arrival_rates(arrival_rates)
        
        # Set CQI matrix
        self._set_cqi_matrix(CQI_base, CQIs_are_equal)
        
        # Initialize solution storage
        self.solution = None
        self.error = None
        
        # Useful constants
        self.zero = torch.tensor([0], dtype=torch.float).to(self.device)
        self.one = torch.tensor([1], dtype=torch.float).to(self.device)
        self.two = torch.tensor([2], dtype=torch.float).to(self.device)
    
    def _create_state_action_spaces(self):
        """Create state and action spaces for the MDP"""
        # State space: all possible combinations of buffer levels
        range_buffer_levels = torch.tensor(range(self.max_size.item() + 1))
        state_combinations = list(product(range_buffer_levels, repeat=self.number_UEs.item()))
        self.states_matrix = torch.tensor(state_combinations).to(self.device)
        
        # Action space: all possible RB assignments
        range_ue_indices = torch.tensor(range(self.number_UEs.item()))
        action_combinations = list(product(range_ue_indices, repeat=self.number_RBs.item()))
        self.actions_matrix = torch.tensor(action_combinations).to(self.device)
        
        # Index ranges
        self.range_state_indices = torch.tensor(range(self.number_states)).to(self.device)
        self.range_action_indices = torch.tensor(range(self.number_actions)).to(self.device)
        self.range_UE_indices = torch.tensor(range(self.number_UEs.item())).to(self.device)
        self.range_RB_indices = torch.tensor(range(self.number_RBs.item())).to(self.device)
        
        # Buffer states
        self.empty_buffer = torch.zeros([self.number_UEs.item()], dtype=torch.int).to(self.device)
        self.full_buffer = self.max_size * torch.ones([self.number_UEs.item()], dtype=torch.int).to(self.device)
    
    def _set_arrival_rates(self, arrival_rates):
        """Set arrival rates for each UE"""
        if arrival_rates is None:
            # Default arrival rates
            self.arrival_rates_vector = (self.max_size.float() / 3) * torch.ones([self.number_UEs.item()]).to(self.device)
        else:
            self.arrival_rates_vector = torch.tensor(arrival_rates, dtype=torch.float).to(self.device)
    
    def _set_cqi_matrix(self, CQI_base, CQIs_are_equal):
        """Set Channel Quality Indicator matrix"""
        if CQIs_are_equal:
            self.CQI_matrix = CQI_base * torch.ones([self.number_UEs.item(), self.number_RBs.item()], 
                                                   dtype=torch.float).to(self.device)
        else:
            # For now, use equal CQIs. Can be extended for different values
            self.CQI_matrix = CQI_base * torch.ones([self.number_UEs.item(), self.number_RBs.item()], 
                                                   dtype=torch.float).to(self.device)
    
    def remainder_fn(self, state, action):
        """
        Calculate remaining bits in each UE's buffer after resource allocation.
        
        Args:
            state (torch.Tensor): Current state (buffer levels)
            action (torch.Tensor): Action (RB assignment)
            
        Returns:
            torch.Tensor: Remaining buffer levels after transmission
        """
        state = state.to(self.device)
        schedule = torch.zeros([self.number_UEs.item()], dtype=torch.float).to(self.device)
        
        # Calculate scheduled transmission for each UE
        for i, j in product(self.range_UE_indices, self.range_RB_indices):
            if action[j] == i:
                schedule[i] += self.CQI_matrix[i, j]
        
        schedule = schedule.to(torch.int)
        remainder = state - schedule
        
        # Ensure non-negative remainders
        return torch.max(self.empty_buffer, remainder)
    
    def transition_probability_fn(self, state1, action, state2):
        """
        Calculate transition probability from state1 to state2 under action.
        
        Args:
            state1 (torch.Tensor): Current state
            action (torch.Tensor): Action taken
            state2 (torch.Tensor): Next state
            
        Returns:
            torch.Tensor: Transition probability
        """
        def ue_transition_probability(rest, rate, next_level):
            """Calculate transition probability for a single UE"""
            max_size = self.max_size.item()
            
            if next_level < rest:
                return torch.tensor([0.0], dtype=torch.float).to(self.device)
            
            if next_level < max_size:
                return torch.tensor([poisson.pmf(next_level - rest, rate)], 
                                  dtype=torch.float).to(self.device)
            
            # For maximum buffer level, calculate tail probability
            probability = torch.tensor([1.0], dtype=torch.float).to(self.device)
            for arrival in range(max_size - rest):
                probability -= torch.tensor([poisson.pmf(arrival, rate)], 
                                          dtype=torch.float).to(self.device)
            
            return probability
        
        remainder = self.remainder_fn(state1, action)
        probability = torch.tensor([1.0], dtype=torch.float).to(self.device)
        
        for i in self.range_UE_indices:
            ue_prob = ue_transition_probability(remainder[i].item(), 
                                              self.arrival_rates_vector[i].item(), 
                                              state2[i].item())
            probability *= ue_prob
        
        return probability
    
    def cost_fn(self, state, action):
        """
        Calculate expected cost for taking action in state.
        
        Args:
            state (torch.Tensor): Current state
            action (torch.Tensor): Action taken
            
        Returns:
            torch.Tensor: Expected cost
        """
        def exact_cost_fn(remainder, next_state):
            """Calculate exact cost for transition"""
            def partial_cost_fn(rest, arrival):
                """Calculate cost for single UE"""
                excess = torch.max(self.zero, arrival + rest - self.max_size.float())
                drop_cost = self.coef_of_drop * (excess ** self.power_of_drop)
                latency_cost = self.coef_of_latency * (rest.float() ** self.power_of_latency)
                return drop_cost + latency_cost
            
            total_cost = torch.tensor([0.0], dtype=torch.float).to(self.device)
            
            for q in self.range_UE_indices:
                if next_state[q] < remainder[q]:
                    return torch.tensor([0.0], dtype=torch.float).to(self.device)
                
                arrival = next_state[q] - remainder[q]
                total_cost += partial_cost_fn(remainder[q], arrival)
            
            return total_cost
        
        remainder = self.remainder_fn(state, action)
        expected_cost = torch.tensor([0.0], dtype=torch.float).to(self.device)
        
        for s in self.range_state_indices:
            possible_state = self.states_matrix[s]
            cost = exact_cost_fn(remainder, possible_state)
            prob = self.transition_probability_fn(state, action, possible_state)
            expected_cost += cost * prob
        
        return expected_cost
    
    def solve(self, progress_callback=None):
        """
        Solve the MDP using value iteration.
        
        Args:
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            tuple: (policy, value_function, error, iterations)
        """
        # Initialize value function and policy
        old_value = torch.zeros([self.number_states], dtype=torch.float).to(self.device)
        new_value = torch.zeros([self.number_states], dtype=torch.float).to(self.device)
        policy = torch.zeros([self.number_states], dtype=torch.int).to(self.device)
        
        gamma = self.discount_factor.item()
        gamma_prime = gamma / (1 - gamma)
        precision = self.precision.item()
        
        error = float('inf')
        iteration = 0
        
        while error > precision:
            iteration += 1
            
            for s in self.range_state_indices:
                state = self.states_matrix[s]
                Q = torch.zeros([self.number_actions], dtype=torch.float).to(self.device)
                
                for a in self.range_action_indices:
                    action = self.actions_matrix[a]
                    
                    # Calculate expected value
                    expected_value = torch.tensor([0.0], dtype=torch.float).to(self.device)
                    for s2 in self.range_state_indices:
                        next_state = self.states_matrix[s2]
                        prob = self.transition_probability_fn(state, action, next_state)
                        expected_value += prob * old_value[s2]
                    
                    Q[a] = self.cost_fn(state, action) + self.discount_factor * expected_value
                
                new_value[s] = torch.min(Q)
                policy[s] = torch.argmin(Q)
                
                # Progress callback
                if progress_callback:
                    progress = (s.item() + 1) / self.number_states
                    progress_callback(progress, iteration, error)
            
            # Calculate error
            diff = torch.abs(new_value - old_value)
            max_diff = torch.max(diff).item()
            error = gamma_prime * max_diff
            
            old_value = new_value.clone()
        
        # Store solution
        self.solution = torch.stack([policy.float(), new_value])
        self.error = error
        
        return policy, new_value, error, iteration
    
    def get_optimal_action(self, state):
        """Get optimal action for a given state"""
        if self.solution is None:
            raise ValueError("Model not solved yet. Call solve() first.")
        
        # Find state index
        for s_idx, s in enumerate(self.states_matrix):
            if torch.equal(s, state):
                action_idx = int(self.solution[0, s_idx].item())
                return self.actions_matrix[action_idx]
        
        raise ValueError("State not found in state space")
    
    def get_state_value(self, state):
        """Get value function for a given state"""
        if self.solution is None:
            raise ValueError("Model not solved yet. Call solve() first.")
        
        # Find state index
        for s_idx, s in enumerate(self.states_matrix):
            if torch.equal(s, state):
                return self.solution[1, s_idx].item()
        
        raise ValueError("State not found in state space")