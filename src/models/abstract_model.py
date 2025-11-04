import torch
import numpy as np
import time
import itertools
from itertools import product
import torch.nn.functional as F
import math

class AbstractModel:
    """
    Implementation of abstract MDP model for radio resource allocation.
    
    This class creates simplified versions of the ground model using various
    abstraction strategies including state aggregation and weight distribution.
    """
    
    def __init__(self, ground_model, number_groups=3, coef_owners="UEs", 
                 coef_distribution_criterion="uniform", variant=None, 
                 selection_mode="one"):
        """
        Initialize abstract model from ground model.
        
        Args:
            ground_model: The ground model to abstract
            number_groups (int): Number of groups for UE aggregation
            coef_owners (str): "UEs" or "groups" - who gets the weights
            coef_distribution_criterion (str): "uniform", "similarity", or "dissimilarity"
            variant (str): For dissimilarity - "sd", "cross", or "gini"
            selection_mode (str): "one", "top", or "all"
        """
        self.model = ground_model
        self.device = ground_model.device
        
        # Validate and set parameters
        self.number_groups = torch.tensor([number_groups], dtype=torch.int).to(self.device)
        self.coef_owners = self._validate_coef_owners(coef_owners)
        self.coef_distribution_criterion = self._validate_criterion(coef_distribution_criterion)
        self.variant = self._validate_variant(variant)
        self.selection_mode = self._validate_selection_mode(selection_mode)
        
        # Create abstraction
        self.abstraction_start_time = time.time()
        self._make_groups_UEs()
        self._make_abstraction_function()
        self._make_weights_distribution()
        self._make_abstract_problem()
        self.abstraction_elapse_time = time.time() - self.abstraction_start_time
        
        # Solution storage
        self.abstract_solution = None
        self.solution = None
        self.precision = None
        self.resolution_elapse_time = None
        self.extrapolation_elapse_time = None
    
    def _validate_coef_owners(self, coef_owners):
        """Validate coefficient owners parameter"""
        if coef_owners.upper() in ["UES", "UE"]:
            return "UEs"
        elif coef_owners.upper() in ["GROUP", "GROUPS"]:
            return "groups"
        else:
            raise ValueError(f"Invalid coef_owners: {coef_owners}. Use 'UEs' or 'groups'")
    
    def _validate_criterion(self, criterion):
        """Validate distribution criterion parameter"""
        if "UNIF" in criterion.upper():
            return "uniform"
        elif "DIS" in criterion.upper():
            return "dissimilarity"
        elif "SIM" in criterion.upper():
            return "similarity"
        else:
            raise ValueError(f"Invalid criterion: {criterion}")
    
    def _validate_variant(self, variant):
        """Validate variant parameter"""
        if variant is None:
            return None
        elif "ST" in variant.upper() or "SD" in variant.upper():
            return "sd"
        elif "CROS" in variant.upper():
            return "cross"
        elif "GINI" in variant.upper() or "TOTAL" in variant.upper():
            return "gini"
        else:
            return variant
    
    def _validate_selection_mode(self, selection_mode):
        """Validate selection mode parameter"""
        if selection_mode.upper() in ["1", "ONE"]:
            return "one"
        elif "TOP" in selection_mode.upper():
            return "top"
        elif selection_mode.upper() in ["ALL", "AL"]:
            return "all"
        else:
            return selection_mode
    
    def _make_groups_UEs(self):
        """Group UEs based on their characteristics"""
        def grouping(N):
            """Group UEs into N groups based on characteristics"""
            # Calculate UE characteristics
            characteristics = (self.model.number_UEs.float() / self.model.number_RBs.float() * 
                             torch.sum(self.model.CQI_matrix, dim=1) - 
                             self.model.arrival_rates_vector)
            
            if N >= self.model.number_UEs.item():
                if characteristics.unique().size(0) == characteristics.size(0):
                    # All UEs have different characteristics
                    sorted_indices = torch.argsort(characteristics)
                    num_elements = characteristics.size(0)
                    vector_groups = torch.zeros(num_elements, dtype=torch.long)
                    vector_groups[sorted_indices] = torch.arange(num_elements)
                    return vector_groups
                else:
                    return grouping(self.model.number_UEs.item() - 1)
            
            # Use quantiles to create groups
            quantiles = torch.linspace(0, 1, N + 1)[1:-1].to(torch.float).to(self.device)
            bounds = torch.quantile(characteristics, quantiles)
            _, group_indices = torch.searchsorted(bounds, characteristics.unsqueeze(1), 
                                                right=True).squeeze(1).unique(return_inverse=True)
            return group_indices
        
        self.group_indices_vector = grouping(self.number_groups.item())
        self.number_groups = torch.tensor([self.group_indices_vector.unique().size(0)], 
                                        dtype=torch.int).to(self.device)
        self.range_group_indices = torch.tensor(range(self.number_groups.item()), 
                                              dtype=torch.int).to(self.device)
        
        # Create groups list
        self.groups_list = []
        for g in self.range_group_indices:
            group_members = torch.where(self.group_indices_vector == g)[0]
            self.groups_list.append(group_members)
        
        # Group sizes
        self.groups_size_vector = torch.tensor([len(group) for group in self.groups_list], 
                                             dtype=torch.int).to(self.device)
    
    def _make_abstraction_function(self):
        """Create abstract states and abstraction function"""
        # Build abstract states
        max_size_groups = torch.zeros([self.number_groups.item()], dtype=torch.int).to(self.device)
        for g in self.range_group_indices:
            max_size_groups[g] = self.model.max_size * len(self.groups_list[g])
        
        # Generate all possible abstract states
        abstract_states = itertools.product(*[range(size.item() + 1) for size in max_size_groups])
        self.states_matrix = torch.tensor(list(abstract_states)).to(self.device)
        self.number_states = self.states_matrix.size(0)
        
        # Build abstraction function
        self.states_vector = torch.zeros([self.model.number_states], dtype=torch.int).to(self.device)
        
        for gs_idx in self.model.range_state_indices:
            ground_state = self.model.states_matrix[gs_idx]
            abstract_state = torch.zeros([self.number_groups.item()], dtype=torch.int).to(self.device)
            
            # Sum buffer levels within each group
            for g in self.range_group_indices:
                abstract_state[g] = ground_state[self.groups_list[g]].sum()
            
            # Find corresponding abstract state index
            for as_idx, row in enumerate(self.states_matrix):
                if torch.all(row == abstract_state):
                    self.states_vector[gs_idx] = as_idx
                    break
        
        # Create classes (sets of ground states mapping to same abstract state)
        self.range_class_indices = torch.tensor(range(self.number_states), dtype=torch.int).to(self.device)
        self.classes_list = []
        for c_idx in self.range_class_indices:
            class_members = torch.where(self.states_vector == c_idx)[0]
            self.classes_list.append(class_members)
    
    def _make_weights_distribution(self):
        """Create weight distribution for states"""
        def coef_fn(state_idx):
            """Calculate coefficients for a given state"""
            state = self.model.states_matrix[state_idx]
            abstract_state_idx = self.states_vector[state_idx]
            abstract_state = self.states_matrix[abstract_state_idx]
            
            if self.coef_distribution_criterion == "uniform":
                if self.coef_owners == "UEs":
                    return torch.ones([self.model.number_UEs.item()], dtype=torch.float).to(self.device)
                else:
                    return torch.ones([self.number_groups.item()], dtype=torch.float).to(self.device)
            
            elif self.coef_owners == "UEs":
                # Weight UEs based on similarity/dissimilarity to group means
                group_means = abstract_state.float() / self.groups_size_vector.float()
                deviations = torch.zeros([self.model.number_UEs.item()], dtype=torch.float).to(self.device)
                
                for ue in self.model.range_UE_indices:
                    g_idx = self.group_indices_vector[ue]
                    deviations[ue] = torch.abs(state[ue].float() - group_means[g_idx])
                
                if self.coef_distribution_criterion == "similarity":
                    return torch.exp(-deviations)
                else:  # dissimilarity
                    return deviations
            
            else:  # coef_owners == "groups"
                coefs = torch.zeros([self.number_groups.item()], dtype=torch.float).to(self.device)
                
                if self.coef_distribution_criterion == "similarity":
                    for g_idx in self.range_group_indices:
                        group_members = self.groups_list[g_idx]
                        state_restriction = state[group_members]
                        num_bits = abstract_state[g_idx]
                        redistribution = torch.full((len(group_members),), num_bits.float()).to(self.device)
                        coefs[g_idx] = F.cosine_similarity(state_restriction.float(), 
                                                         redistribution, dim=0)
                
                else:  # dissimilarity
                    for g_idx in self.range_group_indices:
                        num_bits = abstract_state[g_idx]
                        if num_bits == 0:
                            coefs[g_idx] = 1
                        else:
                            group_members = self.groups_list[g_idx]
                            state_restriction = state[group_members]
                            
                            if self.variant == "sd":
                                # Standard deviation variant
                                mean_bits = num_bits.float() / len(group_members)
                                variance = torch.var(state_restriction.float())
                                coefs[g_idx] = torch.sqrt(variance) / num_bits.float()
                            
                            elif self.variant == "cross":
                                # Cross entropy variant
                                entropy = math.log(len(group_members)) * num_bits.float()
                                coefs[g_idx] = entropy
                            
                            else:  # gini
                                # Gini coefficient variant
                                sum_diff = 0
                                for i, j in itertools.product(group_members, repeat=2):
                                    sum_diff += torch.abs(state[i] - state[j]).float()
                                coefs[g_idx] = sum_diff / (2 * len(group_members) * num_bits.float())
                
                return coefs
        
        # Calculate pre-weights
        preweights = torch.zeros([self.model.number_states], dtype=torch.float).to(self.device)
        for s_idx in self.model.range_state_indices:
            coefs = coef_fn(s_idx)
            preweights[s_idx] = coefs.sum()
        
        # Normalize weights within each class
        self.weight_distribution = torch.zeros([self.model.number_states], dtype=torch.float).to(self.device)
        for c_idx in range(len(self.classes_list)):
            class_states = self.classes_list[c_idx]
            class_weight = preweights[class_states].sum()
            
            if class_weight == 0:
                for s_idx in class_states:
                    self.weight_distribution[s_idx] = 1.0 / len(class_states)
            else:
                for s_idx in class_states:
                    self.weight_distribution[s_idx] = preweights[s_idx] / class_weight
        
        # Apply selection mode
        self._apply_selection_mode()
    
    def _apply_selection_mode(self):
        """Apply selection mode to weight distribution"""
        if self.selection_mode == "all":
            return
        
        final_weights = torch.zeros([self.model.number_states], dtype=torch.float).to(self.device)
        
        for c_idx in range(len(self.classes_list)):
            class_states = self.classes_list[c_idx]
            class_weights = self.weight_distribution[class_states]
            max_weight = class_weights.max()
            
            if self.selection_mode == "one":
                # Select one state randomly from those with maximum weight
                top_indices = torch.where(class_weights == max_weight)[0]
                selected_idx = torch.randint(0, len(top_indices), (1,)).item()
                selected_state = class_states[top_indices[selected_idx]]
                final_weights[selected_state] = 1.0
            
            else:  # selection_mode == "top"
                # Distribute weight equally among top states
                top_indices = torch.where(class_weights == max_weight)[0]
                for idx in top_indices:
                    final_weights[class_states[idx]] = 1.0 / len(top_indices)
        
        self.weight_distribution = final_weights
    
    def _make_abstract_problem(self):
        """Build transition and cost matrices for abstract problem"""
        m = self.model
        NA, NS = m.number_actions, self.number_states
        
        self.transition_matrix = torch.zeros([NA, NS, NS], dtype=torch.float).to(self.device)
        self.cost_matrix = torch.zeros([NA, NS], dtype=torch.float).to(self.device)
        
        for a_idx, c_idx1, c_idx2 in itertools.product(m.range_action_indices, 
                                                      self.range_class_indices, 
                                                      self.range_class_indices):
            
            class1 = self.classes_list[c_idx1]
            class2 = self.classes_list[c_idx2]
            action = m.actions_matrix[a_idx]
            
            # Calculate transition probability
            transition_prob = torch.tensor([0.0], dtype=torch.float).to(self.device)
            cost = torch.tensor([0.0], dtype=torch.float).to(self.device)
            
            for s_idx1 in class1:
                state1 = m.states_matrix[s_idx1]
                weight = self.weight_distribution[s_idx1]
                
                # Transition probability
                prob = torch.tensor([0.0], dtype=torch.float).to(self.device)
                for s_idx2 in class2:
                    state2 = m.states_matrix[s_idx2]
                    prob += m.transition_probability_fn(state1, action, state2)
                
                transition_prob += weight * prob
                
                # Cost
                state_cost = m.cost_fn(state1, action)
                cost += weight * state_cost
            
            self.transition_matrix[a_idx, c_idx1, c_idx2] = transition_prob
            self.cost_matrix[a_idx, c_idx1] = cost
    
    def solve(self, discount_factor=None, precision=None):
        """Solve the abstract MDP using value iteration"""
        # Use model parameters if not provided
        if discount_factor is None:
            discount_factor = self.model.discount_factor.item()
        if precision is None:
            precision = self.model.precision.item()
        
        resolution_start_time = time.time()
        
        # Initialize
        old_value = torch.zeros([self.number_states], dtype=torch.float).to(self.device)
        new_value = torch.zeros([self.number_states], dtype=torch.float).to(self.device)
        policy = torch.zeros([self.number_states], dtype=torch.int).to(self.device)
        
        gamma = discount_factor
        gamma_prime = gamma / (1 - gamma)
        error = float('inf')
        iteration = 0
        
        # Value iteration
        while error > precision:
            iteration += 1
            
            for c_idx in self.range_class_indices:
                Q = torch.zeros([self.model.number_actions], dtype=torch.float).to(self.device)
                
                for a in self.model.range_action_indices:
                    expected_value = torch.dot(self.transition_matrix[a, c_idx], old_value)
                    Q[a] = self.cost_matrix[a, c_idx] + gamma * expected_value
                
                new_value[c_idx] = torch.min(Q)
                policy[c_idx] = torch.argmin(Q)
            
            # Calculate error
            differences = new_value - old_value
            diff_max = torch.max(differences)
            diff_min = torch.min(differences)
            diff_scope = diff_max - diff_min
            error = (gamma_prime * diff_scope).item()
            
            old_value = new_value.clone()
        
        self.abstract_solution = torch.stack([policy.float(), new_value])
        self.resolution_elapse_time = time.time() - resolution_start_time
        self.precision = error
        
        # Extrapolate to ground states
        self._extrapolate_solution()
        
        return policy, new_value, error, iteration
    
    def _extrapolate_solution(self):
        """Extrapolate abstract solution to ground states"""
        extrapolation_start_time = time.time()
        
        solution = torch.zeros([2, self.model.number_states], dtype=torch.float).to(self.device)
        
        for s_idx in self.model.range_state_indices:
            c_idx = self.states_vector[s_idx]
            solution[:, s_idx] = self.abstract_solution[:, c_idx]
        
        self.solution = solution
        self.extrapolation_elapse_time = time.time() - extrapolation_start_time
    
    def get_optimal_action(self, state):
        """Get optimal action for a given state"""
        if self.solution is None:
            raise ValueError("Model not solved yet. Call solve() first.")
        
        # Find state index
        for s_idx, s in enumerate(self.model.states_matrix):
            if torch.equal(s, state):
                action_idx = int(self.solution[0, s_idx].item())
                return self.model.actions_matrix[action_idx]
        
        raise ValueError("State not found in state space")
    
    def get_state_value(self, state):
        """Get value function for a given state"""
        if self.solution is None:
            raise ValueError("Model not solved yet. Call solve() first.")
        
        # Find state index
        for s_idx, s in enumerate(self.model.states_matrix):
            if torch.equal(s, state):
                return self.solution[1, s_idx].item()
        
        raise ValueError("State not found in state space")