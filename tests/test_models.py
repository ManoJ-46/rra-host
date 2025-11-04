import pytest
import torch
from src.models.ground_model import GroundModel
from src.models.abstract_model import AbstractModel

@pytest.fixture
def small_ground_model():
    return GroundModel(max_size=2, number_UEs=2, number_RBs=1)

def test_ground_model_creation(small_ground_model):
    assert small_ground_model.number_states == 9
    assert small_ground_model.number_actions == 2

def test_ground_model_solve(small_ground_model):
    policy, value, error, iterations = small_ground_model.solve()
    assert error < small_ground_model.precision
    assert len(policy) == small_ground_model.number_states
    assert len(value) == small_ground_model.number_states

def test_abstract_model_creation(small_ground_model):
    abstract_model = AbstractModel(small_ground_model, number_groups=1)
    assert abstract_model.number_states == 5  # (0,1,2,3,4)

def test_abstract_model_solve(small_ground_model):
    abstract_model = AbstractModel(small_ground_model, number_groups=1)
    policy, value, error, iterations = abstract_model.solve()
    assert error < small_ground_model.precision
    assert len(policy) == abstract_model.number_states
    assert len(value) == abstract_model.number_states