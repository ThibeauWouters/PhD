from typing import Sequence, Callable
import functools

import jax
import jax.numpy as jnp

from flax import linen as nn  # Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax.training.train_state import TrainState

# from clu import metrics
import optax

"""Neural network architectures"""

class MLP(nn.Module):
    
    layer_sizes: Sequence[int] # sizes of the hidden layers of the neural network
    act_func: Callable # activation function applied on each hidden layer
    
    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]
    
    # @functools.partial(jax.jit, static_argnums=(2, 3))
    @nn.compact
    def __call__(self, x):
        """_summary_

        Args:
            x (data): Input data of the neural network. 
        """
        
        for i, layer in enumerate(self.layers):
            # Apply the linear part of the layer's operation
            x = layer(x)
            # If not the output layer, apply the given activation function
            if i != len(self.layer_sizes) - 1:
                x = self.act_func(x)
                
        return x
                

class NeuralNetwork(nn.Module):
    """A very basic initial neural network used for testing the basic functionalities of Flax.

    Returns:
        NeuralNetwork: The architecture of the neural network
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=24)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


"""Training"""
## TODO what about metrics?
class TrainState(train_state.TrainState):
    
    train_losses: list = []
    val_losses: list = []

def create_train_state(model, test_input, rng, learning_rate):
    """Creates an initial `TrainState`."""
    # Initialize the parameters by passing dummy input
    params = model.init(rng, test_input)['params']
    # Define the optimizer (also called tx in jax/flax)
    optimizer = optax.adam(learning_rate)
    state = TrainState.create(apply_fn = model.apply, params = params, tx = optimizer)
    return state

@jax.jit
def train_step(state, x_train, y_train, x_val = None, y_val = None):
    """Train for a single step."""
    
    def squared_error(params, x, y):
        """For a single datapoint"""
        pred = state.apply_fn({'params': params}, x)
        return jnp.inner(y-pred, y-pred) / 2.0
    
    def loss_fn(params, x, y):
        """Parallelized version for batch data instead of single datapoint"""
        return jnp.mean(jax.vmap(squared_error)(state.params, x, y), axis=0)
    
    # Turn into function that also computes the derivative as well
    grad_fn = jax.value_and_grad(loss_fn)
    
    # Compute the loss and gradients
    train_loss, grads = grad_fn(state.params, x_train, y_train)
    # Compute the loss on validation data if provided
    if x_val is not None:
        val_loss = loss_fn(state.params, x_val, y_val)
        
    # Update the parameters and return new state
    state = state.apply_gradients(grads = grads)
    
    return state

def train_loop(state, number_of_epochs: int = 500, *data):
    
    for i in range(number_of_epochs):
        train_step(state, *data)