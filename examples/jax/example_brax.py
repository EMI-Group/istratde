from tqdm import tqdm
from jax import random
from flax import linen as nn
import jax.numpy as jnp
import jax
import istratde.problems
from istratde.algorithms.jax import IStratDE
from istratde.util import StdSOMonitor, StdWorkflow, TreeAndVector

# 1. Define Policy Network (Swimmer: 8 obs -> 2 actions)
class SwimmerPolicy(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x.reshape(-1)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(2)(x)
        x = nn.tanh(x)
        return x

# 2. Setup Parameter Adapter (Flat Vector <-> PyTree)
key = jax.random.PRNGKey(42)
model = SwimmerPolicy()
weights = model.init(random.PRNGKey(42), jnp.zeros((8,))) # Initialize weights
adapter = TreeAndVector(weights)
vector_form_weights = adapter.to_vector(weights)
D = vector_form_weights.shape[0] # Total weight dimension

# 3. Initialize IStratDE Algorithm
steps = 100
pop_size = 10000
lb = jnp.full((D,), -10.0)
ub = jnp.full((D,), 10.0)

algorithm = IStratDE(
    lb=lb,
    ub=ub,
    pop_size=pop_size,
)

# 4. Setup Problem (Brax Swimmer)
problem = istratde.problems.jax.Brax(
    env_name="swimmer",
    policy=jax.jit(model.apply),
    cap_episode=500,
)

# 5. Initialize Workflow
monitor = StdSOMonitor(record_fit_history=False)

workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    pop_transform=adapter.batched_to_tree, # Convert population back to tree structure
    opt_direction='max', # Maximize reward
)

# 6. Run Loop
key, _ = jax.random.split(key)
state = workflow.init(key)

for i in tqdm(range(steps), desc="Swimmer Evolution"):
    state = workflow.step(state)
    print(f"fitness: {monitor.get_best_fitness()}")

print(f"Best fitness: {monitor.get_best_fitness()}")