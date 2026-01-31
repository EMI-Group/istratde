import jax.numpy as jnp
import jax
from tqdm import tqdm
from istratde.util import StdSOMonitor, StdWorkflow
from istratde.problems.jax import Sphere
from istratde.algorithms.jax import IStratDE

# Problem settings
D = 10
key_start = 42

# Algorithm settings
POP_SIZE = 100000
STEPS = 1000

# Define search space
lb = jnp.full((D,), -100.0)
ub = jnp.full((D,), 100.0)

# Initialize iStratDE (Self-adaptive algorithm)
algorithm = IStratDE(
    lb=lb,
    ub=ub,
    pop_size=POP_SIZE,
)

# Problem to solve
problem = Sphere()

# Monitor for tracking best fitness
monitor = StdSOMonitor(record_fit_history=False)

# Initialize standard workflow (no meta-wrapper needed)
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
)

# Initialize state
key = jax.random.PRNGKey(key_start)
state = workflow.init(key)

# Main optimization loop
for step in tqdm(range(STEPS), desc="Running iStratDE"):
    state = workflow.step(state)

# Print result
print(f"Best fitness: {monitor.get_best_fitness()}")