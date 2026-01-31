import jax.numpy as jnp
import jax
from tqdm import tqdm
from istratde.util import StdSOMonitor, StdWorkflow
from istratde.problems.jax import CEC2022TestSuit
from istratde.algorithms.jax import IStratDE

# Problem setting
D = 10  # Dimension (CEC2022 supports 10D, 20D)
FUNC_LIST = jnp.arange(12) + 1  # Functions 1 to 12
key_start = 42

# IStratDE algorithm settings
POP_SIZE = 100000  # Large population size to leverage JAX parallelism
STEPS = 5000       # Total generations

# Define search space for CEC2022
lb = jnp.full((D,), -100.0)
ub = jnp.full((D,), 100.0)

for func_num in FUNC_LIST:
    # 1. Create specific CEC2022 problem instance
    base_problem = CEC2022TestSuit.create(int(func_num))
    print(f"Testing Problem: CEC2022 F{int(func_num)}")

    # 2. Initialize IStratDE
    algorithm = IStratDE(
        lb=lb,
        ub=ub,
        pop_size=POP_SIZE,
    )

    # 3. Initialize Monitor
    monitor = StdSOMonitor(record_fit_history=False)

    # 4. Initialize Workflow directly with the problem
    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=base_problem,
        monitor=monitor,
    )

    # 5. Initialize State
    key = jax.random.PRNGKey(key_start)
    state = workflow.init(key)

    # 6. Main Optimization Loop
    for i in tqdm(range(STEPS), desc=f"F{int(func_num)} Progress"):
        state = workflow.step(state)

    # 7. Final Results
    print(f"F{int(func_num)} Best_fitness: {monitor.get_best_fitness()}")
    print("-" * 30)