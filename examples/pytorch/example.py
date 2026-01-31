import torch
from tqdm import tqdm
from evox.problems.numerical import Sphere
from istratde.util.workflows import EvalMonitor, StdWorkflow
from istratde.algorithms.pytorch.istratde import IStratDE

# Problem settings
D = 10
seed = 42

# Algorithm settings
POP_SIZE = 100000
STEPS = 1000

# Setup device
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Define search space
lb = -100 * torch.ones(D)
ub = 100 * torch.ones(D)

# Initialize iStratDE (Self-adaptive algorithm)
algorithm = IStratDE(
    lb=lb,
    ub=ub,
    pop_size=POP_SIZE,
)

# Problem to solve
problem = Sphere()

# Monitor for tracking best fitness
monitor = EvalMonitor(full_sol_history=False)

# Initialize standard workflow (no meta-wrapper needed)
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
)

# Initialize state
workflow.init_step()

# Main optimization loop
for step in tqdm(range(STEPS), desc="Running iStratDE"):
    workflow.step()

# Print result
print(f"Best fitness: {monitor.topk_fitness[0]:.5f}")