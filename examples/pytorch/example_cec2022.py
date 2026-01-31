import torch
from tqdm import tqdm
from istratde.util.workflows import EvalMonitor, StdWorkflow
from istratde.algorithms.pytorch.istratde import IStratDE
from istratde.problems.torch.cec2022 import CEC2022 # Follow the original code, fitness minimum is 300, 400...

# Global Settings
D = 10
POP_SIZE = 100000
STEPS = 5000
SEED = 42

# Setup device
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

print(f"Running on device: {device}")

# Iterate Over CEC2022 Problems F1 ~ F12
for problem_number in range(1, 13):
    print(f"\n=== Running CEC2022 Problem F{problem_number} ===")

    # Define search space
    lb = -100 * torch.ones(D)
    ub = 100 * torch.ones(D)

    # Initialize IStratDE (Self-adaptive algorithm)
    # No HPO wrapper needed, it learns parameters automatically
    algorithm = IStratDE(
        pop_size=POP_SIZE,
        lb=lb,
        ub=ub,
    )

    # Problem to solve
    problem = CEC2022(problem_number=problem_number, dimension=D)

    # Monitor for tracking best fitness
    monitor = EvalMonitor(full_sol_history=False)

    # Initialize standard workflow
    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
    )

    workflow.init_step()

    for step in tqdm(range(STEPS), desc=f"Optimizing F{problem_number}"):
        workflow.step()

    best_fitness = monitor.topk_fitness[0].item()
    print(f"Best fitness for F{problem_number}: {best_fitness:.5f}")