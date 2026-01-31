import torch
import torch.nn as nn
from tqdm import tqdm
from istratde.algorithms.pytorch.istratde import IStratDE
from istratde.problems.torch.brax import BraxProblem
from istratde.util.workflows import EvalMonitor, StdWorkflow
from istratde.util import ParamsAndVector

torch.set_float32_matmul_precision('high')

# 1. Global Parameters (Swimmer: 8 obs -> 2 actions)
Env_Name = "swimmer"
Obs_Dim = 8
Act_Dim = 2
Pop_Size = 10000
Steps = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# 2. Define Policy Network
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(Obs_Dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, Act_Dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

model = PolicyNet().to(device)

# 3. Setup Parameter Adapter (Flat Vector <-> State Dict)
adapter = ParamsAndVector(dummy_model=model)
dummy_params = dict(model.named_parameters())
center = adapter.to_vector(dummy_params)
D = center.shape[0]

# 4. Initialize iStratDE Algorithm
lb = torch.full_like(center, -10.0)
ub = torch.full_like(center, 10.0)

algorithm = IStratDE(
    pop_size=Pop_Size,
    lb=lb,
    ub=ub
)

# 5. Setup Problem (Brax Swimmer)
problem = BraxProblem(
    policy=model,
    env_name=Env_Name,
    max_episode_length=500,
    num_episodes=1,
    pop_size=Pop_Size,
)

# 6. Initialize Workflow
monitor = EvalMonitor(full_sol_history=False)

workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
    solution_transform=adapter,
    device=device,
    opt_direction="max" # Maximize reward
)

# 7. Run Loop
workflow.init_step()

for step in tqdm(range(Steps), desc=f"{Env_Name.capitalize()} Evolution"):
    workflow.step()
    print(f"fitness: {-monitor.topk_fitness[0].item()}")
    # Because BraxProblem returns negative rewards as fitness

print(f"Best fitness: {monitor.topk_fitness[0].item()}")