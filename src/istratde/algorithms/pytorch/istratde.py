import torch

from evox.core import Algorithm, Mutable
from evox.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
)
from evox.operators.selection import select_rand_pbest
from evox.utils import clamp


class IStratDE(Algorithm):

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        diff_padding_num: int = 9,
        mean: torch.Tensor | None = None,
        stdev: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize the DE algorithm with the given parameters.

        :param pop_size: The size of the pop.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param mean: The mean for initializing the pop with a normal distribution. Defaults to None.
        :param stdev: The standard deviation for initializing the pop with a normal distribution. Defaults to None.
        :param device: The device to use for tensor computations. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device

        # Validate input parameters
        assert pop_size >= 4
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype

        # Initialize parameters
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.dim = lb.shape[0]
        self.best_index = Mutable(torch.tensor(0, device=device))

        # Move bounds to the specified device and add batch dimension
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub

        values = torch.linspace(0, 1, steps=1001, device=device)
        random_indices = torch.randint(0, 1001, (2*pop_size,), device=device)
        self.Fs = values[random_indices[:pop_size]]
        self.CRs = values[random_indices[pop_size:]]
        """Base vector types: 0 = rand, 1 = best, 2 = pbest_vect, 3 = current_vect"""
        self.basevect_prim_type_array = torch.randint(0, 4, (pop_size,), device=device)
        self.basevect_sec_type_array = torch.randint(0, 4, (pop_size,), device=device)
        self.num_diff_vects_array = torch.randint(0, 4, (pop_size,), device=device)
        """Crossover: 0 = bin, 1 = exp, 2 = arith"""
        self.cross_strategy_array = torch.randint(0, 3, (pop_size,), device=device)

        # Initialize pop
        if mean is not None and stdev is not None:
            # Initialize pop using a normal distribution
            pop = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            pop = clamp(pop, lb=self.lb, ub=self.ub)
        else:
            # Initialize pop uniformly within bounds
            pop = torch.rand(self.pop_size, self.dim, device=device)
            pop = pop * (self.ub - self.lb) + self.lb

        # Mutable attributes to store pop and fit
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.empty(self.pop_size, device=device).fill_(float("inf")))

    def init_step(self):
        """
        Perform the initial evaluation of the pop's fit and proceed to the first optimization step.

        This method evaluates the fit of the initial pop and then calls the `step` method to perform the first optimization iteration.
        """
        self.fit = self.evaluate(self.pop)
        self.step()

    def step(self):

        device = self.pop.device
        # print(device)
        indices = torch.arange(self.pop_size, device=device)

        # Mutation: Generate random permutations for selecting vectors
        difference_sum, rand_vec_idx = DE_differential_sum(self.diff_padding_num, self.num_diff_vects_array, indices, self.pop)

        rand_vecs = self.pop[rand_vec_idx]
        best_vecs = torch.tile(self.pop[self.best_index].unsqueeze(0), (self.pop_size, 1))
        pbest_vecs = select_rand_pbest(0.05, self.pop, self.fit)
        current_vecs = self.pop[indices]

        vectors_merge = torch.stack([rand_vecs, best_vecs, pbest_vecs, current_vecs])

        base_vectors_prim = torch.zeros(self.pop_size, self.dim, device=device)
        base_vectors_sec = torch.zeros(self.pop_size, self.dim, device=device)

        for i in range(4):
            base_vectors_prim = torch.where(self.basevect_prim_type_array.unsqueeze(1) == i, vectors_merge[i], base_vectors_prim)
            base_vectors_sec = torch.where(self.basevect_sec_type_array.unsqueeze(1) == i, vectors_merge[i], base_vectors_sec)

        base_vectors = base_vectors_prim + self.Fs.unsqueeze(1) * (base_vectors_sec - base_vectors_prim)
        mutation_vectors = base_vectors + difference_sum * self.Fs.unsqueeze(1)

        # Crossover: Determine which dimensions to crossover based on the crossover probability
        trial_vectors = torch.zeros(self.pop_size, self.dim, device=device)
        trial_vectors = torch.where(
            self.cross_strategy_array.unsqueeze(1) == 0,
            DE_binary_crossover(mutation_vectors, current_vecs, self.CRs),
            trial_vectors,
        )
        trial_vectors = torch.where(
            self.cross_strategy_array.unsqueeze(1) == 1,
            DE_exponential_crossover(mutation_vectors, current_vecs, self.CRs),
            trial_vectors,
        )
        trial_vectors = torch.where(
            self.cross_strategy_array.unsqueeze(1) == 2,
            DE_arithmetic_recombination(mutation_vectors, current_vecs, self.CRs),
            trial_vectors,
        )
        trial_vectors = clamp(trial_vectors, self.lb, self.ub)
        trial_fitness = self.evaluate(trial_vectors)
        compare = trial_fitness <= self.fit

        self.pop = torch.where(compare[:, None], trial_vectors, self.pop)
        self.fit = torch.where(compare, trial_fitness, self.fit)

        self.best_index = torch.argmin(self.fit)
