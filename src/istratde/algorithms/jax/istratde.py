from istratde.util import Algorithm, State, jit_class
from istratde.util.operators.selection import select_rand_pbest_jax
from istratde.util.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)
import jax
import jax.numpy as jnp
from jax import vmap, lax
from functools import partial
from evox.utils import *


@jit_class
class IStratDE(Algorithm):
    """
    Class representing the iStratDE algorithm, a variant of the Differential Evolution algorithm.
    This class is a child of the Algorithm class from the evox library.

    The class uses JAX for just-in-time compilation to speed up the computations.
    """

    def __init__(
            self,
            lb,
            ub,
            pop_size=10000,
            diff_padding_num=9,
            replace=False,
            mean=None,
            stdvar=None,
    ):
        """
        Constructor for the iStratDE class.

        Parameters:
        lb (np.array): Lower bounds for the population.
        ub (np.array): Upper bounds for the population.
        pop_size (int): Population size.
        diff_padding_num (int): Number of padding for the difference.
        replace (bool): If True, replacement is used in the selection.
        mean (float): Mean for the normal distribution used in the initialization.
        stdvar (float): Standard deviation for the normal distribution used in the initialization.
        """
        assert jnp.all(lb < ub)
        assert pop_size >= 4
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.replace = replace
        self.mean = mean
        self.stdvar = stdvar
        self.diff_padding_num = diff_padding_num

    def setup(self, key):
        """
        Set up the initial state for the algorithm.

        Parameters:
        key (int): Random seed.

        Returns:
        State: Initial state.
        """
        state_key, init_key, param1_key, param2_key, param3_key, param4_key, param5_key = jax.random.split(key, 7)

        if self.mean is not None and self.stdvar is not None:
            population = self.stdvar * jax.random.normal(
                init_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        best_index = 0
        start_index = 0
        params_values = jnp.linspace(0, 1, 1000)
        # params_array = jax.random.uniform(param1_key, shape=(self.pop_size, 2))
        params_array = jax.random.choice(param1_key, params_values, shape=(self.pop_size, 2))

        basevect_prim_type_array = jax.random.choice(param2_key, jnp.array([0, 1, 2, 3]), shape=(self.pop_size,))
        basevect_sec_type_array = jax.random.choice(param3_key, jnp.array([0, 1, 2, 3]), shape=(self.pop_size,))
        num_diff_vects_array = jax.random.choice(param4_key, jnp.array([0, 1, 2, 3]), shape=(self.pop_size,))
        cross_strategy_array = jax.random.choice(param5_key, jnp.array([0, 1, 2]), shape=(self.pop_size,))
        params_list = jnp.column_stack((params_array, basevect_prim_type_array, basevect_sec_type_array,
                                        num_diff_vects_array, cross_strategy_array))

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            params=params_list,
            count=0,
        )

    def ask(self, state):
        """
        Generate trial vectors for the population.

        Parameters:
        state (State): Current state.

        Returns:
        np.array, State: Trial vectors and updated state.
        """
        key, ask_one_key = jax.random.split(state.key, 2)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
        indices = jnp.arange(self.pop_size)

        trial_vectors = vmap(
            partial(
                self._ask_one, state=state
            )
        )(index=indices, key=ask_one_keys)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, state, index, key):
        """
        Generate a trial vector for a single individual.

        Parameters:
        state (State): Current state.
        index (int): Index of the individual.
        key (int): Random seed.

        Returns:
        np.array: Trial vector.
        """
        select_key, pbest_key, crossover_key = jax.random.split(key, 3)
        population = state.population
        best_index = state.best_index
        fitness = state.fitness
        params = state.params
        param = params[index]
        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
            self.diff_padding_num,
            param[4].astype(int),
            index,
            population,
            replace=True
        )
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]
        pbest_vect = select_rand_pbest_jax(pbest_key, 0.05, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[param[2].astype(int)]
        base_vector_sec = vector_merge[param[3].astype(int)]
        base_vector = base_vector_prim + param[0] * (base_vector_sec - base_vector_prim)
        mutation_vector = (base_vector + difference_sum * param[0])
        """Crossover: 0 = bin, 1 = exp, 2 = arith"""
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = jax.lax.switch(
            param[5].astype(int),
            cross_funcs,
            crossover_key,
            mutation_vector,
            current_vect,
            param[1],
        )
        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        return trial_vector

    def tell(self, state, trial_fitness):
        """
        Update the state based on the fitness of the trial vectors.

        Parameters:
        state (State): Current state.
        trial_fitness (np.array): Fitness values of the trial vectors.

        Returns:
        State: Updated state.
        """
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.pop_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.pop_size, axis=0
        )

        compare = trial_fitness <= batch_fitness

        population_update = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.pop_size) % self.pop_size
        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
        )

    def override(self, state, key, params):
        """
        Override the current state with a new one.

        Parameters:
        state (State): Current state.
        key (int): Random seed.
        params (np.array): Parameters for the new state.

        Returns:
        State: New state.
        """
        state = state | self.setup(key)
        return state.update(params=params)