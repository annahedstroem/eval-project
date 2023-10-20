import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
from numpy.random import default_rng


def _spawn_ranking(p1:np.ndarray[int], p2:np.ndarray[int]) -> np.ndarray[int]:
    '''
    Generates a new ranking that's a mixture of r1 and r2

    r1 and r2 are unidimensional and sorted by attribution and contain the indices of the features in the flattened input
    '''
    new_r = np.zeros(len(p1), dtype=int)
    # The next position (i-th) of new_r will be sampled from a pool of candidates: all indices that p1 and p2 rank before i
    candidates:list[int] = []
    # We don't want the same index more than once so we keep track of the ones we use
    used = np.zeros(len(p1),bool)
    for i in range(len(new_r)):
        # Add indices in p1 and p2 to the list of candidates (unless that particular index is already there)
        if not used[p1[i]]:
            used[p1[i]] = True
            candidates.append(p1[i])
        if not used[p2[i]]:
            used[p2[i]] = True
            candidates.append(p2[i])

        # Use a random candidate for the ranking and remove it from the list of candidates
        new_r[i] = candidates.pop(random.randint(0,len(candidates)-1))
    return new_r

def _repopulate(population:np.ndarray[int], num_saved:int) -> np.ndarray[int]:
    '''
    Trims a population to its top `num_saved` performers and repopulates the rest by performing random crossings of the remaining elements

    Population must be sorted across axis 0 in descending order of performance
    '''
    rng = default_rng()
    for i in range(num_saved, population.shape[0]): # Loop through all non-top performers to overwrite them
       # Select two random parents from the num_saved top performers of the population
       parent_indices = rng.permutation(num_saved)[:2]
       parent1 = population[parent_indices[0]]
       parent2 = population[parent_indices[1]]
       # Generate a ranking and add it to the population (overwriting one of the poor performers)
       population[i] = _spawn_ranking(parent1, parent2)


def generate_rankings(num_rankings: int,\
                      element_shape: tuple[int],\
                      fitness_function: Callable[[np.ndarray[int]], float],\
                      population_size: Optional[int] = None,\
                      num_iterations:int = 10):
    '''
    Generates with genetic programming as many rankings as indicated by num_rankings
    The algorithm is as follows:
      1. Generate an initial population of random rankings
      2. For a given number of steps, do:
        - Compute fitness of each element of the current population
        - Take top num_rankings elements according to fitness. Replace the rest of the population with "children" of randomly selected "parents" from the top num_rankings
    '''
    if population_size is None:
       population_size = int(1.4 * num_rankings)

    num_features = 1
    for d in element_shape:
       num_features *= d

    # Generate an initial population of entirely random rankings
    population = np.zeros((population_size, num_features), dtype=int)
    rng = default_rng()
    for i in range(population_size):
        population[i] = rng.permutation(num_features)

    for i in range(num_iterations):
        iteration_fitnesses = []
        for j in range(population_size):
            fitness = fitness_function(population[j].reshape(element_shape) / (num_features - 1))
            iteration_fitnesses.append((fitness, j))
        iteration_fitnesses.sort(reverse=True)
        # Sort population by fitness
        sorted_indices = list(map(lambda x:x[1], iteration_fitnesses))
        population = population[sorted_indices]
        avg_fitness = np.mean(list(map(lambda x:x[0], iteration_fitnesses)))
        print(f'{i+1}/{num_iterations} - Avg. fitness {avg_fitness}')
        _repopulate(population, num_rankings)
    
    return population[:num_rankings].reshape((num_rankings, *element_shape))

if __name__ == '__main__':
  #TESTS
  rng = default_rng()
  p1 = rng.permutation(8)
  p2 = rng.permutation(8)
  print('P1',p1)
  print('P2',p2)
  r_spawn = _spawn_ranking(p1, p2)
  print(r_spawn)