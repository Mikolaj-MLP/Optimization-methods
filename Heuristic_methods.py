import numpy as np
import random

def simulated_annealing(func, x0, bounds, max_iters=1000, temp=1.0, cooling=0.99):
    x = x0.copy()
    fx = func(x)
    best_x, best_fx = x.copy(), fx
    trajectory = [x.copy()]
    
    for i in range(max_iters):
        candidate = x + np.random.uniform(-0.1, 0.1, size=x.shape)
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        f_candidate = func(candidate)
        delta = f_candidate - fx

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x, fx = candidate, f_candidate
            if fx < best_fx:
                best_x, best_fx = x.copy(), fx
        temp *= cooling
        trajectory.append(x.copy())

    return best_x, trajectory


def tabu_search(func, x0, bounds, max_iters=500, tabu_size=25, step=0.1):
    x = x0.copy()
    best_x = x.copy()
    best_fx = func(x)
    tabu_list = [x.copy()]
    trajectory = [x.copy()]
    
    for _ in range(max_iters):
        neighborhood = [x + np.random.uniform(-step, step, size=x.shape) for _ in range(20)]
        neighborhood = [np.clip(n, bounds[:, 0], bounds[:, 1]) for n in neighborhood]
        neighborhood = [n for n in neighborhood if not any(np.allclose(n, t, atol=1e-3) for t in tabu_list)]
        if not neighborhood:
            continue
        candidate = min(neighborhood, key=func)
        fx = func(candidate)

        if fx < best_fx:
            best_x = candidate.copy()
            best_fx = fx
        
        x = candidate
        tabu_list.append(x.copy())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        trajectory.append(x.copy())

    return best_x, trajectory


def genetic_algorithm(func, bounds, pop_size=50, generations=100, mutation_rate=0.1, elite_frac=0.2):
    dim = bounds.shape[0]
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, dim))
    trajectory = []

    for _ in range(generations):
        scores = np.array([func(ind) for ind in pop])
        sorted_idx = np.argsort(scores)
        pop = pop[sorted_idx]
        trajectory.append(pop[0].copy())
        
        elite_count = int(elite_frac * pop_size)
        elites = pop[:elite_count]

        offspring = []
        while len(offspring) < pop_size - elite_count:
            parents = elites[np.random.choice(elite_count, 2, replace=False)]
            cross_point = np.random.randint(1, dim)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(dim)
                child[idx] += np.random.normal(0, 0.1)
            child = np.clip(child, bounds[:, 0], bounds[:, 1])
            offspring.append(child)

        pop = np.vstack((elites, offspring))

    return trajectory[-1], trajectory
