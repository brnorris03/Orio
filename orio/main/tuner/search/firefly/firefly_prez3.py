import numpy as np
import math
import operator
import time
import itertools
import orio.main.tuner.search.search


class FireflyElement:

    def __init__(self, min_bound, max_bound):
        self.position = self.generate_position(min_bound, max_bound)
        self.brightness = None

    @staticmethod
    def generate_position(min_bound, max_bound):
        problem_dim = len(min_bound)
        error = 1e-10 * np.ones(problem_dim)
        return (np.multiply(max_bound - error - min_bound,
                            np.random.random_sample(problem_dim)) + min_bound).round().astype(int)


class Firefly(orio.main.tuner.search.search.Search):

    def __init__(self, params, **kwargs):
        orio.main.tuner.search.search.Search.__init__(self, params)
        self.population_size = int(kwargs.get('population_size', 10))
        self.max_bound = np.array(self.dim_uplimits) - np.ones(len(self.dim_uplimits)).astype(int)
        self.min_bound = np.zeros(len(self.dim_uplimits)).astype(int)
        self.problem_dim = len(self.min_bound)
        self.generations = kwargs.get('generations', 10)
        self.population = []
        self.gamma = kwargs.get('gamma', 0.97)  # absorption coefficient
        self.alpha = kwargs.get('alpha', 0.25)  # randomness [0,1]
        self.alpha_decay = kwargs.get('alpha_decay', 0.9)
        self.beta_init = kwargs.get('beta_init', 1)

    def get_population(self):
        info("Generating population...")
        for i in range(self.population_size):
            while True:
                self.population.append(FireflyElement(self.min_bound, self.max_bound))
                if self.move(i, np.random.standard_normal(self.problem_dim)):
                    info(("|- Created firefly %d" % i))
                    break
                else:
                    self.population.pop()

    def step(self):
        self._modify_alpha()
        current_brightness = [firefly.brightness for firefly in self.population]
        current_positions = [firefly.position for firefly in self.population]
        for i in range(self.population_size):
            tmp_direction = np.zeros(self.problem_dim)
            for j in range(self.population_size):
                if current_brightness[i] < current_brightness[j]:
                    r = math.sqrt(float(np.sum((current_positions[j] - current_positions[i]) ** 2)))
                    beta = self.beta_init * math.exp(-self.gamma * r ** 2)
                    tmp_direction += (current_positions[j] - current_positions[i]) * beta
            self.move(i, tmp_direction)
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))

    def move(self, i, direction):
        noise = np.multiply((np.random.random_sample(self.problem_dim) - 0.5 * np.ones(self.problem_dim)),
                            self.max_bound - self.min_bound) * self.alpha
        new_position = self.population[i].position + direction + noise
        new_position[new_position > self.max_bound] = self.max_bound[new_position > self.max_bound]
        new_position[new_position < self.min_bound] = self.min_bound[new_position < self.min_bound]
        hyper_octant = np.sign(direction).astype(int)
        # round the position toward the old position:
        new_position = new_position.astype(int) + [int(self.population[i].position[j] > new_position[j]) for j in
                                                   range(self.problem_dim)]
        while sum((hyper_octant - np.zeros(self.problem_dim).astype(int)) ** 2) == 0:
            hyper_octant = ((np.random.random_sample(self.problem_dim) - 0.5) * 2).round().astype(int)
        # the indexes with the smallest gradient are on top of the list so tried as last.
        ranked_indexes = np.argsort(abs(direction))
        superset = []
        # directions to try, only 1 hyper octant is tried which is the one in the direction of direction
        for index in ranked_indexes:
            if hyper_octant[index] == 0:
                superset.append([0])
            elif hyper_octant[index] == -1:
                if new_position[index] > self.min_bound[index]:
                    superset.append([0, -1])
                else:
                    superset.append([0])
            elif hyper_octant[index] == 1:
                if new_position[index] < self.max_bound[index]:
                    superset.append([0, 1])
                else:
                    superset.append([0])
            else:
                raise ValueError("Hyper octant has wrong values.")
        for delta_indexes in itertools.product(*superset):
            delta = np.zeros(self.problem_dim).astype(int)
            for index in range(self.problem_dim):
                delta[ranked_indexes[index]] = delta_indexes[index]
            perf_params = self.coordToPerfParams(list(new_position + delta))
            is_valid = eval(self.constraint, perf_params, dict(self.input_params))
            if sum((new_position + delta - self.population[i].position) ** 2) > 0 and is_valid:
                self.population[i].position = new_position + delta
                return True
        # the search is limited to the hypercube of length 1. If nothing is found, the firefly stands in place.

    def searchBestCoord(self, startCoord=None):
        start_time = time.time()
        # initialize all fireflies brightness
        self.get_population()
        for i in range(self.population_size):
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))
        for t in range(self.generations):
            info(('Generation %s, best fitness %s' % (t, -self.population[-1].brightness)))
            self.step()
            self.population.sort(key=operator.attrgetter('brightness'))
        search_time = time.time() - start_time
        return list(self.population[-1].position), -self.population[-1].brightness, search_time, 1

    def _modify_alpha(self):
        self.alpha = self.alpha * self.alpha_decay
