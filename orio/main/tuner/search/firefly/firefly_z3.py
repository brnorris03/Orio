import numpy as np
import math
import operator
import time
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
        self.population_size = int(kwargs.get('population_size', 3))
        self.max_bound = np.array(self.dim_uplimits) - np.ones(len(self.dim_uplimits)).astype(int)
        self.min_bound = np.zeros(len(self.dim_uplimits)).astype(int)
        self.problem_dim = len(self.min_bound)
        self.generations = kwargs.get('generations', 3)
        self.population = []
        self.gamma = kwargs.get('gamma', 0.97)  # absorption coefficient
        self.alpha = kwargs.get('alpha', 0.25)  # randomness [0,1]
        self.alpha_decay = kwargs.get('alpha_decay', 0.9)
        self.beta_init = kwargs.get('beta_init', 1)

    def is_valid(self, coord):
        perf_params = self.coordToPerfParams(coord)
        return eval(self.constraint, perf_params, dict(self.input_params))

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
        # round the position toward the old position:
        new_position = new_position.astype(int) + [int(self.population[i].position[j] > new_position[j]) for j in
                                                   range(self.problem_dim)]
        perf_params = self.coordToPerfParams(list(new_position))
        perf_params = self.getNearestFeasibleZ3(perf_params)
        if perf_params is None:
            raise ValueError("Impossible to find a feasible neighbor.")
        coord = self.perfParamTabToCoord(self.perfParamToCoord(perf_params))
        self.population[i].position = np.array(coord)
        return True

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
