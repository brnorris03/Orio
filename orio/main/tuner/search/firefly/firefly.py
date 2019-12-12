import numpy as np
import math
import operator
import time
import itertools
import orio.main.tuner.search.search


UNROLL_NAMES = 'unroll_variables'  # unroll constraint (U1_I == 1 or U1_J == 1 or U1_K) and (U2_I == 1 or U2_J == 1)
                                   # has to be passed as: 'U1_I U1_J U1_K - U2_I U2_j'


class FireflyElement:

    def __init__(self, min_bound, max_bound, ide):
        self.position = self.generate_position(min_bound, max_bound)
        self.brightness = None
        self.id = ide

    @staticmethod
    def generate_position(min_bound, max_bound):
        problem_dim = len(min_bound)
        error = 1e-10 * np.ones(problem_dim)
        return (np.multiply(max_bound - error - min_bound,
                            np.random.random_sample(problem_dim)) + min_bound).round().astype(int)


class Firefly(orio.main.tuner.search.search.Search):

    def __init__(self, params, **kwargs):
        orio.main.tuner.search.search.Search.__init__(self, params)
        self.population_size = int(kwargs.get('population_size', 100))
        self.max_bound = np.array(self.dim_uplimits) - np.ones(len(self.dim_uplimits)).astype(int)
        self.min_bound = np.zeros(len(self.dim_uplimits)).astype(int)
        self.problem_dim = len(self.min_bound)
        self.generations = kwargs.get('generations', 100000000)
        self.population = []
        self.alpha = kwargs.get('alpha', 0.5)  # randomness [0,1]
        self.alpha_decay = kwargs.get('alpha_decay', 0.9)
        self.beta_init = kwargs.get('beta_init', 1)
        self.unroll_list = []
        self._get_algo_params()
        auto_gamma = kwargs.get('auto_gamma', True)
        if auto_gamma:
            # this ensures that the movement to be on average half of the distance between two fireflies
            self.gamma = self._get_gamma()
        else:
            self.gamma = kwargs.get('gamma', 0.01)  # absorption coefficient

    def _get_gamma(self):
        a = np.random.random_sample((self.population_size, self.problem_dim))
        avg = 0
        for i in range(self.population_size):
            j = i + 1
            while j < self.population_size:
                avg += np.linalg.norm(a[i, :] - a[j, :])
                j += 1
        return np.log(2) / (2 * avg/(self.population_size * (self.population_size - 1))) ** 2

    def get_population(self):
        print("std_pr: Generating population...")
        for i in range(self.population_size):
            while True:
                self.population.append(FireflyElement(self.min_bound, self.max_bound, i))
                if self.move(i, np.random.standard_normal(self.problem_dim)):
                    print("std_pr: |- Created firefly %d" % i)
                    break
                else:
                    self.population.pop()

    def close_bounds_to_unroll(self, position, index):
        """Returns a new position with unroll + min and max bound feasibility, the nearest to position.
        The rounding is done toward its old position before making unroll feasibility."""
        position[position < self.min_bound] = self.min_bound[position < self.min_bound]
        position[position > self.max_bound] = self.max_bound[position > self.max_bound]
        # round the position toward the old position:
        position = position.astype(int) + [int(self.population[index].position[j] > position[j]) for j in
                                           range(self.problem_dim)]
        perf_params = self.coordToPerfParams(list(position))
        for k in range(len(self.unroll_list)):
            index_to_be_constrained = None
            tmp_minimum = float('inf')
            for j, variable in enumerate(self.unroll_list[k]):
                if perf_params[variable] < tmp_minimum:
                    tmp_minimum = perf_params[variable]
                    index_to_be_constrained = j
            perf_params[self.unroll_list[k][index_to_be_constrained]] = 1
        return self.perfParamTabToCoord(self.perfParamToCoord(perf_params))

    def step(self):
        # don't need to be ordered by brightness
        self._modify_alpha()
        current_brightness = [firefly.brightness for firefly in self.population]
        current_positions = [firefly.position for firefly in self.population]
        for i in range(self.population_size):
            tmp_direction = np.zeros(self.problem_dim)
            for j in range(self.population_size):
                if current_brightness[i] < current_brightness[j]:
                    # we go to the [0,1] hypercube to compute the relative distance from two coords
                    rescaled_vector = np.divide(current_positions[j] - current_positions[i], self.max_bound)
                    r = math.sqrt(float(np.sum(rescaled_vector ** 2)))
                    beta = self.beta_init * math.exp(-self.gamma * r ** 2)
                    tmp_direction += (current_positions[j] - current_positions[i]) * beta
            try:
                self.move(i, tmp_direction)
            except ValueError:
                if len(self.population) == self.population_size:
                    pass
                else:
                    raise Exception
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))

    def move(self, i, direction):
        noise = np.multiply((np.random.random_sample(self.problem_dim) - 0.5 * np.ones(self.problem_dim)),
                            self.max_bound - self.min_bound) * self.alpha
        new_position = self.population[i].position + direction + noise
        new_position = self.close_bounds_to_unroll(new_position, i)  # this also push into regular bounds.

        perf_params = self.coordToPerfParams(list(new_position))
        perf_params = self.getNearestFeasibleZ3(perf_params)
        if perf_params is None:
            raise ValueError("std_pr: Impossible to find a feasible neighbor.")
        coord = self.perfParamTabToCoord(self.perfParamToCoord(perf_params))
        self.population[i].position = np.array(coord)
        return True

    def searchBestCoord(self, startCoord=None):
        start_time = time.time()
        # initialize all fireflies brightness
        self.get_population()
        for i in range(self.population_size):
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))
        t = 0
        best_fitness = float('inf')
        best_coord = None
        while t < self.generations and not ((time.time()-start_time) > self.time_limit > 0):
            self.population.sort(key=operator.attrgetter('brightness'))
            print('std_pr: Generation %s, best fitness %s' % (t, -self.population[-1].brightness))
            if -self.population[-1].brightness < best_fitness:
                best_fitness = -self.population[-1].brightness
                best_coord = self.population[-1].position
            self.population.sort(key=operator.attrgetter('id'))
            print("std_pr: " + str([f.brightness for f in self.population][:8]))
            self.step()
            t += 1
        search_time = time.time() - start_time
        self.population.sort(key=operator.attrgetter('brightness'))
        if -self.population[-1].brightness < best_fitness:
            best_fitness = -self.population[-1].brightness
            best_coord = self.population[-1].position
        return list(best_coord), best_fitness, search_time, 1

    def _modify_alpha(self):
        self.alpha = self.alpha * self.alpha_decay

    def _get_algo_params(self):
        for name, value in self.search_opts.iteritems():
            if name == UNROLL_NAMES:
                value = firefly_unroll_variables
                items_list = value.split()
                i = 0
                tmp_list = []
                while i < len(items_list):
                    if items_list[i] == '-':
                        self.unroll_list.append(tmp_list)
                        tmp_list = []
                    else:
                        tmp_list.append(items_list[i])
                    i += 1
                self.unroll_list.append(tmp_list)
                for unrolls in self.unroll_list:
                    if len(unrolls) < 2:
                        raise ValueError('std_pr: Unrolls number unfeasible.')
