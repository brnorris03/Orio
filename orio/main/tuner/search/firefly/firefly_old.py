import numpy as np
import math
import operator
import time
import itertools
import orio.main.tuner.search.search


def generate_position(min_bound, max_bound):
    problem_dim = len(min_bound)
    error = 1e-10 * np.ones(problem_dim)
    return np.multiply(max_bound - error - min_bound, np.random.random_sample()) + min_bound


class FireflyElement:

    def __init__(self, min_bound, max_bound):
        self.position = generate_position(min_bound, max_bound)
        self.brightness = None


class Firefly(orio.main.tuner.search.search.Search):

    def __init__(self, params, **kwargs):
        orio.main.tuner.search.search.Search.__init__(self, params)
        self.population_size = int(kwargs.get('population_size', 10))
        self.max_bound = np.array(self.dim_uplimits) - np.ones(len(self.dim_uplimits)).astype(int)
        self.min_bound = np.zeros(len(self.dim_uplimits)).astype(int)
        self.problem_dim = len(self.min_bound)
        self.generations = kwargs.get('generations', 10)
        self.population = self._population(self.population_size, self.min_bound, self.max_bound)
        self.gamma = kwargs.get('gamma', 0.97)  # absorption coefficient
        self.alpha = kwargs.get('alpha', 0.25)  # randomness [0,1]
        self.alpha_decay = kwargs.get('alpha_decay', 0.9)
        self.beta_init = kwargs.get('beta_init', 1)
        self.chosen = [] # here? or in the check_position() function?
        
    @staticmethod
    def _population(population_size, min_bound, max_bound):
        population = []
        for i in range(population_size):
            population.append(FireflyElement(min_bound, max_bound))
        return population

    def step(self):
        self.population.sort(key=operator.attrgetter('brightness'))
        self._modify_alpha()
        current_brightness = [firefly.brightness for firefly in self.population]
        current_positions = [firefly.position for firefly in self.population]
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.population[i].brightness < current_brightness[j]:
                    r = math.sqrt(float(np.sum((self.population[i].position - current_positions[j]) ** 2)))
                    beta = self.beta_init * math.exp(-self.gamma * r ** 2)
                    noise = np.multiply((np.random.random_sample(self.problem_dim) - 0.5 * np.ones(self.problem_dim)),
                                        self.max_bound - self.min_bound)
                    tmp_position = (1-beta) * self.population[i].position + beta * current_positions[j]
                    self.population[i].position = self.check_position(tmp_position + self.alpha * noise)
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))

    def searchBestCoord(self, startCoord=None):
        start_time = time.time()
        # initialize all fireflies brightness
        for i in range(self.population_size):
            self.population[i].position = self.check_position(self.population[i].position)
            self.population[i].brightness = -sum(self.getPerfCost(list(self.population[i].position)))
        for t in range(self.generations):
            print('Generation %s, best fitness %s' % (t, -self.population[-1].brightness))
            self.step()
        self.population.sort(key=operator.attrgetter('brightness'), reverse=True)
        search_time = time.time() - start_time
        return list(self.population[-1].position), -self.population[-1].brightness, search_time, 1

    def check_position(self, position):
        """First: position is rounded to integer and if it is outside the bounds it will be put at the bounds.
        Second: generate random increment/decrement vectors to move the position into the neighborhood. These vectors
        are made changing a random number of coordinates by a random length between 1 and radius. Radius is incremented
        every 1000 unfeasible positions. Whenever a feasible point is reached, the function returns the position."""
        position = position.round().astype(int)
        position[position > self.max_bound] = self.max_bound[position > self.max_bound]
        position[position < self.min_bound] = self.min_bound[position < self.min_bound]
        radius = 0
        perf_params = self.coordToPerfParams(list(position))
        is_valid = eval(self.constraint, perf_params, dict(self.input_params))
        delta = np.zeros(self.problem_dim).astype(int)
        while True:
            if is_valid:
                return position + delta
            else:
                upperable_dims = []
                lowerable_dims = []
                either_dims = []  # both upperable and lowerable
                radius = radius + 1
                for i in range(self.problem_dim):
                    if position[i] <= self.max_bound[i] - radius:
                        upperable_dims.append(i)
                        if position[i] >= self.min_bound[i] + radius:
                            either_dims.append(i)
                    elif position[i] >= self.min_bound[i] + radius:
                        lowerable_dims.append(i)
                counter = 0
                while counter < 50 and (not is_valid):
                    print(counter)
                    n_change_up = np.random.choice(range(len(upperable_dims) + 1))  # how many dims to up
                    n_change_low = np.random.choice(range(len(lowerable_dims) + 1)) # how many dims to low
                    n_change_either = np.random.choice(range(len(either_dims) + 1))  # how many dims to up or low
                    delta = np.zeros(self.problem_dim).astype(int)
                    if upperable_dims:
                        delta[np.random.choice(upperable_dims, n_change_up, replace=False)] =\
                            1 + round((radius-1)*np.random.random_sample())
                    if lowerable_dims:
                        delta[np.random.choice(lowerable_dims, n_change_low, replace=False)] =\
                            - 1 - round((radius-1)*np.random.random_sample())
                    if either_dims:
                        delta[np.random.choice(either_dims, n_change_either, replace=False)] = \
                            round(2*(0.5-np.random.random_sample()))*(1+round((radius-1)*np.random.random_sample()))
                    counter = counter + 1
                    perf_params = self.coordToPerfParams(list(position+delta))
                    is_valid = eval(self.constraint, perf_params, dict(self.input_params))
                if counter == 50 and radius == 10 and not is_valid: # TODO use parameters here
                    # could not find a valid random point. Iterate until I find one.

                    print self.min_bound
                    print self.max_bound
                    
                    # list all the possible coordinates
                    """ This is really really slow and memory expensive
                    l = [ [ x for x in range( self.min_bound[i], self.max_bound[i] ) ] for i, v in enumerate( self.min_bound ) ]
                    points = [ [x] for x in l[0] ]
                    for i in l[1:]:
                        n = []
                        for k in i:
                            print k
                            for p in points:
                                ne = p + [k]
                                n.append( ne )
                        points = n[:]
                    print len( points ), " possible (or not) points"
                    """

                    """
                    position = self.min_bound[:]
                    i = 0
                    points = []
                    while position[-1] < self.max_bound[-1]:
                        position[i] = position[i] + 1
                     
                        while i < self.problem_dim - 1:
                               if position[i] == self.max_bound[i]:
                                   position[i+1] = position[i+1] + 1
                                   position[i] = self.min_bound[i] 
                                   i = i + 1
                               else:
                                   break
                               
                        print position
                        points.append( position )
                    """
                    found = False
                    print "DELTA: ", delta
                    # very unsure about whether we can shift the boundaries by delta
                    l = [ [ x for x in range( self.min_bound[i] - delta[i], self.max_bound[i] - delta[i] ) ] for i, v in enumerate( self.min_bound ) ]
                    for position in itertools.product(*l):
                        if not str( position ) in self.chosen:
                            perf_params = self.coordToPerfParams( list( position+delta ) )
                            is_valid = eval(self.constraint, perf_params, dict(self.input_params))
                            if is_valid:
                                print "position ", position, " is valid"
                                self.chosen.append( str( position+delta) )
                                found = True
                                break
                    if not found:
                        print "Error: could not find any valid point in the parameter space"

    def _modify_alpha(self):
        self.alpha = self.alpha * self.alpha_decay
