import evolve
from random import uniform, choice, randint
import orio.main.tuner.search.search
class ChaosGA(orio.main.tuner.search.search.Search):
    def __init__(self, params):
        orio.main.tuner.search.search.Search.__init__(self, params)
        self.search_time = -1
        self.numEvals=500

        # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        # TODO: implement startCoord support
        answer = evolve.newSolution("ga_test.txt", 
            self,
            self.numEvals,    #numEvals
            2.8,    #growthrate
            50,    #population max
            0.6,    #initial population proportion
            min,    #minimizing, not maximizing
            )
        return answer.genome, answer.bestYet, self.search_time, self.numEvals

    def fitness(self, x):
        return self.getPerfCost(x)
    
    def generate(self):
        s = []
        for i in range(self.total_dims):
            s.append(randint(0,self.dim_uplimits[i]))
        return s
    
    def crossover(self, g1, g2):
        return [int_crossover(x,y) for (x,y) in zip(g1,g2)]

    def mutate(self, g):
        return [int_mutate(1,0.8, 0, i, a) for (i,a) in zip(self.dim_uplimits,g)]

def int_crossover(x,y):
    return choice([x,y])

def int_mutate(distance, likelihood, lo, hi, x):
    p = uniform(0,1)
    d = randint(-distance, distance)
    if p<likelihood:
        return bound(lo,hi,x+d)
    else:
        return x

def bound(a,b,x):
    return max(a,min(b,x))
