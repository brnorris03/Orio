from random import choice
from pdb import set_trace
from orio.main.util.globals import *

class Problem:
    '''Just to show the necessary functions.  Do not use.'''

    def fitness(self,x):
        'fitness takes a genome and returns a real number'
        return 0
    
    def generate(self):
        'generates a random new genome'
        return None
    
    def crossover(self, g1, g2):
        'takes two genomes and returns a new child genome'
        return None

    def mutate(self, g):
        'mutates a genome'
        return g

class OrganismData:
    def __init__(self, name, genome, fitresult, birth, parents):
        self.name = name
        self.genome = genome
        self.fitresult = fitresult
        self.birth = birth
        self.parents = parents
    def __repr__(self):
        return "OrganismData("+repr(self.name)+","+repr(self.genome)+","+repr(self.fitresult)+","+ repr(self.birth)+","+repr(self.parents)+")"


#this is the orio.main.function to run an experiment
def newSolution(filename, problem, numEvals, growthRate, popMax, popInit, opt=max):
    gensize = int(max(2,round(popMax*popInit)))
    organisms = [OrganismData(n, problem.generate(), 0, 0, []) for n in range(1,gensize+1)]
    if filename:
        logHandle = open(filename,'w')
    else:
        logHandle = None
    fitmap(problem.fitness, organisms, logHandle)
    bestYet = opt(organisms, key = lambda x : x.fitresult)
    return evolve(problem, logHandle, organisms, numEvals - gensize, growthRate, popMax, popInit, gensize+1, bestYet, opt)

def evolve(problem, logHandle, population, numEvals, growthRate, popMax, pop, nameCount, bestYet, opt):
    generationCount = 1
    while numEvals > 0:
        pop = growthRate * pop * (1 - pop)
        nextgensize = int(max(2, round(pop * popMax)))
        nextGen = generation_tournament_elite(problem, population, int(min(numEvals, nextgensize-1)), generationCount, nameCount, opt) #first generate untested
        fitmap(problem.fitness, nextGen, logHandle) #test the new organisms
        population = [bestYet]+nextGen #append previous best
        numEvals -= nextgensize-1
        generationCount += 1
        nameCount += nextgensize-1
        bestYet = opt(population, key = lambda x : x.fitresult)
    if logHandle:
        logHandle.close()
    return bestYet

def fitmap(fitness, population, logHandle):
    for a in population:
        a.fitresult = fitness(a.genome)
    logPop(population, logHandle)

def logPop(population, handle):
    if handle:
        handle.write(repr(population)+"\n")
        handle.flush()

def generation_tournament_elite(problem, population, popsize, currentGeneration,start, opt): 
    return [mutate(problem, binaryTournamentPair(currentGeneration, problem, i, population, opt)) for i in range(start, start+popsize)]

def binaryTournamentPair(currentGeneration, problem, n, pop, opt):
    return newChild(problem, n, currentGeneration, opt(choice(pop), choice(pop), key = lambda x : x.fitresult), betterOrganism(choice(pop), choice(pop)))

def betterOrganism(a,b):
    if a.fitresult >= b.fitresult:  
        return a 
    else:
        return b

def newChild(problem, newname, curgen, o1, o2):
    newgenome = problem.crossover(o1.genome, o2.genome)
    return OrganismData(newname, newgenome, 0, curgen, [o1.name, o2.name])

def mutate(problem, organism):
    organism.genome = problem.mutate(organism.genome)
    return organism

#an example of how to read the text file
def parent_analysis(f):
    data = open(f,'r')
    sc = []
    for line in data:
        sc.append(eval(line))
    for line in sc:
        best = max(line, key = lambda x : x.fitresult)
        info("orio.main.tuner.search.chaosga.evolve: %s,%s,%s" % (best.name,str(best.fitresult), str(best.parents)))
