## Our modifications

### Search algorithms

#### Firefly algorithms

The Firefly Algorithm by Xin-She Yang was implemented for solving this kind of optimization problems. The modifications respect to the orgininal are mainly two: the gamma parameter, since we operate in large dimensions, is automatically provided to avoid the curse of high dimensionality. Also, the unroll bounds were managed separately by a particular function which provide unroll feasibility to the nearest point. For the non-unroll variables, the task is left to z3. The algorithm kwargs parameters are:

``` python
population_size # The number of fireflies
generations # The number of major loops of the algorithm
alpha # The noise coefficient
alpha_decay # The multiplicative decay of alpha
beta_init # beta zero (see Yang paper)
auto_gamma # Say if we auto compute gamma as explained before (default: True)
```

The unroll variables must be passed as an external parameter.

#### Variable neighborhood search

(Camille)

#### Random simple

(Camille)

#### Modification to the Nelder-Mead algorithm:

The Nelder-Mead algorithm interacts with the point selection by taking random points and a centroid. The random points need to be feasible, which can be provided by z3. The centroid needs to be feasible too. We have implemented a function that takes the closest feasible point (see below).

* TODO: store the polyhedra that have already been explored and proceed as long as the problem is SAT (ie, until all the parameter space has been tiled).

### Point finding

#### Reject / Validate

#### SMT solver (z3)

When the parameters are very constrained (eg, adi.ancc.src1.c), finding valid points can be difficult. The basic approach that consists in picking a point and testing whether it is valid can be very expensive and reject a lot of points.

We propose to use a SMT solver to find valid points. Here we have written its integration using z3.

##### `tuner.py`

* Import the z3 module. If the import fails, a boolean variable is set to False and Z3 is never used.
 
* Build the solver and the optimizer

The solver is built by a call to its constructor. An optimizer is created too. The optimizer can be used for instance to get a feasible point that minimizes a distance (e.g., to a random point). 

``` python
if self.have_z3:
   self.solver = z3.Solver()
   self.optim = z3.Optimize()
   self.variables = []
   Globals.z3types = {}
   Globals.z3variables = {}
```

Then the variables are created by `__addVariableNames()` that defines their definition domain is provided to the solver (and the optimizer) by a call to the `__addConstraints()` function. 

* Create the variables

The `__addVariableNames()` function builds the z3 variables and saves them in the `Globals.z3variables` dictionnary. They will be used by functions that manipulate the z3 variables, because they need these variables:
``` python
for i in self.axis_names:
    locals()[i] = Globals.z3variables[i]
```

It also detects non-numeric variables and creates a "pseudo domain" of integers, because z3 does not handle non-numeric variables well. Non-numeric variables are considered as integers and mapped to a domain by the `__defineNonNumeric()` function.

* Non-numeric variables

Non-numeric variables are inserted in the `Globals.z3types` dictionary: after that, looking if a variable name is a key of this dictionary can be used to test whether a variable is numeric. This is what the `z3IsNumeric()` function does (in `search.py`).

Two translation routines between this integer range and the actual non-numeric values are provided: `__numericToNonNumeric()` and  `__numericToNonNumeric()`.

* Create the definition domain of the solver

The `__addDefinitionDomain()` function provides the list of possible values for the variable. ` __addConstraints` inputs the constraints provided by the user. Since they are provided using and infix syntax whereas z3 uses a prefix syntax, the translation is made by the `__infixToPrefix()` function.

* Create the search engine

Pass two additional variables:
```python
'search_z3': self.have_z3,
'solver': ( self.solver, self.optim )
```

##### `search.py`

* Use z3 or not

We try to load the module, but if we cannot, we simply pass. Using or not z3 is obtained in the parameters passed by the tuner:

```python
if 'search_z3' in params.keys(): self.have_z3 = params['search_z3']
        if self.have_z3 and 'solver' in params.keys(): self.solver, self.optim = params['solver']
```

I have put a TODO here, because I would like to let the used chose it as a parameter. NB: it can be done in the search modules.

* Coordinates system

One issue when using z3 is that z3 works with performance parameters (except for non-numeric values) whereas Orio uses coordinates a lot. For instance, if the possible values are [8, 16, 32, 64], z3 really needs these values in order to verify the constraints.

Therefore, the fonctions that convert constraits to performance parameters are called very often. `coordToPerfParams()` is unchanged. Since `coordToPerfParams()` produces a dictionary while sometimes we need a table, we have added the function `coordToTabOfPerfParams()`. `perfParamToCoord()` needs a specific case when using z3.

This can be fixed: when using z3, the `perfParamToCoord()` is called with a list parameter, whereas normally it is called with a dictionary. Later we can make it more uniform.

* Random point selection

Picking a random point in SMT is not trivial. Some work has been done on SAT, we will look at a SAT formulation later.

The function `getRandomCoordZ3()` simply returns a feasible point. Not very random.

The function `getRandomCoordZ3Distance()` takes a random point and gets the closest feasible point. This is where we need the optimizer.

However, this approach does not give a uniform distribution: for instance, if we have the following possible values: [8, 16, 32], 17 will be considered closer to 8 that to 32. We will see later if we can use a different distance function with a mapping between coordinates and real values in z3.

* Centroid

This function is needed by the Nelder-Mead algorithm. It needs a cendroid, but this point obtained by a simple computation on the point coordinates is not alweays feasible. The `getCentroidZ3()` function gets a feasible one by taking the closest feasible point. It also uses the optimizer.

* Neighbors

The `getNeighborsZ3()` selects only feasible points within a certain distance. It takes into account the fact that the notion of distance is eant in terms of coordinates and not in the actual parameter space. For instance if we have the following possible values: [8, 16, 32], 8 and 32 are both at distance 1 of 16.

It gets all the possible points by enumrating them: as long as the problem is SAT, it gets a point and adds it as a constraint.

* Model object produced by z3

z3 returns a model, whose fields can be accessed as a dictionnary but the data in it cannot be manipulated by a regular dictionnary. Therefore, we have added a function to convert the resulting model into a point: `z3ToPoint()`. It gives a table that contains the performance parameters, including non-numeric ones. 

TODO: we can modify it to output a dictionary and improve the homogeneity of data structures. 
