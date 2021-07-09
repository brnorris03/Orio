# Implementation of the ml search algorithm
# Primary author: Prasanna Balaprakash (pbalapra@mcs.anl.gov)
# Bug fixes, updates: Boyana Norris (brnorris03@gmail.com)
#
# The ML model that is used is extra tree regressor. 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
#
# The main limitation: it is a local search method (only exploitation not much exploration). 
# The quality of the local solution will depend on the initialization. 
# One way to overcome this is to specify the use_z3=True (in the tuning spec)  when using this method.
#
# Assumption: The search space is discrete (integer/categorical parameters) 
# (not real, which is the case for most of the autotuning problems).



import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *

from sklearn import ensemble
import numpy as np
import pandas as pd
import copy
import random
import json


# -----------------------------------------------------

class Mlsearch(orio.main.tuner.search.search.Search):
    '''
    The search engine that uses a ml search approach
    the best neighboring coordinate.

    Below is a list of algorithm-specific arguments used to steer the search algorithm.
    '''

    # algorithm-specific argument names
    __LOCAL_DIST = 'local_distance'  # default: 0

    # --------------------------------------------------

    def __init__(self, params):
        '''To instantiate a random search engine'''

        random.seed(1)

        orio.main.tuner.search.search.Search.__init__(self, params)

        # set all algorithm-specific arguments to their default values
        self.local_distance = 0

        self.init_samp = 10000
        self.batch_size = 5

        # read all algorithm-specific arguments
        self.__readAlgoArgs()

        # complain if both the search time limit and the total number of search runs are undefined
        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.mlsearch.mlsearch: %s search requires the ' +
                'total number of search runs (total_runs) to be defined') % self.__class__.__name__)

    # Method required by the search interface
    def searchBestCoord(self, startCoord=None):
        '''
        To explore the search space and retun the coordinate that yields the best performance
        (i.e. minimum performance cost).
        '''
        # TODO: implement startCoord support

        info('\n----- begin ml search -----')

        # get the total number of coordinates to be tested at the same time
        coord_count = 1
        if self.use_parallel_search:
            coord_count = self.num_procs

        # initialize a storage to remember all coordinates that have been explored
        coord_records = {}

        # initialize a list to store the neighboring coordinates
        neigh_coords = []

        # record the best coordinate and its best performance cost
        best_coord = None
        best_perf_cost = self.MAXFLOAT
        old_perf_cost = best_perf_cost

        # record the number of runs
        runs = 0
        sruns = 0
        fruns = 0

        # start the timer
        start_time = time.time()
        init = True

        # randomly pick coordinates to be empirically tested
        coords = {}
        uneval_coords = []
        uneval_params = []

        # default code without transformation
        neigh_coords = [[0] * self.total_dims]

# FIXME BN: The time_limit unfortunately puts the search in an invalid state, so temporarily disabling
#        while (len(uneval_coords) < self.init_samp) and not ((time.time()-start_time) > self.time_limit > 0):
        while (len(uneval_coords) < self.init_samp):
            coord = self.__getNextCoord(coord_records, neigh_coords, init)

            if coord is None:
                break

            coord_key = str(coord)

            if len(coord) == 0:
                break

            if coord_key not in coords:
                coords[coord_key] = coord
                # if the given coordinate is out of the search space
                is_out = False
                for i in range(0, self.total_dims):
                    if coord[i] < 0 or coord[i] >= self.dim_uplimits[i]:
                        is_out = True
                        break
                if is_out:
                    continue
                # test if the performance parameters are valid
                perf_params = self.coordToPerfParams(coord)
                perf_params1 = copy.copy(perf_params)

                try:
                    is_valid = eval(self.constraint, perf_params1, dict(self.input_params))
                except Exception as e:
                    err('failed to evaluate the constraint expression: "%s"\n%s %s' % (
                    self.constraint, e.__class__.__name__, e))
                # if invalid performance parameters
                if not is_valid:
                    continue

                temp = []
                for k in sorted(perf_params):
                    temp.append(perf_params[k])

                uneval_coords.append(coord)
                uneval_params.append(perf_params)

        debug("# of coordinates=%d, unevaluated coords=%d, unevaluated params=%d" % (len(coords), len(uneval_coords), len(uneval_params)), obj=self)

        eval_coords = []
        eval_params = []
        eval_cost = []
        num_eval_best = 0

        indices = random.sample(list(range(1, len(uneval_coords))), self.total_dims)
        # indices=random.sample(range(1,len(uneval_coords)),  5)
        indices.insert(0, 0)
        debug("Indices: %s" % str(indices), obj=self)

        for index in indices:
            coord = uneval_coords[index]
            params = uneval_params[index]
            eval_coords.append(coord)
            eval_params.append(params)

            debug("Params: %s" % str(params), obj=self)
            runs += 1

            perf_costs = {}
            perf_cost = self.MAXFLOAT
            mean_perf_cost = perf_cost
            try:
                perf_costs = self.getPerfCosts([coord])
            except Exception as e:
                perf_costs[str(coords)] = [self.MAXFLOAT]
                info('FAILED: %s %s' % (e.__class__.__name__, e))
                fruns += 1

            # compare to the best result
            pcost_items = sorted(list(perf_costs.items()))
            for i, (coord_str, pcost) in enumerate(pcost_items):
                if type(pcost) == tuple:
                    (perf_cost, _) = pcost  # ignore transfer costs -- GPUs only
                else:
                    perf_cost = pcost

                try:
                    floatNums = [float(x) for x in perf_cost]
                    mean_perf_cost = sum(floatNums) / len(perf_cost)
                except:
                    mean_perf_cost = perf_cost

            # transform_time=self.getTransformTime()
            # compile_time=self.getCompileTime()
            transform_time = self.getTransformTime(coord_key)
            compile_time = self.getCompileTime(coord_key)

            res_obj = {}
            res_obj['run'] = runs
            res_obj['coordinate'] = coord
            res_obj['perf_params'] = params
            res_obj['transform_time'] = transform_time
            res_obj['compile_time'] = compile_time
            res_obj['cost'] = perf_cost
            info('(run %s) | %s | ' % (runs,datetime.datetime.now()) + json.dumps(res_obj))

            # info('(run %s) coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs, coord, params, transform_time, compile_time,perf_cost))

            eval_cost.append(mean_perf_cost)

            if mean_perf_cost < best_perf_cost and mean_perf_cost > 0.0:
                best_coord = coord
                best_perf_cost = mean_perf_cost
                info('>>>> best coordinate found: %s, cost: %e' % (coord, mean_perf_cost))
                num_eval_best = runs

            if not math.isinf(mean_perf_cost):
                sruns += 1

            # remove the indices
        indices = sorted(indices, reverse=True)
        for index in indices:
            uneval_coords.pop(index)
            uneval_params.pop(index)

        debug("# of coordinates=%d, unevaluated coords=%d, unevaluated params=%d, evaluated coords=%d" % (
            len(coords), len(uneval_coords), len(uneval_params), len(eval_coords)), obj=self)

        # sys.exit(0)

        # Create linear regression object
        # regr = linear_model.LinearRegression()
        # regr = GradientBoostingRegressor(n_estimators=1000, random_state=0)# loss='quantile',alpha=0.05)
        # regr = ensemble.BaggingRegressor(n_estimators=1000)
        regr = ensemble.ExtraTreesRegressor(n_estimators=1000, random_state=0)
        while True:
            batch_size = min(self.batch_size, self.total_runs - len(eval_coords))
            debug("+++++++ new iteration; batch size = %d" % batch_size, obj=self, level=6)
            if len(uneval_params) == 0: break

            debug('Initial eval cost %s' % str(eval_cost), obj=self)
            eval_cost = [min(x, 100) for x in eval_cost]
            debug('Eval cost %s' % str(eval_cost), obj=self)

            X_train = pd.DataFrame(eval_params)   #.select_dtypes(include=[np.number,np.bool])
            X_test = pd.DataFrame(uneval_params)
            X_all = pd.concat(objs=[X_train, X_test], axis=0)
            X_preprocessed = pd.get_dummies(X_all)   # one-hot encoding
            X_train_preprocessed = X_preprocessed[:len(eval_params)]
            X_test_preprocessed = X_preprocessed[len(eval_params):]
            # Some parameters, notably CFLAGS are strings; we could convert them to one-hot encoding.
            #X_train = pd.concat([X_train, pd.get_dummies(X_train['CFLAGS'], prefix='CFLAGS')], axis=1)

            #X_train["CFLAGS_cat"] = X_train["CFLAGS"].astype('category').cat.codes  # convert to categorical
            #X_train.drop(['CFLAGS'], axis=1, inplace=True)  # drop original CFLAGS, no longer needed
            debug("Train:\n%s" % str(X_train_preprocessed), obj=self, level=6)
            debug("Test:\n%s" % str(X_test_preprocessed), obj=self, level=6)
            Y_train = np.array(eval_cost)
            #X_test = pd.DataFrame(uneval_params)  #.select_dtypes(include=[np.number,np.bool])
            #X_test["CFLAGS_cat"] = X_test["CFLAGS"].astype('category').cat.codes  # convert to categorical

            # Train the model using the training sets
            debug("X train shape: %s" % str(X_train_preprocessed.shape), obj=self, level=6)
            debug("Y train shape: %s" % str(Y_train.shape), obj=self, level=6)
            debug("X test shape: %s" % str(X_test_preprocessed.shape), obj=self, level=6)

            regr.fit(X_train_preprocessed, 1.0 / Y_train)

            pred = 1.0 / regr.predict(X_test_preprocessed)
            indices = np.argsort(pred)[:batch_size]

            # prepare the batch
            batch_coords = []
            batch_params = []
            batch_cost = []

            debug("Number of unevaluated coordinates: %d" % len(uneval_coords), obj=self, level=6)
            debug("Number of unevaluated parameters: %d" % len(uneval_params), obj=self, level=6)
            indices = sorted(indices, reverse=True)
            for index in indices:
                coords = uneval_coords.pop(index)
                params = uneval_params.pop(index)
                batch_coords.append(coords)
                batch_params.append(params)

            # evaluate the batch
            for i in range(len(batch_coords)):     #range(batch_size):
                coord = batch_coords[i]
                coord_key = str(coord)
                params = batch_params[i]
                mean_perf_cost = [self.MAXFLOAT]

                perf_costs = {}
                try:
                    perf_costs = self.getPerfCosts([coord])
                except Exception as e:
                    perf_costs[str(coords)] = [self.MAXFLOAT]
                    info('FAILED: %s %s' % (e.__class__.__name__, e))
                    fruns += 1

                pcost_items = sorted(list(perf_costs.items()))
                for i, (coord_str, pcost) in enumerate(pcost_items):
                    if type(pcost) == tuple:
                        (perf_cost, _) = pcost  # ignore transfer costs -- GPUs only
                    else:
                        perf_cost = pcost

                    try:
                        floatNums = [float(x) for x in perf_cost]
                        mean_perf_cost = sum(floatNums) / len(perf_cost)
                    except:
                        mean_perf_cost = perf_cost

                batch_cost.append(mean_perf_cost)
                transform_time = self.getTransformTime(coord_key)
                compile_time = self.getCompileTime(coord_key)

                res_obj = {}
                res_obj['run'] = runs
                res_obj['coordinate'] = coord
                res_obj['perf_params'] = params
                res_obj['transform_time'] = transform_time
                res_obj['compile_time'] = compile_time
                res_obj['cost'] = perf_cost
                info('(run %s) | %s | ' % (runs,datetime.datetime.now()) + json.dumps(res_obj))

                if mean_perf_cost < best_perf_cost and mean_perf_cost > 0.0:
                    best_coord = coord
                    best_perf_cost = mean_perf_cost
                    info('>>>> best coordinate found: %s, cost: %e' % (coord, mean_perf_cost))
                    num_eval_best = runs

                if not math.isinf(mean_perf_cost):
                    sruns += 1
                    # info('(run %s) sruns: %s, fruns: %s, coordinate: %s, perf_params: %s, transform_time: %s, compile_time: %s, cost: %s' % (runs, sruns, fruns, coord, p, transform_time, compile_time,mean_perf_cost))

                if self.total_runs > 0 and runs >= self.total_runs: break
                runs += 1

            debug('# of predicted values:\n%s' % str(np.sort(pred)[:batch_size]),obj=self)
            debug('# of observed values=%s, successful runs=%d, total_runs=%d' % (str(batch_cost), sruns, self.total_runs), obj=self)

            # add to the training set
            eval_coords.extend(batch_coords)
            eval_params.extend(batch_params)
            eval_cost.extend(batch_cost)

            # check if the time is up
            # info('%s' % self.time_limit)
            if self.time_limit > 0 and (time.time() - start_time) > self.time_limit:
                break
            # check if the maximum limit of runs is reached
            if self.total_runs > 0 and runs >= self.total_runs:
                break

        sort_ind = np.argsort(eval_cost)
        best_coord = eval_coords[sort_ind[0]]
        best_perf_cost = eval_cost[sort_ind[0]]

        debug("eval_params: %s" % str(eval_params[sort_ind[0]]), obj=self, level=6)
        debug("Best performance cost: %s" % str(best_perf_cost), obj=self, level=6)
        debug("Best coordinate: %s" % str(best_coord), obj=self, level=6)
        end_time = time.time()
        search_time = start_time - end_time
        speedup = float(eval_cost[0]) / float(best_perf_cost)

        # compute the total search time
        search_time = time.time() - start_time

        info('----- end ml search -----')
        info('----- begin ml  search summary -----')
        info(' total completed runs: %s' % runs)
        info(' total successful runs: %s' % sruns)
        info(' total failed runs: %s' % fruns)
        info(' speedup: %s' % speedup)
        info(' found at: %s' % num_eval_best)
        info('----- end ml search summary -----')

        # return the best coordinate
        return best_coord, best_perf_cost, search_time, sruns

    # Private methods
    # --------------------------------------------------

    def __readAlgoArgs(self):
        '''To read all algorithm-specific arguments'''

        # check for algorithm-specific arguments
        for vname, rhs in self.search_opts.items():

            # local search distance
            if vname == self.__LOCAL_DIST:
                if not isinstance(rhs, int) or rhs < 0:
                    err('orio.main.tuner.search.randomsearch: %s argument "%s" must be a positive integer or zero'
                        % (self.__class__.__name__, vname))
                self.local_distance = rhs

            # unrecognized algorithm-specific argument
            else:
                err('orio.main.tuner.search.randomsearch: unrecognized %s algorithm-specific argument: "%s"' %
                    (self.__class__.__name__, vname))

    # --------------------------------------------------

    def __getNextCoord(self, coord_records, neigh_coords, init):
        '''Get the next coordinate to be empirically tested'''

        # if len(coord_records) == 0:
        #  return [0] * self.total_dims

        # info('neighcoords: %s' % neigh_coords)
        # check if all coordinates have been explored
        if len(coord_records) >= self.space_size:
            return None

        # pick the next neighbor coordinate in the list (if exists)
        while len(neigh_coords) > 0:
            coord = neigh_coords.pop(0)
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

        # randomly pick a coordinate that has never been explored before
        while True:
            coord = self.getRandomCoord()
            if str(coord) not in coord_records:
                coord_records[str(coord)] = None
                return coord

    # --------------------------------------------------
