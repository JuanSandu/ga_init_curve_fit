"""
Class which implements a genetic algorithm for finding the best combination
of initial values to get a solution from fit_curve of Scipy.

Author: Juan Sandubete López.
"""
import os
import random
import json
import time
import numpy as np
import pandas as pd

import scipy.stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from math import ceil


dirname = os.path.dirname(__file__)


class GA_fit_curve:
    def __init__(self, init_params, obj_funct, in_data, out_data, timeline=None,
                 debug_print=False):
        self.max_generations = 75
        self.inc_ranges = [15.0] #
        self.dec_rate = 0.05 # Decreasing rate of inc range per generation
        self.max_parent_pop = 15
        self.max_children_pop = 30
        self.children_prop = ceil(self.max_children_pop / self.max_parent_pop)
        self.prev_parents = 3
        self.method = "curve_fit"
        self.obj_funct = obj_funct
        self.eva_data = {}
        self.eva_data["input"] = in_data
        self.eva_data["output"] = out_data
        self.set_size = len(init_params)
        self.init_values = init_params
        self.parents_pop = []
        self.children_pop = []
        self.children_perfm = []
        self.best_perfms = []
        self.debug_print = debug_print
        self.fail_cnt = 0
        self.success_cnt = 0

    def load_config(self, config_name):
        path = os.path.join(dirname, config_name)
        with open(path, 'r') as fp:
            config_dict = json.load( fp )
        self.max_generations = config_dict["max_generations"]
        self.inc_ranges = config_dict["inc_ranges"]
        self.dec_rate = config_dict["dec_rate"]
        self.max_parent_pop = config_dict["max_parent_pop"]
        self.max_children_pop = config_dict["max_children_pop"]
        self.children_prop = ceil(self.max_children_pop / self.max_parent_pop)
        self.prev_parents = config_dict["prev_parents"]
        self.method = config_dict["method"]

    def find_best_children(self, N):
        """
        This algorithm only take the best N candidates. No more exploration.
        """
        self.best_perfms.append( min(self.children_perfm) )
        return sorted( range(len(self.children_perfm)),
                       key = lambda idx: self.children_perfm[idx] )[:N]

    def propagate_gens(self):
        self.children_pop = []
        for parent_set in self.parents_pop:
            # Generate self.children_prop children per parent set
            for idx in range(self.children_prop):
                # Generate first children gen from initial params
                set_k = []
                for i, val in enumerate(parent_set):
                    # Add a random number to each parameter
                    set_k.append( val +\
                           random.uniform(-self.inc_ranges[i], self.inc_ranges[i]) )
            self.children_pop.append( set_k )

    def eval_obj_func(self):
        ev_method = "eval_" + self.method
        method = getattr( self, ev_method,
                          lambda: "Not valid evaluation method." )
        method()

    def eval_curvefit(self):
        self.children_perfm = []
        for idx, child in enumerate(self.children_pop):
            try:
                popt, pcov = curve_fit( self.obj_funct, self.eva_data["input"],
                                        self.eva_data["output"], p0=child ) # maxfev=1000
                self.children_pop[idx] = popt
                perfm = np.sum(np.sqrt(np.power(np.diag(pcov), 2)))
                self.success_cnt += 1
            except NameError:
                if self.debug_print:
                    print("Error: {}".format(NameError))
                perfm = 10e12
                self.fail_cnt += 1
            except:
                perfm = 10e12
                self.fail_cnt += 1
            self.children_perfm.append( perfm )

    def eval_leastsq(self):
        self.children_perfm = []
        for idx, child in enumerate(self.children_pop):
            ret = leastsq( self.obj_funct, child, args=( self.eva_data["output"],
                             self.eva_data["input"] ), full_output=False)
            popt = ret[0]
            pcov = ret[1] # TODO this is not the pcov matrix actually.
            self.children_perfm.append( np.sum(np.sqrt(pcov)) )

    def get_best_params(self):
        return self.children_pop[self.find_best_children(1)[0]]

    def run(self):
        start_t = time.time() # <- This must be close to the routine
        for idx_gen in range(self.max_generations):
            #print("Running Generation {}.".format(idx_gen+1))
            # 1. Create an initial population
            if idx_gen == 0:
                for idx in range(self.max_parent_pop):
                    # Generate first children gen from initial params
                    set_k = []
                    for i in range(self.set_size):
                        # Add a random number to each parameter
                        set_k.append( self.init_values[i] + \
                               random.uniform(-self.inc_ranges[i], self.inc_ranges[i]) )
                    self.parents_pop.append( set_k )

            # 2. Continue population evolution
            # Generate children gen from parents gen
            self.propagate_gens()

            # 3. Evaluate the objective function
            # This internally updates the children population and the associated
            # performance (errors) list
            self.eval_obj_func()

            # 4. Find best and update the parent population
            indices = self.find_best_children(self.max_parent_pop-self.prev_parents)
            prev_parents = self.parents_pop[:self.prev_parents]
            self.parents_pop = [self.children_pop[i] for i in indices] + prev_parents

            # Update the algorithm parameters
            for i in range(len(self.inc_ranges)):
                self.inc_ranges[i] = self.inc_ranges[i] * (1.0 - self.dec_rate)

        elapsed_t = time.time() - start_t
        print("Genetic optimization finished.")
        print("Elapsed time: {}s.".format(round(elapsed_t, 5)))
        print("Success rate: {}%.".format(
                round(100*self.success_cnt/(self.success_cnt+self.fail_cnt), 2)))
