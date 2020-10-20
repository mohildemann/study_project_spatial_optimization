#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:58:17 2020

@author: jessicaruijsch

# --------------------------------------------------
# initialization
# --------------------------------------------------  

Rules:
- A data cell must have exactly one class (between 1 and 13)
- Water, urban area and no data (11, 12, 15) cannot be changed 
- Cerrado, forest, secondary vegetation (1, 3, 13), can be turned into agriculture, but not the other way around
- Combine different soy classes (5, 6, 7, 8, 9) into one class of soy (5)
- Cotton, pasture, soy, sugarcane (2, 4, 5, 10) can be interchanged and replace forest

"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from initial_population import initialize_spatial

default_directory = "/Users/jessicaruijsch/Documents/Internship/study_project_jessica/input_data"

# --------------------------------------------------
# spatial extention pymoo
# --------------------------------------------------

from pymoo import  factory
from pymoo.model.crossover import Crossover
import spatial_extention_pymoo

factory.get_sampling_options = spatial_extention_pymoo._new_get_sampling_options
factory.get_crossover_options = spatial_extention_pymoo._new_get_crossover_options
factory.get_mutation_options = spatial_extention_pymoo._new_get_mutation_options
Crossover.do = spatial_extention_pymoo._new_crossover_do

# --------------------------------------------------
# read data
# --------------------------------------------------

cell_area = 2.5 * 2.5 # in hectares

with open(default_directory + "/Objectives/sugarcane_potential_yield_example.pkl", 'rb') as output:
    sugarcane_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/soy_potential_yield_example.pkl", 'rb') as output:
    soy_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/cotton_potential_yield_example.pkl", 'rb') as output:
    cotton_pot_yield =  pickle.load(output)

with open(default_directory + "/Objectives/pasture_potential_yield_example.pkl", 'rb') as output:
    pasture_pot_yield =  pickle.load(output)

# --------------------------------------------------
# define the problem
# --------------------------------------------------

from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from calculate_objectives import calculate_tot_yield, calculate_above_ground_biomass, calculate_landuse_patches


class MyProblem(Problem):
    
    # by calling the super() function the problem properties are initialized 
    def __init__(self):
        super().__init__(n_var=100,                   # nr of variables
                         n_obj=2,                   # nr of objectives
                         n_constr=0,                # nr of constrains
                         xl=0.0,                   # lower boundaries
                         xu=1.0)                   # upper boundaries

    # the _evaluate function needs to be overwritten from the superclass 
    # the method takes two-dimensional NumPy array x with n rows and n columns as input
    # each row represents an individual and each column an optimization variable 
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = -calculate_tot_yield(X[:], sugarcane_pot_yield,soy_pot_yield,cotton_pot_yield,pasture_pot_yield,cell_area)
        f2 = -calculate_above_ground_biomass(X[:],cell_area)
        # f3 = calculate_landuse_patches(x)

        # after doing the necessary calculations, 
        # the objective values have to be added to the dictionary out
        # with the key F and the constrains with key G 
        out["F"] = np.column_stack([f1, f2])

problem = MyProblem()

# --------------------------------------------------
# initialize the algorithm
# --------------------------------------------------

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("spatial"),
    crossover=get_crossover("spatial_one_point_crossover", n_points = 3),
    mutation=get_mutation("spatial_n_point_mutation", prob = 0.3, point_mutation_probability = 0.1),
    eliminate_duplicates=False
    )
# algorithm.eleminate_duplicates = ElementwiseDuplicateElimination

# --------------------------------------------------
# define the termination criterion
# --------------------------------------------------

from pymoo.factory import get_termination

termination = get_termination("n_gen", 400)

# --------------------------------------------------
# optimize
# --------------------------------------------------

from pymoo.optimize import minimize
 
res = minimize(problem,
               algorithm,
               termination,
               seed=None,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

# --------------------------------------------------
# visualize pareto front
# --------------------------------------------------

from pymoo.visualization.scatter import Scatter

plt.scatter(-res.F[:,0],-res.F[:,1])
plt.title("Objective Space")
plt.xlabel('Total yield [tonnes]')
plt.ylabel('Above ground biomass [tonnes]')
plt.savefig(default_directory+"/figures/objective_space.png",dpi=150)
plt.show()

# --------------------------------------------------
# convergence
# --------------------------------------------------

import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume

# the objective space values in each generation
F = []

# iterate over the deepcopies of algorithms
for algorithm in res.history:
    # retrieve the optimum from the algorithm
    opt = algorithm.opt
    _F = opt.get("F")
    F.append(_F)

# make an array of the number of generations
n_gen = np.array(range(1,len(F)+1))
# set reference point
ref_point = np.array([0.0, 0.0])
# create the performance indicator object with reference point
metric = Hypervolume(ref_point=ref_point, normalize=False)
# calculate for each generation the HV metric
hv = [metric.calc(f) for f in F]

# visualze the convergence curve
plt.plot(n_gen, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Generations")
plt.ylabel("Hypervolume")
plt.ylim(0,1.5*10**11)
plt.savefig(default_directory+"/figures/convergence.png",dpi=150)
plt.show()

# --------------------------------------------------
# visualize land use maps
# --------------------------------------------------

# np.argmax(-res.F[:,0], axis=0) --> optimized for f1
# np.argmax(-res.F[:,1], axis=0) --> optimized for f2

cmap = ListedColormap(["#10773e","#b3cc33", "#0cf8c1", "#a4507d","#877712",
                      "#be94e8","#eeefce","#1b5ee4","#614040","#00000000"])

landuse_max_yield = res.X[np.argmax(-res.F[:,0], axis=0)]
landuse_max_biomass = res.X[np.argmax(-res.F[:,1], axis=0)]

plt.imshow(landuse_max_yield,interpolation='None',cmap=cmap,vmin=0.5,vmax=10.5)
plt.colorbar()
plt.title('Landuse map maximized total yield')
plt.xlabel('Column #')
plt.ylabel('Row #')
plt.savefig(default_directory+"/figures/landuse_max_yield.png",dpi=150)
plt.show()

plt.imshow(landuse_max_biomass, interpolation='None',cmap=cmap,vmin=0.5,vmax=10.5)
plt.colorbar()
plt.title('Landuse map minimized CO2 emissions')
plt.xlabel('Column #')
plt.ylabel('Row #')
plt.savefig(default_directory+"/figures/landuse_min_co2.png",dpi=150)
plt.show()


"""
res.X design space values are
res.F objective spaces values
res.G constraint values
res.CV aggregated constraint violation
res.algorithm algorithm object
res.pop final population object
res.history history of algorithm object. (only if save_history has been enabled during the algorithm initialization)
res.time the time required to run the algorithm
"""


