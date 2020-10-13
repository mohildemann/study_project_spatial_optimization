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

Initialization rules:
- random variation with constrains
- 70% of the initial map stays, 30% random classes with constrains:
    - water, urban area and no data (8, 9, 10) cannot be changed
    - cerrao, forest, secondary vegetation (1, 2, 3) can be turned into agriculture
    - cotton, pasture, soy, sugarcane (4, 5, 6, 7) can be changed
    

Reclassify:
1 = forest                #10773e
2 = cerrado               #b3cc33  
3 = secondary_vegetation  #0cf8c1
4 = soy                   #a4507d
5 = sugarcane             #877712
6 = fallow_cotton         #be94e8
7 = pasture               #eeefce
8 = water                 #1b5ee4
9 = urban                 #614040
10 = no_data              #00000000
    
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from initial_population import initialize_spatial

default_directory = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Study Project\Internship"

# --------------------------------------------------
# read data
# --------------------------------------------------

# read data from tiff file
# read potential yield maps from asc file
# sugarcane_pot_yield = np.loadtxt(default_directory + "/Objectives/sugarcane_new.asc", skiprows=6)[2800:2900,1700:1800]
# soy_pot_yield = np.loadtxt(default_directory + "/Objectives/soy_new.asc", skiprows=6)[2800:2900,1700:1800]
# cotton_pot_yield = np.loadtxt(default_directory + "/Objectives/cotton_new.asc", skiprows=6)[2800:2900,1700:1800]
# pasture_pot_yield = np.loadtxt(default_directory + "/Objectives/grass_new.asc", skiprows=6)[2800:2900,1700:1800]
cell_area = 2.5 * 2.5 # in hectares
#
# with open(default_directory + "\\sugarcane_potential_yield_example.pkl", 'wb') as output:
#     pickle.dump(sugarcane_pot_yield, output, pickle.HIGHEST_PROTOCOL)
#
# with open(default_directory + "\\soy_potential_yield_example.pkl", 'wb') as output:
#     pickle.dump(soy_pot_yield, output, pickle.HIGHEST_PROTOCOL)
#
# with open(default_directory + "\\cotton_potential_yield_example.pkl", 'wb') as output:
#     pickle.dump(cotton_pot_yield, output, pickle.HIGHEST_PROTOCOL)
#
# with open(default_directory + "\\pasture_potential_yield_example.pkl", 'wb') as output:
#     pickle.dump(pasture_pot_yield, output, pickle.HIGHEST_PROTOCOL)
#


with open(default_directory + "\\sugarcane_potential_yield_example.pkl", 'rb') as output:
    sugarcane_pot_yield =  pickle.load(output)

with open(default_directory + "\\soy_potential_yield_example.pkl", 'rb') as output:
    soy_pot_yield =  pickle.load(output)

with open(default_directory + "\\cotton_potential_yield_example.pkl", 'rb') as output:
    cotton_pot_yield =  pickle.load(output)

with open(default_directory + "\\pasture_potential_yield_example.pkl", 'rb') as output:
    pasture_pot_yield =  pickle.load(output)
# --------------------------------------------------
# reclassify
# --------------------------------------------------


# --------------------------------------------------
# define the problem
# --------------------------------------------------

from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from calculate_objectives import calculate_tot_yield, calculate_CO2_emissions, calculate_landuse_patches


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
        print(X.shape)
        f1 = -calculate_tot_yield(X[:], sugarcane_pot_yield,soy_pot_yield,cotton_pot_yield,pasture_pot_yield,cell_area)
        f2 = -calculate_CO2_emissions(X[:],cell_area)
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

termination = get_termination("n_gen", 10)

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




