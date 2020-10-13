#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:42:01 2020

@author: jessicaruijsch

# --------------------------------------------------
# goal of script 
# --------------------------------------------------  
    in this script the 'getting started' of the pymoo website
    is executed. It also contains comment on each step
    http://pymoo.org/getting_started.html

# --------------------------------------------------
# optimization problem
# --------------------------------------------------   
    min f1(x) = (x1^2 + x2^2)
    max f2(x) = -(x1-1)^2 - x2^2
    
    s.t. g1(x) = 2(x1 - 0.1) * (x1 - 0.9) <= 0
         g2(x) = 20(x1-0.4) * (x1-0.6) >= 0 
        
         -2 <= x1 <= 2
         -2 <= x2 <= 2
    
two objective (M=2), two inequality constrains (J=2), two variables (N=2), 
no equality constrains (K=0)

# --------------------------------------------------
# problem definition
# --------------------------------------------------
    only minimization problems
    all constraint functions need to be formulated as <= 0
    normalization of constrains is recommended
    
    min f1(x) = (x1^2 + x2^2)
    min f2(x) = (x1-1)^2 + x2^2
    
    s.t. g1(x) = 2(x1 - 0.1) * (x1 - 0.9) / 0.18 <= 0    because (2*(-0.1)*(-0.9)=0.18)
         g2(x) = -20(x1-0.4) * (x1-0.6) / 4.8 <= 0       because (20*(-0.4)*(-0.6)=4.8)
        
         -2 <= x1 <= 2
         -2 <= x2 <= 2
"""

import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem


# --------------------------------------------------
# define the problem
# --------------------------------------------------
class MyProblem(Problem):
    
    # by calling the super() function the problem properties are initialized 
    def __init__(self):
        super().__init__(n_var=2,                   # nr of variables
                         n_obj=2,                   # nr of objectives
                         n_constr=2,                # nr of constrains
                         xl=np.array([-2,-2]),      # lower boundaries
                         xu=np.array([2,2]))        # upper boundaries

    # the _evaluate function needs to be overwritten from the superclass 
    # the method takes two-dimensional NumPy array x with n rows and n columns as input
    # each row represents an individual and each column an optimization variable 
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2
        f2 = (x[:,0]-1)**2 + x[:,1]**2

        g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8

        # after doing the necessary calculations, 
        # the objective values have to be added to the dictionary out
        # with the key F and the constrains with key G 
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


    # --------------------------------------------------
    # Pareto-front - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_front(self, flatten=True, **kwargs):
        f1_a = np.linspace(0.1**2, 0.4**2, 100)
        f2_a = (np.sqrt(f1_a) - 1)**2

        f1_b = np.linspace(0.6**2, 0.9**2, 100)
        f2_b = (np.sqrt(f1_b) - 1)**2

        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
        return stack(a, b, flatten=flatten)

    # --------------------------------------------------
    # Pareto-set - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_set(self, flatten=True, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a,b, flatten=flatten)

problem = MyProblem()


# --------------------------------------------------
# initialize the algorithm
# --------------------------------------------------
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("(real|bin|int)_one_point", n_points = 3),
    mutation=get_mutation("real_pm",eta=20),
    eliminate_duplicates=True
    )

# --------------------------------------------------
# define the termination criterion
# --------------------------------------------------
from pymoo.factory import get_termination

termination = get_termination("n_gen", 100)

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
# visualize
# --------------------------------------------------
from pymoo.visualization.scatter import Scatter

# get the pareto-set and pareto-front for plotting
ps = problem.pareto_set(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=False)

# Design Space
plot = Scatter(title = "Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
plot.apply(lambda ax: ax.set_ylim(-2, 2))
plot.show()

# Objective Space
plot2 = Scatter(title = "Objective Space")
plot2.add(res.F)
plot2.add(pf, plot_type="line", color="black", alpha=0.7)
plot2.show()

# --------------------------------------------------
# performance tracking
# --------------------------------------------------
# import matplotlib.pyplot as plt
# from pymoo.performance_indicator.hv import Hypervolume

# # create the performance indicator object with reference point (4,4)
# metric = Hypervolume(ref_point=np.array([1.0, 1.0]))

# # collect the population in each generation
# pop_each_gen = [a.pop for a in res.history]

# # receive the population in each generation
# obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

# # calculate for each generation the HV metric
# hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

# # function evaluations at each snapshot
# n_evals = np.array([a.evaluator.n_eval for a in res.history])

# # visualze the convergence curve
# plt.plot(n_evals, hv, '-o')
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("Hypervolume")
# plt.show()


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