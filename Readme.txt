This folder ('Scripts') contains the scripts for the spatial functions of the optimalization.

- nsga2_example.py --> runs a non-spatial nsga2 algorithm from the pymoo library

- calculate_objectives.py --> contains functions to calculate the objective functions
- compute_genome.py --> contains helper functions from CoMOLA to compute patches and genome from the land use map
- crossover.py --> contains spatial crossover and mutation functions
- initial_population --> contains function to make the initial land use maps and create an initial population for pymoo

- run_nsga2_spatial.py --> contains set up of the spatial ngsa2 algorithm and runs the pymoo library

- translate.sh --> takes input maps and translates them to the same resolution and coordinate system using GDAL


Problem:
I am trying to run the NSGA II algorithm using the pymoo library, but with spatial inputs. I wrote the run_nsga2_spatial.py script for this, but I am having problems with defining the population in the correct way. In addition, the algorithm says the number of dimensions of the population array is too small. Judith preferably wants to make the algorithm spatial without changing anything in the pymoo library scripts. I am not sure how to overwrite the variables in the pymoo library from outside. 


To do:
- define the population of spatial maps (correctly)
- overwrite the standard creation of the population with my script
- write crossover script in such a way that it can be read by the pymoo algorithm
- the calculate_objectives.py should work, but I get an error that the dimensions are not right --> solve this
- run algorithm without errors about the number of dimensions that are wrong
