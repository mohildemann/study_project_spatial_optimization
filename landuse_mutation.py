import numpy as np

from pymoo.model.mutation import Mutation
from compute_genome import getNbh, determine_patch_elements, create_patch_ID_map

class SpatialNPointMutation(Mutation):

    def __init__(self, prob=None,point_mutation_probability=0.01):
        super().__init__()
        self.prob = prob
        self.point_mutation_probability = point_mutation_probability

    def _do(self, problem, X, **kwargs):
        # -------------------------------------------------------------------------------------
        # Random reset mutation
        # -------------------------------------------------------------------------------------

        def random_reset_mutation(genome_in, point_mutation_prob):
            genome = list(genome_in)
            for i in range(1, len(genome)):
                if np.random.uniform(0, 1)>point_mutation_prob:
                    randomnumber = np.random.uniform(0, 1)
                    if randomnumber < 0.0025:
                        genome[i] = 4.
                    elif randomnumber < 0.005:
                        genome[i] = 5.
                    elif randomnumber < 0.0075:
                        genome[i] = 6.
                    elif randomnumber < 0.01:
                        genome[i] = 7.

            return (genome)

        shape_landusemaps = [X[0].shape[0], X[0].shape[1]]
        rows = shape_landusemaps[0]
        cols = shape_landusemaps[1]
        offsprings = []
        for i in X:
            if np.random.uniform(0, 1) > self.prob:
                patches, genome = create_patch_ID_map(i, 0, [8, 9], "True")
                copy_genome_i = random_reset_mutation(genome,self.point_mutation_probability)
                #test = np.array(copy_genome_i) == np.array(genome)
                
                # child 1: fill in genome in patches
                mutated_individual = patches
                for x in range(0, cols):
                    for y in range(0, rows):
                        if mutated_individual[x, y] != 0:
                            mutated_individual[x, y] = genome[mutated_individual[x, y] - 1]
                        else:
                            mutated_individual[x, y] = i[x,y]
                            
                mutated_individual = np.where(mutated_individual == 0, i, mutated_individual)
                offsprings.append(mutated_individual)
            else:
                offsprings.append(i)
        offsprings = np.array(offsprings)
        return offsprings
