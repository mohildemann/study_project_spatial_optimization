import numpy as np
from pymoo.model.sampling import Sampling
from pymoo.util.normalization import denormalize
import initial_population

class SpatialSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        default_directory = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Study Project\Internship"
        landusemaps_np = initial_population.initialize_spatial(n_samples, default_directory)
        return landusemaps_np
