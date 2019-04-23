import numpy as np

from spring_mass_model import SpringMassModel
from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness to estimate the expected value of the maximum
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.

Demonstrates the modular ("advanced") usage of MLMCPy where a user splits the
analysis into 3 steps/scripts. It also demonstrates the optional modular cache
functionality. This is script #1 for initialization with MLMCPy to determine how
many model evaluations to do on each level and the input parameters to perform
the evaluations with.
'''

# Step 1 - Compute the optimal sample sizes, setup cache and store model inputs:

# Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)

np.random.seed(1)
stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.,
                                     random_seed=1)

# Initialize spring-mass models for MLMC. Here using three levels with MLMC 
# defined by different time steps:
model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=0.00034791)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=0.00073748)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=0.00086135)

models = [model_level1, model_level2, model_level3]

# Initialize MLMCSimulator:
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

# Calculate optimal sample size for each level:
# Optional - compute cost and variances of model (or user knows beforehand)
initial_sample_size = 100
epsilon = np.sqrt(0.00170890122096)

# Optional - Set the optional parameter cache to true to activate the modular
# cache functionality. Additionally, a specific file name can be passed as a
# string, must be an hdf5 file e.g. 'custom_mlmc_cache.hdf5'.
cache = True

costs, variances = \
    mlmc_simulator.compute_costs_and_variances(initial_sample_size,
                                               cache=cache)

# Calculate optimal sample size for each level from cost/variance/error:
sample_sizes = mlmc_simulator.compute_optimal_sample_sizes(costs, variances,
                                                           epsilon)

# Optional - Pass the cace_files variable to the method
# store_model_inputs_to_run_for_each_level(), this will modify the cache file
# used in Step 3:
mlmc_simulator.store_model_inputs_to_run_for_each_level(sample_sizes,
                                                        cache=cache)
