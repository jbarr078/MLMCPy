import timeit

from MLMCPy.mlmc import MLMCSimulator

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness to estimate the expected value of the maximum
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.

Demonstrates the modular ("advanced") usage of MLMCPy where a user splits the
analysis into 3 steps/scripts. This is script #3 for computing MLMC estimators
using previously calculated model outputs by loading them from file
'''

# Optional - Create a list of cache file names and pass to the method
# load_outputs_for_each_level(), this will merge the saved cache output values
# with the outputs generated in Step 2:
cache_files = ['level0_cache.txt', 'level1_cache.txt', 'level2_cache.txt']

model_outputs_per_level = \
    MLMCSimulator.load_model_outputs_for_each_level(cache_file=cache_files)

# Step 6 - Aggregate model outputs to compute estimators:
mlmc_start = timeit.default_timer()

estimates, variances = \
        MLMCSimulator.compute_estimators(model_outputs_per_level)

mlmc_total_cost = timeit.default_timer() - mlmc_start

# Step 7 - Summarize results:

print
print 'MLMC estimate: %s' % estimates
print 'MLMC precision: %s' % variances
print 'MLMC total cost: %s' % mlmc_total_cost
