import numpy as np

from spring_mass_model import SpringMassModel

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness to estimate the expected value of the maximum
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.

Demonstrates the modular ("advanced") usage of MLMCPy where a user splits the
analysis into 3 steps/scripts. This is script #2 for running the user-defined
model on each level for the inputs prescribed by MLMCPy.

'''

# Step 5 - Run the model on each level using the input data generated by
# store_model_inputs_to_run_for_each_level()

# Initialize models on each level
model_level0 = SpringMassModel(mass=1.5, time_step=1.0, cost=0.00034791)
model_level1 = SpringMassModel(mass=1.5, time_step=0.1, cost=0.00073748)
model_level2 = SpringMassModel(mass=1.5, time_step=0.01, cost=0.00086135)

#Generate outputs for model on level 0:
samples_level0 = np.genfromtxt("level0_inputs.txt")
outputs_level0 = []

for inputsample in samples_level0:
    outputs_level0.append(model_level0.evaluate([inputsample]))

np.savetxt("level0_outputs.txt", np.array(outputs_level0))

#Generate outputs for model on level 1:
samples_level1 = np.genfromtxt("level1_inputs.txt")
outputs_level1 = []

for inputsample in samples_level1:
    outputs_level1.append(model_level1.evaluate([inputsample]))

np.savetxt("level1_outputs.txt", np.array(outputs_level1))

#Generate outputs for model on level 2:
samples_level2 = np.genfromtxt("level2_inputs.txt")
outputs_level2 = []

for inputsample in samples_level2:
    outputs_level2.append(model_level2.evaluate([inputsample]))

np.savetxt("level2_outputs.txt", np.array(outputs_level2))
