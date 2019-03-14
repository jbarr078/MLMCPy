import os
import pytest
import sys
import numpy as np

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.input import RandomInput, InputFromData
from MLMCPy.model import ModelFromData
from tests.testing_scripts import SpringMassModel

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"


def test_modular_costs_and_initial_variances_from_model(spring_mlmc_simulator):
    """
    Tests costs and variances computed by simulator's modular
    compute_costs_and_variances() against expected values based on a
    beta distribution.
    """
    sim = spring_mlmc_simulator

    np.random.seed(1)

    initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(initial_sample_sizes)

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    true_costs = np.array([1., 11., 110.])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))
    

def test_modular_costs_and_initial_variances_from_data(data_input, 
                                                       models_from_data):
    """
    Tests modular costs and variances computed by simulator's
    compute_costs_and_variances() against expected values based on data loaded
    from files.
    """
    np.random.seed(1)
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(sample_sizes)

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_write_cache_to_file():
    """
    Ensures the _write_cache_to_file() method is correctly writing the array to
    text.    
    """
    inputs = np.arange(0, 5)
    outputs = np.arange(5, 10)

    MLMCSimulator._write_cache_to_file(True, outputs, inputs)
    cached_inputs = np.genfromtxt('cache_inputs.txt')
    cached_outputs = np.genfromtxt('cache_outputs.txt')

    assert np.array_equal(cached_inputs, inputs)
    assert np.array_equal(cached_outputs, outputs)

    os.remove('cache_outputs.txt')
    os.remove('cache_inputs.txt')


def test_write_cache_to_custom_file():
    """
    Ensures the _write_cache_to_file() method is correctly writing to custom
    file names.   
    """
    inputs = np.arange(0, 5)
    outputs = np.arange(5, 10)
    cache = ['custom_outputs.txt', 'custom_inputs.txt']
    MLMCSimulator._write_cache_to_file(cache, outputs, inputs)
    cached_outputs = np.genfromtxt(cache[0])
    cached_inputs = np.genfromtxt(cache[1])

    assert np.array_equal(cached_inputs, inputs)
    assert np.array_equal(cached_outputs, outputs)

    for i in cache:
        os.remove(i)


def test_compare_inputs_to_cache():
    """
    Ensures the _compare_inputs_to_cache() method is properly finding similar
    values and removing them.
    """
    fname = 'cache_inputs.txt'
    inputs = np.arange(1, 20)
    np.savetxt(fname, np.arange(1,11).reshape(-1, 2))

    new_inputs, cache_sample_sizes = \
        MLMCSimulator._compare_inputs_to_cache(inputs, fname)

    assert np.array_equal(new_inputs, np.arange(11, 20))

    os.remove('cache_inputs.txt')


def test_modular_compute_costs_and_variances_cache_file(dummy_arange_simulator):
    """
    Ensuress the compute_costs_and_variances() method is correctly calling the
    _write_cache_to_file() method.
    """
    sim = dummy_arange_simulator
    sim.compute_costs_and_variances(10, True)
    inputs = np.arange(10)

    cached_inputs = np.genfromtxt('cache_inputs.txt')

    assert np.array_equal(cached_inputs[0], inputs)

    os.remove('cache_inputs.txt')
    os.remove('cache_outputs.txt')
    

def test_modular_compute_optimal_sample_sizes_models(spring_mlmc_simulator):
    """
    Tests optimal sample sizes computed by simulator's modular
    compute_optimal_sample_sizes() against expected values based on a
    beta distribution.
    """
    sim = spring_mlmc_simulator

    np.random.seed(1)

    initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(initial_sample_sizes)
    epsilon = np.sqrt(0.00170890122096)

    optimal_sample_sizes = sim.compute_optimal_sample_sizes(costs,
                                                            variances,
                                                            epsilon)

    true_optimal_sizes = np.array([6506, 200, 0])

    assert np.all(np.array_equal(true_optimal_sizes, optimal_sample_sizes))


def test_compute_output_sample_sizes(spring_mlmc_simulator):
    """
    Ensures that _compute_output_sample_sizes() is returning the appropriate
    sample sizes per level.
    """
    outputs = {'level0':np.arange(1600),
               'level1':np.arange(900),
               'level2':np.arange(400), 
               'level3':np.arange(150),
               'level4':np.arange(60),
               'level5':np.arange(10)
            }
    
    sizes = MLMCSimulator._compute_output_sample_sizes(outputs)

    assert sizes[0] == 1000
    assert sizes[1] == 600
    assert sizes[2] == 300
    assert sizes[3] == 100
    assert sizes[4] == 50
    assert sizes[5] == 10


def test_compute_differences_per_level_array_return_type(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning the correct
    2D numpy array.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    test_model_outputs = \
        sim._compute_differences_per_level(inputs)

    assert test_model_outputs[0][:3].shape == (3,1)
    assert test_model_outputs[1][:2].shape == (2,1)
    assert test_model_outputs[2][:1].shape == (1,1)


def test_compute_differences_per_level_simple_1D(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1])
               }
    
    sim = spring_mlmc_simulator
    test_model_outputs = sim._compute_differences_per_level(outputs)

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:])


def test_compute_differences_per_level_simple_2D(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2]),
               'level1':np.array([3])
               }
    
    sim = spring_mlmc_simulator
    test_model_outputs = sim._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:] - outputs['level0'][1:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:1])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])


def test_compute_differences_per_level_simple_3D(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])}
    
    sim = spring_mlmc_simulator
    test_model_outputs = sim._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:2] - outputs['level0'][3:]
    subtracted_level2 = outputs['level2'] - outputs['level1'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:3])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])


def test_compute_differences_per_level_simple_4D(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7]),
               'level1':np.array([8,9,10,11,12]),
               'level2':np.array([13,14,15]),
               'level3':np.array([16])}
    
    sim = spring_mlmc_simulator

    test_model_outputs = \
        sim._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:3] - outputs['level0'][4:]
    subtracted_level2 = outputs['level2'][:2] - outputs['level1'][3:]
    subtracted_level3 = outputs['level3'] - outputs['level2'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:4])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])


def test_compute_differences_per_level_simple_5D(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7,8,9]),
               'level1':np.array([10,11,12,13,14,15,16]),
               'level2':np.array([17,18,19,20,21]),
               'level3':np.array([22,23,24]),
               'level4':np.array([25])
               }
    
    sim = spring_mlmc_simulator

    test_model_outputs = \
        sim._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:4] - outputs['level0'][5:]
    subtracted_level2 = outputs['level2'][:3] - outputs['level1'][4:]
    subtracted_level3 = outputs['level3'][:2] - outputs['level2'][3:]
    subtracted_level4 = outputs['level4'] - outputs['level3'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:5])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])
    assert np.array_equal(subtracted_level4, test_model_outputs[4])


def test_compute_differences_per_level_3D_output(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is subtracting and returning 
    the correct values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    test_model_outputs = \
        sim._compute_differences_per_level(inputs)
    subtracted_level1 = inputs['level1'][:2] - inputs['level0'][3:]
    subtracted_level2 = inputs['level2'] - inputs['level1'][2:]

    assert np.array_equal(test_model_outputs[0], inputs['level0'][:3])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])


def test_compute_differences_per_level_4D_output(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is subtracting and returning 
    the correct values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [4, 3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    test_model_outputs = \
        sim._compute_differences_per_level(inputs)
    subtracted_level1 = inputs['level1'][:3] - inputs['level0'][4:]
    subtracted_level2 = inputs['level2'][:2] - inputs['level1'][3:]
    subtracted_level3 = inputs['level3'] - inputs['level2'][2:]

    assert np.array_equal(test_model_outputs[0], inputs['level0'][:4])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])


def test_compute_differences_per_level_5D_output(spring_mlmc_simulator):
    """
    Ensures that _compute_differences_per_level() is subtracting and returning 
    the correct values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [5, 4, 3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    test_model_outputs = \
        sim._compute_differences_per_level(inputs)
    subtracted_level1 = inputs['level1'][:4] - inputs['level0'][5:]
    subtracted_level2 = inputs['level2'][:3] - inputs['level1'][4:]
    subtracted_level3 = inputs['level3'][:2] - inputs['level2'][3:]
    subtracted_level4 = inputs['level4'] - inputs['level3'][2:]

    assert np.array_equal(test_model_outputs[0], inputs['level0'][:5])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])
    assert np.array_equal(subtracted_level4, test_model_outputs[4])


def test_modular_compute_estimators_simple_1D(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3])
            }

    sim = spring_mlmc_simulator
    

    estimates, variances = \
        sim.compute_estimators(outputs)

    assert np.isclose(estimates, 2)
    assert np.isclose(variances, 0.222222222222)


def test_modular_compute_estimators_simple_2D(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3]),
               'level1':np.array([4])
            }

    sim = spring_mlmc_simulator
    

    estimates, variances = \
        sim.compute_estimators(outputs)

    assert np.isclose(estimates, 2.5)
    assert np.isclose(variances, 0.125)


def test_modular_compute_estimators_simple_3D(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])
            }

    sim = spring_mlmc_simulator
    

    estimates, variances = \
        sim.compute_estimators(outputs)

    assert np.isclose(estimates, 5.0)
    assert np.isclose(variances, 0.222222222222)


def test_modular_compute_estimators_simple_4D(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7]),
               'level1':np.array([8,9,10,11,12]),
               'level2':np.array([13,14,15]),
               'level3':np.array([16])
            }

    sim = spring_mlmc_simulator
    

    estimates, variances = \
        sim.compute_estimators(outputs)

    assert np.isclose(estimates, 8.5)
    assert np.isclose(variances, 0.3125)


def test_modular_compute_estimators_simple_5D(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7,8,9]),
               'level1':np.array([10,11,12,13,14,15,16]),
               'level2':np.array([17,18,19,20,21]),
               'level3':np.array([22,23,24]),
               'level4':np.array([25])
            }

    sim = spring_mlmc_simulator
    

    estimates, variances = \
        sim.compute_estimators(outputs)

    assert np.isclose(estimates, 13.0)
    assert np.isclose(variances, 0.4)


def test_modular_compute_estimators_1D_expected_output(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert np.isclose(estimates, 3.17248042)
    assert np.isclose(variances, 0.01724233)


def test_modular_compute_estimators_2D_expected_output(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert np.isclose(estimates, 3.17248042)
    assert np.isclose(variances, 0.01724233)


def test_modular_compute_estimators_3D_expected_output(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert np.isclose(estimates, 3.17248042)
    assert np.isclose(variances, 0.01724233)


def test_modular_compute_estimators_4D_expected_output(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [4, 3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert np.isclose(estimates, 3.18597516)
    assert np.isclose(variances, 0.00983539)


def test_modular_compute_estimators_5D_expected_output(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [5, 4, 3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert np.isclose(estimates, 3.12703401)
    assert np.isclose(variances, 0.0090739)


def test_modular_compute_estimators_return_type(spring_mlmc_simulator):
    """
    Ensures that compute_estimators() is returning a np.ndarray.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert isinstance(variances, np.ndarray)
    assert isinstance(estimates, np.ndarray)


def test_modular_compute_estimators_exception(spring_mlmc_simulator):
    """
    Ensures the outputs parameter is of type np.ndarray.
    """                                          
    sim = spring_mlmc_simulator
    test_dict = {'level0': 10, 'level1': 100.5}
    with pytest.raises(TypeError):
        sim.compute_estimators([3, 2, 1])
    
    with pytest.raises(TypeError):
        sim.compute_estimators(test_dict)


def test_get_model_inputs_for_each_level_return_type(spring_mlmc_simulator):  
    """
    Ensures that the return type is a dictionary of numpy arrays with the 
    correct sample sizes.
    """  
    sample_sizes = [3, 2, 1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    assert isinstance(inputs, dict)
    assert isinstance(inputs['level0'], np.ndarray)
    assert isinstance(inputs['level1'], np.ndarray)
    assert isinstance(inputs['level2'], np.ndarray)
    assert len(inputs['level0']) == 5
    assert len(inputs['level1']) == 3
    assert len(inputs['level2']) == 1


def test_get_model_inputs_for_each_level_equal_array(spring_mlmc_simulator):
    """
    Ensures that each array has the correct values from the array that follows.
    """
    sample_sizes = [10, 6, 3, 1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    assert np.array_equal(inputs['level0'][10:], inputs['level1'][:6])
    assert np.array_equal(inputs['level1'][6:], inputs['level2'][:3])
    assert np.array_equal(inputs['level2'][3:], inputs['level3'])


def test_get_model_inputs_one_sample_expected_output(spring_mlmc_simulator):
    """
    Ensures that the get_model_inputs_to_run_for_each_level() can proceed with
    one sample provided.
    """
    np.random.seed(1)
    
    sample_sizes = [1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 1
    assert np.isclose(inputs['level0'][0], 3.15880039)


def test_get_model_inputs_two_samples_expected_output(spring_mlmc_simulator):
    """
    Ensures that the get_model_inputs_to_run_for_each_level() can proceed with
    two samples provided.
    """
    np.random.seed(1)
    
    sample_sizes = [3, 1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 4
    assert len(inputs['level1']) == 1
    assert np.isclose(inputs['level0'][0], 3.15880039)
    assert np.isclose(inputs['level1'][0], 3.42888628)


def test_get_model_inputs_three_samples_expected_output(spring_mlmc_simulator):
    """
    Ensures that the get_model_inputs_to_run_for_each_level() can proceed with
    three samples provided.
    """
    np.random.seed(1)
    
    sample_sizes = [3, 2, 1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 5
    assert len(inputs['level1']) == 3
    assert len(inputs['level2']) == 1
    assert np.isclose(inputs['level0'][0], 3.15880039)
    assert np.isclose(inputs['level1'][0], 3.42888628)
    assert np.isclose(inputs['level2'][0], 2.89126945)


def test_get_model_inputs_five_samples_expected_output(spring_mlmc_simulator):
    """
    Ensures that the get_model_inputs_to_run_for_each_level() can proceed with
    five samples provided.
    """
    np.random.seed(1)
    
    sample_sizes = [5, 4, 3, 2, 1]
    sim = spring_mlmc_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 9
    assert len(inputs['level1']) == 7
    assert len(inputs['level2']) == 5
    assert len(inputs['level3']) == 3
    assert len(inputs['level4']) == 1
    assert np.isclose(inputs['level0'][0], 3.15880039)
    assert np.isclose(inputs['level1'][0], 2.89126945)
    assert np.isclose(inputs['level2'][0], 2.70076945)
    assert np.isclose(inputs['level3'][0], 2.84713918)
    assert np.isclose(inputs['level4'][0], 2.79495595)


def test_simple_get_model_inputs_1D(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using a simple arrays.
    """
    sample_sizes = [24]
    
    sim = dummy_arange_simulator
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    
    assert len(inputs.keys()) == 1
    assert np.array_equal(inputs["level0"].flatten(), np.arange(24))


def test_simple_get_model_inputs_4D(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using a simple arrays.
    """
    sample_sizes = [5, 3, 3, 2]
    
    sim = dummy_arange_simulator
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
 
    assert len(inputs.keys()) == 4   
    assert np.array_equal(inputs["level0"].flatten(), np.arange(8))
    assert np.array_equal(inputs["level1"].flatten(), np.arange(5,11))
    assert np.array_equal(inputs["level2"].flatten(), np.arange(8,13))
    assert np.array_equal(inputs["level3"].flatten(), np.arange(11,13))


def test_get_model_inputs_param_exceptions(spring_mlmc_simulator):
    """
    Ensures that exceptions are raised by 
    get_model_inputs_to_run_for_each_level().
    """
    sim = spring_mlmc_simulator

    with pytest.raises(TypeError):
        sim.get_model_inputs_to_run_for_each_level(5)

    with pytest.raises(TypeError):
        sim.get_model_inputs_to_run_for_each_level([5.5])

    with pytest.raises(TypeError):
        sim.get_model_inputs_to_run_for_each_level('Not A List')


def test_simple_1D_store_model_inputs_for_each_level(dummy_arange_simulator):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [24]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(24))

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_simple_2D_store_model_inputs_for_each_level(dummy_arange_simulator):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')
    level1 = np.loadtxt('level1_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(8))
    assert np.array_equal(level1.flatten(), np.arange(5, 8))

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_simple_3D_store_model_inputs_for_each_level(dummy_arange_simulator):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')
    level1 = np.loadtxt('level1_inputs.txt')
    level2 = np.loadtxt('level2_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(8))
    assert np.array_equal(level1.flatten(), np.arange(5, 10))
    assert np.array_equal(level2.flatten(), np.arange(8, 10))

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_simple_4D_store_model_inputs_for_each_level(dummy_arange_simulator):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')
    level1 = np.loadtxt('level1_inputs.txt')
    level2 = np.loadtxt('level2_inputs.txt')
    level3 = np.loadtxt('level3_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(8))
    assert np.array_equal(level1.flatten(), np.arange(5, 11))
    assert np.array_equal(level2.flatten(), np.arange(8, 13))
    assert np.array_equal(level3.flatten(), np.arange(11, 13))

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_simple_5D_store_model_inputs_for_each_level(dummy_arange_simulator):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 4, 3, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')
    level1 = np.loadtxt('level1_inputs.txt')
    level2 = np.loadtxt('level2_inputs.txt')
    level3 = np.loadtxt('level3_inputs.txt')
    level4 = np.loadtxt('level4_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(9))
    assert np.array_equal(level1.flatten(), np.arange(5, 12))
    assert np.array_equal(level2.flatten(), np.arange(9, 15))
    assert np.array_equal(level3.flatten(), np.arange(12, 17))
    assert np.array_equal(level4.flatten(), np.arange(15, 17))

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_store_model_inputs_to_run_for_each_level_return(spring_mlmc_simulator):
    """
    Ensures that store_model_inputs_to_run_for_each_level() is properly storing
    the inputs to text files using default file names and transitioning back to 
    np.ndarray.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3,2,1]

    sim.store_model_inputs_to_run_for_each_level(sample_sizes)

    level0 = np.loadtxt('level0_inputs.txt')
    level1 = np.loadtxt('level1_inputs.txt')
    level2 = np.loadtxt('level2_inputs.txt')


    assert isinstance(level0, np.ndarray)
    assert isinstance(level1, np.ndarray)
    assert isinstance(level2, np.ndarray)

    for i in range(len(sample_sizes)):
        os.remove('level%s_inputs.txt' % i)


def test_store_model_inputs_custom_file_names(spring_mlmc_simulator):
    """
    Ensures that store_model_inputs_to_run_for_each_level() is properly storing
    the inputs to text files using custom file names and transitioning back to 
    np.ndarray.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    fnames = ['level0.txt', 'level1.txt', 'level2.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)
    
    level0 = np.loadtxt('level0.txt')
    level1 = np.loadtxt('level1.txt')
    level2 = np.loadtxt('level2.txt')

    assert isinstance(level0, np.ndarray)
    assert isinstance(level1, np.ndarray)
    assert isinstance(level2, np.ndarray)

    for i in range(len(sample_sizes)):
        os.remove('level%s.txt' % i)


def test_store_model_inputs_to_run_for_each_level_except(spring_mlmc_simulator):
    """
    Ensures that store_model_inputs_to_run_for_each_level() is raising
    exceptions.
    """
    sim = spring_mlmc_simulator

    with pytest.raises(TypeError):
        sim.store_model_inputs_to_run_for_each_level(5)
    
    with pytest.raises(TypeError):
        sim.store_model_inputs_to_run_for_each_level([10.5])
    
    with pytest.raises(TypeError):
        sim.store_model_inputs_to_run_for_each_level('Not sample sizes')

    with pytest.raises(TypeError):
        sim.store_model_inputs_to_run_for_each_level([3,2,1], 1)
    
    with pytest.raises(TypeError):
        sim.store_model_inputs_to_run_for_each_level([3,2,1], ['string', 1])


def test_load_model_outputs_for_each_level_one_output(spring_mlmc_simulator):
    """
    Ensures that load model is correctly loading data from files.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3]
    fnames = ['level0_outputs.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)
    model_outputs = sim.load_model_outputs_for_each_level()

    assert np.isclose(model_outputs['level0'][0], 2.87610342)

    for i in range(len(sample_sizes)):
        os.remove('level%s_outputs.txt' % i)


def test_load_model_outputs_for_each_level_two_outputs(spring_mlmc_simulator):
    """
    Ensures that load model is correctly loading data from files.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2]
    fnames = ['level0_outputs.txt', 'level1_outputs.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)
    model_outputs = sim.load_model_outputs_for_each_level()

    assert np.isclose(model_outputs['level0'][0], 2.87610342)
    assert np.isclose(model_outputs['level1'][0], 3.22645934)

    for i in range(len(sample_sizes)):
        os.remove('level%s_outputs.txt' % i)


def test_load_model_outputs_for_each_level_three_outputs(spring_mlmc_simulator):
    """
    Ensures that load_model_outputs_for_each_level() is correctly loading data 
    from files.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    fnames = ['level0_outputs.txt', 'level1_outputs.txt', 'level2_outputs.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)
    model_outputs = sim.load_model_outputs_for_each_level()

    assert np.isclose(model_outputs['level0'][0], 2.87610342)
    assert np.isclose(model_outputs['level1'][0], 3.22645934)
    assert np.isclose(model_outputs['level2'], 1.63840664)

    for i in range(len(sample_sizes)):
        os.remove('level%s_outputs.txt' % i)


def test_load_model_outputs_for_each_level_return_type(spring_mlmc_simulator):
    """
    Ensures that load_model_outputs_for_each_level() is correctly loading data 
    from files.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    fnames = ['level0_outputs.txt', 'level1_outputs.txt', 'level2_outputs.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)
    model_outputs = sim.load_model_outputs_for_each_level()

    assert isinstance(model_outputs, dict)
    assert isinstance(model_outputs['level0'], np.ndarray)
    assert isinstance(model_outputs['level1'], np.ndarray)
    assert isinstance(model_outputs['level2'], np.ndarray)

    for i in range(len(sample_sizes)):
        os.remove('level%s_outputs.txt' % i)


def test_load_model_outputs_for_each_level_custom_fname(spring_mlmc_simulator):
    """
    Ensures that load_model_outputs_for_each_level() is correctly loading data 
    from files with custom file names.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    fnames = ['level0.txt', 'level1.txt', 'level2.txt']
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)

    model_outputs = sim.load_model_outputs_for_each_level(fnames)

    assert isinstance(model_outputs, dict)
    assert isinstance(model_outputs['level0'], np.ndarray)
    assert isinstance(model_outputs['level1'], np.ndarray)
    assert isinstance(model_outputs['level2'], np.ndarray)

    for i in range(len(sample_sizes)):
        os.remove('level%s.txt' % i)


def test_load_model_outputs_merge():
    fname = 'cache_inputs.txt'
    oname = 'level0_outputs.txt'

    np.savetxt(oname, np.arange(11,21))
    np.savetxt(fname, np.arange(11))

    outputs = \
        MLMCSimulator.load_model_outputs_for_each_level(filenames=None,
                                                        output_cache_file=fname)
    
    merged_outputs = np.sort(outputs['level0'])

    assert np.array_equal(merged_outputs, np.arange(21))
    assert np.mean(outputs['level0']) == 10

    os.remove(fname)
    os.remove(oname)



def test_load_model_outputs_for_each_level_exception():
    """
    Ensures that load_model_outputs_for_each_level() throws its exceptions.
    """
    with pytest.raises(TypeError):
        MLMCSimulator.load_model_outputs_for_each_level('Not an Integer.')


def test_merge_model_output_with_cache_1D():
    """
    Ensures that merge_model_output_with_cache() is properly merging the user
    evaluated outputs with the cache created by compute_costs_and_variances().
    """
    fname = 'cache_outputs.txt'
    np.savetxt(fname, np.arange(11))
    outputs = np.arange(11, 21)

    merged_outputs = MLMCSimulator._merge_cache_output(outputs, fname)
    merged_outputs = np.sort(merged_outputs)

    assert np.array_equal(merged_outputs, np.arange(21))
    assert np.mean(merged_outputs) == 10

    os.remove(fname)

