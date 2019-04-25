import os
import pytest
import sys
import h5py

import numpy as np

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.input import RandomInput, InputFromData
from MLMCPy.model import ModelFromData
from MLMCPy.hdf5 import HDF5Storage
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


def test_modular_costs_and_variances_cache(dummy_arange_simulator):
    sim = dummy_arange_simulator
    user_sample_size = 10
    cache = True

    sim.compute_costs_and_variances(user_sample_size, cache)
    
    file_name = 'mlmc_cache.hdf5'
    h5 = h5py.File(file_name, 'r')

    assert np.array_equal(h5['cache_inputs']['level0'][()].flatten(),
                          np.arange(10))

    os.remove(file_name)


def test_modular_compute_optimal_sample_sizes_models(dummy_arange_simulator):
    """
    Tests optimal sample sizes computed by simulator's modular
    compute_optimal_sample_sizes() against expected values based on a
    beta distribution.
    """
    sim = dummy_arange_simulator

    np.random.seed(1)

    costs = np.array([1, 10, 100])
    variances = np.array([[150], [120], [100]])
    epsilon = 1.0

    optimal_sample_sizes = sim.compute_optimal_sample_sizes(costs,
                                                            variances,
                                                            epsilon)

    true_optimal_sizes = np.array([1799, 508, 146])

    assert np.all(np.array_equal(true_optimal_sizes, optimal_sample_sizes))


def test_compute_output_sample_sizes():
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


def test_compute_differences_per_level_array_return_type():
    """
    Ensures that _compute_differences_per_level() is returning the correct
    2D numpy array.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])}

    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    assert test_model_outputs[0][:3].shape == (3,)
    assert test_model_outputs[1][:2].shape == (2,)
    assert test_model_outputs[2][:1].shape == (1,)


def test_compute_differences_per_level_simple_1D():
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1])
               }
    
    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:])


def test_compute_differences_per_level_simple_2D():
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2]),
               'level1':np.array([3])
               }
    
    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:] - outputs['level0'][1:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:1])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])


def test_compute_differences_per_level_simple_3D():
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])}
    
    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:2] - outputs['level0'][3:]
    subtracted_level2 = outputs['level2'] - outputs['level1'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:3])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])


def test_compute_differences_per_level_simple_4D():
    """
    Ensures that _compute_differences_per_level() is returning correct values
    using simple arrays.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7]),
               'level1':np.array([8,9,10,11,12]),
               'level2':np.array([13,14,15]),
               'level3':np.array([16])}

    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:3] - outputs['level0'][4:]
    subtracted_level2 = outputs['level2'][:2] - outputs['level1'][3:]
    subtracted_level3 = outputs['level3'] - outputs['level2'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:4])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])


def test_compute_differences_per_level_simple_5D():
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
    
    test_model_outputs = \
        MLMCSimulator._compute_differences_per_level(outputs)

    subtracted_level1 = outputs['level1'][:4] - outputs['level0'][5:]
    subtracted_level2 = outputs['level2'][:3] - outputs['level1'][4:]
    subtracted_level3 = outputs['level3'][:2] - outputs['level2'][3:]
    subtracted_level4 = outputs['level4'] - outputs['level3'][2:]

    assert np.array_equal(test_model_outputs[0], outputs['level0'][:5])
    assert np.array_equal(subtracted_level1, test_model_outputs[1])
    assert np.array_equal(subtracted_level2, test_model_outputs[2])
    assert np.array_equal(subtracted_level3, test_model_outputs[3])
    assert np.array_equal(subtracted_level4, test_model_outputs[4])


def test_compute_differences_write_to_file():
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])}
    
    _ = \
        MLMCSimulator._compute_differences_per_level(outputs, True)

    true_level0 = outputs['level0'][:3].reshape(1, -1)
    true_level1 = outputs['level1'][:2] - outputs['level0'][3:].reshape(1, -1)
    true_level2 = outputs['level2'] - outputs['level1'][2:].reshape(1, -1)

    test_output_diffs1 = np.genfromtxt('level0_output_diffs.txt').reshape(1, -1)
    test_output_diffs2 = np.genfromtxt('level1_output_diffs.txt').reshape(1, -1)
    test_output_diffs3 = np.genfromtxt('level2_output_diffs.txt').reshape(1, -1)

    assert np.array_equal(test_output_diffs1, true_level0)
    assert np.array_equal(test_output_diffs2, true_level1)
    assert np.array_equal(test_output_diffs3, true_level2)

    for i in range(3):
        os.remove('level%s_output_diffs.txt' % i)


def test_compute_differences_write_to_custom_file(temp_files):
    """
    Ensures that _compute_differences_per_level() is properly writing the data
    to file.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])}
    
    _ = \
        MLMCSimulator._compute_differences_per_level(outputs, temp_files)

    true_level0 = outputs['level0'][:3].reshape(1, -1)
    true_level1 = outputs['level1'][:2] - outputs['level0'][3:].reshape(1, -1)
    true_level2 = outputs['level2'] - outputs['level1'][2:].reshape(1, -1)

    test_output_diffs1 = np.genfromtxt(temp_files[0]).reshape(1, -1)
    test_output_diffs2 = np.genfromtxt(temp_files[1]).reshape(1, -1)
    test_output_diffs3 = np.genfromtxt(temp_files[2]).reshape(1, -1)

    assert np.array_equal(test_output_diffs1, true_level0)
    assert np.array_equal(test_output_diffs2, true_level1)
    assert np.array_equal(test_output_diffs3, true_level2)


def test_modular_compute_estimators_simple_1D():
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3])
            }    

    estimates, variances = \
        MLMCSimulator.compute_estimators(outputs)

    assert np.isclose(estimates, 2)
    assert np.isclose(variances, 0.222222222222)


def test_modular_compute_estimators_simple_2D():
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3]),
               'level1':np.array([4])
            }    

    estimates, variances = \
        MLMCSimulator.compute_estimators(outputs)

    assert np.isclose(estimates, 2.5)
    assert np.isclose(variances, 0.125)


def test_modular_compute_estimators_simple_3D():
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5]),
               'level1':np.array([6,7,8]),
               'level2':np.array([9])
            }    

    estimates, variances = \
        MLMCSimulator.compute_estimators(outputs)

    assert np.isclose(estimates, 5.0)
    assert np.isclose(variances, 0.222222222222)


def test_modular_compute_estimators_simple_4D():
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7]),
               'level1':np.array([8,9,10,11,12]),
               'level2':np.array([13,14,15]),
               'level3':np.array([16])
            }    

    estimates, variances = \
        MLMCSimulator.compute_estimators(outputs)

    assert np.isclose(estimates, 8.5)
    assert np.isclose(variances, 0.3125)


def test_modular_compute_estimators_simple_5D():
    """
    Ensures that compute_estimators() is returning accurate values.
    """
    outputs = {'level0':np.array([1,2,3,4,5,6,7,8,9]),
               'level1':np.array([10,11,12,13,14,15,16]),
               'level2':np.array([17,18,19,20,21]),
               'level3':np.array([22,23,24]),
               'level4':np.array([25])
            }    

    estimates, variances = \
        MLMCSimulator.compute_estimators(outputs)

    assert np.isclose(estimates, 13.0)
    assert np.isclose(variances, 0.4)


def test_modular_compute_estimators_return_type(dummy_arange_simulator):
    """
    Ensures that compute_estimators() is returning a np.ndarray.
    """
    sim = dummy_arange_simulator
    sample_sizes = [3, 2, 1]
    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    estimates, variances = \
        sim.compute_estimators(inputs)

    assert isinstance(variances, np.ndarray)
    assert isinstance(estimates, np.ndarray)


def test_modular_compute_estimators_exception():
    """
    Ensures the outputs parameter is of type np.ndarray.
    """                                          
    test_dict = {'level0': 10, 'level1': 100.5}
    with pytest.raises(TypeError):
        MLMCSimulator.compute_estimators([3, 2, 1])
    
    with pytest.raises(TypeError):
        MLMCSimulator.compute_estimators(test_dict)


def test_get_model_inputs_for_each_level_return_type(dummy_arange_simulator):  
    """
    Ensures that the return type is a dictionary of numpy arrays with the 
    correct sample sizes.
    """  
    sample_sizes = [3, 2, 1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    assert isinstance(inputs, dict)
    assert isinstance(inputs['level0'], np.ndarray)
    assert isinstance(inputs['level1'], np.ndarray)
    assert isinstance(inputs['level2'], np.ndarray)
    assert len(inputs['level0']) == 5
    assert len(inputs['level1']) == 3
    assert len(inputs['level2']) == 1


def test_get_model_inputs_for_each_level_equal_array(dummy_arange_simulator):
    """
    Ensures that each array has the correct values from the array that follows.
    """
    sample_sizes = [10, 6, 3, 1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)

    assert np.array_equal(inputs['level0'][10:], inputs['level1'][:6])
    assert np.array_equal(inputs['level1'][6:], inputs['level2'][:3])
    assert np.array_equal(inputs['level2'][3:], inputs['level3'])


def test_get_model_inputs_one_sample_expected_output(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using simple arrays.
    """
    np.random.seed(1)
    
    sample_sizes = [1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 1
    assert np.isclose(inputs['level0'][0], 0)


def test_get_model_inputs_two_samples_expected_output(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using simple arrays.
    """
    np.random.seed(1)
    
    sample_sizes = [3, 1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 4
    assert len(inputs['level1']) == 1
    assert np.array_equal(inputs['level0'].ravel(), np.arange(4))
    assert np.array_equal(inputs['level1'].ravel(), np.arange(3,4))


def test_get_model_inputs_three_samples_expected_output(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using simple arrays.
    """
    np.random.seed(1)
    
    sample_sizes = [3, 2, 1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 5
    assert len(inputs['level1']) == 3
    assert len(inputs['level2']) == 1
    assert np.array_equal(inputs['level0'].ravel(), np.arange(5))
    assert np.array_equal(inputs['level1'].ravel(), np.arange(3,6))
    assert np.array_equal(inputs['level2'].ravel(), np.arange(5,6))


def test_get_model_inputs_five_samples_expected_output(dummy_arange_simulator):
    """
    Ensures get_model_inputs_to_run_for_each_level() is returning the correct
    values using simple arrays.
    """
    np.random.seed(1)
    
    sample_sizes = [5, 4, 3, 2, 1]
    sim = dummy_arange_simulator

    inputs = sim.get_model_inputs_to_run_for_each_level(sample_sizes)
    assert len(inputs['level0']) == 9
    assert len(inputs['level1']) == 7
    assert len(inputs['level2']) == 5
    assert len(inputs['level3']) == 3
    assert len(inputs['level4']) == 1
    assert np.array_equal(inputs['level0'].ravel(), np.arange(9))
    assert np.array_equal(inputs['level1'].ravel(), np.arange(5,12))
    assert np.array_equal(inputs['level2'].ravel(), np.arange(9,14))
    assert np.array_equal(inputs['level3'].ravel(), np.arange(12,15))
    assert np.array_equal(inputs['level4'].ravel(), np.arange(14,15))


def test_get_model_inputs_cache(dummy_arange_simulator, tmpdir):
    p = tmpdir.mkdir('sub')
    sim = dummy_arange_simulator
    cache_path = str(p.join('test_cache.hdf5'))
    test_inputs = [np.arange(100)]
    test_outputs = [np.arange(125)]
    HDF5Storage.write_cache_to_hdf5(cache_path, test_inputs, test_outputs)
    sample_sizes = np.array([125])
    
    inputs = \
        sim.get_model_inputs_to_run_for_each_level(sample_sizes, cache_path)

    assert np.array_equal(inputs['level0'], np.arange(100, 125))


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

    sim.store_model_inputs_to_run_for_each_level(sample_sizes, True)

    level0 = np.loadtxt('level0_inputs.txt')

    assert np.array_equal(level0.flatten(), np.arange(24))

    os.remove('level0_inputs.txt')


def test_simple_2D_store_model_inputs_for_each_level(dummy_arange_simulator,
                                                     temp_files):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes, temp_files)

    level = []
    for i in range(2):
        level.append(np.loadtxt(temp_files[i]))

    assert np.array_equal(level[0].flatten(), np.arange(8))
    assert np.array_equal(level[1].flatten(), np.arange(5, 8))


def test_simple_3D_store_model_inputs_for_each_level(dummy_arange_simulator,
                                                     temp_files):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes, temp_files)

    level = []
    for i in range(3):
        level.append(np.loadtxt(temp_files[i]))

    assert np.array_equal(level[0].flatten(), np.arange(8))
    assert np.array_equal(level[1].flatten(), np.arange(5, 10))
    assert np.array_equal(level[2].flatten(), np.arange(8, 10))


def test_simple_4D_store_model_inputs_for_each_level(dummy_arange_simulator,
                                                     temp_files):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 3, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes, temp_files)

    level = []
    for i in range(4):
        level.append(np.loadtxt(temp_files[i]))

    assert np.array_equal(level[0].flatten(), np.arange(8))
    assert np.array_equal(level[1].flatten(), np.arange(5, 11))
    assert np.array_equal(level[2].flatten(), np.arange(8, 13))
    assert np.array_equal(level[3].flatten(), np.arange(11, 13))


def test_simple_5D_store_model_inputs_for_each_level(dummy_arange_simulator,
                                                     temp_files):
    """
    Ensures the store_model_inputs_to_run_for_each_level method is properly
    allocating values in the array by using a simple arange function.  
    """
    sample_sizes = [5, 4, 3, 3, 2]

    sim = dummy_arange_simulator

    sim.store_model_inputs_to_run_for_each_level(sample_sizes, temp_files)

    level = []
    for i in range(5):
        level.append(np.loadtxt(temp_files[i]))

    assert np.array_equal(level[0].flatten(), np.arange(9))
    assert np.array_equal(level[1].flatten(), np.arange(5, 12))
    assert np.array_equal(level[2].flatten(), np.arange(9, 15))
    assert np.array_equal(level[3].flatten(), np.arange(12, 17))
    assert np.array_equal(level[4].flatten(), np.arange(15, 17))


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


# def test_load_model_outputs_cache(tmpdir):
#     cache_inputs = [np.arange(10)]
#     cache_outputs = [np.arange(100, 150)]

#     HDF5Storage.write_cache_to_hdf5(True, cache_inputs, cache_outputs)
#     data = [np.arange(100)]

#     for i in range(len(data)):
#         np.savetxt('level%s_outputs.txt' % i, data[i])

#     outputs_dict = \
#         MLMCSimulator.load_model_outputs_for_each_level(cache=True)

#     assert outputs_dict == (1)

#     for i in range(len(data)):
#         os.remove('level%s_outputs.txt' % i)
#     os.remove('mlmc_cache.hdf5')


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


def test_load_model_outputs_for_each_level_custom_fname(spring_mlmc_simulator,
                                                        temp_files):
    """
    Ensures that load_model_outputs_for_each_level() is correctly loading data 
    from files with custom file names.
    """
    sim = spring_mlmc_simulator
    sample_sizes = [3, 2, 1]
    fnames = temp_files[:3]
    sim.store_model_inputs_to_run_for_each_level(sample_sizes, fnames)

    model_outputs = sim.load_model_outputs_for_each_level(fnames)

    assert isinstance(model_outputs, dict)
    assert isinstance(model_outputs['level0'], np.ndarray)
    assert isinstance(model_outputs['level1'], np.ndarray)
    assert isinstance(model_outputs['level2'], np.ndarray)


def test_load_model_outputs_for_each_level_exception():
    """
    Ensures that load_model_outputs_for_each_level() throws its exceptions.
    """
    with pytest.raises(TypeError):
        MLMCSimulator.load_model_outputs_for_each_level('Not an Integer.')


def test_write_data_to_files(temp_files):
    """
    Ensures that write_data_to_file() is properly writing to file.    
    """
    output_diffs = [np.arange(5), np.arange(10), np.arange(20)]

    MLMCSimulator._write_to_file('suffix.txt', True,
                                 len(output_diffs), output_diffs)

    output_diffs1 = np.genfromtxt('level0suffix.txt')
    output_diffs2 = np.genfromtxt('level1suffix.txt')
    output_diffs3 = np.genfromtxt('level2suffix.txt')
    
    assert np.array_equal(output_diffs1, np.arange(5))
    assert np.array_equal(output_diffs2, np.arange(10))
    assert np.array_equal(output_diffs3, np.arange(20))

    for i in range(3):
        os.remove('level%ssuffix.txt' % i)

