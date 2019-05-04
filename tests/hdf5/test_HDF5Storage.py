import os
import pytest
import h5py
import numpy as np

from MLMCPy.utils import HDF5Storage


@pytest.fixture
def cache_file_paths(tmpdir):
    p = tmpdir.mkdir('sub')

    return [str(p.join('test_cache_file.hdf5')), str(p.join('test_merge.hdf5'))]


@pytest.fixture
def dummy_cache(cache_file_paths):
    inputs = [np.arange(100), np.arange(125, 150), np.arange(180, 190)]
    outputs = [np.arange(200), np.arange(225, 250), np.arange(280, 290)]
    HDF5Storage.write_cache_to_hdf5(cache_file_paths[0], inputs, outputs)

    dummy_hdf5storage = HDF5Storage(cache_file_paths[0])

    return dummy_hdf5storage


def test_write_cache_to_hdf5():
    cache_inputs = [np.arange(100),
                    np.arange(200),
                    np.arange(300)]

    cache_outputs = [np.arange(400, 500),
                     np.arange(500, 700),
                     np.arange(700, 1000)]
    file_name = 'mlmc_cache.hdf5'

    HDF5Storage.write_cache_to_hdf5(True, cache_inputs, cache_outputs)

    h5 = h5py.File(file_name, 'r')
    
    assert np.array_equal(h5['cache_inputs']['level0'][()], cache_inputs[0])
    assert np.array_equal(h5['cache_inputs']['level1'][()], cache_inputs[1])
    assert np.array_equal(h5['cache_inputs']['level2'][()], cache_inputs[2])
    assert np.array_equal(h5['cache_outputs']['level0'][()], cache_outputs[0])
    assert np.array_equal(h5['cache_outputs']['level1'][()], cache_outputs[1])
    assert np.array_equal(h5['cache_outputs']['level2'][()], cache_outputs[2])

    os.remove(file_name)


def test_write_cache_to_custom_hdf5(cache_file_paths):
    cache_inputs = [np.arange(100),
                    np.arange(200),
                    np.arange(300)]

    cache_outputs = [np.arange(400, 500),
                     np.arange(500, 700),
                     np.arange(700, 1000)]

    HDF5Storage.write_cache_to_hdf5(cache_file_paths[0], cache_inputs,
                                    cache_outputs)

    h5 = h5py.File(cache_file_paths[0], 'r')
    
    assert np.array_equal(h5['cache_inputs']['level0'][()], cache_inputs[0])
    assert np.array_equal(h5['cache_inputs']['level1'][()], cache_inputs[1])
    assert np.array_equal(h5['cache_inputs']['level2'][()], cache_inputs[2])
    assert np.array_equal(h5['cache_outputs']['level0'][()], cache_outputs[0])
    assert np.array_equal(h5['cache_outputs']['level1'][()], cache_outputs[1])
    assert np.array_equal(h5['cache_outputs']['level2'][()], cache_outputs[2])


def test_compare_inputs(dummy_cache):
    cache = dummy_cache
    test_sample_sizes = [125, 50, 25]
    test_inputs = np.arange(200)

    updated_inputs, cache_sample_sizes, indicies = \
        cache.compare_inputs(test_sample_sizes, test_inputs)

    expected_inputs = np.concatenate((np.arange(100, 125),
                                      np.arange(150, 180),
                                      np.arange(190, 200)))

    assert np.array_equal(updated_inputs, expected_inputs)
    assert cache_sample_sizes == [100, 25, 10]


def test_remove_unused_outputs_hdf5(dummy_cache, cache_file_paths):
    cache = dummy_cache
    test_indicies = [(np.arange(100), np.arange(100, 200)),
                     (np.arange(15), np.arange(25, 50)),
                     (np.arange(5), np.arange(10, 20))]

    cache.remove_unused_outputs_hdf5(test_indicies)

    h5 = h5py.File(cache_file_paths[0], 'r')
    level0 = np.arange(100)
    level1 = np.arange(225, 240)
    level2 = np.arange(280, 285)

    assert np.array_equal(h5['updated_outputs']['level0'][()], level0)
    assert np.array_equal(h5['updated_outputs']['level1'][()], level1)
    assert np.array_equal(h5['updated_outputs']['level2'][()], level2)


def test_merge_cache_output_hdf5():
    file_name = 'mlmc_cache.hdf5'
    h5 = h5py.File(file_name, 'w')
    h5.create_group('updated_outputs')
    h5_data = [np.arange(10), np.arange(20, 30), np.arange(100)]
    for i in range(len(h5_data)):
        h5['updated_outputs'].create_dataset('level%s' % i,
                                             data=h5_data[i],
                                             shape=h5_data[i].shape)
    
    test_outputs = []
    outputs = [np.arange(10, 20), np.arange(30, 60), np.arange(100, 300)]
    for level in range(len(outputs)):
        merged_outputs = \
            HDF5Storage.merge_cache_output(outputs[level], level, file_name)
        test_outputs.append(merged_outputs)

    assert np.array_equal(test_outputs[0], np.arange(20))
    assert np.array_equal(test_outputs[1], np.arange(20, 60))
    assert np.array_equal(test_outputs[2], np.arange(300))
    
    os.remove(file_name)


def test_merge_cache_output_custom_hdf5(cache_file_paths):
    h5 = h5py.File(cache_file_paths[1], 'w')
    h5.create_group('updated_outputs')
    h5_data = [np.arange(10), np.arange(20, 30), np.arange(100)]
    for i in range(len(h5_data)):
        h5['updated_outputs'].create_dataset('level%s' % i,
                                             data=h5_data[i],
                                             shape=h5_data[i].shape)
    
    test_outputs = []
    outputs = [np.arange(10, 20), np.arange(30, 60), np.arange(100, 300)]
    for level in range(len(outputs)):
        merged_outputs = \
            HDF5Storage.merge_cache_output(outputs[level], level,
                                           cache_file_paths[1])
        test_outputs.append(merged_outputs)

    assert np.array_equal(test_outputs[0], np.arange(20))
    assert np.array_equal(test_outputs[1], np.arange(20, 60))
    assert np.array_equal(test_outputs[2], np.arange(300))

