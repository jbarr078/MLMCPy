import h5py
import numpy as np

class HDF5Storage(object):
    def __init__(self, filename):
        if isinstance(filename, str):
            h5_filename = filename
        else:
            h5_filename = 'mlmc_cache.hdf5'

        self._h5 = h5py.File(h5_filename, mode='r+')

    def close(self):
        self._h5.close()
    
    @staticmethod
    def write_cache_to_hdf5(file_name, inputs, outputs):
        if isinstance(file_name, str):
            h5 = h5py.File(file_name, 'w')
        else:
            h5 = h5py.File('mlmc_cache.hdf5', 'w')

        group1 = h5.create_group('cache_inputs')
        group2 = h5.create_group('cache_outputs')

        for i in range(len(inputs)):
            group1.create_dataset('level%s' % i, inputs[i].shape,
                                  data=inputs[i])
            group2.create_dataset('level%s' % i, outputs[i].shape,
                                  data=outputs[i])
        
        h5.close()

    def compare_inputs(self, sample_sizes, inputs):
        cache_inputs = []

        for key in self._h5['cache_inputs'].keys():
            cache_inputs.append(self._h5['cache_inputs'][key][()])
        
        inputs = HDF5Storage._get_input_list(inputs, sample_sizes)
        indices = []
        cache_sample_sizes = []

        for i in range(len(sample_sizes)):
            indices.append(np.where(inputs[i].flatten() == cache_inputs[i].reshape(-1, 1)))
            cache_sample_sizes.append(len(indices[i][1]))

        updated_inputs = HDF5Storage._delete_inputs(inputs, indices)

        return updated_inputs, cache_sample_sizes, indices

    def remove_unused_outputs_hdf5(self, indices_to_save):
        new_group = self._h5.create_group('updated_outputs')
        cache = self._h5['cache_outputs']

        for i, key in enumerate(cache.keys()):
            # Want indices_to_save[i][0], because that is the cache - right?
            outputs = np.take(cache[key][()], indices_to_save[i][0])
            new_group.create_dataset('level%s' % i, shape=outputs.shape,
                                    data=outputs)

    @staticmethod
    def merge_cache_output(outputs, level, cache):
        """
        Takes in outputs and merges them with the cached outputs generated by
        compute_costs_and_variances().
        """
        if isinstance(cache, str):
            cache_file = cache
        else:
            cache_file = 'mlmc_cache.hdf5'
    
        h5 = h5py.File(cache_file, 'r')

        # with np.warnings.catch_warnings():
        #     np.warnings.simplefilter('ignore')
        cache_outputs = h5['updated_outputs']['level%s' % level][()]

        merged_outputs = np.concatenate((cache_outputs, outputs))

        return merged_outputs

    @staticmethod    
    def _get_input_list(inputs, sample_sizes):
        inputs_list = []
        sample_index = 0
        for value in sample_sizes:
            inputs_list.append(inputs[sample_index:value+sample_index])
            sample_index += value

        return inputs_list

    @staticmethod
    def _delete_inputs(inputs, indices):
        updated_inputs = []

        for i in range(len(inputs)):
            updated_inputs.append(np.delete(inputs[i], indices[i][1]))

        return np.hstack(updated_inputs)