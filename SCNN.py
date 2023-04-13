import tensorflow as tf
import time as tm
import sys
# import numpy as np
# import cudnn as cd
# from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt
# from tensorflow.python.client import device_lib

class Integrator_layer(tf.keras.layers.Layer):
    def __init__(self, n_steps=100, integration_window=100, time_constant=1.0, leakyness=0.0,
                 V_m_threshold = 2.0, refractory_period=0, amplitude=1.0, V_m_min=0, V_cm=2.5, device='cuda'):
        super(Integrator_layer, self).__init__()
        # self.threshold = nn.Threshold(V_m_threshold, 0)
        # self.zero = torch.tensor(0, dtype=torch.float, device=device)
        self.Vm_threshold = V_m_threshold
        self.integration_window = integration_window
        self.refractory_period = refractory_period
        self.time_constant = time_constant
        # self.epsilon = 0.001
        self.epsilon = tf.keras.backend.epsilon
        self.amplitude = amplitude
        # self.threshold = nn.Threshold(V_m_threshold - self.epsilon, 0) ### Thresholding function
        # self.threshold = tf.nn.relu(V_m_threshold - self.epsilon, 0)  ### Thresholding function
        self.V_m_min = V_m_min
        self.device = device

    @tf.function
    def chunk_sizes(self, length, chunk_size):
        chunks = [chunk_size for x in range(length//chunk_size)]
        if length % chunk_size != 0:
            chunks.append(length % chunk_size)
        return chunks

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.timesteps = input_shape[1]
        self.image_shape = input_shape[2:]
        # self.chunk_sizes = self.chunk_sizes(self.timesteps, self.integration_window)
        self.chunk_sizes = self.chunk_sizes(input_shape[1], self.integration_window)
        ###
        ###
        ###
        # self.list_of_indices = [[x, 0] for x in tf.range(input_shape[0])]
        # self.list_of_indices = tf.range(input_shape[0])

    @tf.function
    def call(self, inputs):
        ### List of indices - list of indices to replace very first timestep with zero after the roll operation
        list_of_indices = tf.pad(tf.expand_dims(tf.range(tf.shape(inputs)[0]), axis=1),
                                 paddings=[[0, 0], [0, 1]],
                                 mode="CONSTANT")
        images_chunks = tf.split(inputs, self.chunk_sizes, axis=1) ### Fragment current sample into multiple chunks with length equal to the integration window
        first_chunk = True
        # zero = torch.tensor(0, dtype=torch.float, device=self.device)
        for chunk, n_timesteps in zip(images_chunks, self.chunk_sizes):
            ### n_timesteps - the number of timesteps for current chunk of integration window
            Spikes_out = tf.zeros([tf.shape(chunk)[0], *self.image_shape, n_timesteps + 1])
            ### V_m_out - array for storing membrane potential
            V_m_out = tf.zeros_like(chunk)
            V_m_temp = tf.zeros_like(chunk)
            while tf.math.count_nonzero(V_m_temp) != 0:
                ### V_m_chunk - cumulative summation (integration) along time dimension
                V_m_chunk = tf.math.cumsum(tf.math.multiply(chunk, self.time_constant), axis=1)
                ### Thresholding chunks, all values bellow threshold value are zeroed
                V_m_temp = tf.nn.relu(V_m_chunk - self.Vm_threshold)
                if tf.math.count_nonzero(V_m_temp) == 0: ### if Vm did not cross threshold, break the cycle
                    V_m_out = V_m_out + V_m_chunk
                    break
                ### Cumsum of the thresholded cumsum - to avoid any future threshold crossings (additional zeroes) that can occur after threshold is hit:
                V_m_temp = tf.math.cumsum(V_m_temp, axis=1)
                ### V_m_temp == 0 The amount of zero values before function crosses the threshold. Used to calculated how many timesteps it took for an integrator to fire an output spike
                Spikes_out = Spikes_out + tf.one_hot(tf.reduce_sum(tf.cast((V_m_temp == 0), tf.int32), axis=1), depth=n_timesteps + 1)### One hotted zero counts
                ### TF roll operation is used to shift the vector values by 1, other timestep which crossed threshold is not included:
                V_m_temp = tf.roll(V_m_temp, shift=1, axis=1)
                ###Since roll operation will shift the last value to the first place, the first value should be 0'ed for a proper counting of 0 in the next code fragments.
                V_m_temp = tf.tensor_scatter_nd_update(V_m_temp, indices=list_of_indices,
                                                       updates=tf.zeros([1, *self.image_shape]))
                V_m_out = tf.where(V_m_temp == 0, V_m_out + V_m_chunk, 0) ### Resets V_m to 0 after firing
                if self.refractory_period!=0: ### Resets (=0) number of timesteps after output spike is fired
                    V_m_temp = tf.roll(chunk, shift=self.refractory_period, axis=1)
                    V_m_temp[:, 0:(self.refractory_period-1), :, :, :] = 0
                chunk = tf.where(V_m_temp == 0.0, 0.0, chunk) ### Removes spikes before firing. So new V_m can be calculated for a next spike.
            # Spikes_out = torch.narrow(Spikes_out, dim=-1, start=0, length= n_timesteps)
            Spikes_out, _ = tf.split(Spikes_out, [n_timesteps, 1], axis=-1) ### Onehot operation adds back time dimension to the last place, so it must be popped out
            if first_chunk:
                V_m_final = V_m_out
                Spikes_out_final = Spikes_out
                first_chunk = False
            else:
                V_m_final = tf.concat((V_m_final, V_m_out), axis=1)
                Spikes_out_final = tf.concat((Spikes_out_final, Spikes_out), axis=-1)

        # Spikes_out_final = torch.movedim(Spikes_out_final, source=-1, destination=1)
        Spikes_out_final = tf.experimental.numpy.swapaxes(Spikes_out_final, axis1=-1,
                                         axis2=1)  ### Onehotting puts time as the last tensor dimension. 'movedim' moves time dimension to the 2nd place, after the batch number, as it was before.

        # return V_m_final, Spikes_out_final
        # print('LIF forward end:')
        # print(f'{datetime.now().time().replace(microsecond=0)} --- ')
        # print(Spikes_out.type())
        if self.amplitude !=1.0:
            Spikes_out_final = Spikes_out_final*self.amplitude
        return Spikes_out_final


def sparse_data_generator_non_spiking(input_images, input_labels, batch_size=32, nb_steps=100, shuffle=True, flatten= False):
    """ This generator takes datasets in analog format and generates network input as constant currents.
        If repeat=True, encoding is rate-based, otherwise it is a latency encoding
    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

# def argument_free_generator():
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    data_loader_original = tf.data.Dataset.from_tensor_slices((tf.cast(input_images, tf.float32), input_labels))
    if shuffle:
        data_loader_original = data_loader_original.shuffle(buffer_size=100)
    data_loader = data_loader_original.batch(batch_size=batch_size, drop_remainder=False)

    number_of_batches = input_labels.__len__() // batch_size
    counter = 0
    time = tm.time()

    for X, y in data_loader:
        if flatten:
            X = X.reshape(X.shape[0], -1)
        # sample_dims = np.array(X.shape[1:], dtype=int)
        sample_dims = X.shape[1:]
        # X = torch.unsqueeze(X, dim=1)
        X = tf.expand_dims(X, axis=1)
        X = tf.repeat(X, repeats=nb_steps, axis=1)
        time_taken = tm.time() - time
        time = tm.time()
        ETA = time_taken * (number_of_batches - counter)
        sys.stdout.write(
            "\rBatch: {0}/{1}, Progress: {2:0.2f}%, Time to process last batch: {3:0.2f} seconds, Estimated time to finish epoch: {4:0.2f} seconds | {5}:{6} minutes".format(
                counter, number_of_batches, (counter / number_of_batches) * 100, time_taken, ETA, int(ETA // 60),
                int(ETA % 60)))
        sys.stdout.flush()
        # X_batch = torch.tensor(X, device=device, dtype=torch.float)
        # yield X.expand(-1, nb_steps, *sample_dims).to(device), y.to(device)  ### Returns this values after each batch
        counter += 1
        yield X, y  ### Returns this values after each batch

# return argument_free_generator()


class Reduce_sum(tf.keras.layers.Layer):
    def __init__(self):
        super(Reduce_sum, self).__init__()

    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=1, keepdims=True)