# from matplotlib import pyplot as plt
import numpy as np
# import cv2
import time
import nest
import h5py
import os
import sys
from mnist_data import create_hdf5_data

height = 28
width = 28


class SNN:
    def __init__(self, input_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = 100
        self.output_layer_size = output_layer_size
        self.base_frequency = 100
        self.target_frequency = 100
        self.sim_time = 100
        self.episislon = 10
        self.learning_rate = 200

    def create_snn(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')

        # input layer, poisson neuron * 784
        self.input_neurons = nest.Create("poisson_generator", self.input_layer_size)
        # print(self.input_neurons)

        # hidder layer,iaf neuron
        self.hidden_neurons = nest.Create("iaf_psc_alpha", self.hidden_layer_size)
        self.spike_dectector_hidden = nest.Create("spike_detector", self.hidden_layer_size)

        # output layer,iaf neuron
        self.output_neurons = nest.Create("iaf_psc_alpha", self.output_layer_size)
        self.spike_dectector_output = nest.Create("spike_detector", self.output_layer_size)

        nest.Connect(self.input_neurons, self.hidden_neurons, "all_to_all", syn_spec={"model": "static_synapse",
                                                                                      "weight": {
                                                                                          "distribution": "normal",
                                                                                          "mu": 10.0,
                                                                                          "sigma": 5.0}})
        nest.Connect(self.hidden_neurons, self.output_neurons, "all_to_all", syn_spec={"model": "static_synapse",
                                                                                       "weight": {
                                                                                           "distribution": "normal",
                                                                                           "mu": 10.0,
                                                                                           "sigma": 5.0}})
        nest.Connect(self.output_neurons, self.spike_dectector_output, "one_to_one")
        nest.Connect(self.hidden_neurons, self.spike_dectector_hidden, 'one_to_one')

    def train(self, img_array, label):
        # nest.ResetKernel()
        # for i in range(self.input_layer_size):
        #     nest.SetStatus([self.input_neurons[i]],{"rate":np.float(img_array[i] / 255.0 * self.base_frequency)})
        nest.SetStatus(self.input_neurons, [{"rate": i / 255.0 * self.base_frequency} for i in img_array])
        start_time = time.time()
        nest.Simulate(self.sim_time)
        end_time = time.time()
        print("    Nest Simulate Time:", end_time - start_time, "s")
        start_time = time.time()
        sd_status_output = nest.GetStatus(self.spike_dectector_output)
        sd_status_hidden = nest.GetStatus(self.spike_dectector_hidden)

        # hidden_error is backpropagated from output layer
        hidden_error = np.zeros(self.hidden_layer_size)
        for output_neuron_id in range(self.output_layer_size):
            # z_j = 1 , check r_i
            ksai = 0.0
            if (output_neuron_id == label):
                if (sd_status_output[output_neuron_id]["n_events"] == 0):
                    ksai = 1.0
            else:
                if (sd_status_output[output_neuron_id]["n_events"] != 0):
                    ksai = -1.0
            for hidden_neuron_id in range(self.hidden_layer_size):
                spike_sum = sd_status_hidden[hidden_neuron_id]["n_events"]
                delta_weight = self.learning_rate * spike_sum * ksai
                conn = nest.GetConnections([self.hidden_neurons[hidden_neuron_id]],
                                           [self.output_neurons[output_neuron_id]])
                old_weight = nest.GetStatus(conn)[0]["weight"]
                nest.SetStatus(conn, params={"weight": old_weight + delta_weight})

                hidden_error[hidden_neuron_id] += ksai * old_weight
        for hidden_neuron_id in range(self.hidden_layer_size):
            if (sd_status_hidden[hidden_neuron_id]["n_events"] > 0):
                for input_neuron_id in range(self.input_layer_size):
                    sigma_s_j = nest.GetStatus([self.input_neurons[input_neuron_id]])[0]['rate'] * self.sim_time
                    delta_weight = self.learning_rate * hidden_error[hidden_neuron_id] * sigma_s_j
                    conn = nest.GetConnections([self.input_neurons[input_neuron_id]],
                                               [self.hidden_neurons[hidden_neuron_id]])
                    old_weight = nest.GetStatus(conn)[0]["weight"]
                    nest.SetStatus(conn, params={"weight": old_weight + delta_weight})
        max_index = 0
        total_spikes = sd_status_output[max_index]["n_events"]
        events = [sd_status_output[max_index]["n_events"]]
        for output_neuron_id in range(1, self.output_layer_size):
            if (sd_status_output[output_neuron_id]["n_events"] > sd_status_output[max_index]["n_events"]):
                max_index = output_neuron_id
            total_spikes += sd_status_output[output_neuron_id]["n_events"]
            events.append(sd_status_output[output_neuron_id]["n_events"])
        if (total_spikes == 0):
            possiblity = 1
        else:
            possiblity = sd_status_output[max_index]["n_events"] / total_spikes
        end_time = time.time()
        print("    Weight Update Time:", end_time - start_time, "s")
        # print(max_index,",",possiblity)
        print("  output events:", events)
        return (max_index, possiblity)

    def test(self, imgs, labels):
        pass

    def save_weight(self):
        conns = nest.GetConnections(self.input_neurons, self.hidden_neurons)
        statuses = nest.GetStatus(conns)
        weight_1 = []
        for i in range(len(statuses)):
            weight_1.append(statuses[i]["weight"])
        conns = nest.GetConnections(self.hidden_neurons, self.output_neurons)
        statuses = nest.GetStatus(conns)
        weight_2 = []
        for i in range(len(statuses)):
            weight_2.append(statuses[i]["weight"])
        weights = []
        weights.append(weight_1)
        weights.append(weight_2)
        f = h5py.File("WEIGHTS_TRAIN.h5", "w")
        f["weights"] = weights
        f.close()

    def load_weight(self, weight_file_name):
        f = h5py.File(weight_file_name, 'r')
        weights = f["weights"][:]
        weights_1 = weights[0]
        weights_2 = weights[1]
        conns = nest.GetConnections(self.input_neurons, self.hidden_neurons)
        nest.SetStatus(conns, [{"weights": x} for x in weights_1])

        conns = nest.GetConnections(self.hidden_neurons, self.output_neurons)
        nest.SetStatus(conns, [{"weights": x} for x in weights_2])


if __name__ == "__main__":
    create_hdf5_data()
    is_train = False
    if len(sys.argv) > 1:
        is_train = sys.argv[1] == 'train'
    if is_train:
        with h5py.File("HDF5_MNIST_TRAIN.h5", 'r') as f:
            imgs = f["img"][:]
            labels = f["label"][:]
    else:
        with h5py.File("HDF5_MNIST_TEST.h5", 'r') as f:
            imgs = f["img"][:]
            labels = f["label"][:]
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # disp_img = np.zeros((height,width))
    # for img_ct in range(len(img)):
    #     disp_img = img[img_ct]
    #     cv2.setWindowTitle('img',chr(label[img_ct]+ord('0')))
    #     cv2.imshow('img', disp_img)
    #     cv2.waitKey(0)
    # exit(0)
    snn = SNN()
    snn.create_snn()
    for i in range(100):
        print("train for img[%d] label[%d]" % (i, label[i]))
        snn.train(img[i].flatten(), label[i])
    # snn.save_weight()
