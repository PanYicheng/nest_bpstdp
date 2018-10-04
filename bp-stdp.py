from matplotlib import pyplot as plt
import numpy as np
import time
import nest
import h5py
import os


class SNN:

    def __init__(self):
        nest.ResetKernel()
        self.input_layer_size = 784
        self.hidden_layer_size = 100
        self.output_layer_size = 10
        self.base_frequency = 250
        self.target_frequency = 100
        self.time_step = 100
        self.test_steps = 5
        self.episislon = 4
        self.learning_rate = 0.00001
        self.resolution = 1.0
        self.cita_h = 0.9
        self.cita_o = 0.025*self.hidden_layer_size
        self.parrallel_parameter = 1

    def create_snn(self):
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads": self.parrallel_parameter,
                              "resolution":self.resolution})
        nest.set_verbosity('M_ERROR')

        # input layer, poisson neuron * 784
        self.input_neurons = nest.Create("poisson_generator",self.input_layer_size)
        self.spike_detector_input = nest.Create("spike_detector", self.input_layer_size)

        # hidden layer,iaf neuron
        self.hidden_neurons = nest.Create("iaf_psc_alpha", self.hidden_layer_size)
        self.spike_detector_hidden = nest.Create("spike_detector", self.hidden_layer_size)

        # output layer,iaf neuron
        self.output_neurons = nest.Create("iaf_psc_alpha", self.output_layer_size)
        self.spike_detector_output = nest.Create("spike_detector", self.output_layer_size)

        # Set Status

        # Connect the network
        nest.Connect(self.input_neurons, self.hidden_neurons, "all_to_all", syn_spec={"model":"static_synapse",
                                                                                      "weight":{"distribution":"normal",
                                                                                                "mu":10.0,
                                                                                                "sigma":10.0}})
        nest.Connect(self.hidden_neurons, self.output_neurons, "all_to_all", syn_spec={"model":"static_synapse",
                                                                                       "weight":{"distribution":"normal",
                                                                                                "mu":10.0,
                                                                                                "sigma":10.0}})
        nest.Connect(self.input_neurons, self.spike_detector_input, 'one_to_one')
        nest.Connect(self.hidden_neurons, self.spike_detector_hidden, 'one_to_one')
        nest.Connect(self.output_neurons, self.spike_detector_output, "one_to_one")

        self.conns_i_h = nest.GetConnections(self.input_neurons,self.hidden_neurons)
        self.conns_h_o = nest.GetConnections(self.hidden_neurons,self.output_neurons)

    def train(self, sim_time, img_array, label):
        print("Current the label is : %d" % label)
        # reset poisson generator
        kernel_time = nest.GetKernelStatus("time")
        nest.SetStatus(self.input_neurons, {"origin": kernel_time})
        nest.SetStatus(self.input_neurons, {"start": 0.0})
        nest.SetStatus(self.input_neurons, {"stop": float(sim_time)})

        nest.SetStatus(self.input_neurons, [{"rate": np.float(x) / 255.0 * self.base_frequency} for x in img_array])
        
        # train the SNN     
        for f in range(int(sim_time / self.time_step)):
            start = time.time()
            # get the parameters before Simulate
            last_spike_input = nest.GetStatus(self.spike_detector_input)
            last_spike_hidden = nest.GetStatus(self.spike_detector_hidden)
            last_spike_output = nest.GetStatus(self.spike_detector_output)

            # Simulate one time_step
            nest.Simulate(self.time_step)

            # get the parameters after Simulate
            now_spike_input = nest.GetStatus(self.spike_detector_input)
            now_spike_hidden = nest.GetStatus(self.spike_detector_hidden)
            now_spike_output = nest.GetStatus(self.spike_detector_output)

            # compute ksai_output
            ksai_output = np.zeros(self.output_layer_size)
            spike_sum_output = np.zeros(self.output_layer_size)
            for output_neuron_id in range(self.output_layer_size):
                spike_sum_output[output_neuron_id] = now_spike_output[output_neuron_id]["n_events"]-last_spike_output[output_neuron_id]["n_events"]
                if(spike_sum_output[output_neuron_id] >= 1):
                    if(output_neuron_id != label):
                        ksai_output[output_neuron_id] = -1
                else:
                    if(output_neuron_id == label):
                        ksai_output[output_neuron_id] = 1

            # count spikes of input & hidden layer in one timestep
            spike_sum_input = np.zeros(self.input_layer_size)
            for input_neuron_id in range(self.input_layer_size):
                spike_sum_input[input_neuron_id] += now_spike_input[input_neuron_id]["n_events"]-last_spike_input[input_neuron_id]["n_events"]
            spike_sum_hidden = np.zeros(self.hidden_layer_size)
            for hidden_neuron_id in range(self.hidden_layer_size):
                spike_sum_hidden[hidden_neuron_id] += now_spike_hidden[hidden_neuron_id]["n_events"]-last_spike_hidden[hidden_neuron_id]["n_events"]
            print("Spikes Sum:(%d,%d,%d)" % (np.sum(spike_sum_input),np.sum(spike_sum_hidden),np.sum(spike_sum_output)))
            # when output spikes decrease to 0,means it cann't train anymore
            if(np.sum(spike_sum_hidden) == 0 ):
                exit(1)

            # update the weights from hidden to output layer
            status_h_o = nest.GetStatus(self.conns_h_o)
            weights_h_o = np.array([x["weight"] for x in status_h_o]).reshape(
                (self.hidden_layer_size, self.output_layer_size))
            delta_weight_h_o = np.dot(spike_sum_hidden.reshape((self.hidden_layer_size,1)),
                                      ksai_output.reshape((1,self.output_layer_size))) * self.learning_rate
            weights_h_o_new = weights_h_o + delta_weight_h_o
            weights_h_o_new_flatten = weights_h_o_new.flatten()
            nest.SetStatus(self.conns_h_o,[{"weight":x} for x in weights_h_o_new_flatten])


            # update the weights from input to hidden layer
            derivatives_h = spike_sum_hidden.copy()
            derivatives_h[derivatives_h>0] = 1
            ksai_hidden = np.dot(weights_h_o,ksai_output) * derivatives_h
            delta_weight_i_h = np.dot(spike_sum_input.reshape((self.input_layer_size,1)),
                                      ksai_hidden.reshape((1,self.hidden_layer_size))) * self.learning_rate
            status_i_h = nest.GetStatus(self.conns_i_h)
            weight_i_h = np.array([x["weight"] for x in status_i_h]).reshape(self.input_layer_size,self.hidden_layer_size)
            weight_i_h = weight_i_h + delta_weight_i_h
            weight_i_h_flatten = weight_i_h.flatten()
            nest.SetStatus(self.conns_i_h,[{"weight":x} for x in weight_i_h_flatten])
            
            end = time.time()
            for u in range(self.output_layer_size):
                result = now_spike_output[u]["n_events"] - last_spike_output[u]["n_events"]
                print("%d " % result,end=" ")
            print("trianing for one step cost: %.2f" % (end - start))
        return (weight_i_h_flatten,weights_h_o_new_flatten)

    def test(self,imgs,labels):
        correct_num = 0
        num = len(labels)
        for img_ct in range(num):
            img_array = imgs[img_ct].flatten()
            img_label = labels[img_ct]
            kernel_time = nest.GetKernelStatus("time")
            nest.SetStatus(self.input_neurons, {"origin": kernel_time})
            nest.SetStatus(self.input_neurons, {"start": 0.0})
            nest.SetStatus(self.input_neurons, {"stop": float(self.test_steps * self.time_step)})
            nest.SetStatus(self.input_neurons, [{"rate": np.float(x) / 255.0 * self.base_frequency} for x in img_array])

            nest.SetStatus(self.spike_detector_output, {"n_events": 0})
            # Simulate one time_step
            nest.Simulate(self.time_step * self.test_steps)
            now_spike_output = [x["n_events"] for x in nest.GetStatus(self.spike_detector_output)]
            network_label = np.argmax(now_spike_output)
            correct_num += int(img_label == network_label)
        print(" Accuracy is: %f" % (np.float(correct_num)/np.float(num)),end="; ")
        print([x for x in labels])
        record_file = open('test_record.txt', 'a+')
        localtime = time.asctime(time.localtime(time.time()))
        record_file.writelines('%s , Accuarcy : %d\n' % (localtime, np.float(correct_num)/np.float(num)))
        record_file.close()

    def save_weight(self):
        print(" # Saving weights...")
        conns = nest.GetConnections(self.input_neurons,self.hidden_neurons)
        status = nest.GetStatus(conns)
        weight_1 = [x["weight"] for x in status]
        conns = nest.GetConnections(self.hidden_neurons, self.output_neurons)
        status = nest.GetStatus(conns)
        weight_2 = [x["weight"] for x in status]
        f = h5py.File("WEIGHTS_TRAIN.h5","w")
        f["weight_1"] = weight_1
        f["weight_2"] = weight_2
        f.close()
    def load_weight(self,weight_file_name):
        print(" # Loading weights...")
        f = h5py.File(weight_file_name,'r')
        weights_1 = f["weight_1"][:]
        weights_2 = f["weight_2"][:]
        conns = nest.GetConnections(self.input_neurons, self.hidden_neurons)
        nest.SetStatus(conns,[{"weight":x} for x in weights_1])

        conns = nest.GetConnections(self.hidden_neurons, self.output_neurons)
        nest.SetStatus(conns, [{"weight": x} for x in weights_2])

if __name__ == "__main__":
    f = h5py.File("HDF5_MNIST_TRAIN.h5", 'r')
    img = f["img"][:]
    label = f["label"][:]
    f.close()
    f = h5py.File("HDF5_MNIST_TEST.h5", 'r')
    test_img = f["img"][:]
    test_label = f["label"][:]
    f.close()
    snn = SNN()
    snn.create_snn()
    if(os.path.exists('WEIGHTS_TRAIN.h5')):
        snn.load_weight("WEIGHTS_TRAIN.h5")
    for i in range(10000):
        # print("train for img[%d] label: %d" % (i,label[i]))
        (i_h_weight_list,h_o_weight_list) = snn.train(500, img[i].flatten(), label[i])
        snn.save_weight()
        # plt.subplot(211).hist(i_h_weight_list,bins = 500,color='steelblue',normed=True)
        # plt.subplot(212).hist(h_o_weight_list,bins = 500,color='steelblue',normed=True)
        # plt.show()
        start_index = int(np.random.random(1)[0]*9990)
        snn.test(test_img[start_index:(start_index+10)],test_label[start_index:(start_index+10)])
    snn.save_weight()
