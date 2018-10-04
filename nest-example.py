import nest
from matplotlib import pyplot as plt
import numpy as np

syn_ex = {"weight":1.2}
syn_in = {"weight":-2.0}

neuron = nest.Create("iaf_psc_alpha")
voltmeter = nest.Create("multimeter")
spike_detector = nest.Create("spike_detector",params={"withtargetgid":True,"to_file":False,"label":"spike-detector-out"})
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")


nest.SetStatus(neuron,{"I_e":0.})
nest.SetStatus(voltmeter,{"withtime":True,"record_from":["V_m"]})
nest.SetStatus(noise_ex,{"rate":80000.})
nest.SetStatus(noise_in,{"rate":15000.})


nest.Connect(noise_ex,neuron,syn_spec=syn_ex)
nest.Connect(noise_in,neuron,syn_spec=syn_in)
nest.Connect(voltmeter,neuron)
nest.Connect(neuron,spike_detector)

nest.Simulate(1000.0)

dmm = nest.GetStatus(voltmeter,keys="events")[0]
# print(neuron)
# print(dmm)
Vms = dmm["V_m"]
max_Vms = np.max(Vms)
ts = dmm["times"]

plt.figure(1)
plt.plot(ts,Vms)

dmm2 = nest.GetStatus(spike_detector)[0]
spike_times = dmm2["events"]["times"]
# plt.figure(2)
# plt.subplot(212)
plt.plot(spike_times,np.ones(len(spike_times))*max_Vms,'bo')
print(dmm2)
plt.show()
