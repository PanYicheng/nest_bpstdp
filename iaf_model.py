import nest
import nest.voltage_trace
from matplotlib import pyplot as plt

nest.ResetKernel()
res = 0.1
nest.SetKernelStatus({"resolution": res})
neuron = nest.Create("iaf_psc_alpha")

nest.SetStatus(neuron, {'V_m':0.0, 'E_L':0.0, 'C_m':100.0, 'tau_m':100.0,
                        't_ref':0.0, 'V_th':1.0, 'V_reset':0.0, 'I_e':0.0,
                        'V_min':0.0})

sg = nest.Create("spike_generator")
nest.SetStatus(sg, {'spike_times':[2.0]})

nest.Connect(sg, neuron, 'all_to_all',syn_spec={"model": "static_synapse",
                                                "weight":10
                                                })
voltmeter = nest.Create("voltmeter")
nest.SetStatus(voltmeter, {'interval': 0.1, "withgid": True, "withtime": True})

nest.Connect(voltmeter, neuron)

nest.Simulate(100.0)

nest.voltage_trace.from_device(voltmeter)
plt.axis([0, 20, 0, 1])
plt.show()