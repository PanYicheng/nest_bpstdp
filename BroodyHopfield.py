import nest
import nest.raster_plot
import pylab

N = 1000           # number of neurons
bias_begin = 140.  # minimal value for the bias current injection [pA]
bias_end = 200.    # maximal value for the bias current injection [pA]
T = 600            # simulation time (ms)

driveparams = {'amplitude': 50., 'frequency': 35.}
noiseparams = {'mean': 0.0, 'std': 200.}
neuronparams = {'tau_m': 20.,  # membrane time constant
                'V_th': 20.,  # threshold potential
                'E_L': 10.,  # membrane resting potential
                't_ref': 2.,  # refractory period
                'V_reset': 0.,  # reset potential
                'C_m': 200.,  # membrane capacitance
                'V_m': 0.}      # initial membrane potential

neurons = nest.Create('iaf_psc_alpha', N)
sd = nest.Create('spike_detector')
noise = nest.Create('noise_generator')
drive = nest.Create('ac_generator')

nest.SetStatus(drive, driveparams)
nest.SetStatus(noise, noiseparams)

nest.SetStatus(neurons, neuronparams)
nest.SetStatus(neurons, [{'I_e':
                          (n * (bias_end - bias_begin) / N + bias_begin)}
                         for n in neurons])

nest.SetStatus(sd, {"withgid": True, "withtime": True})

nest.Connect(drive, neurons)
nest.Connect(noise, neurons)
nest.Connect(neurons, sd)

nest.Simulate(T)

nest.raster_plot.from_device(sd, hist=True)
pylab.show()
