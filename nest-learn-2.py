import nest
import nest.topology as topp
import numpy as np
from matplotlib import pyplot as plt

jit = 0.03
xs = np.arange(-0.5,.501,0.1)
poss = [[x,y] for y in xs for x in xs]
poss = [[p[0]+np.random.uniform(-jit,jit),p[1]+np.random.uniform(-jit,jit)] for p in poss]

my_layer_dict_off_grid = {"positions":poss,
                          "elements":"iaf_psc_alpha",
                          "extent":[1.1,1.1]}

my_layer_dict_on_grid = {"rows":11,
                          "columns":11,
                          "extent":[11.0,11.0],
                          "elements":'iaf_psc_alpha'}

# connectivity specifications with a mask
conndict = {'connection_type': 'divergent',
             'mask': {'rectangular': {'lower_left'  : [-2.0, -1.0],
                                      'upper_right' : [2.0, 1.0]}}}

my_layer = topp.CreateLayer(my_layer_dict_on_grid)
topp.PlotLayer(my_layer)
topp.ConnectLayers(my_layer,my_layer,conndict)

nest.PrintNetwork(depth=1)
topp.PlotTargets([5],my_layer)
plt.show()


