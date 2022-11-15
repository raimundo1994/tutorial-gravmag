import numpy as np
from numpy.testing import assert_array_less as aal
import build_model as mod

def test_model_edge_size():

    area = [-32, -24.5, -0.5, 2]
    shape = (4,5)
    y = np.linspace(area[0], area[1], shape[1])
    x = np.linspace(area[3],area[2], shape[0])
    y,x = np.meshgrid(y,x)
    z = np.random.rand(4,5)

    model = mod.build_model(x,y,z,dz=1000)

    aal(model[:,0],y.ravel()),"The west(y1) boundary can't be greater than the y component of TCC"
    aal(y.ravel(),model[:,1]),"The y component of TCC can't be greater than the east(y2) boundary "
    aal(model[:,2],x.ravel()),"The south(x1) boundary can't be greater than the x component of TCC"
    aal(x.ravel(),model[:,3]),"The x component of TCC can't be greater than the north(x2) boundary "
    aal(model[:,4],model[:,5])," Top can't be greater than the bottom"
