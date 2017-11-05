import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from adaptivity import *

check = []

for n in range(2, 20):
    mesh = UnitSquareMesh(mpi_comm_world(), n,n)

    M = mesh_metric(mesh)

    A = M.vector().array().reshape((mesh.num_cells(),2,2))

    # the metric should be constant on each element
    if abs(A - A[0,:,:]).max()/A.max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not constant on each element')

    a = A[0,:,:]

    check.append(a.max()*mesh.hmax()**2)

# the metric should scale like O(1/h^2)
if max(check) - min(check) > 1.0e-14:
    raise RuntimeError('Test failed: the metric does not scale like O(1/h^2)')

print("TEST PASSED!")
