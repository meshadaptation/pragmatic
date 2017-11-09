import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from adaptivity import *
from numpy import array

check = []
test_metric = array([[1.0, -0.5], [-0.5, 1.0]])

for n in range(2, 20):
    mesh = UnitSquareMesh(mpi_comm_world(), n,n)

    M = mesh_metric(mesh)

    A = M.vector().array().reshape((mesh.num_cells(),2,2))

    # the metric should be constant on each element
    if abs(A - A[0,:,:]).max()/A.max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not constant on each element')

    a = A[0,:,:]*mesh.hmax()**2/2.

    if abs(a-test_metric).max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not correct')

    check.append(a.max())

# the metric should scale like O(1/h^2). This test is redundant because we are checking whether the metric is exact.
if max(check) - min(check) > 1.0e-14:
    raise RuntimeError('Test failed: the metric does not scale like O(1/h^2)')

print("TEST PASSED!")
