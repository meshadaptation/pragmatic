import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from adaptivity import *
from numpy import array, sort

#############################################################################################################
### Test 1: Unit Square
#############################################################################################################

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

#############################################################################################################
### Test 2: Unit Cube
#############################################################################################################

check = []
test_metric = array([-0.5, -0.5, -0.5, -0.5, -0. , -0. ,  1. ,  1. ,  1. ])

for n in range(2, 10):
    mesh = UnitCubeMesh(mpi_comm_world(), n, n, n)

    M = mesh_metric(mesh)

    A = M.vector().array().reshape((mesh.num_cells(),3,3))

    # the metric should be constant on each element. Currently the metric depends on the mesh ordering
    # so we sort the entries
    if abs(sort(A.flatten()) - sort(A[0,:,:].flatten().repeat(mesh.num_cells()))).max()/A.max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not constant on each element')

    a = sort(A[0,:,:].flatten()*mesh.hmax()**2/3.)

    if abs(a-test_metric).max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not correct')

    check.append(a.max())

# the metric should scale like O(1/h^2). This test is redundant because we are checking whether the metric is exact.
if max(check) - min(check) > 1.0e-14:
    raise RuntimeError('Test failed: the metric does not scale like O(1/h^2)')

print("TEST PASSED!")
