import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from adaptivity import *
from numpy import array, sort
import numpy as np

#############################################################################################################
### Test 1: Unit Square
#############################################################################################################

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

#############################################################################################################
### Test 2: Unit Cube
#############################################################################################################

# in 3D a cube can be divided into 6 different tetrahedra. Given the structure of UnitCubeMesh
# there are only 3 types of tetrahedra in this case
exact_metrics = array([[[ 1. , -0.5,  0. ], [-0.5,  1. , -0.5], [ 0. , -0.5,  1. ]],\
                       [[ 1. ,  0., -0.5 ], [ 0. ,  1. , -0.5], [-0.5, -0.5,  1. ]],\
                       [[ 1. ,-0.5, -0.5 ], [-0.5,  1. ,  0. ], [-0.5,  0. ,  1. ]]])

for n in range(2, 10):
    mesh = UnitCubeMesh(mpi_comm_world(), n, n, n)

    M = mesh_metric(mesh)

    A = M.vector().array().reshape((mesh.num_cells(),3,3))

    # the following checks that all the sorted entries of all the metrics are correct
    if abs(sort(A.flatten()) - sort(A[0,:,:].flatten().repeat(mesh.num_cells()))).max()/A.max() > 1.0e-14:
        raise RuntimeError('Test failed: the metric is not correct')

    a = A*mesh.hmax()**2/3.

    # the following checks that each computed metric is one of the three possible exact metrics
    if np.prod(array([np.max(abs(a - exact_metrics[i,:,:]) > 1.0e-14, (1,2)) for i in range(3)]), 1).max() != 0:
        raise RuntimeError('Test failed: the metric is not correct')

print("TEST PASSED!")
