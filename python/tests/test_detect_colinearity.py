import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from adaptivity import *
from mshr import Polygon, generate_mesh
from numpy import cos, sin, pi, empty, unique

#############################################################################################################
### Test 1: Hexagon
#############################################################################################################

for N in range(4,21):
    x = empty((N,))
    y = empty((N,))
    for n in range(N):
        x[n] = cos(2.*pi*n/N)
        y[n] = sin(2.*pi*n/N)

    interior_angles = 180.*(N - 2)/N
    if interior_angles > 90:
        interior_angles = 180 - interior_angles

    mesh = generate_mesh(Polygon([Point(xx,yy) for xx,yy in zip(x,y)]), N)

    markers1 = detect_colinearity(mesh, interior_angles - 1)

    markers2 = detect_colinearity(mesh, interior_angles + 1)

    # number of tags should be equal to the number of sides of the polygon plus the interior tag
    if len(unique(markers1.array())) != N+1:
        raise RuntimeError('Test failed: number of tags assigned by colinearity_detection is wrong')

    # number of tags should be equal to the number of sides of the polygon divided by 2 plus the interior tag
    # this is because colinearity is checked with respect to the first untagged facet found
    if len(unique(markers2.array())) != int(N/2)+1:
        raise RuntimeError('Test failed: number of tags assigned by colinearity_detection is wrong')

#############################################################################################################
### Test 2: Unit Cube
#############################################################################################################

mesh = UnitCubeMesh(mpi_comm_world(), 2, 2, 2)

markers1 = detect_colinearity(mesh, 89)
markers2 = detect_colinearity(mesh, 91)

# number of tags should be equal to the number of sides of the cube plus the interior tag
if len(unique(markers1.array())) != 6+1:
    raise RuntimeError('Test failed: number of tags assigned by colinearity_detection is wrong')

# number of tags should be equal to the number of sides of the cube divided by 3 plus the interior tag
if len(unique(markers2.array())) != 2+1:
    raise RuntimeError('Test failed: number of tags assigned by colinearity_detection is wrong')


print("TEST PASSED!")
