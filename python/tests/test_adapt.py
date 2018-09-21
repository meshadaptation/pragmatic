import sys
import os
sys.path.append(os.path.abspath('..'))
from dolfin import *
from mshr import *
from adaptivity import *
from numpy import unique

#############################################################################################################
### Test 1: Unit Square
#############################################################################################################

mesh = UnitSquareMesh(MPI.comm_world, 5, 5)

V = TensorFunctionSpace(mesh, 'CG', 1)

M_expr = Expression((('1. + 500.0*x[0]','0.0'), ('0.0', '1.0 + 500.0*x[1]')), degree = 1)
M = interpolate(M_expr, V)

new_mesh,tags = adapt(M)

# 4 tags for the boundary + 1 tag for the interior
if len(unique(tags.array())) != 5:
    raise RuntimeError('Test failed: number of tags of generated mesh is different from 5')

if abs(new_mesh.num_vertices() - 296) > 50:
    raise RuntimeError('Test failed: number of vertices of generated mesh is different from what expected')

#############################################################################################################
### Test 2: Unit Cube
#############################################################################################################

mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)

V = TensorFunctionSpace(mesh, 'CG', 1)

M_expr = Expression((('1. + 100.0*x[0]','0.0', '0.0'), ('0.0', '1.0 + 100.0*x[1]', '0.0'), ('0.0', '0.0', '1.0 + 100.0*x[2]')), degree = 1)
M = interpolate(M_expr, V)

new_mesh,tags = adapt(M)

# 4 tags for the boundary + 1 tag for the interior
if len(unique(tags.array())) != 7:
    raise RuntimeError('Test failed: number of tags of generated mesh is different from 7')

if abs(new_mesh.num_vertices() - 905) > 80:
    raise RuntimeError('Test failed: number of vertices of generated mesh is different from what expected')

#############################################################################################################
### Test 4: Unit Circle
#############################################################################################################

try: 
    mesh = UnitDiscMesh(MPI.comm_world, 10, 1, 2)
except TypeError:
    circ = Circle(Point(0.0,0.0), 1)
    mesh = generate_mesh(circ, 10)

V = TensorFunctionSpace(mesh, 'CG', 1)

M_expr = Constant((('20.0','0.0'), ('0.0', '20.0')))
M = interpolate(M_expr, V)

new_mesh,tags = adapt(M, coarsen = True)

area     = assemble(Constant(1.0)*dx(domain=mesh))
new_area = assemble(Constant(1.0)*dx(domain=new_mesh))

relative_change = abs(area - new_area)/area

if relative_change > 0.012:
    raise RuntimeError('Test failed: mesh volume not preserved in the right way')


print("TEST PASSED!")
