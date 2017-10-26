from dolfin import *
from adaptivity import *
from numpy import unique

#############################################################################################################
### Test 1: Unit Square
#############################################################################################################

mesh = UnitSquareMesh(mpi_comm_world(), 5, 5)

V = TensorFunctionSpace(mesh, 'CG', 1)

M_expr = Expression((('1. + 500.0*x[0]','0.0'), ('0.0', '1.0 + 500.0*x[1]')), degree = 1)
M = interpolate(M_expr, V)

new_mesh,tags = adapt(M)

# 4 tags for the boundary + 1 tag for the interior
if len(unique(tags.array())) != 5:
    raise RuntimeError('Test failed: number of tags of generated mesh is different from 5')

if abs(new_mesh.num_vertices() - 339) > 30:
    raise RuntimeError('Test failed: number of vertices of generated mesh is different from what expected')

#############################################################################################################
### Test 2: Unit Cube
#############################################################################################################

mesh = UnitCubeMesh(mpi_comm_world(), 5, 5, 5)

V = TensorFunctionSpace(mesh, 'CG', 1)

M_expr = Expression((('1. + 100.0*x[0]','0.0', '0.0'), ('0.0', '1.0 + 100.0*x[1]', '0.0'), ('0.0', '0.0', '1.0 + 100.0*x[2]')), degree = 1)
M = interpolate(M_expr, V)

new_mesh,tags = adapt(M)

# 4 tags for the boundary + 1 tag for the interior
if len(unique(tags.array())) != 7:
    raise RuntimeError('Test failed: number of tags of generated mesh is different from 5')

if abs(new_mesh.num_vertices() - 833) > 80:
    raise RuntimeError('Test failed: number of vertices of generated mesh is different from what expected')

