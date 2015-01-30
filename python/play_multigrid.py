### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Gerard Gorman, Imperial College London
### the purpose of the test case is to illustrate the use of the refine_metric function.
### The idea is to use the functionality in a multigrid algorithm.

from dolfin import *
from adaptivity import refine_metric, adapt, metric_pnorm, mesh_metric
set_log_level(WARNING)

def test_refine_metric():
  #  from mpi4py import MPI
  import sys

#  comm = MPI.COMM_WORLD

  # mesh = Mesh("greenland.xml.gz")
  mesh = UnitSquareMesh(100, 100)

  V = FunctionSpace(mesh, "CG", 2)
  f = interpolate(Expression("0.1*sin(50.*(2*x[0]-1)) + atan2(-0.1, (2.0*(2*x[0]-1) - sin(5.*(2*x[1]-1))))"), V)

  eta = 0.01
  #Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
  #mesh = adapt(Mp)

  if True:
    level = 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh1 = adapt(Mp)

    level *= 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh2 = adapt(Mp)

    level *= 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh3 = adapt(Mp)

    level *= 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh4 = adapt(Mp)

    level *= 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh5 = adapt(Mp)

    level *= 0.5
    Mp = refine_metric(mesh_metric(mesh), level)
    new_mesh6 = adapt(Mp)
  else:
    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh1 = adapt(Mp)

    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh2 = adapt(Mp)

    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh3 = adapt(Mp)

    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh4 = adapt(Mp)

    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh5 = adapt(Mp)

    eta *= 2
    Mp = metric_pnorm(f, mesh, eta, max_edge_ratio=5)
    new_mesh6 = adapt(Mp)

  # plot(Mp[0,0])
  # from IPython import embed
  # embed()

  plot(mesh, title="initial mesh")
  plot(new_mesh1, title="coarsen 1")
  plot(new_mesh2, title="coarsen 2")
  plot(new_mesh3, title="coarsen 3")
  plot(new_mesh4, title="coarsen 4")
  plot(new_mesh5, title="coarsen 5")
  plot(new_mesh6, title="coarsen 6")

  interactive()
  
if __name__=="__main__":
 test_refine_metric()
