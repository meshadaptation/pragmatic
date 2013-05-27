#!/usr/bin/env python

# Copyright (C) 2010 Imperial College London and others.
#
# Please see the AUTHORS file in the main source directory for a
# full list of copyright holders.
#
# Gerard Gorman
# Applied Modelling and Computation Group
# Department of Earth Science and Engineering
# Imperial College London
#
# g.gorman@imperial.ac.uk
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# Many thanks to:
# James Maddinson for the original version of Dolfin interface.
# Davide Longoni for p-norm function.
"""@package PRAgMaTIc

The python interface to PRAgMaTIc (Parallel anisotRopic Adaptive Mesh
ToolkIt) provides anisotropic mesh adaptivity for meshes of
simplexes. The target applications are finite element and finite
volume methods although the it can also be used as a lossy compression
algorithm for data (e.g. image compression). It takes as its input the
mesh and a metric tensor field which encodes desired mesh element size
anisotropically.
"""

import ctypes
import ctypes.util

import numpy
from numpy import array, zeros, matrix, linalg

from dolfin import *

__all__ = ["_libpragmatic",
           "InvalidArgumentException",
           "LibraryException",
           "NotImplementedException",
           "ParameterException",
           "adapt",
           "edge_lengths",
           "mesh_metric",
           "refine_metric"]

class InvalidArgumentException(TypeError):
  pass
class LibraryException(SystemError):
  pass
class NotImplementedException(Exception):
  pass
class ParameterException(Exception):
  pass

try:
  _libpragmatic = ctypes.cdll.LoadLibrary("libpragmatic.so")
except:
  raise LibraryException("Failed to load libpragmatic.so")

def mesh_metric(mesh):
  cells = mesh.cells()
  coords = mesh.coordinates()

  class CellExpression(Expression):
    def eval_cell(self, value, x, ufc_cell):
      x = coords[cells[ufc_cell.index]]
      A = numpy.empty([3, 3])
      b = numpy.ones([3])
      r = numpy.empty([3, 2])
      for i, pair in enumerate([[0, 1], [0, 2], [1, 2]]):
        r[i, :] = x[pair[0], :] - x[pair[1], :]
      for i in range(3):
        A[i, 0] = r[i, 0] ** 2
        A[i, 1] = 2.0 * r[i, 0] * r[i, 1]
        A[i, 2] = r[i, 1] ** 2
      X = numpy.linalg.solve(A, b)
      value[0] = X[0]
      value[1] = value[2] = X[1]
      value[3] = X[2]

      return

    def value_shape(self):
      return (2, 2)

  space = TensorFunctionSpace(mesh, "DG", 0)
  M = interpolate(CellExpression(), space)

  M2 = project(M, TensorFunctionSpace(mesh, "CG", 1), solver_type="lu")
  M2.rename("mesh_metric", "mesh_metric")

  return M2

def refine_metric(M, factor):
  class RefineExpression(Expression):
    def eval(self, value, x):
      value[:] = M(x) * (factor * factor)
      return
    def value_shape(self):
      return (2, 2)

  space = M.function_space()
  M2 = interpolate(RefineExpression(), space)
  name = "mesh_metric_refined_x%.6g" % factor
  M2.rename(name, name)

  return M2

def edge_lengths(M):
  class EdgeLengthExpression(Expression):
    def eval(self, value, x):
      mat = M(x)
      mat.shape = (2, 2)
      evals, evecs = numpy.linalg.eig(mat)
      value[:] = 1.0 / numpy.sqrt(evals)
      return
    def value_shape(self):
      return (2,)
  e = interpolate(EdgeLengthExpression(), VectorFunctionSpace(M.function_space().mesh(), "CG", 1))
  name = "%s_edge_lengths" % M.name()
  e.rename(name, name)

  return e

def adapt(metric, fields=[]):
  if not isinstance(fields, list):
    return adapt(metric, [fields])

  mesh = metric.function_space().mesh()
  space = FunctionSpace(mesh, "CG", 1)
  element = space.ufl_element()

  # Sanity checks
  if not mesh.geometry().dim() == 2 \
        or not element.cell().geometric_dimension() == 2 \
        or not element.cell().topological_dimension() == 2 \
        or not element.family() == "Lagrange" \
        or not element.degree() == 1:
    raise InvalidArgumentException("Require 2D P1 function space for metric tensor field")

  # Create an ordered array of all the node id's in the mesh.  It is
  # not clear to me that we should be going to all this effort. Will
  # we only be adapting parts of the mesh? This would cause problems
  # on the boundary etc. Ask James Maddinson about his reasoning here.
  dof = space.dofmap()
  nodes = set()
  for i in range(mesh.num_cells()):
    cell =  dof.cell_dofs(i)
    for node in cell:
      nodes.add(node)
  nodes = array(sorted(list(nodes)), dtype = numpy.intc)

  # Create the list of cells
  cells = numpy.empty([mesh.num_cells(), 3], dtype = numpy.intc)
  for i in range(mesh.num_cells()):
    cells[i, :] = dof.cell_dofs(i)

  # Gather x and y coordinates
  x = interpolate(Expression("x[0]"), space).vector().gather(nodes)
  y = interpolate(Expression("x[1]"), space).vector().gather(nodes)

  # Create facets and associated data. This is quite slow and a real
  # pain. Need to think harder.
  facets = []
  i = 0
  for cell in cells:
    for pair in [(cell[0], cell[1]), (cell[0], cell[2]), (cell[1], cell[2])]:
      facets.append(pair)
  def facet_cmp(a, b):
    a = min(a[0], a[1]), max(a[0], a[1])
    b = min(b[0], b[1]), max(b[0], b[1])
    if a[0] < b[0]:
      return -1
    elif a[0] > b[0]:
      return 1
    elif a[1] < b[1]:
      return -1
    elif a[1] > b[1]:
      return 1
    else:
      return 0
  facets.sort(cmp = facet_cmp)

  faces = []
  i = 0
  while i < len(facets) - 1:
    j = i + 1
    while j < len(facets) and facet_cmp(facets[i], facets[j]) == 0:
      j += 1
    if i + 1 == j:
      faces.append(facets[i])
    i = j
  assert(len(facets) > 1)
  if not facet_cmp(facets[-2], facets[-1]) == 0:
    faces.append(facets[-1])
  del(facets)

  n = x.shape[0]
  nf_list = numpy.empty([n, 2], dtype = numpy.int)
  nf_list[:] = -1
  for i, face in enumerate(faces):
    if nf_list[face[0], 0] < 0:
      nf_list[face[0], 0] = i
    else:
      assert(nf_list[face[0], 1] < 0)
      nf_list[face[0], 1] = i
    if nf_list[face[1], 0] < 0:
      nf_list[face[1], 0] = i
    else:
      assert(nf_list[face[1], 1] < 0)
      nf_list[face[1], 1] = i

  normals = numpy.empty([n, 2])
  for i, face in enumerate(faces):
    normals[i, 0] = -(y[face[1]] - y[face[0]])
    normals[i, 1] = x[face[1]] - x[face[0]]
    normals[i, :] /= numpy.sqrt(normals[i, 0] ** 2 + normals[i, 1] ** 2)

  colinear_ids = numpy.empty(len(faces), dtype = numpy.intc)
  colinear_ids[:] = -1
  index = 0
  id = 1
  i = 0
  node = -1
  while True:
    colinear_ids[i] = id
    if faces[i][0] == node:
      assert(not faces[i][1] == node)
      node = faces[i][1]
    else:
      node = faces[i][0]
    if nf_list[node, 0] == i:
      assert(not nf_list[node, 1] == i)
      j = nf_list[node, 1]
    else:
      j = nf_list[node, 0]
    dot = normals[i, 0] * normals[j, 0] + normals[i, 1] * normals[j, 1]
    err = abs(abs(dot) - 1.0)
    if err < 1.0e-12:
      if colinear_ids[j] == id:
        seek = True
      else:
        i = j
        seek = False
    else:
      if colinear_ids[j] > 0:
        seek = True
      else:
        id += 1
        i = j
        seek = False
    if seek:
      while index < len(faces) and colinear_ids[index] >= 0:
        index += 1
      if index == len(faces):
        break
      id += 1
      i = index
      node = -1
  assert((colinear_ids >= 0).all())

  sorted_colinear_ids = sorted(colinear_ids)
  assert(len(sorted_colinear_ids) > 0)
  nids = 1
  for i in range(len(faces) - 1):
    if not sorted_colinear_ids[i] == sorted_colinear_ids[i + 1]:
      nids += 1
  info("Found %i co-linear edges" % nids)

  info("Beginning PRAgMaTIc adapt")
  info("Initialising PRAgMaTIc ...")
  NNodes = ctypes.c_int(x.shape[0])
  NElements = ctypes.c_int(cells.shape[0])
  _libpragmatic.pragmatic_2d_init(ctypes.byref(NNodes), 
                                  ctypes.byref(NElements), 
                                  cells.ctypes.data, 
                                  x.ctypes.data, 
                                  y.ctypes.data)

  info("Setting surface ...")
  nfacets = ctypes.c_int(len(faces))
  facets = numpy.empty(2 * nfacets.value, numpy.intc)
  for i in range(nfacets.value):
    facets[i * 2    ] = faces[i][0]
    facets[i * 2 + 1] = faces[i][1]
  boundary_ids = numpy.zeros(nfacets.value, dtype = numpy.intc)
  _libpragmatic.pragmatic_set_surface(ctypes.byref(nfacets),
                                      facets.ctypes.data,
                                      boundary_ids.ctypes.data, 
                                      colinear_ids.ctypes.data)

  info("Setting metric tensor field ...")
  # Dolfin stores the tensor as:
  # |dyy dxy|
  # |dyx dxx|
  metric_arr = numpy.empty(metric.vector().array().size, dtype = numpy.float64)
  for i in range(0, metric.vector().array().size, 4):
    metric_arr[i  ] = metric.vector().array()[i+3]
    metric_arr[i+1] = metric.vector().array()[i+2]
    metric_arr[i+2] = metric.vector().array()[i+1]
    metric_arr[i+3] = metric.vector().array()[i]

  _libpragmatic.pragmatic_set_metric(metric_arr.ctypes.data)

  info("Entering adapt ...")
  _libpragmatic.pragmatic_adapt()

  n_NNodes = ctypes.c_int()
  n_NElements = ctypes.c_int()
  n_NSElements = ctypes.c_int()

  info("Querying output ...")
  _libpragmatic.pragmatic_get_info(ctypes.byref(n_NNodes), 
                                   ctypes.byref(n_NElements),
                                   ctypes.byref(n_NSElements))

  n_enlist = numpy.empty(3 * n_NElements.value, numpy.intc)
  n_x = numpy.empty(n_NNodes.value)
  n_y = numpy.empty(n_NNodes.value)
  info("Extracting output ...")
  _libpragmatic.pragmatic_get_coords_2d(n_x.ctypes.data,
                                        n_y.ctypes.data)
  _libpragmatic.pragmatic_get_elements(n_enlist.ctypes.data)

  info("Finalising PRAgMaTIc ...")
  _libpragmatic.pragmatic_finalize()
  info("PRAgMaTIc adapt complete")

  n_mesh = Mesh()
  ed = MeshEditor()
  ed.open(n_mesh, 2, 2)
  ed.init_vertices(n_NNodes.value)
  for i in range(n_NNodes.value):
    ed.add_vertex(i, n_x[i], n_y[i])
  ed.init_cells(n_NElements.value)
  for i in range(n_NElements.value):
    ed.add_cell(i, n_enlist[i * 3], n_enlist[i * 3 + 1], n_enlist[i * 3 + 2])
  ed.close()

  # Sanity check to be deleted or made optional
  n_space = FunctionSpace(n_mesh, "CG", 1)

  area = assemble(Constant(1.0) * dx, mesh = mesh)
  n_area = assemble(Constant(1.0) * dx, mesh = n_mesh)
  err = abs(area - n_area)
  info("Donor mesh area : %.17e" % area)
  info("Target mesh area: %.17e" % n_area)
  info("Change          : %.17e" % err)
  info("Relative change : %.17e" % (err / area))
  # assert(err < 2.0e-13 * area)

  n_fields = []
  for field in fields:
    n_field = Function(n_space)
    n_field.rename(field.name(), field.name())
    val = numpy.empty(1)
    coord = numpy.empty(2)
    nx = interpolate(Expression("x[0]"), n_space).vector().array()
    ny = interpolate(Expression("x[1]"), n_space).vector().array()
    n_field_arr = numpy.empty(n_NNodes.value)
    for i in range(n_NNodes.value):
      coord[0] = nx[i]
      coord[1] = ny[i]
      field.eval(val, coord)
      n_field_arr[i] = val
    n_field.vector().set_local(n_field_arr)
    n_field.vector().apply("insert")
    n_fields.append(n_field)

  if len(n_fields) > 0:
    return n_fields
  else:
    return n_mesh

# p-norm scaling to the metric, as in Chen, Sun and Xu, Mathematics of
# Computation, Volume 76, Number 257, January 2007, pp. 179-204.
def metric_pnorm(f, mesh, eta, sigma=1.0e-6, p=2):
  # Sanity checks
  n = mesh.geometry().dim()
  if not n == 2:
    raise InvalidArgumentException("Currently only 2D is supported")

  element = f.function_space().ufl_element()
  if not element.family() == "Lagrange" \
        or not element.degree() == 2:
    raise InvalidArgumentException("Require Lagrange P2 function spaces")

  gradf = project(grad(f), VectorFunctionSpace(mesh, "DG", 1))
  H = project(grad(gradf), TensorFunctionSpace(mesh, "DG", 0))
  
  # Make H positive definite and calculate the p-norm.
  space = H.function_space()
  cbig=numpy.zeros((H.vector().array()).size)

  for i in range(mesh.num_cells()):
    indold = space.dofmap().cell_dofs(i)
    ind = numpy.array(indold)
    
    # Enforce symmetry
    ind[1]=ind[2]
    
    v,w=linalg.eig(numpy.matrix(H.vector().gather(ind).reshape(2,2)))
    
    diags=numpy.diag(abs(v))

    temph=w*diags*w.T # + sigma*numpy.identity(2)

    # Deal with zero eigenvalues.
    if linalg.det(temph) == 0:
      if v[0]<v[1]:
        v[0] = 0.1*v[1]
      elif v[0]>v[1]:
        v[1] = 0.1*v[0]
      else:
        v = sigma
      diags=numpy.diag(v)
      temph=w*diags*w.T    
    temph=1./eta*(linalg.det(temph)**(-1.0/(2*p + n)))*temph
    # HACK!
    # temph[1,1] = 1.0

    cbig[indold]=temph.reshape(1,4)
  H.vector().set_local(cbig)

  Mp = project(H, TensorFunctionSpace(mesh, "CG", 1))
  return Mp

if __name__=="__main__":
  from mpi4py import MPI
  import sys

  comm = MPI.COMM_WORLD

  mesh = UnitSquareMesh(50, 50)
  V = FunctionSpace(mesh, "CG", 2)
  f = interpolate(Expression("0.1*sin(50.*(2*x[0]-1)) + atan2(-0.1, (2.0*(2*x[0]-1) - sin(5.*(2*x[1]-1))))"), V)
  #f = interpolate(Expression("pow(x[0]-0.5, 2)+pow(x[1]-0.5, 2)"), V)
  #f = interpolate(Expression("pow(x[0]-0.5, 2)"), V)

  #Mp = metric_pnorm(f, mesh, 0.001)
  Mp = refine_metric(mesh_metric(mesh), 0.5)

  new_mesh = adapt(Mp)

  # plot(Mp[0,0])
  # from IPython import embed
  # embed()

  # plot(new_f[0], title="adapted mesh")
  # plot(new_f[0].function_space().mesh(), title="adapted mesh")
  plot(mesh, title="old mesh")
  plot(new_mesh, title="adapted mesh")
  interactive()
