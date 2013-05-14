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
# Many thanks to James Maddinson for the original version of this.
#
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
from numpy import array

from dolfin import *

__all__ = ["_libpragmatic",
           "pragmatic_begin",
           "pragmatic_add_field",
           "pragmatic_set_surface",
           "pragmatic_set_metric",
           "pragmatic_adapt",
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

__pragmatic_dimension = -1

def pragmatic_begin(NNodes, NElements, enlist, x, y, z=None):
  """ Initialise pragmatic with mesh to be adapted. pragmatic_end must
  be called before this can be called again, i.e. cannot adapt
  multiple meshes at the same time.
  """
  ctype_NNodes = ctypes.c_int(NNodes)
  ctype_NElements = ctypes.c_int(NElements)
  if z:
    _libpragmatic.pragmatic_3d_begin(ctypes.byref(ctype_NNodes),
                                     ctypes.byref(ctype_NElements),
                                     enlist.ctypes.data,
                                     x.ctypes.data,
                                     y.ctypes.data,
                                     z.ctypes.data)
    __pragmatic_dimension = 3
  else:
    _libpragmatic.pragmatic_2d_begin(ctypes.byref(ctype_NNodes),
                                     ctypes.byref(ctype_NElements),
                                     enlist.ctypes.data,
                                     x.ctypes.data,
                                     y.ctypes.data)
    __pragmatic_dimension = 2
  return

def pragmatic_add_field(psi, error, pnorm=-1):
  """ Add field which should be adapted to. The optional argument
    pnorm applies the p-norm scaling to the metric, as in Chen, Sun
    and Xu, Mathematics of Computation, Volume 76, Number 257, January
    2007. Default (-1) specifies the absolute error measure.
  """
  ctype_error = ctypes.c_double(error)
  ctype_pnorm = ctypes.c_int(pnorm)
  _libpragmatic.pragmatic_add_field(psi.ctypes.data,
                                    ctypes.byref(ctype_error),
                                    ctypes.byref(ctype_pnorm))
  return

def pragmatic_set_surface(nfacets, facets, boundary_ids, coplanar_ids):
  """ Set the surface boundary.

  """
  ctype_nfacets = ctypes.c_int(nfacets)
  _libpragmatic.pragmatic_set_surface(ctypes.byref(ctype_nfacets),
                                      facets.ctypes.data,
                                      boundary_ids.ctypes.data,
                                      coplanar_ids.ctypes.data)
  return

def pragmatic_set_metric(metric):
  """ Set the node centred metric field
  """
  _libpragmatic.pragmatic_set_metric(metric.ctypes.data)

  return

def pragmatic_adapt():
  """ Adapt the mesh.
  """
  _libpragmatic.pragmatic_adapt()

  # Get information about the new size of the mesh.
  NNodes = ctypes.c_int()
  NElements = ctypes.c_int()
  NSElements = ctypes.c_int()

  _libpragmatic.pragmatic_get_info(ctypes.byref(NNodes),
                                   ctypes.byref(NElements), 
                                   ctypes.byref(NSElements))

  # Get out the new mesh.
  if __pragmatic_dimension == 2:
    x = numpy.empty(NNodes)
    y = numpy.empty(NNodes)
    _libpragmatic.pragmatic_get_coords_2d(x.ctypes.data,
                                          y.ctypes.data)

    enlist = numpy.empty(NElements*3)
    _libpragmatic.pragmatic_get_elements(elements.ctypes.data)

    facets = numpy.empty(NElements*2)
    boundary_ids = numpy.empty(NElements)
    coplanar_ids = numpy.empty(NElements)
    _libpragmatic.pragmatic_get_surface(facets.ctypes.data,
                                        boundary_ids.ctypes.data,
                                        coplanar_ids.ctypes.data)

    return x, y, enlist, facets, boundary_ids, coplanar_ids
  else:
    assert(False)

  return

def pragmatic_end():
  """ Free up pragmatic data structures.
  """
  _libpragmatic.pragmatic_end()
  return

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

  space = TensorFunctionSpace(mesh, "CG", 1)
  M2 = Function(space)
  M2.rename("mesh_metric", "mesh_metric")

  test, trial = TestFunction(space), TrialFunction(space)
  solver = LUSolver()
  solver.solve(assemble(inner(test, trial) * dx), M2.vector(), assemble(inner(test, M) * dx))

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

def adapt(fields, eps, gradation = None, bounds = None):
  if isinstance(fields, Function):
    return adapt([fields], [eps], gradation = gradation, bounds = bounds)[0]

  if not isinstance(fields, list):
    raise InvalidArgumentException("fields must be a Function or a list of Function s")
  if not isinstance(eps, list):
    raise InvalidArgumentException("eps must be a non-negative float or a list of non-negative floats or None")
  elif not len(eps) == len(fields):
    raise InvalidArgumentException("Invalid length for eps")
  for field in fields:
    if not isinstance(field, Function):
      raise InvalidArgumentException("fields must be a Function or a list of Function s")
  neps = 0
  for e in eps:
    if not e is None:
      if not isinstance(e, float) or e <= 0.0:
        raise InvalidArgumentException("eps must be a non-negative float a list of non-negative floats or None")
      neps += 1
  if neps == 0:
    raise InvalidArgumentException("eps must contain at least one non-negative float")

  if not gradation is None:
    if not isinstance(gradation, float) or gradation <= 0.0:
      raise InvalidArgumentException("gradation must be a non-negative float")
  if not bounds is None:
    if not isinstance(bounds, (list, tuple)) or not len(bounds) == 2 \
      or not isinstance(bounds[0], float) or not isinstance(bounds[1], float) \
      or bounds[0] <= 0.0 or bounds[1] <= 0.0:
      raise InvalidArgumentException("bounds must be a list of 2 non-negative floats")

  if len(fields) == 0:
    raise InvalidArgumentException("Require at least one field")
  space = fields[0].function_space()
  mesh = space.mesh()
  e = space.ufl_element()
  if not mesh.geometry().dim() == 2 \
    or not e.cell().geometric_dimension() == 2 or not e.cell().topological_dimension() == 2 \
    or not e.family() == "Lagrange" or not e.degree() == 1:
    raise InvalidArgumentException("Require 2D P1 function spaces")
  for field in fields[1:]:
    e = space.ufl_element()
    if not mesh.geometry().dim() == 2 \
      or not e.cell().geometric_dimension() == 2 or not e.cell().topological_dimension() == 2 \
      or not e.family() == "Lagrange" or not e.degree() == 1:
      raise InvalidArgumentException("Require 2D P1 function spaces")

  dof = space.dofmap()
  nodes = set()
  for i in range(mesh.num_cells()):
    cell =  dof.cell_dofs(i)
    for node in cell:
      nodes.add(node)
  nodes = array(sorted(list(nodes)), dtype = numpy.intc)

  cells = numpy.empty([mesh.num_cells(), 3], dtype = numpy.intc)
  for i in range(mesh.num_cells()):
    cells[i, :] = dof.cell_dofs(i)

  x = interpolate(Expression("x[0]"), space).vector().gather(nodes)
  y = interpolate(Expression("x[1]"), space).vector().gather(nodes)
  n = x.shape[0]

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

  NNodes = ctypes.c_int(n)
  NElements = ctypes.c_int(cells.shape[0])

  info("Beginning PRAgMaTIc adapt")
  info("Initialising PRAgMaTIc ...")
  _libpragmatic.cpragmatic_initialise_2d(ctypes.byref(NNodes), ctypes.byref(NElements), cells.ctypes.data, x.ctypes.data, y.ctypes.data)

  nfacets = ctypes.c_int(len(faces))
  facets = numpy.empty(2 * nfacets.value, numpy.intc)
  for i in range(nfacets.value):
    facets[i * 2    ] = faces[i][0]
    facets[i * 2 + 1] = faces[i][1]
  boundary_ids = numpy.zeros(nfacets.value, dtype = numpy.intc)
  info("Setting surface ...")
  _libpragmatic.cpragmatic_set_surface(ctypes.byref(nfacets), facets.ctypes.data, boundary_ids.ctypes.data, colinear_ids.ctypes.data)

  for field, e in zip(fields, eps):
    if not e is None:
      field_arr = field.vector().array()
      error = ctypes.c_double(e)
      pnorm = ctypes.c_int(-1)
      info("Adding field %s ..." % field.name())
      _libpragmatic.cpragmatic_metric_add_field(field_arr.ctypes.data, ctypes.byref(error), ctypes.byref(pnorm))

  if not bounds is None:
    min_len = ctypes.c_double(min(bounds[0], bounds[1]))
    max_len = ctypes.c_double(max(bounds[0], bounds[1]))
    info("Bounding metric...")
    _libpragmatic.cpragmatic_apply_metric_bounds(ctypes.byref(min_len), ctypes.byref(max_len))

  if not gradation is None:
    gradation = ctypes.c_double(gradation)
    info("Applying metric gradation ...")
    _libpragmatic.cpragmatic_apply_metric_gradation(ctypes.byref(gradation))

  smooth = ctypes.c_int(0)
  info("Entering adapt ...")
  _libpragmatic.cpragmatic_adapt(ctypes.byref(smooth))

  n_NNodes = ctypes.c_int()
  n_NElements = ctypes.c_int()
  info("Querying output ...")
  _libpragmatic.cpragmatic_query_output(ctypes.byref(n_NNodes), ctypes.byref(n_NElements))

  n_enlist = numpy.empty(3 * n_NElements.value, numpy.intc)
  n_x = numpy.empty(n_NNodes.value)
  n_y = numpy.empty(n_NNodes.value)
  info("Extracting output ...")
  _libpragmatic.cpragmatic_get_output_2d(n_enlist.ctypes.data, n_x.ctypes.data, n_y.ctypes.data)

  info("Finalising PRAgMaTIc ...")
  _libpragmatic.cpragmatic_finalise()
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

  n_space = FunctionSpace(n_mesh, "CG", 1)

  area = assemble(Constant(1.0) * dx, mesh = mesh)
  n_area = assemble(Constant(1.0) * dx, mesh = n_mesh)
  err = abs(area - n_area)
  info("Donor mesh area : %.17e" % area)
  info("Target mesh area: %.17e" % n_area)
  info("Change          : %.17e" % err)
  info("Relative change : %.17e" % (err / area))
  assert(err < 2.0e-13 * area)

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
    n_field.vector().set_local(n_field_arr);  n_field.vector().apply("insert")
    n_fields.append(n_field)

  return n_fields

if __name__=="__main__":
  # Create a dolfin mesh
  mesh = UnitSquareMesh(4, 4)

  # Import dolfin mesh into pragmatic
  enlist_list=[]
  for c in cells(mesh):
    for v in vertices(c):
      enlist_list.append(v.index())
      print enlist_list
      enlist = array(enlist_list, dtype=numpy.int)

  x_list = []
  y_list = []
  for v in vertices(mesh):
    x_list.append(mesh.coordinates()[v.index()][0])
    y_list.append(mesh.coordinates()[v.index()][1])
  x = array(x_list, dtype=numpy.float64)
  y = array(y_list, dtype=numpy.float64)

  # Initialise pragmatic
  pragmatic_begin(mesh.num_vertices(), mesh.num_cells(), enlist, x, y)

  # End pragmatic
  pragmatic_end()

  new_mesh = Mesh()
  editor = MeshEditor()
  editor.open(new_mesh, mesh.topology().dim(), mesh.geometry().dim())
  editor.init_vertices(mesh.num_vertices())
  editor.init_cells(mesh.num_cells())
  for c in cells(mesh):
    editor.add_cell(c.index(), array([v.index() for v in vertices(c)], dtype="uintp"))
    
  for v in vertices(mesh):
    editor.add_vertex(v.index(), mesh.coordinates()[v.index()])

  editor.close()

  #plot(mesh, title="Old mesh")
  #plot(new_mesh, title="New mesh")
  #interactive()

  V = TensorFunctionSpace(mesh, "CG", 1)
  id = interpolate(Expression((("1", "0"), ("0.0", "1.0"))), V)
  
  dofmap = V.dofmap()
  print "mesh.num_vertices(): ", mesh.num_vertices()
  vmap = dofmap.vertex_to_dof_map(mesh)
  dmap = dofmap.dof_to_vertex_map(mesh)
  print "len(vmap): ", len(vmap)
  print id.vector().array()[vmap[0:4]]
  print id(mesh.coordinates()[0])
