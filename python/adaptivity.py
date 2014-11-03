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
# Patrick Farrell    for mesh based metric used for coarsening.
# James Maddinson    for the original version of Dolfin interface.
# Davide Longoni     for p-norm function.
# Kristian E. Jensen for ellipse function, test cases, vectorization opt., 3D glue and gradation

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

import numpy, scipy.sparse, scipy.sparse.linalg
from numpy import array, zeros, ones, any, arange
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
#  _libpragmatic = ctypes.cdll.LoadLibrary("/home/kjensen/projects/pragmatic/src/.libs/libpragmatic.so")
  _libpragmatic = ctypes.cdll.LoadLibrary("libpragmatic.so")
except:
  raise LibraryException("Failed to load libpragmatic.so")

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

def polyhedron_surfmesh(mesh):
    cells = mesh.cells()
    coords = mesh.coordinates()
    #this function calculates a surface mesh assuming a polyhedral geometry, i.e. not suitable for
    #curved geometries and the output will have to be modified for problems colinear faces.
    #a surface mesh is required for the adaptation, so this function is called, if no surface mesh
    #is provided by the user, but the user can load this function herself, use it, modify the output
    #and provide the modified surface mesh to adapt()
    ntri = len(cells)
    v1 = coords[cells[:,1],:]-coords[cells[:,0],:]
    v2 = coords[cells[:,2],:]-coords[cells[:,0],:]
    v3 = coords[cells[:,3],:]-(coords[cells[:,0],:]+coords[cells[:,1],:]+coords[cells[:,2],:])/3.
    crossprod = array([v1[:,1]*v2[:,2]-v1[:,2]*v2[:,1], \
                       v1[:,2]*v2[:,0]-v1[:,0]*v2[:,2], \
                       v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]]).T
    badcells = (crossprod*v3).sum(1)>0
    tri = cells.flatten()
    R = range(0,4*ntri,4); m1 = ones(ntri,dtype=numpy.int64)
    tri = array([tri[R+badcells],tri[R+m1-badcells],tri[R+2*m1],tri[R+3*m1]])
    fac = array([tri[0],tri[1],tri[2],tri[3],\
                 tri[1],tri[0],tri[3],tri[2],\
                 tri[3],tri[2],tri[1],tri[0]]).reshape([3,ntri*4])
    #putting large node number in later row, smaller in first
    C = fac.argsort(0)
    Cgood = (C[0,:] == 0)*(C[0,:] == 1)+(C[1,:] == 1)*(C[0,:] == 2)+(C[1,:] == 2)*(C[0,:] == 0)
    Cinv  = Cgood==False
    R = arange(ntri*4)
    fac = fac.transpose().flatten()
    fac = array([fac[R*3+C[0,:]],fac[R*3+C[1,:]],fac[R*3+C[2,:]]])
    #sort according to first node number (with fall back to second node number)
    I2 = numpy.argsort(array(zip(fac[0,:],fac[1,:],fac[2,:]),dtype=[('e1',int),('e2',int),('e3',int)]),order=['e1','e2','e3'])
    fac = fac[:,I2]
    #find unique faces
    d = array([any(fac[:,0] != fac[:,1])] +\
         (any((fac[:,range(1,ntri*4-1)] != fac[:,range(2,ntri*4)]),0) *\
          any((fac[:,range(1,ntri*4-1)] != fac[:,range(0,ntri*4-2)]),0)).tolist() +\
              [any(fac[:,ntri*4-1] != fac[:,ntri*4-2])])
    fac = fac[:,d]
    #rearrange face to correct orientation
    R = arange(sum(d))
    fac = fac.transpose().flatten()
    bfaces = array([fac[R*3+Cgood[I2[d]]],fac[R*3+Cinv[I2[d]]],fac[R*3+2]]).transpose()
    # compute normal vector (n)
    v1 = coords[bfaces[:,1],:]-coords[bfaces[:,0],:]
    v2 = coords[bfaces[:,2],:]-coords[bfaces[:,0],:]
    n = array([v1[:,1]*v2[:,2]-v1[:,2]*v2[:,1], \
               v1[:,2]*v2[:,0]-v1[:,0]*v2[:,2], \
               v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]]).T
    n = n/numpy.sqrt(n[:,0]**2+n[:,1]**2+n[:,2]**2).repeat(3).reshape([len(n),3])
    # compute sets of co-linear faces (this is specific to polyhedral geometries)
    IDs = zeros(len(n), dtype = numpy.intc)
    while True:
        nn = IDs.argmin()
        IDs[nn] = IDs.max() + 1
        I = arange(0,len(IDs))
        notnset = I != nn * ones(len(I),dtype=numpy.int64)
        dists = abs(n[notnset,0]*(coords[bfaces[notnset,0],0] - ones(sum(notnset))*coords[bfaces[nn,0],0])+\
                    n[notnset,1]*(coords[bfaces[notnset,0],1] - ones(sum(notnset))*coords[bfaces[nn,0],1])+\
                    n[notnset,2]*(coords[bfaces[notnset,0],2] - ones(sum(notnset))*coords[bfaces[nn,0],2])) < 1e-12
        angles = ones(sum(notnset))-abs(n[notnset,0]*n[nn,0]+n[notnset,1]*n[nn,1]+n[notnset,2]*n[nn,2])<1e-12 # angles = arccos(abs(t[notnset,0]*t[n,0]+t[notnset,1]*t[n,1]))<1e-12
        IDs[I[notnset][angles*dists]] = IDs[nn]
        if all(IDs != zeros(len(IDs),dtype=numpy.int64)):
            info("Found %i co-linear faces" % IDs.max())
            break
    
    #compatibility fixes
    IDs += 1
    bfaces_pair = zip(bfaces[:,0],bfaces[:,1],bfaces[:,2])
    return [bfaces,IDs]

def polygon_surfmesh(mesh):
    cells = mesh.cells()
    coords = mesh.coordinates()
    #this function calculates a surface mesh assuming a polygonal geometry, i.e. not suitable for
    #curved geometries and the output will have to be modified for problems colinear faces.
    #a surface mesh is required for the adaptation, so this function is called, if no surface mesh
    #is provided by the user, but the user can load this function herself, use it, modify the output
    #and provide the modified surface mesh to adapt()
    ntri = len(cells)
    v1 = coords[cells[:,1],:]-coords[cells[:,0],:]
    v2 = coords[cells[:,2],:]-coords[cells[:,0],:]
    badcells = v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]>0
    tri = cells.flatten()
    R = range(0,3*ntri,3); m1 = ones(ntri,dtype=numpy.int64)
    tri = array([tri[R+badcells],tri[R+m1-badcells],tri[R+2*m1]])
    edg = array([tri[0],tri[1],tri[2],\
                 tri[1],tri[2],tri[0]]).reshape([2,ntri*3])
    #putting large node number in later row
    C = edg.argsort(0)
    R = arange(ntri*3)
    edg = edg.transpose().flatten()
    edg = array([edg[R*2+C[0,:]],edg[R*2+C[1,:]]])
    #sort according to first node number (with fall back to second node number)
    I2 = numpy.argsort(array(zip(edg[0,:],edg[1,:]),dtype=[('e1',int),('e2',int)]),order=['e1','e2'])
    edg = edg[:,I2] 
    #find unique edges
    d = array([any(edg[:,0] != edg[:,1])] +\
         (any((edg[:,range(1,ntri*3-1)] != edg[:,range(2,ntri*3)]),0) *\
          any((edg[:,range(1,ntri*3-1)] != edg[:,range(0,ntri*3-2)]),0)).tolist() +\
              [any(edg[:,ntri*3-1] != edg[:,ntri*3-2])])
    edg = edg[:,d]
    #put correct node number back in later row
    R = arange(sum(d))
    edg = edg.transpose().flatten()
    bfaces = array([edg[R*2+C[0,I2[d]]],edg[R*2+C[1,I2[d]]]]).transpose()
    # compute normalized tangent (t)
    t = coords[bfaces[:,0],:]-coords[bfaces[:,1],:]
    t = t/numpy.sqrt(t[:,0]**2+t[:,1]**2).repeat(2).reshape([len(t),2])
    # compute sets of co-linear edges (this is specific to polygonal geometries)
    IDs = zeros(len(t), dtype = numpy.intc)
    while True:
        n = IDs.argmin()
        IDs[n] = IDs.max() + 1
        I = arange(0,len(IDs))
        notnset = I != n * ones(len(I),dtype=numpy.int64)
        dists = abs(t[notnset,1]*(coords[bfaces[notnset,0],0] - ones(sum(notnset))*coords[bfaces[n,0],0])-\
                    t[notnset,0]*(coords[bfaces[notnset,0],1] - ones(sum(notnset))*coords[bfaces[n,0],1])) < 1e-12
        angles = ones(sum(notnset))-abs(t[notnset,0]*t[n,0]+t[notnset,1]*t[n,1])<1e-12 # angles = arccos(abs(t[notnset,0]*t[n,0]+t[notnset,1]*t[n,1]))<1e-12
        IDs[I[notnset][angles*dists]] = IDs[n]
        if all(IDs != zeros(len(IDs),dtype=numpy.int64)):
            info("Found %i co-linear edges" % IDs.max())
            break
    #compatibility fixes
    IDs += 1
    #bfaces_pair = zip(bfaces[:,0],bfaces[:,1])
    return [bfaces,IDs]

def set_mesh(n_xy, n_enlist, mesh=None, dx=None, debugon=False):
  #this function generates a mesh 2D DOLFIN mesh given coordinates(nx,ny) and cells(n_enlist).
  #it is used in the adaptation, but can also be used in the context of debugging, i.e. if one
  #one saves the mesh coordinates and cells using pickle between iterations.
  startTime = time()
  nvtx = n_xy.shape[1]
  n_mesh = Mesh()
  ed = MeshEditor()
  ed.open(n_mesh, len(n_xy), len(n_xy))
  ed.init_vertices(nvtx) #n_NNodes.value
  if len(n_xy) == 1:
   for i in range(nvtx):
    ed.add_vertex(i, n_xy[0,i])
   ed.init_cells(len(n_enlist)/2)
   for i in range(len(n_enlist)/2): #n_NElements.value
    ed.add_cell(i, n_enlist[i * 2], n_enlist[i * 2 + 1])
  elif len(n_xy) == 2:
   for i in range(nvtx): #n_NNodes.value  
     ed.add_vertex(i, n_xy[0,i], n_xy[1,i])
   ed.init_cells(len(n_enlist)/3) #n_NElements.value
   for i in range(len(n_enlist)/3): #n_NElements.value
     ed.add_cell(i, n_enlist[i * 3], n_enlist[i * 3 + 1], n_enlist[i * 3 + 2])
  else: #3D
   for i in range(nvtx): #n_NNodes.value  
     ed.add_vertex(i, n_xy[0,i], n_xy[1,i], n_xy[2,i])
   ed.init_cells(len(n_enlist)/4) #n_NElements.value
   for i in range(len(n_enlist)/4): #n_NElements.value
     ed.add_cell(i, n_enlist[i * 4], n_enlist[i * 4 + 1], n_enlist[i * 4 + 2], n_enlist[i * 4 + 3])
  ed.close()
  info("mesh definition took %0.1fs (not vectorized)" % (time()-startTime))
  if debugon==True and dx is not None:
    # Sanity check to be deleted or made optional
    area = assemble(interpolate(Constant(1.0),FunctionSpace(mesh,'DG',0)) * dx)
    n_area = assemble(interpolate(Constant(1.0),FunctionSpace(n_mesh,'DG',0)) * dx)
    err = abs(area - n_area)
    info("Donor mesh area : %.17e" % area)
    info("Target mesh area: %.17e" % n_area)
    info("Change          : %.17e" % err)
    info("Relative change : %.17e" % (err / area))
    
    assert(err < 2.0e-11 * area)
  return n_mesh
  
  
def impose_maxN(metric, maxN):
    gdim = metric.function_space().ufl_element().cell().geometric_dimension()
    targetN = assemble(sqrt(det(metric))*dx)
    fak = 1.
    if targetN > maxN:
      fak = (targetN/maxN)**(gdim/2)
      metric.vector().set_local(metric.vector().array()/fak)
      info('metric coarsened to meet target node number')
    return [metric,fak]
      
def adapt(metric, bfaces=None, bfaces_IDs=None, debugon=True, eta=1e-2, grada=False, maxN=None):
  #this is the actual adapt function. It currently works with vertex 
  #numbers rather than DOF numbers. 
  mesh = metric.function_space().mesh()
  
  #check if input is not a metric
  if metric.function_space().ufl_element().num_sub_elements() == 0:
     metric = metric_pnorm(metric, eta=eta, CG1out=True)
  
  if metric.function_space().ufl_element().degree() == 0 and metric.function_space().ufl_element().family()[0] == 'D':
   metric = project(metric,  TensorFunctionSpace(mesh, "CG", 1)) #metric = logproject(metric)
  metric = fix_CG1_metric(metric) #fixes negative eigenvalues
  if grada is not None:
      metric = gradate(metric,grada)
  if maxN is not None:
      [metric,fak] = impose_maxN(metric, maxN)
  
  # warn before generating huge mesh
  targetN = assemble(sqrt(det(metric))*dx)
  if targetN < 1e6:
    info("target mesh has %0.0f nodes" % targetN)  
  else:
    warning("target mesh has %0.0f nodes" % targetN)  
    
  space = metric.function_space() #FunctionSpace(mesh, "CG", 1)
  element = space.ufl_element()

  # Sanity checks
  if not (mesh.geometry().dim() == 2 or mesh.geometry().dim() == 3)\
        or not (element.cell().geometric_dimension() == 2 \
        or element.cell().geometric_dimension() == 3) \
        or not (element.cell().topological_dimension() == 2 \
        or element.cell().topological_dimension() == 3) \
        or not element.family() == "Lagrange" \
        or not element.degree() == 1:
    raise InvalidArgumentException("Require 2D P1 function space for metric tensor field")
  
  nodes = array(range(0,mesh.num_vertices()),dtype=numpy.intc) 
  cells = mesh.cells()
  coords = mesh.coordinates()
  # create boundary mesh and associated list of co-linear edges
  if bfaces is None:
    if element.cell().geometric_dimension() == 2:
      [bfaces,bfaces_IDs] = polygon_surfmesh(mesh)
    else:
      [bfaces,bfaces_IDs] = polyhedron_surfmesh(mesh)
    
  x = coords[nodes,0]
  y = coords[nodes,1]
  if element.cell().geometric_dimension() == 3:
    z = coords[nodes,2]
  cells = array(cells,dtype=numpy.intc)
  
  # Dolfin stores the tensor as:
  # |dxx dxy|
  # |dyx dyy|
    ## THE (CG1-)DOF NUMBERS ARE DIFFERENT FROM THE VERTEX NUMBERS (and we wish to work with the former)
  if dolfin.__version__ != '1.2.0':
      dof2vtx = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
  else:
      dof2vtx = FunctionSpace(mesh,'CG',1).dofmap().vertex_to_dof_map(mesh).argsort()
  
  metric_arr = numpy.empty(metric.vector().array().size, dtype = numpy.float64)
  if element.cell().geometric_dimension() == 2:
    metric_arr[range(0,metric.vector().array().size,4)] = metric.vector().array()[arange(0,metric.vector().array().size,4)[dof2vtx]]
    metric_arr[range(1,metric.vector().array().size,4)] = metric.vector().array()[arange(2,metric.vector().array().size,4)[dof2vtx]]
    metric_arr[range(2,metric.vector().array().size,4)] = metric.vector().array()[arange(2,metric.vector().array().size,4)[dof2vtx]]
    metric_arr[range(3,metric.vector().array().size,4)] = metric.vector().array()[arange(3,metric.vector().array().size,4)[dof2vtx]]
  else:
    metric_arr[range(0,metric.vector().array().size,9)] = metric.vector().array()[arange(0,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(1,metric.vector().array().size,9)] = metric.vector().array()[arange(3,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(2,metric.vector().array().size,9)] = metric.vector().array()[arange(6,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(3,metric.vector().array().size,9)] = metric.vector().array()[arange(3,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(4,metric.vector().array().size,9)] = metric.vector().array()[arange(4,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(5,metric.vector().array().size,9)] = metric.vector().array()[arange(7,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(6,metric.vector().array().size,9)] = metric.vector().array()[arange(6,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(7,metric.vector().array().size,9)] = metric.vector().array()[arange(7,metric.vector().array().size,9)[dof2vtx]]
    metric_arr[range(8,metric.vector().array().size,9)] = metric.vector().array()[arange(8,metric.vector().array().size,9)[dof2vtx]]
  info("Beginning PRAgMaTIc adapt")
  info("Initialising PRAgMaTIc ...")
  NNodes = ctypes.c_int(x.shape[0])
  
  NElements = ctypes.c_int(cells.shape[0])
  
  if element.cell().geometric_dimension() == 2:
      _libpragmatic.pragmatic_2d_init(ctypes.byref(NNodes), 
                                  ctypes.byref(NElements), 
                                  cells.ctypes.data, 
                                  x.ctypes.data, 
                                  y.ctypes.data)
  else:
      _libpragmatic.pragmatic_3d_init(ctypes.byref(NNodes), 
                                  ctypes.byref(NElements), 
                                  cells.ctypes.data, 
                                  x.ctypes.data, 
                                  y.ctypes.data, 
                                  z.ctypes.data)
  info("Setting surface ...")
  nfacets = ctypes.c_int(len(bfaces))
  facets = array(bfaces.flatten(),dtype=numpy.intc)
  
  _libpragmatic.pragmatic_set_boundary(ctypes.byref(nfacets),
                                       facets.ctypes.data,
                                       bfaces_IDs.ctypes.data)
  
  info("Setting metric tensor field ...")
  _libpragmatic.pragmatic_set_metric(metric_arr.ctypes.data)
  
  info("Entering adapt ...")
  startTime = time()
  _libpragmatic.pragmatic_adapt()
  
  info("adapt took %0.1fs" % (time()-startTime))
  n_NNodes = ctypes.c_int()
  n_NElements = ctypes.c_int()
  n_NSElements = ctypes.c_int()

  info("Querying output ...")
  _libpragmatic.pragmatic_get_info(ctypes.byref(n_NNodes), 
                                   ctypes.byref(n_NElements),
                                   ctypes.byref(n_NSElements))
  
  if element.cell().geometric_dimension() == 2:
      n_enlist = numpy.empty(3 * n_NElements.value, numpy.intc)
  else:
      n_enlist = numpy.empty(4 * n_NElements.value, numpy.intc)
  
  info("Extracting output ...")
  n_x = numpy.empty(n_NNodes.value)
  n_y = numpy.empty(n_NNodes.value)
  if element.cell().geometric_dimension() == 3:
      n_z = numpy.empty(n_NNodes.value)    
      _libpragmatic.pragmatic_get_coords_3d(n_x.ctypes.data,
                                            n_y.ctypes.data,
                                            n_z.ctypes.data)
  else:
      _libpragmatic.pragmatic_get_coords_2d(n_x.ctypes.data,
                                        n_y.ctypes.data)
                                        
  _libpragmatic.pragmatic_get_elements(n_enlist.ctypes.data)

  info("Finalising PRAgMaTIc ...")
  _libpragmatic.pragmatic_finalize()
  info("PRAgMaTIc adapt complete")
  
  if element.cell().geometric_dimension() == 2:
      n_mesh = set_mesh(array([n_x,n_y]),n_enlist,mesh=mesh,dx=dx,debugon=debugon)
  else:
      n_mesh = set_mesh(array([n_x,n_y,n_z]),n_enlist,mesh=mesh,dx=dx,debugon=debugon)
  
  return n_mesh

def consistent_interpolation(mesh, fields=[]):
  if not isinstance(fields, list):
    return consistent_interpolation(mesh, [fields])

  n_space = FunctionSpace(n_mesh, "CG", 1)
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

def fix_CG1_metric(Mp):
 #this function makes the eigenvalues of a metric positive (this property is
 #lost during the projection step)
 [H,cell2dof] = get_dofs(Mp)
 [eigL,eigR] = analytic_eig(H)
# if any(lambda1<zeros(len(lambda2))) or any(lambda2<zeros(len(lambda2))):
#  warning('negative eigenvalue in metric fixed')
 eigL = numpy.abs(eigL)
 H = analyt_rot(fulleig(eigL),eigR)
 out = sym2asym(H).transpose().flatten()
 Mp.vector().set_local(out)
 return Mp   

# p-norm scaling to the metric, as in Chen, Sun and Xu, Mathematics of
# Computation, Volume 76, Number 257, January 2007, pp. 179-204.
# the DG0 hessian can be extracted in three different ways (controlled CG0H option)
def metric_pnorm(f, eta, max_edge_length=None, min_edge_length=None, max_edge_ratio=10, p=2, CG1out=False, CG0H=3):
  mesh = f.function_space().mesh()
  # Sanity checks
  if max_edge_ratio is not None and max_edge_ratio < 1.0:
    raise InvalidArgumentException("The maximum edge ratio must be greater greater or equal to 1")
  else:
    if max_edge_ratio is not None:
     max_edge_ratio = max_edge_ratio**2 # ie we are going to be looking at eigenvalues

  n = mesh.geometry().dim()
 
  if f.function_space().ufl_element().degree() == 2 and f.function_space().ufl_element().family() == 'Lagrange':
     if CG0H == 0:
        S = VectorFunctionSpace(mesh,'DG',1) #False and 
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(f), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.array())
        gradf = Function(S)
        gradf.vector()[:] = b * A_diag
        
        S = TensorFunctionSpace(mesh,'DG',0)
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(gradf), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.array())
     elif CG0H == 1:
        S = TensorFunctionSpace(mesh,'DG',0)
        A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
        b = assemble(inner(grad(grad(f)), TestFunction(S))*dx)
        ones_ = Function(S)
        ones_.vector()[:] = 1
        A_diag = A * ones_.vector()
        A_diag.set_local(1.0/A_diag.array())
        H = Function(S)
        H.vector()[:] = b * A_diag
     else:
        H = project(grad(grad(f)), TensorFunctionSpace(mesh, "DG", 0))
  else:
    gradf = project(grad(f), VectorFunctionSpace(mesh, "CG", 1))
    H = project(sym(grad(gradf)), TensorFunctionSpace(mesh, "DG", 0))
  
  if CG1out or dolfin.__version__ >= '1.4.0':
   H = project(H,TensorFunctionSpace(mesh,'CG',1))
  # EXTRACT HESSIAN
  [HH,cell2dof] = get_dofs(H)
  # CALCULATE EIGENVALUES 
  [eigL,eigR] = analytic_eig(HH)
  
  # Make H positive definite and calculate the p-norm.
  #enforce hardcoded min and max contraints
  min_eigenvalue = 1e-20; max_eigenvalue = 1e20
  onesC = ones(eigL.shape)
  eigL = array([numpy.abs(eigL),onesC*min_eigenvalue]).max(0)
  eigL = array([numpy.abs(eigL),onesC*max_eigenvalue]).min(0)
  #enforce constraint on aspect ratio 
  if max_edge_ratio is not None:
   RR = arange(HH.shape[1]) 
   CC = eigL.argmax(0)
   I_ = array([False]).repeat(array(eigL.shape).prod())
   I_[CC+(RR-1)*eigL.shape[0]] = True
   I_ = I_.reshape(eigL.shape)
   eigL[I_==False] = array([eigL[I_==False],eigL[I_].repeat(eigL.shape[0]-1)/max_edge_ratio]).max(0)
  
  #check (will not trigger with min_eigenvalue > 0)
  det = eigL.prod(0)
  if any(det==0):
    raise FloatingPointError("Eigenvalues are zero")
  
  #compute metric
  exponent = -1.0/(2*p + n)
  eigL *= 1./eta*(det**exponent).repeat(eigL.shape[0]).reshape([eigL.shape[1],eigL.shape[0]]).T
  
#  HH = analyt_rot(fulleig(eigL),eigR)
#  HH *= 1./eta*det**exponent 
#  [eigL,eigR] = analytic_eig(HH)  

  #enforce min and max contraints
  if max_edge_length is not None:
    min_eigenvalue = 1.0/max_edge_length**2
    if eigL.flatten().min()<min_eigenvalue:
     info('upper bound on element edge length is active')
  if min_edge_length is not None:
    max_eigenvalue = 1.0/min_edge_length**2
    if eigL.flatten().max()>max_eigenvalue:
     info('lower bound on element edge length is active')
  eigL = array([eigL,onesC*min_eigenvalue]).max(0)
  eigL = array([eigL,onesC*max_eigenvalue]).min(0)
  HH = analyt_rot(fulleig(eigL),eigR)
  
  Hfinal = sym2asym(HH) 
  cbig=zeros((H.vector().array()).size)
  cbig[cell2dof.flatten()] = Hfinal.transpose().flatten()
  H.vector().set_local(cbig)
  return H

def metric_ellipse(H1, H2, method='in', qualtesting=False):
  #this function calculates the inner or outer ellipse (depending on the value of the method input)
  #of two the two input metrics.
  [HH1,cell2dof] = get_dofs(H1)
  [HH2,cell2dof] = get_dofs(H2)
  cbig = zeros((H1.vector().array()).size)
  
  # CALCULATE EIGENVALUES using analytic expression numpy._version__>1.8.0 can do this more elegantly
  [eigL1,eigR1] = analytic_eig(HH1)
  # convert metric2 to metric1 space  
  tmp = analyt_rot(HH2, transpose_eigR(eigR1))
  tmp = prod_eig(tmp, 1/eigL1)
  [eigL2,eigR2] = analytic_eig(tmp)
  # enforce inner or outer ellipse
  if method == 'in':
    if qualtesting:
     HH = Function(FunctionSpace(H1.function_space().mesh(),'DG',0))
     HH.vector().set_local((eigL2<ones(eigL2.shape)).sum(0)-ones(eigL2.shape[1]))
     return HH
    else:
     eigL2 = array([eigL2 ,ones(eigL2.shape)]).max(0)
  else:
    eigL2 = array([eigL2, ones(eigL2.shape)]).min(0)

  #convert metric2 back to original space
  tmp = analyt_rot(fulleig(eigL2), eigR2)
  tmp = prod_eig(tmp, eigL1)
  HH = analyt_rot(tmp,eigR1)
  HH = sym2asym(HH)
  #set metric
  cbig[cell2dof.flatten()] = HH.transpose().flatten()
  H1.vector().set_local(cbig)
  return H1

def get_dofs(H):
  mesh = H.function_space().mesh()
  n = mesh.geometry().dim()
  if H.function_space().ufl_element().degree() == 0 and H.function_space().ufl_element().family()[0] == 'D':
      cell2dof = c_cell_dofs(mesh,H.function_space())
      cell2dof = cell2dof.reshape([mesh.num_cells(),n**2])
  else: #CG1 metric
      cell2dof = arange(mesh.num_vertices()*n**2)
      cell2dof = cell2dof.reshape([mesh.num_vertices(),n**2])
  if n == 2:   
   H11 = H.vector().array()[cell2dof[:,0]]
   H12 = H.vector().array()[cell2dof[:,1]] #;H21 = H.vector().array()[cell2dof[:,2]]
   H22 = H.vector().array()[cell2dof[:,3]]
   return [array([H11,H12,H22]),cell2dof]
  else: #n==3
   H11 = H.vector().array()[cell2dof[:,0]]
   H12 = H.vector().array()[cell2dof[:,1]] #;H21 = H.vector().array()[cell2dof[:,3]]
   H13 = H.vector().array()[cell2dof[:,2]] #;H31 = H.vector().array()[cell2dof[:,6]]
   H22 = H.vector().array()[cell2dof[:,4]]
   H23 = H.vector().array()[cell2dof[:,5]] #H32 = H.vector().array()[cell2dof[:,7]]
   H33 = H.vector().array()[cell2dof[:,8]]
   return [array([H11,H12,H22,H13,H23,H33]),cell2dof]

def transpose_eigR(eigR):
    if eigR.shape[0] == 4:
     return array([eigR[0,:],eigR[2,:],\
                   eigR[1,:],eigR[3,:]])
    else: #3D
     return array([eigR[0,:],eigR[3,:],eigR[6,:],\
                   eigR[1,:],eigR[4,:],eigR[7,:],\
                   eigR[2,:],eigR[5,:],eigR[8,:]])

def sym2asym(HH):
    if HH.shape[0] == 3:
        return array([HH[0,:],HH[1,:],\
                      HH[1,:],HH[2,:]])
    else:
        return array([HH[0,:],HH[1,:],HH[3,:],\
                      HH[1,:],HH[2,:],HH[4,:],\
                      HH[3,:],HH[4,:],HH[5,:]])

def fulleig(eigL):
    zeron = zeros(eigL.shape[1])
    if eigL.shape[0] == 2:
        return array([eigL[0,:],zeron,eigL[1,:]])
    else: #3D
        return array([eigL[0,:],zeron,eigL[1,:],zeron,zeron,eigL[2,:]])
        
def analyt_rot(H,eigR):
  #this function rotates a 2x2 symmetric matrix
  if H.shape[0] == 3: #2D
   inds  = array([[0,1],[1,2]])
   indA = array([[0,1],[2,3]])
  else: #3D
   inds  = array([[0,1,3],[1,2,4],[3,4,5]])
   indA = array([[0,1,2],[3,4,5],[6,7,8]])
  indB = indA.T
  A = zeros(H.shape)
  for i in range(len(inds)):
    for j in range(len(inds)):
      for m in range(len(inds)):
        for n in range(len(inds)):
          if i<n:
           continue
          A[inds[i,n],:] += eigR[indB[i,j],:]*H[inds[j,m],:]*eigR[indA[m,n],:]
  return A

def prod_eig(H, eigL):
    if H.shape[0] == 3:
        return array([H[0,:]*eigL[0,:], H[1,:]*numpy.sqrt(eigL[0,:]*eigL[1,:]), \
                                        H[2,:]*eigL[1,:]])
    else:
        return array([H[0,:]*eigL[0,:], H[1,:]*numpy.sqrt(eigL[0,:]*eigL[1,:]), H[2,:]*eigL[1,:], \
                                        H[3,:]*numpy.sqrt(eigL[0,:]*eigL[2,:]), H[4,:]*numpy.sqrt(eigL[2,:]*eigL[1,:]),\
                                        H[5,:]*eigL[2,:]])

def analytic_eig(H, tol=1e-12):
  #this function calculates the eigenvalues and eigenvectors using explicit analytical
  #expression for a 2x2 symmetric matrix. 
  # numpy._version__>1.8.0 can do this more elegantly
  H11 = H[0,:]
  H12 = H[1,:]
  H22 = H[2,:]
  onesC = ones(len(H11))
  if H.shape[0] == 3:
      lambda1 = 0.5*(H11+H22-numpy.sqrt((H11-H22)**2+4*H12**2))
      lambda2 = 0.5*(H11+H22+numpy.sqrt((H11-H22)**2+4*H12**2))        
      v1x = ones(len(H11)); v1y = zeros(len(H11))
      #identical eigenvalues
      I2 = numpy.abs(lambda1-lambda2)<onesC*tol;
      #diagonal matrix
      I1 = numpy.abs(H12)<onesC*tol
      lambda1[I1] = H11[I1]
      lambda2[I1] = H22[I1]
      #general case
      nI = (I1==False)*(I2==False)
      v1x[nI] = -H12[nI]
      v1y[nI] = H11[nI]-lambda1[nI]
      L1 = numpy.sqrt(v1x**2+v1y**2)
      v1x /= L1
      v1y /= L1
      eigL = array([lambda1,lambda2])
      eigR = array([v1x,v1y,-v1y,v1x])
  else: #3D
      H13 = H[3,:]
      H23 = H[4,:]
      H33 = H[5,:]
      p1 = H12**2 + H13**2 + H23**2
      zeroC = zeros(len(H11))
      eig1 = array(H11); eig2 = array(H22); eig3 = array(H33) #do not modify input
      v1 = array([onesC, zeroC, zeroC])
      v2 = array([zeroC, onesC, zeroC])
      v3 = array([zeroC, zeroC, onesC])
      # A is not diagonal.                       
      nI = (numpy.abs(p1) > tol**2)
      p1 = p1[nI]
      H11 = H11[nI]; H12 = H12[nI]; H22 = H22[nI];
      H13 = H13[nI]; H23 = H23[nI]; H33 = H33[nI];
      q = array((H11+H22+H33)/3.)
      H11 /= q; H12 /= q; H22 /= q; H13 /= q; H23 /= q; H33 /= q
      p1 /= q**2; qold = q; q = ones(len(H11))
      p2 = (H11-q)**2 + (H22-q)**2 + (H33-q)**2 + 2.*p1
      p = numpy.sqrt(p2 / 6.)
      I = array([onesC,zeroC,onesC,zeroC,zeroC,onesC])#I = array([1., 0., 1., 0., 0., 1.]).repeat(len(H11)).reshape(6,len(H11)) #identity matrix
      HH = array([H11,H12,H22,H13,H23,H33])
      B = (1./p) * (HH-q.repeat(6).reshape(len(H11),6).T*I[:,nI]) 
      #detB = B11*B22*B33+2*(B12*B23*B13)-B13*B22*B13-B12*B12*B33-B11*B23*B23
      detB = B[0,:]*B[2,:]*B[5,:]+2*(B[1,:]*B[4,:]*B[3,:])-B[3,:]*B[2,:]*B[3,:]-B[1,:]*B[1,:]*B[5,:]-B[0,:]*B[4,:]*B[4,:]
      
      #calc r
      r = detB / 2. 
      rsmall = r<=-1.
      rbig   = r>= 1.
      rgood = (rsmall==False)*(rbig==False)
      phi = zeros(len(H11))
      phi[rsmall] = pi / 3. 
      phi[rbig]   = 0. 
      phi[rgood]  = numpy.arccos(r[rgood]) / 3.
      
      eig1[nI] = q + 2.*p*numpy.cos(phi)
      eig3[nI] = q + 2.*p*numpy.cos(phi + (2.*pi/3.))
      eig2[nI] = array(3.*q - eig1[nI] - eig3[nI])
      eig1[nI] *= qold; eig2[nI] *= qold; eig3[nI] *= qold
      v1[0,nI] = H22*H33 - H23**2 + eig1[nI]*(eig1[nI]-H33-H22)
      v1[1,nI] = H12*(eig1[nI]-H33)+H13*H23
      v1[2,nI] = H13*(eig1[nI]-H22)+H12*H23
      v2[0,nI] = H12*(eig2[nI]-H33)+H23*H13
      v2[1,nI] = H11*H33 - H13**2 + eig2[nI]*(eig2[nI]-H11-H33)
      v2[2,nI] = H23*(eig2[nI]-H11)+H12*H13
      v3[0,nI] = H13*(eig3[nI]-H22)+H23*H12
      v3[1,nI] = H23*(eig3[nI]-H11)+H13*H12
      v3[2,nI] = H11*H22 - H12**2 + eig3[nI]*(eig3[nI]-H11-H22)
      L1 = numpy.sqrt((v1[:,nI]**2).sum(0))
      L2 = numpy.sqrt((v2[:,nI]**2).sum(0))
      L3 = numpy.sqrt((v3[:,nI]**2).sum(0))
      v1[:,nI] /= L1.repeat(3).reshape(len(L1),3).T
      v2[:,nI] /= L2.repeat(3).reshape(len(L1),3).T
      v3[:,nI] /= L3.repeat(3).reshape(len(L1),3).T
      eigL = array([eig1,eig2,eig3])
      eigR = array([v1[0,:],v1[1,:],v1[2,:],\
                    v2[0,:],v2[1,:],v2[2,:],\
                    v3[0,:],v3[1,:],v3[2,:]])
      bad = (numpy.abs(analyt_rot(fulleig(eigL),eigR)-H).sum(0) > tol) | isnan(eigR).any(0) | isnan(eigL).any(0)
      if any(bad):
       log(INFO,'%0.0f problems in eigendecomposition' % bad.sum())
       for i in numpy.where(bad)[0]:
           [eigL_,eigR_] = pyeig(array([[H[0,i],H[1,i],H[3,i]],\
                                        [H[1,i],H[2,i],H[4,i]],\
                                        [H[3,i],H[4,i],H[5,i]]]))
           eigL[:,i] = eigL_
           eigR[:,i] = eigR_.T.flatten()
  return [eigL,eigR]
    
def logexpmetric(Mp,logexp='log'):
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    if logexp=='log':
      eigL = numpy.log(eigL)
    elif logexp=='sqrt':
      eigL = numpy.sqrt(eigL)
    elif logexp=='inv':
      eigL = 1./eigL
    elif logexp=='sqr':
      eigL = eigL**2
    elif logexp=='sqrtinv':
      eigL = numpy.sqrt(1./eigL)
    elif logexp=='sqrinv':
      eigL = 1./eigL**2
    elif logexp=='exp':
      eigL = numpy.exp(eigL)
    else:
      error('logexp='+logexp+' is an invalid value')
    HH = analyt_rot(fulleig(eigL),eigR)
    out = sym2asym(HH).transpose().flatten()
    Mp.vector().set_local(out)
    return Mp

def minimum_eig(Mp):
    mesh = Mp.function_space().mesh()
    element = Mp.function_space().ufl_element()
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    out = Function(FunctionSpace(mesh,element.family(),element.degree()))
    out.vector().set_local(eigL.min(0))
    return out
    
def get_rot(Mp):
    mesh = Mp.function_space().mesh()
    element = Mp.function_space().ufl_element()
    [H,cell2dof] = get_dofs(Mp)
    [eigL,eigR] = analytic_eig(H)
    out = Function(TensorFunctionSpace(mesh,element.family(),element.degree()))
    out.vector().set_local(eigR.transpose().flatten())
    return out

def logproject(Mp):
    mesh = Mp.function_space().mesh()
    logMp = project(logexpmetric(Mp),TensorFunctionSpace(mesh,'CG',1))
    return logexpmetric(logMp,logexp='exp')

def mesh_metric(mesh):
  # this function calculates a mesh metric (or perhaps a square inverse of that, see mesh_metric2...)
  cell2dof = c_cell_dofs(mesh,TensorFunctionSpace(mesh, "DG", 0))
  cells = mesh.cells()
  coords = mesh.coordinates()
  p1 = coords[cells[:,0],:];
  p2 = coords[cells[:,1],:];
  p3 = coords[cells[:,2],:];
  r1 = p1-p2; r2 = p1-p3; r3 = p2-p3
  Nedg = 3
  if mesh.geometry().dim() == 3:
      Nedg = 6
      p4 = coords[cells[:,3],:];
      r4 = p1-p4; r5 = p2-p4; r6 = p3-p4
  rall = zeros([p1.shape[0],p1.shape[1],Nedg])
  rall[:,:,0] = r1; rall[:,:,1] = r2; rall[:,:,2] = r3;
  if mesh.geometry().dim() == 3:
      rall[:,:,3] = r4; rall[:,:,4] = r5; rall[:,:,5] = r6
  All = zeros([p1.shape[0],Nedg**2])
  inds = arange(Nedg**2).reshape([Nedg,Nedg])
  for i in range(Nedg):
    All[:,inds[i,0]] = rall[:,0,i]**2; All[:,inds[i,1]] = 2.*rall[:,0,i]*rall[:,1,i]; All[:,inds[i,2]] = rall[:,1,i]**2
    if mesh.geometry().dim() == 3:
      All[:,inds[i,3]] = 2.*rall[:,0,i]*rall[:,2,i]; All[:,inds[i,4]] = 2.*rall[:,1,i]*rall[:,2,i]; All[:,inds[i,5]] = rall[:,2,i]**2
  Ain = zeros([Nedg*2-1,Nedg*p1.shape[0]])
  ndia = zeros(Nedg*2-1)
  for i in range(Nedg):
      for j in range(i,Nedg):
          iks1 = arange(j,Ain.shape[1],Nedg)
          if i==0:
              Ain[i,iks1] = All[:,inds[j,j]]
          else:
              iks2 = arange(j-i,Ain.shape[1],Nedg)
              Ain[2*i-1,iks1] = All[:,inds[j-i,j]]
              Ain[2*i,iks2]   = All[:,inds[j,j-i]]
              ndia[2*i-1] = i
              ndia[2*i]   = -i
    
  A = scipy.sparse.spdiags(Ain, ndia, Ain.shape[1], Ain.shape[1]).tocsr()
  b = ones(Ain.shape[1])
  X = scipy.sparse.linalg.spsolve(A,b)
  #set solution
  XX = sym2asym(X.reshape([mesh.num_cells(),Nedg]).transpose())
  M = Function(TensorFunctionSpace(mesh,"DG", 0))
  M.vector().set_local(XX.transpose().flatten()[cell2dof])
  return M

def mesh_metric1(mesh):
  #this is just the inverse of mesh_metric2, and it is useful for projecting the ellipse
  #in a certain (velocity) direction, which is usefull for stabilization terms.
  M = mesh_metric(mesh)
  #M = logexpmetric(M,logexp='sqrt')
  [MM,cell2dof] = get_dofs(M)
  [eigL,eigR] = analytic_eig(MM)
  eigL = numpy.sqrt(eigL)
  MM = analyt_rot(fulleig(eigL),eigR)
  MM = sym2asym(MM).transpose().flatten()
  M.vector().set_local(MM[cell2dof.flatten()])
  return M

def mesh_metric2(mesh):
  #this function calculates a metric field, which when divided by sqrt(3) corresponds to the steiner
  #ellipse for the individual elements, see the test case mesh_metric2_example
  #the sqrt(3) ensures that the unit element maps to the identity tensor
  M = mesh_metric(mesh)
  #M = logexpmetric(M,logexp='sqrtinv')
  [MM,cell2dof] = get_dofs(M)
  [eigL,eigR] = analytic_eig(MM)
  eigL = numpy.sqrt(1./eigL)
  MM = analyt_rot(fulleig(eigL),eigR)
  MM = sym2asym(MM).transpose().flatten()
  M.vector().set_local(MM[cell2dof.flatten()])
  return M

def gradate(H, grada, itsolver=False):
    #this function provides anisotropic Helm-holtz smoothing on the logarithm
    #of a metric based on the metric of the mesh times a scaling factor(grada)
    if itsolver:
        solverp = {"linear_solver": "cg", "preconditioner": "ilu"}
    else:
        solverp = {"linear_solver": "lu"}
    mesh = H.function_space().mesh()
    grada = Constant(grada)
    mm2 = mesh_metric2(mesh)
    mm2sq = dot(grada*mm2,grada*mm2)
    Hold = Function(H); H = logexpmetric(H) #avoid logexpmetric side-effect
    V = TensorFunctionSpace(mesh,'CG',1); H_trial = TrialFunction(V); H_test = TestFunction(V); Hnew=Function(V)    
    a = (inner(grad(H_test),dot(mm2sq,grad(H_trial)))+inner(H_trial,H_test))*dx
    L = inner(H,H_test)*dx
    solve(a==L,Hnew,[], solver_parameters=solverp)
    Hnew = metric_ellipse(logexpmetric(Hnew,logexp='exp'), Hold)
    return Hnew


def c_cell_dofs(mesh,V):
  if dolfin.__version__ >= '1.3.0':
   if V.ufl_element().is_cellwise_constant():
    return arange(mesh.num_cells()*mesh.coordinates().shape[1]**2)
   else:
    return arange(mesh.num_vertices()*mesh.coordinates().shape[1]**2)
  else:
      #this function returns the degrees of freedom numbers in a cell
      code = """
      void cell_dofs(boost::shared_ptr<GenericDofMap> dofmap,
                     const std::vector<std::size_t>& cell_indices,
                     std::vector<std::size_t>& dofs)
      {
        assert(dofmap);
        std::size_t local_dof_size = dofmap->cell_dofs(0).size();
        const std::size_t size = cell_indices.size()*local_dof_size;
        dofs.resize(size);
        for (std::size_t i=0; i<cell_indices.size(); i++)
           for (std::size_t j=0; j<local_dof_size;j++)
               dofs[i*local_dof_size+j] = dofmap->cell_dofs(cell_indices[i])[j];
      }
      """
      module = compile_extension_module(code)
      return module.cell_dofs(V.dofmap(), arange(mesh.num_cells(), dtype=numpy.uintp))


if __name__=="__main__":
 testcase = 3
 if testcase == 0:
   from minimal_example import minimal_example
   minimal_example(width=5e-2)
 elif testcase == 1:
   from minimal_example_minell import check_metric_ellipse 
   check_metric_ellipse(width=2e-2)
 elif testcase == 2:
   from play_multigrid import test_refine_metric
   test_refine_metric()
 elif testcase == 3:
   from mesh_metric2_example import test_mesh_metric
   test_mesh_metric()
 elif testcase == 4:
   from circle_convergence import circle_convergence
   circle_convergence()
 elif testcase == 5:
   from maximal_example import maximal_example
   maximal_example()
