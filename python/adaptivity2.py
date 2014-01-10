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
# Kristian E. Jensen for ellipse function, test cases and vectorization opt.

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
from numpy import array, zeros, ones, matrix, linalg, any
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
#  _libpragmatic = ctypes.cdll.LoadLibrary("libpragmatic.so") 
  _libpragmatic = ctypes.cdll.LoadLibrary("/home/kjensen/projects/scaling_optimisation/src/.libs/libpragmatic.so")
  #_libpragmatic = ctypes.cdll.LoadLibrary("libpragmatic.so")
except:
  raise LibraryException("Failed to load libpragmatic.so")

def c_cell_dofs(mesh,V):
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
  return module.cell_dofs(V.dofmap(), array(range(mesh.num_cells()), dtype=numpy.uintp))
  
def mesh_metric(mesh):
  cell2dof = c_cell_dofs(mesh,TensorFunctionSpace(mesh, "DG", 0))
  cells = mesh.cells()
  coords = mesh.coordinates()
  p1 = coords[cells[:,0],:];
  p2 = coords[cells[:,1],:];
  p3 = coords[cells[:,2],:];
  r1 = p1-p2; r2 = p1-p3; r3 = p2-p3
  A11 = r1[:,0]**2; A12 = 2.*r1[:,0]*r1[:,1]; A13 = r1[:,1]**2
  A21 = r2[:,0]**2; A22 = 2.*r2[:,0]*r2[:,1]; A23 = r2[:,1]**2
  A31 = r3[:,0]**2; A32 = 2.*r3[:,0]*r3[:,1]; A33 = r3[:,1]**2
  #DEFINE AND SOLVE MANY SMALL PROBLEMS AS SINGLE SPARSE (BIG) PROBLEM numpy 1.8.0 allows a better implementation
  R = array(range(mesh.num_cells()*3)).repeat(3)
  C = array(range(mesh.num_cells()*3)).repeat(3).reshape([mesh.num_cells(),3,3]).transpose([0,2,1]).flatten()
  RCdata = array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]]).transpose([2,0,1]).flatten()
  A = scipy.sparse.csr_matrix((RCdata,(R,C)),shape=(3*mesh.num_cells(),3*mesh.num_cells()))
  b = ones(3*mesh.num_cells())
  X = scipy.sparse.linalg.spsolve(A,b)
  #set solution
  X11 = X[range(0,mesh.num_cells()*3,3)]
  X12 = X[range(1,mesh.num_cells()*3,3)]
  X22 = X[range(2,mesh.num_cells()*3,3)]
  M = Function(TensorFunctionSpace(mesh,"DG", 0))
  M.vector().set_local(array([X11,X12,X12,X22]).transpose().flatten()[cell2dof])
  return M
  
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

def gen_polygon_surfmesh(cells,coords):
    ntri = len(cells)
    
    v1 = coords[cells[:,1],:]-coords[cells[:,0],:]
    v2 = coords[cells[:,2],:]-coords[cells[:,0],:]
    badcells = v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]>0
    tri = cells.flatten()
    R = range(0,3*ntri,3); m1 = ones(ntri,dtype=numpy.int64)
    tri = array([tri[R+badcells].tolist(),tri[R+m1-badcells].tolist(),tri[R+2*m1].tolist()])
    edg = array([tri[0],tri[1],tri[2],tri[1],tri[2],tri[0]]).reshape([2,ntri*3])
    #putting large node number in later row
    C = edg.argsort(0)
    R = array(range(ntri*3))
    edg = edg.transpose().flatten()
    edg = array([edg[R*2+C[0,:]].tolist(),edg[R*2+C[1,:]].tolist()])
    #sort according to first node number
    I2 = (edg[0,:]*(2+edg.max())+edg[1,:]).argsort()
    edg = edg[:,I2] 
    #find unique edges
    d = array([any(edg[:,0] != edg[:,1])] +\
         (any((edg[:,range(1,ntri*3-1)] != edg[:,range(2,ntri*3)]),0) *\
          any((edg[:,range(1,ntri*3-1)] != edg[:,range(0,ntri*3-2)]),0)).tolist() +\
              [any(edg[:,ntri*3-1] != edg[:,ntri*3-2])])
    edg = edg[:,d]
    #put correct node number back in later row
    R = array(range(sum(d)))
    edg = edg.transpose().flatten()
    bfaces = array([edg[R*2+C[0,I2[d]]].tolist(),edg[R*2+C[1,I2[d]]].tolist()]).transpose()
    # compute normalized tangent (t) and normal vector (n)
    t = coords[bfaces[:,0],:]-coords[bfaces[:,1],:]
    t = t/numpy.sqrt(t[:,0]**2+t[:,1]**2).repeat(2).reshape([len(t),2])
#    normals = array([t[:,1].tolist(),(-t[:,0]).tolist()]).transpose()
    # compute sets of co-linear edges (this is specific to polygonal geometries)
    IDs = zeros(len(t), dtype = numpy.intc)
    while True:
        n = IDs.argmin()
        IDs[n] = IDs.max() + 1
        I = array(range(0,len(IDs)))
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
#    IDs = IDs.tolist()
    bfaces_pair = zip(bfaces[:,0],bfaces[:,1])
    return [bfaces,IDs]

def set_mesh(n_x,n_y,n_enlist,mesh=None,dx=None, debug=False):
  startTime = time()
  n_mesh = Mesh()
  ed = MeshEditor()
  ed.open(n_mesh, 2, 2)
  ed.init_vertices(len(n_x)) #n_NNodes.value
  for i in range(len(n_x)): #n_NNodes.value
    ed.add_vertex(i, n_x[i], n_y[i])
  ed.init_cells(len(n_enlist)/3) #n_NElements.value
  for i in range(len(n_enlist)/3): #n_NElements.value
    ed.add_cell(i, n_enlist[i * 3], n_enlist[i * 3 + 1], n_enlist[i * 3 + 2])
  ed.close()
  info("mesh definition took %0.1fs" % (time()-startTime))
  if debug and dx is not None:
    # Sanity check to be deleted or made optional
    n_space = FunctionSpace(n_mesh, "CG", 1)

    area = assemble(Constant(1.0) * dx, mesh = mesh)
    n_area = assemble(Constant(1.0) * dx, mesh = n_mesh)
    err = abs(area - n_area)
    info("Donor mesh area : %.17e" % area)
    info("Target mesh area: %.17e" % n_area)
    info("Change          : %.17e" % err)
    info("Relative change : %.17e" % (err / area))
    assert(err < 2.0e-11 * area)
  return n_mesh
  
def adapt(metric, bfaces=None, bfaces_IDs=None, debug=True):
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
  
  nodes = array(range(0,mesh.num_vertices()),dtype=numpy.intc) 
  cells = mesh.cells()
  coords = mesh.coordinates()
  # create boundary mesh and associated list of co-linear edges
  if bfaces is None:
    [bfaces,bfaces_IDs] = gen_polygon_surfmesh(cells,coords)
    
  x = coords[nodes,0]
  y = coords[nodes,1]
    
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
  nfacets = ctypes.c_int(len(bfaces))
  facets = array(bfaces.flatten(),dtype=numpy.intc)
  
  _libpragmatic.pragmatic_set_surface(ctypes.byref(nfacets),
                                      facets.ctypes.data,
                                      bfaces_IDs.ctypes.data)
  
  info("Setting metric tensor field ...")
  # Dolfin stores the tensor as:
  # |dxx dxy|
  # |dyx dyy|
    ## THE (CG1-)DOF NUMBERS ARE DIFFERENT FROM THE VERTEX NUMBERS (and we wish to work with the former)
  if dolfin.__version__ != '1.2.0':
      dof2vtx = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
  else:
      dof2vtx = FunctionSpace(mesh,'CG',1).dofmap().vertex_to_dof_map(mesh).argsort()
  
  metric_arr = numpy.empty(metric.vector().array().size, dtype = numpy.float64)
  metric_arr[range(0,metric.vector().array().size,4)] = metric.vector().array()[array(range(0,metric.vector().array().size,4))[dof2vtx]]
  metric_arr[range(1,metric.vector().array().size,4)] = metric.vector().array()[array(range(2,metric.vector().array().size,4))[dof2vtx]]
  metric_arr[range(2,metric.vector().array().size,4)] = metric.vector().array()[array(range(2,metric.vector().array().size,4))[dof2vtx]]
  metric_arr[range(3,metric.vector().array().size,4)] = metric.vector().array()[array(range(3,metric.vector().array().size,4))[dof2vtx]]
  
  #from IPython import embed
  #embed()

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
  
  n_mesh = set_mesh(n_x,n_y,n_enlist,mesh=mesh,dx=dx,debug=debug)
  
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

def analytic_eig(H11,H12,H22):
  onesC = ones(len(H11))
  lambda1 = 0.5*(H11+H22-numpy.sqrt((H11-H22)**2+4*H12**2))
  lambda2 = 0.5*(H11+H22+numpy.sqrt((H11-H22)**2+4*H12**2))
  I = numpy.abs(H12)<onesC*1e-12; nI = I==False #myeps
  v1x = zeros(len(H11)); v1y = zeros(len(H11))
  v1x[I] = 1; v1y[I] = 0; lambda1[I] = H11[I]; lambda2[I] = H22[I]
  v1x[nI] = -H12[nI]; v1y[nI] = H11[nI]-lambda1[nI]
  v1xn = v1x/numpy.sqrt(v1x**2+v1y**2)
  v1yn = v1y/numpy.sqrt(v1x**2+v1y**2)
  return [lambda1,lambda2,v1xn,v1yn]

def analyt_rot(H11,H12,H22,v1x,v1y):
  A11 = v1x**2*H11 + v1y**2*H22 - 2*v1x*v1y*H12
  A12 = v1x*v1y*(H11-H22) + H12*(v1x**2-v1y**2)
  A22 = v1y**2*H11 + v1x**2*H22 + 2*v1x*v1y*H12
  return [A11,A12,A22]

def fix_CG1_metric(Mp):
 H11 = Mp.vector().array()[range(0,Mp.vector().array().size,4)]
 H12 = Mp.vector().array()[range(1,Mp.vector().array().size,4)]
 H22 = Mp.vector().array()[range(3,Mp.vector().array().size,4)]
 [lambda1,lambda2,v1xn,v1yn] = analytic_eig(H11,H12,H22)
# if any(lambda1<zeros(len(lambda2))) or any(lambda2<zeros(len(lambda2))):
#  warning('negative eigenvalue in metric fixed')
 lambda1 = numpy.abs(lambda1)
 lambda2 = numpy.abs(lambda2)
 [H11,H12,H22] = analyt_rot(lambda1,zeros(len(lambda1)),lambda2,v1xn,v1yn)
 out = zeros(Mp.vector().array().size)
 out[range(0,Mp.vector().array().size,4)] = H11
 out[range(1,Mp.vector().array().size,4)] = H12
 out[range(2,Mp.vector().array().size,4)] = H12
 out[range(3,Mp.vector().array().size,4)] = H22
 Mp.vector().set_local(out)
 return Mp
 
# p-norm scaling to the metric, as in Chen, Sun and Xu, Mathematics of
# Computation, Volume 76, Number 257, January 2007, pp. 179-204.
def metric_pnorm(f, mesh, eta, max_edge_length=None, min_edge_length=None, max_edge_ratio=10, p=2):
  # Sanity checks
  if max_edge_ratio < 1.0:
    raise InvalidArgumentException("The maximum edge ratio must be greater greater or equal to 1")
  else:
    max_edge_ratio = max_edge_ratio**2 # ie we are going to be looking at eigenvalues

  n = mesh.geometry().dim()
 
  if f.function_space().ufl_element().degree() == 2 and f.function_space().ufl_element().family() == 'Lagrange':
#    S = VectorFunctionSpace(mesh,'DG',1) #False and 
#    A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
#    b = assemble(inner(grad(f), TestFunction(S))*dx)
#    ones_ = Function(S)
#    ones_.vector()[:] = 1
#    A_diag = A * ones_.vector()
#    A_diag.set_local(1.0/A_diag.array())
#    gradf = Function(S)
#    gradf.vector()[:] = b * A_diag
##    
#    S = TensorFunctionSpace(mesh,'DG',0)
#    A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
#    b = assemble(inner(grad(gradf), TestFunction(S))*dx)
#    ones_ = Function(S)
#    ones_.vector()[:] = 1
#    A_diag = A * ones_.vector()
#    A_diag.set_local(1.0/A_diag.array())
#    H = Function(S)
#    H.vector()[:] = b * A_diag*4
#    startTime = time()
#    S = TensorFunctionSpace(mesh,'DG',0)
##    print("space definition took %0.1fs" % (time()-startTime))
#    A = assemble(inner(TrialFunction(S), TestFunction(S))*dx)
#    b = assemble(inner(grad(grad(f)), TestFunction(S))*dx)
#    ones_ = Function(S)
#    ones_.vector()[:] = 1
#    A_diag = A * ones_.vector()
#    A_diag.set_local(1.0/A_diag.array())
#    H = Function(S)
#    H.vector()[:] = b * A_diag
    H = project(grad(grad(f)), TensorFunctionSpace(mesh, "DG", 0))
  else:
    gradf = project(grad(f), VectorFunctionSpace(mesh, "DG", 1))
    H = project(grad(gradf), TensorFunctionSpace(mesh, "DG", 0))
  # Make H positive definite and calculate the p-norm.
  cbig=zeros((H.vector().array()).size)
  exponent = -1.0/(2*p + n)

  min_eigenvalue = 1e-8; max_eigenvalue = 1e8
  if max_edge_length is not None:
    min_eigenvalue = 1.0/max_edge_length**2
  if min_edge_length is not None:
    max_eigenvalue = 1.0/min_edge_length**2
  
  # EXTRACT HESSIAN
  cell2dof = c_cell_dofs(mesh,H.function_space())
  cell2dof = cell2dof.reshape([mesh.num_cells(),4])
  H11 = H.vector().array()[cell2dof[:,0]]
  H12 = H.vector().array()[cell2dof[:,1]] #;H21 = H.vector().array()[cell2dof[:,2]]
  H22 = H.vector().array()[cell2dof[:,3]]
  # CALCULATE EIGENVALUES using analytic expression numpy._version__>1.8.0 can do this more elegantly
  [lambda1,lambda2,v1xn,v1yn] = analytic_eig(H11,H12,H22)
  
  #enforce contraints
  onesC = ones(mesh.num_cells())
  lambda1 = array([numpy.abs(lambda1),onesC*min_eigenvalue]).max(0)
  lambda2 = array([numpy.abs(lambda2),onesC*min_eigenvalue]).max(0)
  lambda1 = array([lambda1,onesC*max_eigenvalue]).min(0)
  lambda2 = array([lambda2,onesC*max_eigenvalue]).min(0)
  L1b = lambda1 > lambda2; nL1b = L1b == True
  lambda1[nL1b] = array([lambda1[nL1b],lambda2[nL1b]/max_edge_ratio]).max(0)
  lambda2[L1b]  = array([lambda2[L1b] ,lambda1[L1b] /max_edge_ratio]).max(0)
  
  #check (will not trigger with min_eigenvalue > 0)
  det = lambda1*lambda2
  if any(det==0):
    raise FloatingPointError("Eigenvalues are zero")
  
  #compute metric
  [H11,H12,H22] = analyt_rot(lambda1,zeros(len(H11)),lambda2,v1xn,v1yn)
  H11 *= 1./eta*det**exponent
  H12 *= 1./eta*det**exponent
  H22 *= 1./eta*det**exponent
  
  cbig[cell2dof.flatten()] = array([H11,H12,H12,H22]).transpose().flatten()
  H.vector().set_local(cbig)
  return H


def metric_ellipse(H1, H2, mesh, method='in'):
  # Sanity checks
  cell2dof = c_cell_dofs(mesh,H1.function_space())
  cell2dof = cell2dof.reshape([mesh.num_cells(),4])
  H1aa = H1.vector().array()[cell2dof[:,0]]
  H1ab = H1.vector().array()[cell2dof[:,1]] #;H1ba = H1.vector().array()[cell2dof[:,2]]
  H1bb = H1.vector().array()[cell2dof[:,3]] 
  H2aa = H2.vector().array()[cell2dof[:,0]]
  H2ab = H2.vector().array()[cell2dof[:,1]] #;H2ba = H2.vector().array()[cell2dof[:,2]]
  H2bb = H2.vector().array()[cell2dof[:,3]]

  cbig = zeros((H1.vector().array()).size)
  
  # CALCULATE EIGENVALUES using analytic expression numpy._version__>1.8.0 can do this more elegantly
  [lambda1a,lambda1b,v1xn,v1yn] = analytic_eig(H1aa,H1ab,H1bb)
  # convert metric2 to metric1 space  
  [tmp11,tmp12,tmp22] = analyt_rot(H2aa,H2ab,H2bb,v1xn,-v1yn)
  tmp11 /= lambda1a
  tmp12 /= numpy.sqrt(lambda1a*lambda1b)
  tmp22 /= lambda1b
  [lambda2a,lambda2b,v2xn,v2yn] = analytic_eig(tmp11,tmp12,tmp22)
  # enforce inner or outer ellipse
  if method == 'in':
    lambda2a = array([lambda2a,ones(len(lambda2a))]).max(0)
    lambda2b = array([lambda2b,ones(len(lambda2b))]).max(0)
  else:
    lambda2a = array([lambda2a,ones(len(lambda2a))]).min(0)
    lambda2b = array([lambda2b,ones(len(lambda2b))]).min(0)

  #convert metric2 back to original space
  [tmp11,tmp12,tmp22] = analyt_rot(lambda2a,zeros(len(lambda2a)),lambda2b,v2xn,v2yn)
  tmp11 *= lambda1a
  tmp12 *= numpy.sqrt(lambda1a*lambda1b)
  tmp22 *= lambda1b
  [H11,H12,H22] = analyt_rot(tmp11,tmp12,tmp22,v1xn,v1yn)
  #set metric
  cbig[cell2dof.flatten()] = array([H11,H12,H12,H22]).transpose().flatten()
  H1.vector().set_local(cbig)
  return H1

if __name__=="__main__":
 testcase = 1
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
   from play_metric_adapt import test_mesh_metric
   test_mesh_metric()
 elif testcase == 4:
   from circle_convergence import circle_convergence
   circle_convergence()
 elif testcase == 5:
   from maximal_example import maximal_example
   maximal_example()
