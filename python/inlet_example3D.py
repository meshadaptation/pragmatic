### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to
### illustrate the need for explicit definition corner nodes

from dolfin import *
from adaptivity2 import metric_pnorm, logproject, adapt, polyhedron_surfmesh
import numpy

set_log_level(INFO+1)

def minimal_example3D(meshsz=20, Nadapt=10, dL=0.05, eta = 0.01, returnmesh=False, hax=False):
    
    class inlet(SubDomain):
            def inside(self, x, on_boundary):
              return 0.5-DOLFIN_EPS < x[0] \
              and -dL-DOLFIN_EPS < x[1] and x[1] < dL+DOLFIN_EPS \
              and -dL-DOLFIN_EPS < x[2] and x[2] < dL+DOLFIN_EPS
    def boundary(x, on_boundary): #not(boundary(x))
          return on_boundary and ((0.5-DOLFIN_EPS >= x[0] \
          or -dL-DOLFIN_EPS >= x[1] or x[1] >= dL+DOLFIN_EPS \
          or -dL-DOLFIN_EPS >= x[2] or x[2] >= dL+DOLFIN_EPS))
    
    def get_bnd_mesh(mesh):
        coords = mesh.coordinates()
        [bfaces,bfaces_IDs] = polyhedron_surfmesh(mesh.cells(),coords)
        bcoords = (coords[bfaces[:,0],:]+coords[bfaces[:,1],:]+coords[bfaces[:,2],:])/3.
        I = (bcoords[:,0] > 0.5-DOLFIN_EPS) & (bcoords[:,1] < dL) & (bcoords[:,1] > -dL) \
                                 & (bcoords[:,2] < dL) & (bcoords[:,2] > -dL)
        I2 = (bcoords[:,0] > 0.5-DOLFIN_EPS) & (bcoords[:,1] < dL) & (bcoords[:,1] > -dL) \
                                  & (bcoords[:,2] > dL)
        I3 = (bcoords[:,0] > 0.5-DOLFIN_EPS) & (bcoords[:,1] < dL) & (bcoords[:,1] > -dL) \
                                  & (bcoords[:,2] < -dL)
        bfaces_IDs[I] = 97
        if hax:
         bfaces_IDs[I2] = 98 
         bfaces_IDs[I3] = 99 #problems

        c1 = numpy.array([0.5, -dL, -dL]).repeat(coords.shape[0]).reshape([3,coords.shape[0]]).T
        c2 = numpy.array([0.5, -dL,  dL]).repeat(coords.shape[0]).reshape([3,coords.shape[0]]).T
        c3 = numpy.array([0.5,  dL, -dL]).repeat(coords.shape[0]).reshape([3,coords.shape[0]]).T
        c4 = numpy.array([0.5,  dL,  dL]).repeat(coords.shape[0]).reshape([3,coords.shape[0]]).T
        crnds = numpy.where((numpy.sqrt(((coords - c1)**2).sum(1)) < 1e3*DOLFIN_EPS) | \
                            (numpy.sqrt(((coords - c2)**2).sum(1)) < 1e3*DOLFIN_EPS) | \
                            (numpy.sqrt(((coords - c3)**2).sum(1)) < 1e3*DOLFIN_EPS) | \
                            (numpy.sqrt(((coords - c4)**2).sum(1)) < 1e3*DOLFIN_EPS))[0]
#        return [bfaces,bfaces_IDs,crnds]
        return [bfaces,bfaces_IDs]
    
    ### SETUP MESH
    mesh = BoxMesh(-0.5,-0.5,-0.5,0.5,0.5,0.5,meshsz,meshsz,meshsz)
    fid  = File("out.pvd")
    # PERFORM TEN ADAPTATION ITERATIONS
    for iii in range(Nadapt):
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
     a = inner(grad(dis), grad(dus))*dx
     boundaries = FacetFunction("size_t",mesh)
     Inlet = inlet()
     boundaries.set_all(0)
     Inlet.mark(boundaries, 1)
     ds = Measure("ds")[boundaries]
     
     L = dus*ds(1) #+dus*dx
     bc = DirichletBC(V, Constant(0.), boundary)
     solve(a == L, u, bc)
     fid << u
     startTime = time()
     H = metric_pnorm(u, eta, max_edge_length=2., max_edge_ratio=50, CG1out=True)
     #H = logproject(H)
     if iii != Nadapt-1:
      [bfaces,bfaces_IDs] = get_bnd_mesh(mesh)
      mesh = adapt(H, bfaces=bfaces, bfaces_IDs=bfaces_IDs)
      log(INFO+1,"total (adapt+metric) time was %0.1fs, nodes: %0.0f" % (time()-startTime, mesh.num_vertices()))
    
    plot(u,interactive=True)
    plot(mesh,interactive=True)


if __name__=="__main__":
# minimal_example3D(meshsz=10, dL=0.1, hax=True)
 minimal_example3D(meshsz=10, dL=0.1, hax=False)
 