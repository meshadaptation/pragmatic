### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to illustrate the mesh_metric2 function
### first a non structured mesh is generated (inspired from the minimal example),
### then the mesh_metric2 function is called and an element (i) is plotted together
### with the corresponding mesh_metric2 (divided by sqrt(3)) illustrated as the steiner ellipse,
### i.e. the smallest ellipse intersecting the vertices, while still having the same center as the 
### triangle. A unit triangle (side lengths equal to one) will give a identity tensor for the
### mesh_metric2 (that is the reason for the sqrt(3) factor)

from dolfin import *
from pylab import hold, show, axis
from pylab import plot as pyplot
from numpy import array, linspace, linalg, diag, cross
from numpy import cos as pycos
from numpy import sin as pysin
from numpy import sqrt as pysqrt
from numpy import abs as pyabs
from adaptivity import metric_pnorm, mesh_metric, adapt, mesh_metric2
set_log_level(WARNING)

def test_mesh_metric():
    mesh = RectangleMesh(0,0,1,1,20,20)
    mesh = adapt(interpolate(Constant(((10.,0.),(0.,10.))),TensorFunctionSpace(mesh,'CG',1)))
    #extract mesh metric
    MpH = mesh_metric2(mesh)
    # Plot element i
    i = 20; t = linspace(0,2*pi,101)
    ind = MpH.function_space().dofmap().cell_dofs(i)
    thecell = mesh.cells()[i]
    centerxy = mesh.coordinates()[thecell,:].mean(0).repeat(3).reshape([2,3]).T
    cxy = mesh.coordinates()[thecell,:]-centerxy
    pyplot(cxy[:,0],cxy[:,1],'-b')
    H = MpH.vector().gather(ind).reshape(2,2);# H = array([[H[1],H[0]],[H[0],H[2]]])
    #H = MpH.vector().gather(ind); H = array([[H[1],H[0]],[H[0],H[2]]])
    #H = MpH.vector().array()[ind]; H = array([[H[1],H[0]],[H[0],H[2]]])
    [v,w] = linalg.eig(H); v /= pysqrt(3) #v = 1/pysqrt(v)/pysqrt(3)
    elxy = array([pycos(t),pysin(t)]).T.dot(w).dot(diag(v)).dot(w.T)
    hold('on'); pyplot(elxy[:,0],elxy[:,1],'-r'); hold('off'); axis('equal')
    print('triangle area: %0.6f, ellipse axis product(*3*sqrt(3)/4): %0.6f' % (pyabs(linalg.det(array([cxy[1,:]-cxy[0,:],cxy[2,:]-cxy[0,:]])))/2,v[0]*v[1]*3*sqrt(3)/4))
    show()
    
def test_mesh_metric3D():    
    mesh = BoxMesh(0,0,0,1,1,1,20,20,20)
    mesh = adapt(interpolate(Constant(((100.,0.,0.),(0.,100.,0.),(0.,0.,100.))),TensorFunctionSpace(mesh,'CG',1)))
    
    #extract mesh metric
    MpH = mesh_metric2(mesh)
    for i in range(0,40):
        thecell = mesh.cells()[i]
        ind = MpH.function_space().dofmap().cell_dofs(i)
        coords = mesh.coordinates()[thecell,:]
        r1 = coords[1,:]-coords[0,:]
        r2 = coords[2,:]-coords[0,:]
        r3 = coords[0:3,:].mean(0)-coords[3,:]
        r12c = cross(r2,r1)/6.
        det1234 = pyabs((r12c*r3).sum())
        
        H = MpH.vector().gather(ind).reshape(3,3)# H = array([[H[1],H[0]],[H[0],H[2]]])
        [v,w] = linalg.eig(H)
        print('tetrahedron volume: %0.6f, ellipsoid axis product(*sqrt(2)/12): %0.6f' % (det1234,v.prod()*sqrt(2)/12))
        print v.prod()*sqrt(2)/12/(det1234)

if __name__=="__main__":
# test_mesh_metric()
 test_mesh_metric3D()