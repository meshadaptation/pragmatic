from dolfin import *
from pylab import hold, show, axis
from pylab import plot as pyplot
from numpy import array, linspace, linalg, diag
from numpy import cos as pycos
from numpy import sin as pysin
from numpy import sqrt as pysqrt
from numpy import abs as pyabs
from adaptivity2 import metric_pnorm, mesh_metric, adapt
set_log_level(WARNING)

def test_mesh_metric():
    meshsz = 40
    mesh = RectangleMesh(-0.5,-0.5,0.5,0.5,1*meshsz,1*meshsz,"left/right")
    
    hd = Constant(1e-2)
    testsol = 'tanh(x[0]/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    ddtestsol = '2*'+testsol+'*(1-pow('+testsol+',2))/'+str(float(hd)**2)
    
    # PERFORM ONE ADAPTATION ITERATION
    V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
    R = interpolate(Expression(ddtestsol),V)
    a = inner(grad(dis), grad(dus))*dx
    L = R*dus*dx
    solve(a == L, u, [])
    eta = 0.01; H = metric_pnorm(u, mesh, eta, max_edge_ratio=50);   Mp =  project(H,  TensorFunctionSpace(mesh, "CG", 1))
    mesh = adapt(Mp) 
     
    MpH = mesh_metric(mesh)
    # Plot element i
    i = 40; t = linspace(0,2*pi,101)
    ind = MpH.function_space().dofmap().cell_dofs(i)
    thecell = mesh.cells()[i]
    centerxy = mesh.coordinates()[thecell,:].mean(0).repeat(3).reshape([2,3]).T
    cxy = mesh.coordinates()[thecell,:]-centerxy
    pyplot(cxy[:,0],cxy[:,1],'-b')
    H = MpH.vector().gather(ind).reshape(2,2);# H = array([[H[1],H[0]],[H[0],H[2]]])
    #H = MpH.vector().gather(ind); H = array([[H[1],H[0]],[H[0],H[2]]])
    #H = MpH.vector().array()[ind]; H = array([[H[1],H[0]],[H[0],H[2]]])
    [v,w] = linalg.eig(H); v = 1/pysqrt(v)/pysqrt(3)
    elxy = array([pycos(t),pysin(t)]).T.dot(w).dot(diag(v)).dot(w.T)
    hold('on'); pyplot(elxy[:,0],elxy[:,1],'-r'); hold('off'); axis('equal')
    print('triangle area: %0.6f, ellipse axis product(*3*sqrt(3)/4): %0.6f' % (pyabs(linalg.det(array([cxy[1,:]-cxy[0,:],cxy[2,:]-cxy[0,:]])))/2,v[0]*v[1]*3*sqrt(3)/4))
    show()

if __name__=="__main__":
 test_mesh_metric()