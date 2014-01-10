from dolfin import *
from adaptivity2 import metric_pnorm, adapt
from pylab import hold, show, triplot, tricontourf, colorbar, axis, box, rand, get_cmap
from pylab import plot as pyplot
from numpy import array, ones
import numpy
set_log_level(WARNING)
from sympy import Symbol, diff
from sympy import tanh as pytanh
from sympy import cos as pysin
from sympy import sin as pycos

def minimal_example(width=5e-2):
    ### CONSTANTS
    meshsz = 40
    hd = Constant(width)
    ### SETUP MESH
    mesh = RectangleMesh(-0.5,-0.5,0.5,0.5,1*meshsz,1*meshsz,"left/right")
    ### SETUP SOLUTION
    angle = pi/8 #rand*pi/2
    #testsol = 'tanh(x[0]/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    #sx = Symbol('sx'); sy = Symbol('sy'); width_ = Symbol('ww'); angle_ = Symbol('aa')
    #testsol = pytanh((pycos(angle_)*sx+pysin(angle_)*sy)/width_)    
    testsol = 'tanh((' + str(cos(angle)) + '*x[0]+'+ str(sin(angle)) + '*x[1])/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    ddtestsol = str(cos(angle)+sin(angle))+'*2*'+testsol+'*(1-pow('+testsol+',2))/'+str(float(hd)**2)
    
    # PERFORM TEN ADAPTATION ITERATIONS
    for iii in range(10):
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
     R = interpolate(Expression(ddtestsol),V)
     a = inner(grad(dis), grad(dus))*dx
     L = R*dus*dx
     solve(a == L, u, [])
     startTime = time()
     eta = 0.01; H = metric_pnorm(u, mesh, eta, max_edge_ratio=50);   Mp =  project(H,  TensorFunctionSpace(mesh, "CG", 1))
     mesh = adapt(Mp) 
     print("total (adapt+metric) time was %0.1fs" % (time()-startTime))
    
    # PLOT MESH
    testf = interpolate(Expression(testsol),FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
    zz = testf.vector().array()[vtx2dof]; zz[zz==1] -= 1e-16
    hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
    colorbar(hh)
    
    hold('on'); triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),color='r',linewidth=0.5); hold('off')
    axis('equal'); box('off'); show()

if __name__=="__main__":
 minimal_example()
 