### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to
### 1. derive a forcing term that gives rise to an oscilatting function with a shock (see Gerards pragmatic article)
### 2. solve a poisson equation with the forcing term using 2D anisotropic adaptivity
### 3. plot the numerical solution, the analitical solution and the difference between them
### the error level(eta), the time and period of the function as well the number of solition<->adaptation iterations
### are optional input parameters.

from dolfin import *
from adaptivity2 import metric_pnorm, adapt
from pylab import hold, show, axis, box, figure, tricontourf, triplot, colorbar, savefig, title, get_cmap
from pylab import plot as pyplot
from numpy import array
from sympy import Symbol, diff, Subs
from sympy import sin as pysin
from sympy import atan as pyatan
set_log_level(INFO+1)
#parameters["allow_extrapolation"] = True

def maximal_example(eta = 0.001, Nadapt=5, timet=1., period=2*pi):
    ### CONSTANTS
    meshsz = 40
        ### SETUP MESH
    mesh = RectangleMesh(-0.75,-0.25,0.,0.75,1*meshsz,1*meshsz,"left/right")
    ### SETUP SOLUTION
    #testsol = '0.1*sin(50*x+2*pi*t/T)+atan(-0.1/(2*x - sin(5*y+2*pi*t/T)))';
    sx = Symbol('sx'); sy = Symbol('sy'); sT = Symbol('sT'); st = Symbol('st');  spi = Symbol('spi')
    testsol = 0.1*pysin(50*sx+2*spi*st/sT)+pyatan(-0.1/(2*sx - pysin(5*sy+2*spi*st/sT)))
    ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('spi','pi')
    
    # replacing **P with pow(,P)
    ddtestsol = ddtestsol.replace("(2*x[0] - sin(5*x[1] + 2*pi*st/sT))**2","pow(2*x[0] - sin(5*x[1] + 2*pi*st/sT),2.)")
    ddtestsol = ddtestsol.replace("cos(5*x[1] + 2*pi*st/sT)**2","pow(cos(5*x[1] + 2*pi*st/sT),2.)")
    ddtestsol = ddtestsol.replace("(2*x[0] - sin(5*x[1] + 2*pi*st/sT))**3","pow(2*x[0] - sin(5*x[1] + 2*pi*st/sT),3.)")
    ddtestsol = ddtestsol.replace("(1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*st/sT),2.))**2","pow(1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*st/sT),2.),2.)")
    ddtestsol = ddtestsol.replace("(2*x[0] - sin(5*x[1] + 2*pi*st/sT))**5","pow(2*x[0] - sin(5*x[1] + 2*pi*st/sT),5.)")
    #insert values
    ddtestsol = ddtestsol.replace('sT',str(period)).replace('st',str(timet))
    testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('spi','pi').replace('sT',str(period)).replace('st',str(timet))
    ddtestsol = "-("+ddtestsol+")"
    
    def boundary(x):
          return near(x[0],mesh.coordinates()[:,0].min()) or near(x[0],mesh.coordinates()[:,0].max()) \
          or near(x[1],mesh.coordinates()[:,1].min()) or near(x[1],mesh.coordinates()[:,1].max())
    # PERFORM ONE ADAPTATION ITERATION
    for iii in range(Nadapt):
     startTime = time()
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
#     R = interpolate(Expression(ddtestsol),V)
     a = inner(grad(dis), grad(dus))*dx
     L = Expression(ddtestsol)*dus*dx #
     bc = DirichletBC(V, Expression(testsol), boundary)
     solve(a == L, u, bc)
     soltime = time()-startTime
     
     startTime = time()
     H = metric_pnorm(u, eta, max_edge_ratio=50, CG0H=3)
     metricTime = time()-startTime
     if iii != Nadapt-1:
      mesh = adapt(H) 
      TadaptTime = time()-startTime
      L2error = errornorm(Expression(testsol), u, degree_rise=4, norm_type='L2')
      print("%5.0f elements, %0.0e L2error, adapt took %0.0f %% of the total time, (%0.0f %% of which was the metric calculation)" \
      % (mesh.num_cells(),L2error,TadaptTime/(TadaptTime+soltime)*100,metricTime/TadaptTime*100))
    
#    # PLOT MESH
#    figure()
    coords = mesh.coordinates().transpose()
#    triplot(coords[0],coords[1],mesh.cells(),linewidth=0.1)
#    #savefig('mesh.png',dpi=300) #savefig('mesh.eps'); 
            
    figure() #solution
    testf = interpolate(Expression(testsol),FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
    zz = testf.vector().array()[vtx2dof]
    hh=tricontourf(coords[0],coords[1],mesh.cells(),zz,100)
    colorbar(hh)
    #savefig('solution.png',dpi=300) #savefig('solution.eps'); 
    
    figure() #analytical solution
    testfe = interpolate(u,FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
    zz = testf.vector().array()[vtx2dof]
    hh=tricontourf(coords[0],coords[1],mesh.cells(),zz,100)
    colorbar(hh)
    #savefig('analyt.png',dpi=300) #savefig('analyt.eps');
    
    figure() #error
    zz -= testf.vector().array()[vtx2dof]; zz[zz==1] -= 1e-16
    hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
    colorbar(hh)
    
    hold('on'); triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),color='r',linewidth=0.5); hold('off')
    axis('equal'); box('off'); title('error')
    show()

if __name__=="__main__":
 maximal_example()
