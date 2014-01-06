from dolfin import *
from adaptivity2 import *
from adaptivity2 import metric_pnorm
from pylab import hold, show, figure, tricontourf, triplot, colorbar, savefig
from pylab import plot as pyplot
from numpy import array
from sympy import Symbol, diff, Subs
from sympy import sin as pysin
from sympy import atan as pyatan
set_log_level(WARNING)

def maximal_example():
    ### CONSTANTS
    meshsz = 40
    hd = Constant(1e-2)
    ### SETUP MESH
    mesh = RectangleMesh(-0.75,-0.25,0.,0.75,1*meshsz,1*meshsz,"left/right")
    ### SETUP SOLUTION
    #testsol = '0.1*sin(50*x+2*pi*t/T)+atan(-0.1/(2*x - sin(5*y+2*pi*t/T)))';
    sx = Symbol('sx'); sy = Symbol('sy'); sT = Symbol('sT'); st = Symbol('st');  spi = Symbol('spi')
    testsol = 0.1*pysin(50*sx+2*spi*st/sT)+pyatan(-0.1/(2*sx - pysin(5*sy+2*spi*st/sT)))
    ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('spi','pi')
    ddtestsol = ddtestsol.replace('sT','2*pi').replace('st','1.')
    testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('spi','pi').replace('sT','2*pi').replace('st','1.')
    # replacing **P with pow(,P)
    ddtestsol = '-250.0*sin(50*x[0] + 2*pi*1./2*pi) + 2.5*sin(5*x[1] + 2*pi*1./2*pi)' \
    +'/((1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2))*pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2)) ' \
    +'- 5.0*pow(cos(5*x[1] + 2*pi*1./2*pi),2)/((1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2))'\
    +'*pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),3)) - 0.8/((1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2))'\
    +'*pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),3)) + 0.05*pow(cos(5*x[1] + 2*pi*1./2*pi),2)'\
    +'/(pow(1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2),2)*pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),5)) '\
    +'+0.008/(pow(1 + 0.01/pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),2),2)*pow(2*x[0] - sin(5*x[1] + 2*pi*1./2*pi),5))'
    # PERFORM ONE ADAPTATION ITERATION
    for iii in range(5):
     startTime = time()
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
     R = interpolate(Expression(ddtestsol),V)
     a = inner(grad(dis), grad(dus))*dx
     L = R*dus*dx
     solve(a == L, u, [])
     soltime = time()-startTime
     
     startTime = time()
     eta = 0.001; H = metric_pnorm(u, mesh, eta, max_edge_ratio=50);   Mp =  project(H,  TensorFunctionSpace(mesh, "CG", 1))
     metricTime = time()-startTime
     mesh = adapt(Mp) 
     TadaptTime = time()-startTime
     print("%5.0f elements: adapt took %0.0f %% of the total time, (%0.0f %% of which was the metric calculation)" % (mesh.num_cells(),TadaptTime/(TadaptTime+soltime)*100,metricTime/TadaptTime*100))
    
    # PLOT MESH
    figure(1)
    coords = mesh.coordinates().transpose()
    triplot(coords[0],coords[1],mesh.cells(),linewidth=0.1)
    #savefig('mesh.png',dpi=300) #savefig('mesh.eps'); 
    
    testf = interpolate(Expression(testsol),FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
        
    figure(2)
    zz = testf.vector().array()[vtx2dof]
    hh=tricontourf(coords[0],coords[1],mesh.cells(),zz,100)
    colorbar(hh)
    #savefig('solution.png',dpi=300) #savefig('solution.eps'); 
    show()

if __name__=="__main__":
 maximal_example()