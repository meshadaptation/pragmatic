### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to
### 1. derive a forcing term that gives a step function solution
### 2. solve a poisson equation with the forcing term using 3D anisotropic adaptivity
### 3. calculate and plot the L2error of the resulting solution.
### the width of the step, the number of solition<->adaptation iterations as well as the
### error level (eta) are optional input parameters

from dolfin import *
from adaptivity import metric_pnorm, logproject, adapt
from sympy import Symbol, diff
from sympy import tanh as pytanh
from sympy import cos as pysin
from sympy import sin as pycos
set_log_level(INFO+1)
#parameters["allow_extrapolation"] = True

def minimal_example3D(width=2e-2, Nadapt=10, eta = 0.04):
    ### CONSTANTS
    meshsz = 10
    ### SETUP MESH
    mesh = BoxMesh(Point(-0.5,-0.5,-0.5),Point(0.5,0.5,0.5),meshsz,meshsz,meshsz)
    ### DERIVE FORCING TERM
    angle = pi/8 #rand*pi/2
    angle2 = pi/8 #rand*pi/2
    sx = Symbol('sx'); sy = Symbol('sy'); sz = Symbol('sz'); width_ = Symbol('ww'); aa = Symbol('aa'); bb = Symbol('bb')
    testsol = pytanh((sx*pycos(aa)*pysin(bb)+sy*pysin(aa)*pysin(bb)+sz*pycos(bb))/width_)
    ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)+diff(testsol,sz,sz)).replace('sx','x[0]').replace('sy','x[1]').replace('sz','x[2]')
    #replace ** with pow
    ddtestsol = ddtestsol.replace('tanh((x[0]*sin(aa)*cos(bb) + x[1]*cos(aa)*cos(bb) + x[2]*sin(bb))/ww)**2','pow(tanh((x[0]*sin(aa)*sin(bb) + x[1]*cos(aa)*sin(bb) + x[2]*cos(bb))/ww),2.)')
    ddtestsol = ddtestsol.replace('cos(aa)**2','pow(cos(aa),2.)').replace('sin(aa)**2','pow(sin(aa),2.)').replace('ww**2','(ww*ww)').replace('cos(bb)**2','(cos(bb)*cos(bb))').replace('sin(bb)**2','(sin(bb)*sin(bb))')
    #insert vaulues
    ddtestsol = ddtestsol.replace('aa',str(angle)).replace('ww',str(width)).replace('bb',str(angle2))
    testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('aa',str(angle)).replace('ww',str(width)).replace('bb',str(angle2)).replace('sz','x[2]')
    ddtestsol = "-("+ddtestsol+")"
    def boundary(x):
          return x[0]-mesh.coordinates()[:,0].min() < DOLFIN_EPS or mesh.coordinates()[:,0].max()-x[0] < DOLFIN_EPS \
          or mesh.coordinates()[:,1].min() < DOLFIN_EPS or mesh.coordinates()[:,1].max()-x[1] < DOLFIN_EPS \
          or mesh.coordinates()[:,2].min() < DOLFIN_EPS or mesh.coordinates()[:,2].max()-x[2] < DOLFIN_EPS
    # PERFORM TEN ADAPTATION ITERATIONS
    fid  = File("out.pvd")
    for iii in range(Nadapt):
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
     a = inner(grad(dis), grad(dus))*dx
     L = Expression(ddtestsol)*dus*dx
     bc = DirichletBC(V, Expression(testsol), boundary)
     solve(a == L, u, bc)
     fid << u
     startTime = time()
     H = metric_pnorm(u, eta, max_edge_length=2., max_edge_ratio=50, CG1out=True)
     #H = logproject(H)
     if iii != Nadapt-1:
      mesh = adapt(H) 
      L2error = errornorm(Expression(testsol), u, degree_rise=4, norm_type='L2')
      log(INFO+1,"total (adapt+metric) time was %0.1fs, L2error=%0.0e, nodes: %0.0f" % (time()-startTime,L2error,mesh.num_vertices()))
    

    plot(u,interactive=True)
    plot(mesh,interactive=True)


if __name__=="__main__":
 minimal_example3D()
