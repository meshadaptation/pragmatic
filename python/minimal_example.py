### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to
### 1. derive a forcing term that gives a step function solution
### 2. solve a poisson equation with the forcing term using 2D anisotropic adaptivity
### 3. calculate and plot the L2error of the resulting solution.
### the width of the step, the number of solition<->adaptation iterations as well as the
### error level (eta) are optional input parameters

from fenics import *
from adaptivity import metric_pnorm, logproject, adapt

from time import time
from numpy import array, ones
import numpy
from sympy import Symbol, diff
from sympy import tanh as pytanh
from sympy import cos as pysin
from sympy import sin as pycos
#parameters["allow_extrapolation"] = True

import matplotlib.pyplot as plt

def minimal_example(width=2e-2, Nadapt=10, eta = 0.01):
    ### CONSTANTS
    meshsz = 40
    hd = Constant(width)

    ### SETUP MESH
    mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5),1*meshsz,1*meshsz,"left/right")

    ### DERIVE FORCING TERM
    angle = pi/8 #rand*pi/2
    sx = Symbol('sx'); sy = Symbol('sy'); width_ = Symbol('ww'); aa = Symbol('aa')
    testsol = pytanh((sx*pycos(aa)+sy*pysin(aa))/width_)
    ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]')

    #replace ** with pow
    ddtestsol = ddtestsol.replace('tanh((x[0]*sin(aa) + x[1]*cos(aa))/ww)**2','pow(tanh((x[0]*sin(aa) + x[1]*cos(aa))/ww),2.)')
    ddtestsol = ddtestsol.replace('cos(aa)**2','pow(cos(aa),2.)').replace('sin(aa)**2','pow(sin(aa),2.)').replace('ww**2','(ww*ww)')
    
    #insert vaulues
    ddtestsol = ddtestsol.replace('aa',str(angle)).replace('ww',str(width))
    testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('aa',str(angle)).replace('ww',str(width))
    ddtestsol = "-("+ddtestsol+")"
    
    def boundary(x):
        return x[0]-mesh.coordinates()[:,0].min() < DOLFIN_EPS or mesh.coordinates()[:,0].max()-x[0] < DOLFIN_EPS \
        or mesh.coordinates()[:,1].min()+0.5 < DOLFIN_EPS or mesh.coordinates()[:,1].max()-x[1] < DOLFIN_EPS  

    # PERFORM TEN ADAPTATION ITERATIONS
    for iii in range(Nadapt):
        V = FunctionSpace(mesh, "CG" ,2)
        dis = TrialFunction(V)
        dus = TestFunction(V)
        u = Function(V)
        a = inner(grad(dis), grad(dus))*dx
        L = Expression(ddtestsol, degree=2)*dus*dx
        bc = DirichletBC(V, Expression(testsol, degree=2), boundary)
        solve(a == L, u, bc)
        startTime = time()
        H = metric_pnorm(u, eta, max_edge_length=3., max_edge_ratio=None)
        H = logproject(H)
        if iii != Nadapt-1:
            mesh, mf = adapt(H)
            L2error = errornorm(Expression(testsol, degree=2), u, degree_rise=4, norm_type='L2')
            print("total (adapt+metric) time was %0.1fs, L2error=%0.0e, nodes: %0.0f" % (time()-startTime,L2error,mesh.num_vertices()))
    
    #    # PLOT MESH
    coords = mesh.coordinates().transpose()

    import os
    save_dir = 'output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    plt.figure() #solution
    testf = interpolate(Expression(testsol, degree=2),FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
    zz = testf.vector()[vtx2dof]
    hh=plt.tricontourf(coords[0],coords[1],mesh.cells(),zz,100)
    plt.colorbar(hh)
    plt.savefig(f'{save_dir}/solution.png',dpi=300)
    
    plt.figure() #analytical solution
    testfe = interpolate(u,FunctionSpace(mesh,'CG',1))
    zz = testfe.vector()[vtx2dof]
    hh=plt.tricontourf(coords[0],coords[1],mesh.cells(),zz,100)
    plt.colorbar(hh)
    plt.savefig(f'{save_dir}/analyt.png',dpi=300)
    
    plt.figure() #error
    zz -= testf.vector()[vtx2dof]; zz[zz==1] -= 1e-16
    hh=plt.tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap='binary')
    plt.colorbar(hh)
    plt.triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),color='r',linewidth=0.5)
    plt.savefig(f'{save_dir}/error.png',dpi=300)

if __name__=="__main__":
    minimal_example()
 
