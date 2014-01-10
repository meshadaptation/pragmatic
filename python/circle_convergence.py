from dolfin import *
from adaptivity2 import metric_pnorm, fix_CG1_metric, adapt
from pylab import polyfit, hold, show, triplot, tricontourf, colorbar, axis, box, get_cmap, figure, legend, savefig, xlabel, ylabel, title
from pylab import loglog as pyloglog
from numpy import array
import pickle, os
from sympy import Symbol, diff
from sympy import exp as pyexp
from sympy import tanh as pytanh
from sympy import sqrt as pysqrt
from numpy import log as pylog
from numpy import exp as pyexp2
from numpy import abs as pyabs

set_log_level(WARNING)

def circle_convergence(width=20e-2, Nadapt=10, use_adapt=False, problem=1):
    ### SETUP SOLUTION
    sx = Symbol('sx'); sy = Symbol('sy'); width_ = Symbol('ww')
    if problem == 1:
        testsol = pytanh(-((sx*sx+sy*sy)**2-0.25**2)/width_/width_)
        ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = testsol.replace('ww**2','(ww*ww)')
        #REPLACE ** with pow
        testsol = testsol.replace('((x[0]*x[0]) + (x[1]*x[1]))**2','pow(x[0]*x[0] + x[1]*x[1],2.)')
        ddtestsol = ddtestsol.replace('((x[0]*x[0]) + (x[1]*x[1]))**2','pow(x[0]*x[0] + x[1]*x[1],2.)')
        ddtestsol = ddtestsol.replace('tanh((-pow(x[0]*x[0] + x[1]*x[1],2.) + 0.0625)/ww**2)**2','pow(tanh((-pow(x[0]*x[0] + x[1]*x[1],2.) + 0.0625)/ww**2),2.)')
    elif problem == 2:
        testsol = pytanh(-(pysqrt(sx*sx+sy*sy)-0.25)**2/width_/width_)
        ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = testsol.replace('ww**2','(ww*ww)')
        #REPLACE ** with pow
        testsol = testsol.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
        ddtestsol = ddtestsol.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
        ddtestsol = ddtestsol.replace('tanh(pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)/ww**2)**2','pow(tanh(pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)/ww**2),2.)')
        ddtestsol = ddtestsol.replace('((x[0]*x[0]) + (x[1]*x[1]))**(3/2)','pow(x[0]*x[0] + x[1]*x[1],1.5)')
    else:
        testsol = pyexp(-(pysqrt(sx*sx+sy*sy)-0.25)**2/width_/width_)
        ddtestsol = str(diff(testsol,sx,sx)+diff(testsol,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = str(testsol).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        testsol = testsol.replace('ww**2','(ww*ww)')
        #REPLACE ** with pow
        testsol = testsol.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
        ddtestsol = ddtestsol.replace('((x[0]*x[0]) + (x[1]*x[1]))**(3/2)','pow(x[0]*x[0] + x[1]*x[1],1.5)')
        ddtestsol = ddtestsol.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
    ddtestsol = ddtestsol.replace('ww**2','(ww*ww)').replace('ww**4','pow(ww,4.)')
    ddtestsol = ddtestsol.replace('ww',str(width))
    testsol = testsol.replace('ww',str(width))
    ddtestsol = '-('+ddtestsol+')'
    def boundary(x):
          return x[0]+0.51 < DOLFIN_EPS or 0.5-x[0] < DOLFIN_EPS or x[1]+0.5 < DOLFIN_EPS or 0.5-x[1] < DOLFIN_EPS  
    
    for CGorder in [2, 3]:
        dofs = []
        L2errors = []
        for eta in [0.16, 0.08, 0.04, 0.02, 0.01, 0.005]:     
            ### SETUP MESH
            meshsz = int(round(80*0.005/eta))
            mesh = RectangleMesh(-0.51,-0.5,0.5,0.5,meshsz,meshsz,"left/right")
            # PERFORM TEN ADAPTATION ITERATIONS
            for iii in range(Nadapt):
             V = FunctionSpace(mesh, "CG", CGorder); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
             V2 = FunctionSpace(mesh, "CG", CGorder+2)
             R = interpolate(Expression(ddtestsol),V2)
             a = inner(grad(dis), grad(dus))*dx
             L = R*dus*dx
             bc = DirichletBC(V, Expression(testsol), boundary) #Constant(0.)
             solve(a == L, u, bc)
             if not use_adapt:
                 break
             H = metric_pnorm(u, mesh, eta, max_edge_ratio=50, max_edge_length=1, p=2)
             Mp =  project(H,  TensorFunctionSpace(mesh, "CG", 1))
             Mp = fix_CG1_metric(Mp)
             if iii != Nadapt-1:
              mesh = adapt(Mp)
            
            L2error = errornorm(Expression(testsol), u, degree_rise=CGorder+2, norm_type='L2')
            dofs.append(len(u.vector().array()))
            L2errors.append(L2error)
            print("%1dX ADAPT<->SOLVE complete: DOF=%5d, error=%0.0e" % (Nadapt, dofs[len(dofs)-1],
                                                                                      L2error))
        # PLOT MESH + solution
        figure()
        testf = interpolate(u,FunctionSpace(mesh,'CG',1))
        vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
        zz = testf.vector().array()[vtx2dof]; zz[zz==1] -= 1e-16
        hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
        colorbar(hh)
        hold('on'); triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),color='r',linewidth=0.5); hold('off')
        axis('equal'); box('off')
        #PLOT ERROR
        figure()
        testf  = interpolate(u                  ,FunctionSpace(mesh,'CG',1))
        testfe = interpolate(Expression(testsol),FunctionSpace(mesh,'CG',1))
        vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
        zz = pyabs(testf.vector().array()-testfe.vector().array())[vtx2dof]; zz[zz==1] -= 1e-16
        hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
        colorbar(hh); axis('equal'); box('off'); title('error')
    #    savefig('final_mesh_CG2.png',dpi=300) #; savefig('final_mesh_CG2.eps',dpi=300)
        # PLOT L2error graph
        figure()
        pyloglog(dofs,L2errors,'-b.',linewidth=2,markersize=16); xlabel('Degree of freedoms'); ylabel('L2 error')
        # SAVE SOLUTION
        fid = open("DOFS_L2errors_CG"+str(CGorder)+".mpy",'w')
        dofs = array(dofs); L2errors = array(L2errors)
        pickle.dump([dofs,L2errors],fid)
        fid.close()
    
    #LOAD SAVED SOLUTIONS
    fid = open("DOFS_L2errors_CG2.mpy",'r')
    [dofs,L2errors] = pickle.load(fid)
    fid.close()
    fid = open("DOFS_L2errors_CG3.mpy",'r')
    [dofs_old,L2errors_old] = pickle.load(fid)
    fid.close()
    
    # PERFORM FITS ON LAST THREE POINTS
    I = array(range(len(dofs)-3,len(dofs)))
    slope,ints   = polyfit(pylog(dofs[I]), pylog(L2errors[I]), 1) 
    slope2,ints2 = polyfit(pylog(dofs_old[I]), pylog(L2errors_old[I]), 1) 
    #PLOT THEM TOGETHER
    figure()
    pyloglog(dofs,L2errors,'-b.',dofs_old,L2errors_old,'--b.',linewidth=2,markersize=16)
    hold('on'); pyloglog(dofs,pyexp2(ints)*dofs**slope,'-r',dofs_old,pyexp2(ints2)*dofs_old**slope2,'--r',linewidth=1); hold('off')
    xlabel('Degree of freedoms'); ylabel('L2 error')
    legend(['CG2','CG3',"%0.2f*log(DOFs)" % slope, "%0.2f*log(DOFs)" % slope2]) #legend(['new data','old_data'])
    savefig('comparison.png',dpi=300) #savefig('comparison.eps'); 
    show()
    
if __name__=="__main__":
 circle_convergence()
 