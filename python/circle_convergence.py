### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to
### 1. derive a forcing term that gives rise to a step function with a curvature (hence the name)
### 2. solve a poisson equation with the forcing term using 2D anisotropic adaptivity
### 3. calculate and plot the L2error of the resulting solution as function of the number of degrees
### of freedom for 2nd and third order discretization.
### the width of the step and the number of solition<->adaptation iterations
### are optional input parameters. Furthermore the continuity of the step function
### can be controlled by, i.e. problem=2 gives continuity of 2nd order derivatives 
### the options use_adapt, noplot and outname relate to the possibility of making a concise graph comparing
### no, isotropic and anisotropic adaptation convergence as a function of the step width.

from dolfin import *
from adaptivity3 import metric_pnorm, adapt
from pylab import polyfit, hold, show, triplot, tricontourf, colorbar, axis, box, get_cmap, figure, legend, savefig, xlabel, ylabel, title
from pylab import loglog as pyloglog
from numpy import array
import pickle, os
from sympy import Symbol, diff
from sympy import sqrt as pysqrt
from numpy import abs as pyabs
from numpy import log as pylog
from numpy import exp as pyexp2

#set_log_level(WARNING)
set_log_level(INFO+1)

def circle_convergence(width=1e-2, Nadapt=10, use_adapt=True, problem=2, outname='', CGorderL = [2, 3], noplot=False):
    ### SETUP SOLUTION
    sx = Symbol('sx'); sy = Symbol('sy'); width_ = Symbol('ww')
    if problem == -2:
        stepfunc = 0.5+165./104./width_*sx-20./13./width_**3*sx**3-102./13./width_**5*sx**5+240./13./width_**7*sx**7
        ddstepfunc = str(diff(stepfunc,sx,sx)).replace('sx','x[0]').replace('x[0]**2','(x[0]*x[0])')
        stepfunc = str(stepfunc).replace('sx','x[0]').replace('x[0]**2','(x[0]*x[0])')
        #REPLACE ** with pow
        ddstepfunc = ddstepfunc.replace('x[0]**3','pow(x[0],3.)')
        ddstepfunc = ddstepfunc.replace('x[0]**5','pow(x[0],5.)')
        stepfunc   = stepfunc.replace('x[0]**3','pow(x[0],3.)')
        stepfunc   = stepfunc.replace('x[0]**5','pow(x[0],5.)')
        stepfunc   = stepfunc.replace('x[0]**7','pow(x[0],7.)')
        stepfunc     = '(-ww/2 < x[0] && x[0] < ww/2 ? ' + stepfunc   +' : 0) + (ww/2<x[0] ? 1 : 0)'
        ddstepfunc   = '(-ww/2 < x[0] && x[0] < ww/2 ? ' + ddstepfunc +' : 0)'
        stepfuncy   =   stepfunc.replace('x[0]','x[1]')
        ddstepfuncy = ddstepfunc.replace('x[0]','x[1]')
        testsol     = '(x[0] > x[1] ? (' + stepfunc     + ') : 0) + (x[0] <= x[1] ? (' + stepfuncy   + ') : 0)'
        ddtestsol   = '(x[0] > x[1] ? (' + ddstepfunc   + ') : 0) + (x[0] <= x[1] ? (' + ddstepfuncy + ') : 0)'
    elif problem == 2:
#        A = array([[0.5,0.5**3,0.5**5,0.5**7],[1,3*0.5**2,5*0.5**4,7*0.5**6],[0,6*0.5,20*0.5**3,42*0.5**5],[0,6,60*0.5**2,110*0.5**4]]); b = array([0.5,0,0,0])
#        from numpy.linalg import solve as pysolve #165/104, -20/13, -102/13,240/13
#        X = pysolve(A,b); from numpy import linspace; xx = linspace(-0.5,0.5,100)
#        from pylab import plot as pyplot; pyplot(xx,X[0]*xx+X[1]*xx**3+X[2]*xx**5+X[3]*xx**7,'-b')
        rrpy = pysqrt(sx*sx+sy*sy)
        stepfunc = 0.5+165./104./width_*(rrpy-0.25)-20./13./width_**3*(rrpy-0.25)**3-102./13./width_**5*(rrpy-0.25)**5+240./13./width_**7*(rrpy-0.25)**7
        ddstepfunc = str(diff(stepfunc,sx,sx)+diff(stepfunc,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        stepfunc = str(stepfunc).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        #REPLACE ** with pow
        ddstepfunc = ddstepfunc.replace('((x[0]*x[0]) + (x[1]*x[1]))**(3/2)','pow(x[0]*x[0] + x[1]*x[1],1.5)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**3','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,3.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**4','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,4.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**5','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,5.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**6','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,6.)')
        stepfunc   = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**3','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,3.)')
        stepfunc   = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**5','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,5.)')
        stepfunc   = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**7','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,7.)')
        testsol   = '(0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + stepfunc   + ') : 0) + (0.25+ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) ? 1 : 0)'
#        testsol   = '(0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + stepfunc   + ')) : (0.25<sqrt(x[0]*x[0]+x[1]*x[1]) ? 1 : 0)'
        ddtestsol =  '0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + ddstepfunc + ') : 0' 
    elif problem == 1:
#        A = array([[0.5,0.5**3,0.5**5],[1,3*0.5**2,5*0.5**4],[0,6*0.5,20*0.5**3]]); b = array([0.5,0,0])
#        from numpy.linalg import solve as pysolve #15/8,-5,6
#        X = pysolve(A,b); from numpy import linspace; xx = linspace(-0.5,0.5,100)
#        from pylab import plot as pyplot; pyplot(xx,X[0]*xx+X[1]*xx**3+X[2]*xx**5,'-b')
        rrpy = pysqrt(sx*sx+sy*sy)
        stepfunc = 0.5+15./8./width_*(rrpy-0.25)-5./width_**3*(rrpy-0.25)**3+6./width_**5*(rrpy-0.25)**5 
        ddstepfunc = str(diff(stepfunc,sx,sx)+diff(stepfunc,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        stepfunc = str(stepfunc).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        #REPLACE ** with pow
        ddstepfunc = ddstepfunc.replace('((x[0]*x[0]) + (x[1]*x[1]))**(3/2)','pow(x[0]*x[0] + x[1]*x[1],1.5)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,2.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**3','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,3.)')
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**4','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,4.)')
        stepfunc   = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**3','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,3.)')
        stepfunc   = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**5','pow(sqrt(x[0]*x[0] + x[1]*x[1]) - 0.25,5.)')
        testsol   = '(0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + stepfunc   + ') : 0) + (0.25+ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) ? 1 : 0)'
#        testsol   = '(0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + stepfunc   + ')) : (0.25<sqrt(x[0]*x[0]+x[1]*x[1]) ? 1 : 0)'
        ddtestsol =  '0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + ddstepfunc + ') : 0' 
    else: # problem == 0:
        rrpy = pysqrt(sx*sx+sy*sy)
        stepfunc = 0.5+1.5/width_*(rrpy-0.25)-2/width_**3*(rrpy-0.25)**3 #'if(t<2*WeMax,0,if(t<4*WeMax,0.5+3/2/(2*WeMax)*(t-3*WeMax)-2/(2*WeMax)^3*(t-3*WeMax)^3,1))'; %0.5+3/2/dx*(x-xc)-2/dx^3*(x-xc)^3
        ddstepfunc = str(diff(stepfunc,sx,sx)+diff(stepfunc,sy,sy)).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        stepfunc = str(stepfunc).replace('sx','x[0]').replace('sy','x[1]').replace('x[0]**2','(x[0]*x[0])').replace('x[1]**2','(x[1]*x[1])')
        #REPLACE ** with pow
        ddstepfunc = ddstepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**2','pow(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25,2.)')
        ddstepfunc = ddstepfunc.replace('((x[0]*x[0]) + (x[1]*x[1]))**(3/2)','pow((x[0]*x[0]) + (x[1]*x[1]),1.5)')
        stepfunc = stepfunc.replace('(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25)**3','pow(sqrt((x[0]*x[0]) + (x[1]*x[1])) - 0.25,3.)')
        testsol   = '0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + stepfunc   + ') : (0.25<sqrt(x[0]*x[0]+x[1]*x[1]) ? 1 : 0)'
        ddtestsol = '0.25-ww/2<sqrt(x[0]*x[0]+x[1]*x[1]) && sqrt(x[0]*x[0]+x[1]*x[1]) < 0.25+ww/2 ? (' + ddstepfunc + ') : 0' 
    
    ddtestsol = ddtestsol.replace('ww**2','(ww*ww)').replace('ww**3','pow(ww,3.)').replace('ww**4','pow(ww,4.)').replace('ww**5','pow(ww,5.)').replace('ww**6','pow(ww,6.)').replace('ww**7','pow(ww,7.)')
    ddtestsol = ddtestsol.replace('ww',str(width))
    testsol = testsol.replace('ww**2','(ww*ww)').replace('ww**3','pow(ww,3.)').replace('ww**5','pow(ww,5.)').replace('ww**7','pow(ww,7.)')
    testsol = testsol.replace('ww',str(width))
    ddtestsol = '-('+ddtestsol+')'
    def boundary(x):
          return x[0]+0.5 < DOLFIN_EPS or 0.5-x[0] < DOLFIN_EPS or x[1]+0.5 < DOLFIN_EPS or 0.5-x[1] < DOLFIN_EPS  
    
    for CGorder in CGorderL:
        dofs = []
        L2errors = []
        #for eta in [0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025] #, 0.0025/2, 0.0025/4, 0.0025/8]: #
        for eta in 0.04*pyexp2(-array(range(15))*pylog(2)/2):
            ### SETUP MESH
            meshsz = int(round(80*0.005/(eta*(bool(use_adapt)==False)+0.05*(bool(use_adapt)==True))))
            if (not bool(use_adapt)) and meshsz > 80:
                continue
            
            mesh = RectangleMesh(-0.0,-0.0,0.5,0.5,meshsz,meshsz,"left/right")
            # PERFORM TEN ADAPTATION ITERATIONS
            for iii in range(Nadapt):
             V = FunctionSpace(mesh, "CG", CGorder); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V)
             #V2 = FunctionSpace(mesh, "CG", CGorder+2)
             R = Expression(ddtestsol) #interpolate(Expression(ddtestsol),V2)
             a = inner(grad(dis), grad(dus))*dx
             L = R*dus*dx
             bc = DirichletBC(V, Expression(testsol), boundary) #Constant(0.)
             solve(a == L, u, bc)
             if not bool(use_adapt):
                 break
             H = metric_pnorm(u, eta, max_edge_ratio=1+49*(use_adapt!=2), p=2)
             if iii != Nadapt-1:
              mesh = adapt(H)
            
            L2error = errornorm(Expression(testsol), u, degree_rise=CGorder+2, norm_type='L2')
            dofs.append(len(u.vector().array()))
            L2errors.append(L2error)
            log(INFO+1,"%1dX ADAPT<->SOLVE complete: DOF=%5d, error=%0.0e" % (Nadapt, dofs[len(dofs)-1], L2error))
        
        # PLOT MESH + solution
        figure()
        testf  = interpolate(u                  ,FunctionSpace(mesh,'CG',1))
        testfe = interpolate(Expression(testsol),FunctionSpace(mesh,'CG',1))
        vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
        zz = testf.vector().array()[vtx2dof]; zz[zz==1] -= 1e-16
        hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
        colorbar(hh)
        hold('on'); triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),color='r',linewidth=0.5); hold('off')
        axis('equal'); box('off')
#        savefig(outname+'final_mesh_CG2.png',dpi=300) #; savefig('outname+final_mesh_CG2.eps',dpi=300)
        #PLOT ERROR
        figure()
        zz = pyabs(testf.vector().array()-testfe.vector().array())[vtx2dof]; zz[zz==1] -= 1e-16
        hh=tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100,cmap=get_cmap('binary'))
        colorbar(hh); axis('equal'); box('off'); title('error')
        # PLOT L2error graph
        figure()
        pyloglog(dofs,L2errors,'-b.',linewidth=2,markersize=16); xlabel('Degree of freedoms'); ylabel('L2 error')
        # SAVE SOLUTION
        dofs = array(dofs); L2errors = array(L2errors)
        fid = open("DOFS_L2errors_CG"+str(CGorder)+outname+".mpy",'w')
        pickle.dump([dofs,L2errors],fid)
        fid.close();
    
    #LOAD SAVED SOLUTIONS
    fid = open("DOFS_L2errors_CG2"+outname+".mpy",'r')
    [dofs,L2errors] = pickle.load(fid)
    fid.close()
    
    # PERFORM FITS ON LAST THREE POINTS
    NfitP = 9
    I = array(range(len(dofs)-NfitP,len(dofs)))
    slope,ints   = polyfit(pylog(dofs[I]), pylog(L2errors[I]), 1) 
    if slope < -0.7:
     fid = open("DOFS_L2errors_CG2_fit"+outname+".mpy",'w')
     pickle.dump([dofs,L2errors,slope,ints],fid)
     fid.close()
     log(INFO+1,'succes')
    else:
     os.system('rm '+outname+'.lock')
     log(INFO+1,'fail')
    #PLOT THEM TOGETHER
    if CGorderL != [2]:
     fid = open("DOFS_L2errors_CG3.mpy",'r')
     [dofs_old,L2errors_old] = pickle.load(fid)
     fid.close()
     slope2,ints2 = polyfit(pylog(dofs_old[I]), pylog(L2errors_old[I]), 1) 
     figure()
     pyloglog(dofs,L2errors,'-b.',dofs_old,L2errors_old,'--b.',linewidth=2,markersize=16)
     hold('on'); pyloglog(dofs,pyexp2(ints)*dofs**slope,'-r',dofs_old,pyexp2(ints2)*dofs_old**slope2,'--r',linewidth=1); hold('off')
     xlabel('Degree of freedoms'); ylabel('L2 error')
     legend(['CG2','CG3',"%0.2f*log(DOFs)" % slope, "%0.2f*log(DOFs)" % slope2]) #legend(['new data','old_data'])
#     savefig('comparison.png',dpi=300) #savefig('comparison.eps'); 
    if not noplot:
     show()

if __name__=="__main__":
 circle_convergence()
 
