### this a testcase for use with DOLFIN/FEniCS and PRAgMaTIc 
### by Kristian Ejlebjerg Jensen, January 2014, Imperial College London
### the purpose of the test case is to illustrate the use of the mesh_metric2 function in connection
### with a stabilization term in an advective problem.
### the codes simulates a pressure driven stokes flow with a contraction implemented by means of 
### spatially varying damping/permeability field.
### The velocity field is used in advective problem to convect step function from the inlet and since
### the geometry(and stokes equation) is symmetric the profile at the outlet should converge towards
### the inlet profile
### the step width, number of solution<->adaptation iterations are optional input 
### paramters. The input parameters delta, relp and use_reform are related to a reformulation of the
### advective equation aiming to enforce upper and lower bounds. One however have to allow some 
### deviation (intrinsic to the reformulation), and the problem becomes so non-linear that the solver fails

from dolfin import *
from adaptivity import metric_pnorm, adapt, mesh_metric2, metric_ellipse
from pylab import polyfit, hold, show, triplot, tricontourf, colorbar, axis, box, get_cmap, figure, legend, savefig, xlabel, ylabel, title
from pylab import loglog as pyloglog
from pylab import plot as pyplot
from numpy import array, argsort, min, max
import pickle, os
from sympy import Symbol, diff
from sympy import sqrt as pysqrt
from numpy import abs as pyabs
from numpy import log as pylog
from numpy import exp as pyexp2
from numpy import sqrt as pysqrt2

#set_log_level(WARNING)
set_log_level(INFO+1)

def bnderror(u,ue,ds):
    mesh = u.function_space().mesh()
    V = FunctionSpace(mesh,'CG',4)
    u = interpolate(u,V); ue = interpolate(ue,V)
    u.vector().set_local(u.vector().array()-ue.vector().array())
    area = assemble(interpolate(Constant(1.),FunctionSpace(mesh,'CG',1))*dx) #assemble(Constant(1)*ds(1))
    error = pysqrt2(assemble(u**2*ds(1)))
    return error/area

def adv_convergence(width=2e-2, delta=1e-2, relp=1, Nadapt=10, use_adapt=True, problem=3, outname='', use_reform=False, CGorderL = [2, 3], noplot=False, Lx=3.):
    ### SETUP SOLUTION
    sy = Symbol('sy'); width_ = Symbol('ww')
    if   problem == 3:
        stepfunc = 0.5+165./104./width_*sy-20./13./width_**3*sy**3-102./13./width_**5*sy**5+240./13./width_**7*sy**7
    elif problem == 2:
        stepfunc = 0.5+15./8./width_*sy-5./width_**3*sy**3+6./width_**5*sy**5 
    elif problem == 1:
        stepfunc = 0.5+1.5/width_*sy-2/width_**3*sy**3
    stepfunc = str(stepfunc).replace('sy','x[1]').replace('x[1]**2','(x[1]*x[1])')
    #REPLACE ** with pow
    stepfunc   = stepfunc.replace('x[1]**3','pow(x[1],3.)')
    stepfunc   = stepfunc.replace('x[1]**5','pow(x[1],5.)')
    stepfunc   = stepfunc.replace('x[1]**7','pow(x[1],7.)')
    testsol    = '(-ww/2 < x[1] && x[1] < ww/2 ? ' + stepfunc   +' : 0) + (ww/2<x[1] ? 1 : 0)'
    testsol = testsol.replace('ww**2','(ww*ww)').replace('ww**3','pow(ww,3.)').replace('ww**5','pow(ww,5.)').replace('ww**7','pow(ww,7.)')
    testsol = testsol.replace('ww',str(width))
        
    dp = Constant(1.)
    fac = Constant(1+2.*delta)
    delta = Constant(delta)
    
    def left(x, on_boundary): 
        return x[0] + Lx/2. < DOLFIN_EPS
    def right(x, on_boundary): 
        return x[0] - Lx/2. > -DOLFIN_EPS
    def top_bottom(x, on_boundary):
        return x[1] - 0.5 > -DOLFIN_EPS or x[1] + 0.5 < DOLFIN_EPS
    class Inletbnd(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] + Lx/2. < DOLFIN_EPS
    for CGorder in [2]: #CGorderL:
        dofs = []
        L2errors = []
        for eta in 0.04*pyexp2(-array(range(9))*pylog(2)/2):
            ### SETUP MESH
            meshsz = int(round(80*0.005/(eta*(bool(use_adapt)==False)+0.05*(bool(use_adapt)==True))))
            if not bool(use_adapt) and meshsz > 80:
                continue
            
            mesh = RectangleMesh(-Lx/2.,-0.5,Lx/2.,0.5,meshsz,meshsz,"left/right")
            # PERFORM TEN ADAPTATION ITERATIONS
            for iii in range(Nadapt):
             V = VectorFunctionSpace(mesh, "CG", CGorder); Q = FunctionSpace(mesh, "CG", CGorder-1)
             W = V*Q
             (u, p) = TrialFunctions(W)
             (v, q) = TestFunctions(W)
             alpha = Expression("-0.25<x[0] && x[0]<0.25 && 0. < x[1] ? 1e4 : 0")
    
             boundaries = FacetFunction("size_t",mesh)                         
             #outletbnd = Outletbnd()
             inletbnd = Inletbnd()
             boundaries.set_all(0)
             #outletbnd.mark(boundaries, 1)
             inletbnd.mark(boundaries, 1)
             ds = Measure("ds")[boundaries]
             bc0 = DirichletBC(W.sub(0), Constant((0., 0.)), top_bottom)
             bc1 = DirichletBC(W.sub(1), dp         ,  left)
             bc2 = DirichletBC(W.sub(1), Constant(0), right)
             bc3 = DirichletBC(W.sub(0).sub(1), Constant(0.), left)
             bc4 = DirichletBC(W.sub(0).sub(1), Constant(0.), right)
             bcs = [bc0,bc1,bc2,bc3,bc4]
             
             bndterm = dp*dot(v,Constant((-1.,0.)))*ds(1)
             
             a = eta*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx+alpha*dot(u,v)*dx#+bndterm
             L = inner(Constant((0.,0.)), v)*dx -bndterm
             U = Function(W)
             solve(a == L, U,  bcs)
             u, ps = U.split()
             
             #SOLVE CONCENTRATION
             mm = mesh_metric2(mesh)
             vdir = u/sqrt(inner(u,u)+DOLFIN_EPS)             
             if iii == 0 or use_reform == False:
                 Q2 = FunctionSpace(mesh,'CG',2); c = Function(Q2)
             q = TestFunction(Q2); p = TrialFunction(Q2)
             newq = (q+dot(vdir,dot(mm,vdir))*inner(grad(q),vdir)) #SUPG
             if use_reform:
                 F = newq*(fac/((1+exp(-c))**2)*exp(-c))*inner(grad(c),u)*dx
                 J = derivative(F,c)
                 bc = DirichletBC(Q2, Expression("-log("+str(float(fac)) +"/("+testsol+"+"+str(float(delta))+")-1)"), left)
    #                 bc = DirichletBC(Q, -ln(fac/(Expression(testsol)+delta)-1), left)
                 problem = NonlinearVariationalProblem(F,c,bc,J)
                 solver = NonlinearVariationalSolver(problem)
                 solver.parameters["newton_solver"]["relaxation_parameter"] = relp
                 solver.solve()
             else:
                 a2 = newq*inner(grad(p),u)*dx
                 bc = DirichletBC(Q2, Expression(testsol), left)
                 L2 = Constant(0.)*q*dx
                 solve(a2 == L2, c, bc)
    
             if (not bool(use_adapt)) or iii == Nadapt-1:
                 break
             um = project(sqrt(inner(u,u)),FunctionSpace(mesh,'CG',2))
             H  = metric_pnorm(um, eta, max_edge_ratio=1+49*(use_adapt!=2), p=2)
             H2 = metric_pnorm(c, eta, max_edge_ratio=1+49*(use_adapt!=2), p=2)
             #H3 = metric_pnorm(ps , eta, max_edge_ratio=1+49*(use_adapt!=2), p=2)
             H4 = metric_ellipse(H,H2)
             #H5 = metric_ellipse(H3,H4,mesh)
             mesh = adapt(H4)
             if use_reform:
                Q2 = FunctionSpace(mesh,'CG',2)
                c = interpolate(c,Q2)
         
            if use_reform:
             c = project(fac/(1+exp(-c))-delta,FunctionSpace(mesh,'CG',2))
            L2error = bnderror(c,Expression(testsol),ds)
            dofs.append(len(c.vector().array())+len(U.vector().array()))
            L2errors.append(L2error)
#            fid = open("DOFS_L2errors_mesh_c_CG"+str(CGorder)+outname+".mpy",'w')
#            pickle.dump([dofs[0],L2errors[0],c.vector().array().min(),c.vector().array().max()-1,mesh.cells(),mesh.coordinates(),c.vector().array()],fid)
#            fid.close();
            log(INFO+1,"%1dX ADAPT<->SOLVE complete: DOF=%5d, error=%0.0e, min(c)=%0.0e,max(c)-1=%0.0e" % (Nadapt, dofs[len(dofs)-1], L2error,c.vector().array().min(),c.vector().array().max()-1))
        
        # PLOT MESH + solution
        figure()
        testf  = interpolate(c                  ,FunctionSpace(mesh,'CG',1))
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
        xe = interpolate(Expression("x[0]"),FunctionSpace(mesh,'CG',1)).vector().array()
        ye = interpolate(Expression("x[1]"),FunctionSpace(mesh,'CG',1)).vector().array()
        I = xe - Lx/2 > -DOLFIN_EPS; I2 = ye[I].argsort()
        pyplot(ye[I][I2],testf.vector().array()[I][I2]-testfe.vector().array()[I][I2],'-b'); ylabel('error')
        # PLOT L2error graph
        figure()
        pyloglog(dofs,L2errors,'-b.',linewidth=2,markersize=16); xlabel('Degree of freedoms'); ylabel('L2 error')
        # SAVE SOLUTION
        dofs = array(dofs); L2errors = array(L2errors)
        fid = open("DOFS_L2errors_CG"+str(CGorder)+outname+".mpy",'w')
        pickle.dump([dofs,L2errors],fid)
        fid.close();
#        #show()
    
#    #LOAD SAVED SOLUTIONS
#    fid = open("DOFS_L2errors_CG2"+outname+".mpy",'r')
#    [dofs,L2errors] = pickle.load(fid)
#    fid.close()
#    
    # PERFORM FITS ON LAST THREE POINTS
    NfitP = 5
    I = array(range(len(dofs)-NfitP,len(dofs)))
    slope,ints   = polyfit(pylog(dofs[I]), pylog(L2errors[I]), 1) 
    fid = open("DOFS_L2errors_CG2_fit"+outname+".mpy",'w')
    pickle.dump([dofs,L2errors,slope,ints],fid)
    fid.close()
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
 adv_convergence()
 
