from dolfin import *
from adaptivity2 import *
from adaptivity2 import metric_pnorm, metric_ellipse
from pylab import hold, show, figure, triplot, rand, tricontourf, axis, box
from pylab import plot as pyplot
from numpy import array, ones

def check_metric_ellipse(width=2e-2):
    set_log_level(WARNING)
    parameters["allow_extrapolation"] = True
    
    ### CONSTANTS
    meshsz = 40
    hd = Constant(width)
    ### SETUP MESH
    mesh = RectangleMesh(-0.5,-0.5,0.5,0.5,1*meshsz,1*meshsz,"left/right")
    ### SETUP SOLUTION
    angle = pi/8#rand()*pi/2 
    #testsol = 'tanh(x[0]/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    testsol = 'tanh((' + str(cos(angle)) + '*x[0]+'+ str(sin(angle)) + '*x[1])/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    ddtestsol = str(cos(angle)+sin(angle))+'*2*'+testsol+'*(1-pow('+testsol+',2))/'+str(float(hd)**2)
    #testsol2 = 'tanh(x[1]/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    testsol2 = 'tanh((' + str(cos(angle)) + '*x[1]-'+ str(sin(angle)) + '*x[0])/' + str(float(hd)) + ')' #tanh(x[0]/hd)
    ddtestsol2 = str(cos(angle)-sin(angle))+'*2*'+testsol2+'*(1-pow('+testsol2+',2))/'+str(float(hd)**2)
    
    # PERFORM ONE ADAPTATION ITERATION
    for iii in range(6):
     V = FunctionSpace(mesh, "CG" ,2); dis = TrialFunction(V); dus = TestFunction(V); u = Function(V); u2 = Function(V)
     R = interpolate(Expression(ddtestsol),V)
     R2 = interpolate(Expression(ddtestsol2),V)
     a = inner(grad(dis), grad(dus))*dx
     L = R*dus*dx
     L2 = R2*dus*dx
     solve(a == L, u, [])
     solve(a == L2, u2, [])
     
     eta = 0.02; H = metric_pnorm(u, mesh, eta, max_edge_ratio=50);   Mp =  project(H,  TensorFunctionSpace(mesh, "CG", 1))
     eta = 0.02; H2 = metric_pnorm(u2, mesh, eta, max_edge_ratio=50); Mp2 = project(H2, TensorFunctionSpace(mesh, "CG", 1))
     H3 = metric_ellipse(H,H2,mesh); Mp3 = project(H3, TensorFunctionSpace(mesh, "CG", 1))
     print("H11: %0.0f, H22: %0.0f, V: %0.0f,E: %0.0f" % (assemble(abs(H3[0,0])*dx),assemble(abs(H3[1,1])*dx),mesh.num_vertices(),mesh.num_cells()))
     startTime = time()
     if iii != 6:
     # mesh2 = Mesh(adapt(Mp2))
       mesh = Mesh(adapt(Mp3))
     # mesh3 = adapt(Mp)
      
     print("total time was %0.1fs" % (time()-startTime))
    
    # PLOT MESH
    figure(1); triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells()); axis('equal'); axis('off'); box('off')
    #figure(2); triplot(mesh2.coordinates()[:,0],mesh2.coordinates()[:,1],mesh2.cells()) #mesh = mesh2
    #figure(3); triplot(mesh3.coordinates()[:,0],mesh3.coordinates()[:,1],mesh3.cells()) #mesh = mesh3
    figure(4); testf = interpolate(u2,FunctionSpace(mesh,'CG',1))
    vtx2dof = vertex_to_dof_map(FunctionSpace(mesh, "CG" ,1))
    zz = testf.vector().array()[vtx2dof]; zz[zz==1] -= 1e-16
    tricontourf(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells(),zz,100)
    show()

if __name__=="__main__":
 check_metric_ellipse() 