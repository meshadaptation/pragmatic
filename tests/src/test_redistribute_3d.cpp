#include <iostream>
#include <vector>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif

#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif

#include "MetricField.h"


#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"



int main(int argc, char **argv)
{
    int rank=0, num_processes=1;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    bool verbose = false;
    if(argc>1) {
        verbose = std::string(argv[1])=="-v";
    }

#ifdef HAVE_VTK
    char name[256];

    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10x10.vtu");
    mesh->create_boundary();

    MetricField<double,3> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

#if 1
    // define metric analytically
    double m[3] = {0};
    for(size_t i=0; i<NNodes; i++) {
        double lmax = 1/(0.05*0.05);
        m[0] = m[3] = m[5] = lmax;
        m[1] = m[2] = m[4] = 0;
//        double x = mesh->get_coords(i)[0];
//        double h = 0.5*fabs(1-exp(-fabs(x-0.5))) + 0.003;
//        double lbd = 1/(h*h);
//        double lmax = 1/(0.5*0.5);
//        m[0] = lbd;
//        m[1] = 0.0;
//        m[2] = lmax;
        metric_field.set_metric(m, i);
    }
#endif
#if 0
    // define metric from function
    std::vector<double> psi(NNodes);
    for(size_t i=0; i<NNodes; i++) {
        double x = mesh->get_coords(i)[0]-0.5;
        double y = mesh->get_coords(i)[1]-0.5;

        psi[i] = (50*fabs(x*y) >= 2*M_PI) ? 0.01*sin(50*x*y) : sin(50*x*y);
    }
    double eta=0.01;
    metric_field.add_field(&(psi[0]), eta, 2);
#endif
    metric_field.update_mesh();
    
//    GMFTools<double>::export_gmf_mesh("ex_metric2dV2", mesh);
//    GMFTools<double>::export_gmf_metric2d("ex_metric2dV2", &metric_field, mesh);
//    return 0;

    double L_up = sqrt(2.0);
    double L_low = L_up*0.5;

    Coarsen<double, 3> coarsen(*mesh);
    Smooth<double, 3> smooth(*mesh);
    Refine<double, 3> refine(*mesh);
    Swapping<double, 3> swapping(*mesh);

    double L_max = mesh->maximal_edge_length();

    double alpha = sqrt(2.0)/2.0;
    size_t i=0;
    for(i=0; i<10; i++) {
        if (rank==0) printf("DEBUG(%d)  ite adapt: %lu\n", rank, i);
        double L_ref = std::max(alpha*L_max, L_up);

        sprintf(name, "beforeadp.%lu", i);
        mesh->print_mesh(name);
        mesh->print_halo(name);
        
        refine.refine(L_ref);

        sprintf(name, "afterref.%lu", i);
        mesh->print_mesh(name);
        mesh->print_halo(name);

        if (i==2) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 3;
        }

        coarsen.coarsen(L_low, L_ref, false);

        sprintf(name, "aftercoa.%lu", i);
        mesh->print_mesh(name);
        mesh->print_halo(name);

        swapping.swap(0.95);

        sprintf(name, "afterswp.%lu", i);
        mesh->print_mesh(name);
        mesh->print_halo(name);

        L_max = mesh->maximal_edge_length();

        int ite_red = 3;
        if (i>0 && i%ite_red==0) {

            sprintf(name, "beforedef.%lu", i/ite_red);
            mesh->print_mesh(name);
            mesh->print_halo(name);

            if (rank==0) printf("DEBUG(%d)  %lu-th redistribution\n", rank, i/ite_red);

            mesh->fix_halos();

            sprintf(name, "beforered.%lu", i/ite_red);
            mesh->print_mesh(name);
            mesh->print_halo(name);

//            int tag = 2*(i%2)-1;
            int tag = (i/ite_red)%4 <2 ? 1 : -1;
            if (rank==0) printf("DEBUG  resdistribute to %s\n", (tag==1) ? "greater" : "lower");
//            int tag = 0;
            mesh->redistribute_halo(tag);

            sprintf(name, "afterred.%lu", i/ite_red);
            mesh->print_mesh(name);
            mesh->print_halo(name);

            MetricField<double,3> metric_field_new(*mesh);
            metric_field_new.set_metric(mesh->get_metric());
            metric_field_new.update_mesh();

            mesh->recreate_boundary();

            sprintf(name, "aftermetu.%lu", i/ite_red);
            mesh->print_mesh(name);
            mesh->print_halo(name);

            smooth.smart_laplacian(10);
            smooth.optimisation_linf(10);

            sprintf(name, "aftersmored.%lu", i/ite_red);
            mesh->print_mesh(name);
            mesh->print_halo(name);
        }

    }

    mesh->defragment();

    smooth.smart_laplacian(10);
    smooth.optimisation_linf(10);

    VTKTools<double>::export_vtu("../data/test_redistribute_3d", mesh);
    
//    GMFTools<double>::export_gmf_mesh("ex_metric2d_resV2", mesh);
//    GMFTools<double>::export_gmf_metric2d("ex_metric2d_resV2", &metric_field, mesh);
//    return 0;
    
    if (rank==0) printf("DEBUG  verify at the end\n");
    if (mesh->verify())
        std::cout<<"pass"<<std::endl;

#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}
