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

    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box200x200.vtu");
    mesh->create_boundary();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

#if 0
    // define metric analytically
    double m[3] = {0};
    for(size_t i=0; i<NNodes; i++) {
//        double lmax = 1/(0.005*0.005);
//        m[0] = m[2] = lmax;
//        m[1] = 0;
        double x = mesh->get_coords(i)[0];
        double h = 0.5*fabs(1-exp(-fabs(x-0.5))) + 0.003;
        double lbd = 1/(h*h);
        double lmax = 1/(0.5*0.5);
        m[0] = lbd;
        m[1] = 0.0;
        m[2] = lmax;
        metric_field.set_metric(m, i);
    }
#endif
#if 1
    // define metric from function
    std::vector<double> psi(NNodes);
    for(size_t i=0; i<NNodes; i++) {
        double x = mesh->get_coords(i)[0]-0.5;
        double y = mesh->get_coords(i)[1]-0.5;

        psi[i] = (50*fabs(x*y) >= 2*M_PI) ? 0.01*sin(50*x*y) : sin(50*x*y);
    }
    double eta=0.0009;
    metric_field.add_field(&(psi[0]), eta, 2);
#endif
    metric_field.update_mesh();
    
//    GMFTools<double>::export_gmf_mesh("ex_metric2dV2", mesh);
//    GMFTools<double>::export_gmf_metric2d("ex_metric2dV2", &metric_field, mesh);
//    return 0;

    double L_up = sqrt(2.0);
    double L_low = L_up*0.5;

    Coarsen<double, 2> coarsen(*mesh);
    Smooth<double, 2> smooth(*mesh);
    Refine<double, 2> refine(*mesh);
    Swapping<double, 2> swapping(*mesh);

    double L_max = mesh->maximal_edge_length();

    double alpha = sqrt(2.0)/2.0;
    size_t i=0;
	int cnt_red = 0;
    for(i=0; i<24; i++) {
        if (rank==0) printf("DEBUG(%d)  ite adapt: %lu\n", rank, i);
        double L_ref = std::max(alpha*L_max, L_up);

        for (int iElm = 0; iElm < mesh->get_number_elements(); ++iElm) {
            const int *n = mesh->get_element(iElm);
            int gnnElm[3] = {mesh->get_global_numbering(n[0]), mesh->get_global_numbering(n[1]), mesh->get_global_numbering(n[2])};
            if (gnnElm[0] == 345675 && gnnElm[1] == 345679 && gnnElm[2] == 345724) {
                printf("DEBUG(%d)  -- ite %d Wanted element appeared on this proc with iElm: %d\n", rank, i, iElm);
            }
        }

        char name[200];
        sprintf(name, "ite_%d", i);
        mesh->print_mesh(name);
        mesh->print_halo(name);
        
        coarsen.coarsen(L_low, L_ref, false);
        swapping.swap(0.7);
        refine.refine(L_ref);

        L_max = mesh->maximal_edge_length();

//        for (int iElm = 0; iElm < mesh->get_number_elements(); ++iElm) {
//            const int *n = mesh->get_element(iElm);
//            int gnnElm[3] = {mesh->get_global_numbering(n[0]), mesh->get_global_numbering(n[1]), mesh->get_global_numbering(n[2])};
//            if (gnnElm[0] == 345675 && gnnElm[1] == 345679 && gnnElm[2] == 345724) {
//                printf("DEBUG(%d)  ++ ite %d Wanted element appeared on this proc with iElm: %d\n", rank, i, iElm);
//            }
//        }

        int ite_red = 100;
        //if (i>0 && i%ite_red==0) {
        if (i==1 || i==3 || i==5 || i==10 || i==17 || i==25) {
            if (rank==0) printf("DEBUG(%d)  %lu-th redistribution\n", rank, i/ite_red);

//            mesh->print_mesh("before17");
//            mesh->print_halo("before17");

            mesh->fix_halos();

//            mesh->print_mesh("after17");
//            mesh->print_halo("after17");

//            int tag = 2*(i%2)-1;
//            int tag = (i/ite_red)%4 <2 ? 1 : -1;
			int tag = cnt_red%4 <2 ? 1 : -1;
            if (rank==0) printf("DEBUG  resdistribute to %s\n", (tag==1) ? "greater" : "lower");
			if (num_processes>1) {
                mesh->redistribute_halo(tag);

                MetricField<double,2> metric_field_new(*mesh);
                metric_field_new.set_metric(mesh->get_metric());
                metric_field_new.update_mesh();

                mesh->recreate_boundary();
			}

            smooth.smart_laplacian(20);
            //smooth.optimisation_linf(20);
            cnt_red++;
        }
        if (i==17)
            break;
//		mesh->compute_print_quality();
//        mesh->compute_print_NNodes_global();

    }

    mesh->defragment();

    smooth.smart_laplacian(20);
    smooth.optimisation_linf(20);

    VTKTools<double>::export_vtu("../data/test_redistribute_2d", mesh);
    
//    GMFTools<double>::export_gmf_mesh("ex_metric2d_resV2", mesh);
//    GMFTools<double>::export_gmf_metric2d("ex_metric2d_resV2", &metric_field, mesh);
//    return 0;
    
    if (rank==0) printf("DEBUG  verify at the end\n");
//    if (mesh->verify())
//        std::cout<<"pass"<<std::endl;

	mesh->compute_print_quality();
    mesh->compute_print_NNodes_global();

#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}
