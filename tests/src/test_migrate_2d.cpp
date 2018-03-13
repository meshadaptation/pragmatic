#include <iostream>
#include <vector>
#include <unistd.h>

#include <mpi.h>

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"





int main(int argc, char **argv)
{
    int rank=0, num_processes=1;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (num_processes==1) 
        return 0;

    bool verbose = false;
    if(argc>1) {
        verbose = std::string(argv[1])=="-v";
    }

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10.vtu");
    mesh->create_boundary();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();
    double eta=0.001;

    double m[3] = {0};
    for(size_t i=0; i<NNodes; i++) {
        double x = mesh->get_coords(i)[0];
        double y = mesh->get_coords(i)[1];
        m[0] = m[2] = x+y;
        metric_field.set_metric(m, i);
    }
    metric_field.update_mesh();

    if(verbose) {
        std::cout<<"Initial quality:\n";
        mesh->verify();
    }

//    mesh->print_mesh("before");
//    mesh->print_halo("before");

    VTKTools<double>::export_vtu("globalnumbering", mesh);

    std::vector<int> new_owners;
    new_owners.resize(NNodes);
    for (int iVer=0; iVer<NNodes; ++iVer) {
        new_owners[iVer] = mesh->get_node_owner(iVer);
    }

    size_t NElements = mesh->get_number_elements();
    for (index_t iElm=0; iElm<NElements; ++iElm) {
        const int *elm = mesh->get_element(iElm);
        int owner = mesh->get_node_owner(elm[0]);
        for (int k=1; k<3; ++k) {
            owner = std::min(owner, mesh->get_node_owner(elm[k]));
        }
        for (int k=0; k<3; ++k) {
            new_owners[elm[k]] = std::min(new_owners[elm[k]], owner);
        }
    }

    for (int iVer=0; iVer<NNodes; ++iVer) {
        switch (mesh->get_global_numbering(iVer)) { // this one is for 10x10 on 5 procs with gappy numbering
        case 2234:
            new_owners[iVer] = 0;
            break;
        case 4108:
            new_owners[iVer] = 0;
            break;
        case 4109:
            new_owners[iVer] = 0;
            break;
        case 4121:
            new_owners[iVer] = 0;
            break;
        case 5600:
            new_owners[iVer] = 0;
            break;
        case 5605:
            new_owners[iVer] = 0;
            break;
        case 6150:
            new_owners[iVer] = 0;
            break;
        case 6153:
            new_owners[iVer] = 0;
            break;
        case 6155:
            new_owners[iVer] = 0;
            break;
        case 6172:
            new_owners[iVer] = 0;
            break;
        case 7792:
            new_owners[iVer] = 2;
            break;
        case 7795:
            new_owners[iVer] = 2;
            break;
        case 8342:
            new_owners[iVer] = 0;
            break;
        case 8345:
            new_owners[iVer] = 2;
            break;
        }
    }
//    if (rank == 2)
//        printf("DEBUG(%d) new owner of vertex 7 (%d): %d\n", rank, mesh->get_global_numbering(7), new_owners[7]);
//    if (rank == 3)
//        printf("DEBUG(%d) new owner of vertex 4 (%d): %d\n", rank, mesh->get_global_numbering(4), new_owners[4]);
//    if (rank == 3)
//        printf("DEBUG(%d) new owner of vertex 5 (%d): %d\n", rank, mesh->get_global_numbering(5), new_owners[5]);

    for (int iVer=0; iVer<NNodes; ++iVer) {
        int gnn = mesh->get_global_numbering(iVer);
        if (gnn == 5612 || gnn == 7795 || gnn == 5610)
            printf("DEBUG(%d)  new owner of %d (%d) is %d\n", rank, iVer, mesh->get_global_numbering(iVer), new_owners[iVer]);
    }


    mesh->migrate_mesh(&new_owners[0]); 

    if (mesh->verify())
        std::cout<<"pass"<<std::endl;


#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}