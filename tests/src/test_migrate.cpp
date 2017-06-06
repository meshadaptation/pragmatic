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

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10.vtu");
    mesh->create_boundary();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    std::vector<double> psi(NNodes);
    metric_field.add_field(&(psi[0]), 0.0001);
//    metric_field.update_mesh();

//	mesh->reduce_print_mesh(); MPI_Barrier(MPI_COMM_WORLD); exit(23);

//    mesh->verify();

    mesh->print_mesh("before");
    mesh->print_halo("before");

	fflush(stdout); MPI_Barrier(MPI_COMM_WORLD); //exit(12);

    std::vector<int> new_owners;
    new_owners.resize(NNodes);
    for (int iVer=0; iVer<NNodes; ++iVer) {
        new_owners[iVer] = mesh->get_node_owner(iVer);
    }

#if 0
    switch (rank) {
    case 0:
        new_owners[2] = 1;
        break;
    case 1:
        new_owners[2] = 0;
        break;
    case 2:
        new_owners[6] = 0;
        new_owners[10] = 1;
        break;
    }
#endif
#if 0
    switch (rank) {
    case 0:
        new_owners[43] = 0;
        new_owners[35] = 1;
        new_owners[34] = 2;
        break;
    case 1:
        new_owners[45] = 1;
        new_owners[43] = 2;
        new_owners[31] = 2;
        break;
    case 2:
        new_owners[16] = 0;
        new_owners[44] = 2;
        new_owners[29] = 0;
        break;
    }
#endif

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
//    printf("TM(%d) NNodes: %d\n", rank, NNodes);
    for (int iVer=0; iVer<NNodes; ++iVer) {
//        if (mesh->get_global_numbering(iVer) == 80) new_owners[iVer] = 0;
		switch (mesh->get_global_numbering(iVer)) {
		case 34:
			new_owners[iVer] = 0;
			break;
		case 57:
            new_owners[iVer] = 0;
            break;
		case 58:
            new_owners[iVer] = 0;
            break;
		case 70:
            new_owners[iVer] = 0;
            break;
		case 73:
            new_owners[iVer] = 0;
            break;
		case 76:
            new_owners[iVer] = 0;
            break;
		case 78:
            new_owners[iVer] = 0;
            break;
		case 95:
            new_owners[iVer] = 0;
            break;
		case 114:
            new_owners[iVer] = 0;
            break;
		case 117:
            new_owners[iVer] = 2;
            break;
		}
        printf("TM(%d)  gid: %d  new_owner: %d\n", rank, mesh->get_global_numbering(iVer), new_owners[iVer]);
    }
    fflush(stdout); MPI_Barrier(MPI_COMM_WORLD); //exit(12);


    mesh->migrate_mesh(&new_owners[0]); 

    mesh->verify();


    mesh->print_mesh("after");
    mesh->print_halo("after");



#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}



#if 0

int main2(int argc, char **argv)
{
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box3x3.vtu");
    mesh->create_boundary();

    mesh->print_mesh("After creation");
    mesh->print_halo("After creation");

    int NNodes = mesh->get_number_nodes();
    std::vector<int> new_owners;
    new_owners.resize(NNodes);
    for (int iVer=0; iVer<NNodes; ++iVer) {
        new_owners[iVer] = mesh->get_node_owner(iVer);
    }
    if (rank == 0) {
        new_owners[3] = 1;
        new_owners[5] = 1;
        new_owners[6] = 1;
        new_owners[7] = 1;
        new_owners[4] = 1;
    }
    else if (rank == 1) {
        new_owners[8] = 1;
        new_owners[9] = 1;
        new_owners[10] = 1;
        new_owners[11] = 1;
    }

    mesh->migrate_mesh(&new_owners[0]); 

    if (rank == 0) {
        mesh->set_node_owner(3, 1);
        mesh->set_node_owner(5, 1);
        mesh->set_node_owner(6, 1);
        mesh->set_node_owner(7, 1);
        mesh->set_node_owner(4, 1);
    }
    else if (rank == 1) {
        mesh->set_node_owner(8, 1);
        mesh->set_node_owner(9, 1);
        mesh->set_node_owner(10, 1);
        mesh->set_node_owner(11, 1);

    }


    mesh->print_mesh("After update");
    mesh->print_halo("After update");



#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}


int main3(int argc, char **argv)
{
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box3x3.vtu");
    mesh->create_boundary();

    mesh->print_mesh("After creation");
    mesh->print_halo("After creation");

    int NNodes = mesh->get_number_nodes();
    std::vector<int> new_owners;
    new_owners.resize(NNodes);
    for (int iVer=0; iVer<NNodes; ++iVer) {
        new_owners[iVer] = mesh->get_node_owner(iVer);
    }
    if (rank == 0) {
        new_owners[2] = 1;
    }
    else if (rank == 1) {
        new_owners[2] = 0;
    }
    else if (rank == 2) {
        new_owners[6] = 0;
        new_owners[10] = 1;
    }

    mesh->migrate_mesh(&new_owners[0]); 



    mesh->print_mesh("After update");
    mesh->print_halo("After update");



#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;

}

#endif

