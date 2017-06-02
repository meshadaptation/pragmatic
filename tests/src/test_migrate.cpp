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
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box3x3.vtu");
    mesh->create_boundary();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    std::vector<double> psi(NNodes);
    metric_field.add_field(&(psi[0]), 0.0001);
//    metric_field.update_mesh();


    mesh->verify();

    mesh->print_mesh("After creation");
    mesh->print_halo("After creation");

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

    mesh->verify();


    mesh->print_mesh("After update");
    mesh->print_halo("After update");



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

