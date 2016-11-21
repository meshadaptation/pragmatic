#include "Mesh.h"
#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif

#include "ticker.h"


int main(int argc, char **argv)
{

#ifdef HAVE_MPI
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);
#endif
    
    Mesh<double> *mesh2 = GMFTools<double>::import_gmf_mesh("../data/mesh2d");


    index_t Nnodes = mesh2->get_number_nodes();
    index_t Nelements = mesh2->get_number_elements();

    printf("DEBUG  Number of vertices: %d   Number of elements: %d\n", Nnodes, Nelements);


    Mesh<double> *mesh3 = GMFTools<double>::import_gmf_mesh("../data/mesh3d");


    Nnodes = mesh3->get_number_nodes();
    Nelements = mesh3->get_number_elements();

    printf("DEBUG  Number of vertices: %d   Number of elements: %d\n", Nnodes, Nelements);





    return 0;
}