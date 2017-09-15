#include "Mesh.h"
#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif

#include "ticker.h"


int main(int argc, char **argv)
{

    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

#ifdef HAVE_LIBMESHB
    Mesh<double> *mesh2 = GMFTools<double>::import_gmf_mesh("../data/mesh2d");
    index_t Nnodes = mesh2->get_number_nodes();
    index_t Nelements = mesh2->get_number_elements();
    printf("DEBUG  Number of vertices: %d   Number of elements: %d\n", Nnodes, Nelements);
    printf("pass\n");

    MetricField<double,2> *metric2 = GMFTools<double>::import_gmf_metric2d("../data/mesh2d", *mesh2);
    printf("pass\n");

    GMFTools<double>::export_gmf_mesh("../data/test_gmf_2d", mesh2);
    printf("pass\n");

    GMFTools<double>::export_gmf_metric2d("../data/test_gmf_2d", metric2, mesh2);
    printf("pass\n");


    Mesh<double> *mesh3 = GMFTools<double>::import_gmf_mesh("../data/mesh3d");
    Nnodes = mesh3->get_number_nodes();
    Nelements = mesh3->get_number_elements();
    printf("DEBUG  Number of vertices: %d   Number of elements: %d\n", Nnodes, Nelements);
    printf("pass\n");

    MetricField<double,3> *metric3 = GMFTools<double>::import_gmf_metric3d("../data/mesh3d", *mesh3);
    printf("pass\n");

    GMFTools<double>::export_gmf_mesh("../data/test_gmf_3d", mesh3);
    printf("pass\n");

    GMFTools<double>::export_gmf_metric3d("../data/test_gmf_3d", metric3, mesh3);
    printf("pass\n");
#else
    std::cerr<<"Pragmatic was configured without libMeshb"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
