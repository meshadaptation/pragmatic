#include "Mesh.h"

#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif

#include "MetricField.h"
#include "cpragmatic.h"

int main(int argc, char **argv)
{

    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    char filename_in[256];
    sprintf(filename_in, "../data/cube-cylinder");

    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh(filename_in);
    pragmatic_init_light((void*)mesh);
//    MetricField<double,3> *metric = GMFTools<double>::import_gmf_metric3d(filename_in, *mesh);
//    metric->update_mesh();
#ifdef HAVE_EGADS
    int res = mesh->analyzeCAD("../data/cube-cylinder.step");
    if (!res) printf("pass\n");

    mesh->associate_CAD_with_Mesh();
    printf("pass\n");
#endif

    MPI_Finalize();
    return 0;
}
