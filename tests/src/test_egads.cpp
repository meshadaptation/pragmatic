#include "Mesh.h"

#include "ticker.h"


int main(int argc, char **argv)
{

#ifdef HAVE_MPI
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);
#endif

    int tri[3] = {0, 1, 2};
    double x[3] = {0, 0, 1}, y[3] = {0, 1, 0};
    Mesh<double> mesh = Mesh<double>(3, 1, tri, x, y);
#ifdef HAVE_EGADS
    int res = mesh.analyzeCAD("../data/cube-cylinder.step");
    if (!res) printf("pass\n");

    mesh.associate_CAD_with_Mesh();
    printf("pass\n");
#endif

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}