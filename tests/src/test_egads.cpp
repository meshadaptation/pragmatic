#include "Mesh.h"

#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif

#include "MetricField.h"
#include "Refine.h"
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
    MetricField<double,3> metric(*mesh);
#ifdef HAVE_EGADS
    int res = mesh->analyzeCAD("../data/cube-cylinder.step");
    if (!res) printf("pass\n");

    mesh->associate_CAD_with_Mesh();
    printf("pass\n");
#else
    printf("ERROR  test configured without EGADS\n");
    return 1;
#endif

    size_t NNodes = mesh->get_number_nodes();
    for(size_t i=0; i<NNodes; i++) {
        double h0 = 0.01;    
        double m[6] = {0};

        double x = mesh->get_coords(i)[0];
        double y = mesh->get_coords(i)[1];
        double r = sqrt(x*x+y*y);
        double t = atan2(y,x);
        double h_r = h0 + 2*(0.1-h0)*fabs(r-0.5);
        double d = (0.6 - r) * 10;
        double h_t = (d < 0) ? 0.01 : (d * (0.025) + (1 - d) * 0.1);
        double h_z = 0.01;
        double lr = 1./(h_r*h_r); double lt = 1./(h_t*h_t); double lz = 1./(h_z*h_z);
        double ct = cos(t); double st = sin(t);
        m[0] = lr*ct*ct+lt*st*st;
        m[1] = (lr-lt)*ct*st;
        m[3] = lr*st*st+lt*ct*ct;
        m[5] = lz;
        metric.set_metric(m, i);
    }
    metric.update_mesh();

    Refine<double,3> adapt(*mesh);

    for(int i=0; i<5; i++) {
        printf("DEBUG  refine pass %d\n", i);
        adapt.refine(sqrt(2.0));
//        mesh->defragment();
        char filename_out[256];
        sprintf(filename_out, "../data/test_egads_refine.%d", i);
        GMFTools<double>::export_gmf_mesh(filename_out, mesh);
    }
    printf("pass\n");

    mesh->defragment();
    char filename_out[256];
    sprintf(filename_out, "../data/test_egads_refine");
    GMFTools<double>::export_gmf_mesh(filename_out, mesh);

    MPI_Finalize();
    return 0;
}
