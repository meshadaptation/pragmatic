/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Gerard Gorman
 *  Applied Modelling and Computation Group
 *  Department of Earth Science and Engineering
 *  Imperial College London
 *
 *  g.gorman@imperial.ac.uk
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials provided
 *  with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 *  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#include <cmath>
#include <iostream>
#include <vector>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

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
#include "ticker.h"
#include "cpragmatic.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif




void set_metric(Mesh<double> *mesh, MetricField<double,3> &metric, double h0) {

//    printf("DEBUG  metric 0\n");
    size_t NNodes = mesh->get_number_nodes();

    for(size_t i=0; i<NNodes; i++) {
        double z = mesh->get_coords(i)[2] - 0.5;

        double h = h0 + 2*(0.1-h0)*fabs(z);
        double lbd = 1./(h*h);
        double lmax = 1/(0.1*0.1);
        double lambda1 = lmax;
        double lambda2 = lmax;
        double lambda3 = lbd;

        double m[] = {lambda1, 0, 0, lambda2, 0, lambda3};


        metric.set_metric(m, i);
    }

//    printf("DEBUG  metric 1\n");
    metric.update_mesh();
//    printf("DEBUG  metric 2\n");
}



int main(int argc, char **argv)
{
    int rank=0;
#ifdef HAVE_MPI
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif


#ifdef HAVE_LIBMESHB
    double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0, tic;
    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh("../data/cube-linear-00");

    pragmatic_init_light((void*)mesh);

    double h0 = 0.1;

    for (int ite=0; ite<8; ++ite) {

        h0 = (ite < 4) ? h0*0.33 : 0.001;
        printf("DEBUG  ite: %d   h0: %1.4f\n", ite, h0);

        MetricField<double,3> metric_field(*mesh);
        set_metric(mesh, metric_field, h0);

        pragmatic_adapt(0);

        char name[256];
        sprintf(name, "../data/test_ugawg_cube.%d", ite);
        GMFTools<double>::export_gmf_mesh(name, mesh);
    }

#ifdef HAVE_VTK
    VTKTools<double>::export_vtu("../data/test_ugawg_cube", mesh);
#else
    std::cerr<<"Warning: Pragmatic was configured without VTK support"<<std::endl;
#endif


#else
    std::cerr<<"Pragmatic was configured without libMeshb support"<<std::endl;
#endif

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
