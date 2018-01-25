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

#include <mpi.h>


void set_metric(Mesh<double> *mesh, MetricField<double,2> &metric) {
    
    double m[3] = {0};

    size_t NNodes = mesh->get_number_nodes();

    metric.generate_mesh_metric(0.95);

    for(size_t i=0; i<NNodes; i++) {

        double x = mesh->get_coords(i)[0];

        if (x > 8.e5) {
            const double * m0 = metric.get_metric(i);
            double cof = (1 - 2.4e-6*(x - 8e5)) * 0.13333;
            m[0] = m0[0] * cof;
            m[1] = m0[1] * cof;
            m[2] = m0[2] * cof;
            metric.set_metric(m, i);
        }
    }
    metric.update_mesh();
}


int main(int argc, char **argv)
{    
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_LIBMESHB
    char filename_in[256];
    sprintf(filename_in, "../data/tohoku");
    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh(filename_in);

    pragmatic_init_light((void*)mesh);

    MetricField<double,2> metric_field(*mesh);
    set_metric(mesh, metric_field);

    pragmatic_adapt(0);

    mesh->compute_print_quality();
    mesh->compute_print_NNodes_global();

    char filename_out[256];
    sprintf(filename_out, "../data/test_tohoku");
    GMFTools<double>::export_gmf_mesh(filename_out, mesh);
    printf("pass");

#else
    std::cerr<<"Pragmatic was configured without libMeshb support, cannot run this test"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
