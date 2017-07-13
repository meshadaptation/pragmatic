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

#include <iostream>
#include <vector>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"
#include "Coarsen.h"
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv)
{
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    bool verbose = false;
    if(argc>1) {
        verbose = std::string(argv[1])=="-v";
    }

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10x10.vtu");
    mesh->create_boundary();

    MetricField<double,3> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();
    for(size_t i=0; i<10; i++) {
        double m[] = {
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            0.5
        };

        metric_field.set_metric(m, i);
    }

    for(size_t i=10; i<NNodes; i++) {
        double m[] = {
            0.005,
            0.0,
            0.0,
            0.005,
            0.0,
            0.005
        };

        metric_field.set_metric(m, i);
    }
    metric_field.gradation(1.2, 1.0);
    metric_field.update_mesh();

    VTKTools<double>::export_vtu("../data/test_gradation_3d", mesh);

    std::cout<<"pass"<<std::endl;

    delete mesh;
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
