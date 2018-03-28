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

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv)
{
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box20x20x20.vtu");
    mesh->create_boundary();

    MetricField<double, 3> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    std::vector<double> psi(NNodes);
    for(size_t i=0; i<NNodes; i++)
        psi[i] =
            pow(mesh->get_coords(i)[0]+0.1, 2) +
            pow(mesh->get_coords(i)[1]+0.1, 2) +
            pow(mesh->get_coords(i)[2]+0.1, 2);

    double start_tic = get_wtime();
    metric_field.add_field(&(psi[0]), 1.0);
    metric_field.update_mesh();
    std::cout<<"Hessian loop time = "<<get_wtime()-start_tic<<std::endl;

    std::vector<double> metric(NNodes*6);
    metric_field.get_metric(&(metric[0]));

    double rms[] = {0., 0., 0., 0., 0., 0.};
    int ncnt=0;
    for(size_t i=0; i<NNodes; i++) {
        rms[0] += pow(2.0-metric[i*6  ], 2);
        rms[1] += pow(metric[i*6+1], 2);
        rms[2] += pow(metric[i*6+2], 2);
        rms[3] += pow(2.0-metric[i*6+3], 2);
        rms[4] += pow(metric[i*6+4], 2);
        rms[5] += pow(2.0-metric[i*6+5], 2);

        ncnt++;
    }

    double max_rms = 0;
    for(size_t i=0; i<6; i++) {
        rms[i] = sqrt(rms[i]/ncnt);
        max_rms = std::max(max_rms, rms[i]);
    }

    for(size_t i=0; i<NNodes; i++)
        psi[i] =
            pow(mesh->get_coords(i)[0]+0.1, 2) +
            pow(mesh->get_coords(i)[1]+0.1, 2) +
            pow(mesh->get_coords(i)[2]+0.1, 2);

    VTKTools<double>::export_vtu("../data/test_hessian_3d", mesh, &(psi[0]));

    delete mesh;

    std::cout<<"RMS = "<<rms[0]<<", "<<rms[1]<<", "<<rms[2]<<", "<<rms[3]<<", "<<rms[4]<<", "<<rms[5]<<std::endl;
    if(max_rms>0.01)
        std::cout<<"fail\n";
    else
        std::cout<<"pass\n";
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
