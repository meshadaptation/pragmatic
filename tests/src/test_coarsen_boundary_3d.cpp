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

#include <getopt.h>
#include <sstream>

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"

#include "Coarsen.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

void cout_quality(const Mesh<double> *mesh, std::string operation)
{
    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();

    int rank=0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
        std::cout<<operation<<": step in quality (mean, min): ("<<qmean<<", "<<qmin<<")"<<std::endl;
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

    bool verbose = false;
    if(argc>1) {
        verbose = std::string(argv[1])=="-v";
    }

    // Benchmark times.
#ifdef HAVE_VTK
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/bearskull.vtu");
    mesh->create_boundary();
    mesh->defragment();

    for(int icoarsen=0; icoarsen<10; icoarsen++) {
        double time_coarsen=0, time_refine=0, time_swapping=0, time_adapt=0, tic;

        if(verbose)
            std::cout<<"INFO: coarsen level "<<icoarsen<<std::endl;
        MetricField<double,3> metric_field(*mesh);

        double time_metric = get_wtime();
        metric_field.generate_mesh_metric(2);
        time_metric = (get_wtime() - time_metric);

        metric_field.update_mesh();

        if(verbose) {
            if(icoarsen==0)
                VTKTools<double>::export_vtu("../data/coarsen_boundary_3d_c0", mesh);
            mesh->verify();
        }

        double L_up = sqrt(2.0);

        Coarsen<double, 3> coarsen(*mesh);
        // Smooth<double, 3> smooth(*mesh);
        Swapping<double, 3> swapping(*mesh);

        for(size_t i=0; i<1; i++) {
            if(verbose)
                std::cout<<"INFO: Sweep "<<i<<std::endl;

            double tic = get_wtime();
            coarsen.coarsen(L_up, L_up, true);
            time_coarsen += (get_wtime()-tic);

            if(verbose) {
                mesh->verify();
                cout_quality(mesh, "Quality after coarsening");
            }

            tic = get_wtime();
            swapping.swap(0.1);
            time_swapping += (get_wtime()-tic);
            if(verbose) {
                mesh->verify();
                cout_quality(mesh, "Quality after swapping");
            }
        }

        mesh->defragment();

        double time_smooth=get_wtime();
        // smooth.smart_laplacian(20);
        // smooth.optimisation_linf(20);
        time_smooth = (get_wtime() - time_smooth);
        if(verbose) {
            mesh->verify();
            cout_quality(mesh, "Quality after final smoothening");
        }

        if(verbose)
            std::cout<<"Times for metric, coarsen, swapping, smoothing = "<<time_metric<<", "<<time_coarsen<<", "<<time_swapping<<", "<<time_smooth<<std::endl;

        if(mesh->get_number_elements()==0)
            break;

        std::stringstream cid;
        cid<<(icoarsen+1);
        std::string filename("../data/coarsen_boundary_3d_c"+cid.str());
        VTKTools<double>::export_vtu(filename.c_str(), mesh);
    }

    mesh->verify();
    
    delete mesh;
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
