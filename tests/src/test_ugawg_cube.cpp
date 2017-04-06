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




void set_metric(Mesh<double> *mesh, MetricField<double,3> &metric, int metric_choice, double h0) {

    double x, y, z, r, t, d, ct, st, lr, lt, lz;
    double h, h_r, h_t, h_z, lbd, lmax;
    
    double m[6] = {0};

    size_t NNodes = mesh->get_number_nodes();

    for(size_t i=0; i<NNodes; i++) {

        switch (metric_choice){
        case 0:
            z = mesh->get_coords(i)[2] - 0.5;
            h = h0 + 2*(0.1-h0)*fabs(z);
            lbd = 1./(h*h);
            lmax = 1/(0.1*0.1);
            m[0] = lmax;
            m[3] = lmax;
            m[5] = lbd;
            break;

        case 1:
            x = mesh->get_coords(i)[0];
            y = mesh->get_coords(i)[1];
            r = sqrt(x*x+y*y);
            t = atan2(y,x);
            h_r = h0 + 2*(0.1-h0)*fabs(r-0.5); 
            h_t = h_z = 0.1;
            lr = 1./(h_r*h_r); lt = 1./(h_t*h_t); lz = 1./(h_z*h_z);
            ct = cos(t); st = sin(t);
            m[0] = lr*ct*ct+lt*st*st;
            m[1] = (lr-lt)*ct*st;
            m[3] = lr*st*st+lt*ct*ct;
            m[5] = lz;
            break;

        case 2:
            x = mesh->get_coords(i)[0];
            y = mesh->get_coords(i)[1];
            r = sqrt(x*x+y*y);
            t = atan2(y,x);
            h_r = h0 + 2*(0.1-h0)*fabs(r-0.5);
            d = (0.6 - r) * 10;
            h_t = (d < 0) ? 0.1 : (d * (0.025) + (1 - d) * 0.1);
            h_z = 0.1;
            lr = 1./(h_r*h_r); lt = 1./(h_t*h_t); lz = 1./(h_z*h_z);
            ct = cos(t); st = sin(t);
            m[0] = lr*ct*ct+lt*st*st;
            m[1] = (lr-lt)*ct*st;
            m[3] = lr*st*st+lt*ct*ct;
            m[5] = lz;
            break;

        default :
            std::cerr<<"Error: wrong metric choice."<<std::endl;
            exit(1);
        }

        metric.set_metric(m, i);
    }

    metric.update_mesh();
}



int main(int argc, char **argv)
{

    // Options:
    int mesh_choice = 0; // 0- cube, 1- cube-cylinder
    int metric_choice = 0;  // 0- linear, 1- polar-1, 2- polar-2
    double h0_ratio = 0.33;
    int h0_prog_ite_max = 4;
    int num_ite_adapt = 10;
    int test_case = 0; // 0- cube linear, 1- cube-cylinder linear, 2- cube-cyl polar 1, 3- cube-cyl polar 2
    

    if (argc > 0) 
        test_case = atoi(argv[1]);

    switch (test_case){
    case 0:
        // options for test case cube linear:
        mesh_choice=0; metric_choice=0; h0_ratio=0.33; h0_prog_ite_max=4; num_ite_adapt=10;
        break;
    case 1:
        // options for test case cube-cylinder linear:
        mesh_choice=1; metric_choice=0; h0_ratio=0.5; h0_prog_ite_max=7; num_ite_adapt=12;
        break;
    case 2:
        // options for test case cube-cylinder polar1:
        mesh_choice=1; metric_choice=1; h0_ratio=0.5; h0_prog_ite_max=7; num_ite_adapt=12;
        break;
    case 3:
        // options for test case cube-cylinder polar2:
        mesh_choice=1; metric_choice=2; h0_ratio=0.5; h0_prog_ite_max=7; num_ite_adapt=12;
        break;
    default:
        // default values for the parameters are used, or hard code yours somewhere
        break;
    }



    int rank=0;
#ifdef HAVE_MPI
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif


#ifdef HAVE_LIBMESHB
    char filename_in[256];
    switch (mesh_choice){
    case 0:
        sprintf(filename_in, "../data/cube-linear-00");
        break;

    case 1:
        sprintf(filename_in, "../data/cube-cylinder");
        break;

    default :
        std::cerr<<"Error: wrong metric choice."<<std::endl;
        exit(1);
    }
    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh(filename_in);

    pragmatic_init_light((void*)mesh);

    double h0 = 0.1;

    for (int ite=0; ite<num_ite_adapt; ++ite) {

        h0 = (ite < h0_prog_ite_max) ? h0*h0_ratio : 0.001;
        printf("DEBUG  ite: %d   h0: %1.4f\n", ite, h0);

        MetricField<double,3> metric_field(*mesh);
        set_metric(mesh, metric_field, metric_choice, h0);

        pragmatic_adapt(0);

        char filename_out[256];
        sprintf(filename_out, "../data/test_ugawg_cube.%d", ite);
        GMFTools<double>::export_gmf_mesh(filename_out, mesh);
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
