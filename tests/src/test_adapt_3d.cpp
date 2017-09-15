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
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#include <mpi.h>

void cout_quality(const Mesh<double> *mesh, std::string operation)
{
    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();

    int rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0)
        std::cout<<operation<<": step in quality (mean, min): ("<<qmean<<", "<<qmin<<")"<<std::endl;
}

int main(int argc, char **argv)
{
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool verbose = false;
    if(argc>1) {
        verbose = std::string(argv[1])=="-v";
    }

    // Benchmark times.
#ifdef HAVE_VTK
    double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0, tic;
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box50x50x50.vtu");
    mesh->create_boundary();

    MetricField<double,3> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();
    double eta=0.1;

    for(size_t i=0; i<NNodes; i++) {
        double x = 2*mesh->get_coords(i)[0] - 1;
        double y = 2*mesh->get_coords(i)[1];

        double m[] = {0.2*(-8*x + 4*sin(5*y))/pow(pow(2*x - sin(5*y), 2) + 0.01, 2) - 250.0*sin(50*x),  2.0*(2*x - sin(5*y))*cos(5*y)/pow(pow(2*x - sin(5*y), 2) + 0.01, 2),                                                        0,
                      -5.0*(2*x - sin(5*y))*pow(cos(5*y), 2)/pow(pow(2*x - sin(5*y), 2) + 0.01, 2) + 2.5*sin(5*y)/(pow(2*x - sin(5*y), 2) + 0.01), 0,
                      0
                     };


        for(int j=0; j<5; j++)
            m[j]/=eta;
        m[5] = 1.0;

        metric_field.set_metric(m, i);
    }
    metric_field.apply_max_aspect_ratio(10);
    metric_field.update_mesh();

    VTKTools<double>::export_vtu("../data/test_adapt_3d-initial", mesh);

    cout_quality(mesh, "Initial quality");

    // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
    double L_up = sqrt(2.0);
    double L_low = L_up/2;

    Coarsen<double, 3> coarsen(*mesh);
    Smooth<double, 3> smooth(*mesh);
    Refine<double, 3> refine(*mesh);
    Swapping<double, 3> swapping(*mesh);

    time_adapt = get_wtime();

    double L_max = mesh->maximal_edge_length();
    double alpha = sqrt(2.0)/2;
    double L_ref = std::max(alpha*L_max, L_up);

    if(verbose)
        std::cout<<"Phase I\n";

    for(size_t i=0; i<10; i++) {
        // Coarsen
        tic = get_wtime();
        coarsen.coarsen(L_low, L_ref);
        time_coarsen += get_wtime() - tic;

        if(verbose)
            cout_quality(mesh, "Coarsen");

        // Refine
        tic = get_wtime();
        refine.refine(L_ref);
        time_refine += get_wtime() - tic;

        if(verbose)
            cout_quality(mesh, "refine");

        // Swap
        tic = get_wtime();
        swapping.swap(0.1);
        time_swap += get_wtime() - tic;

        if(verbose)
            cout_quality(mesh, "Swap");

        // Smooth
        tic = get_wtime();
        smooth.smart_laplacian(1);
        time_smooth += get_wtime()-tic;

        if(verbose)
            cout_quality(mesh, "Smooth");

        alpha = (1.0-1e-2*i*i)*sqrt(2.0)/2;
        L_max = mesh->maximal_edge_length();
        L_ref = std::max(alpha*L_max, L_up);


        if(L_max>1.0 and (L_max-L_up)<0.01)
            break;
    }

    if(verbose)
        std::cout<<"Phase II\n";

    for(size_t i=0; i<5; i++) {
        tic = get_wtime();
        coarsen.coarsen(L_up, L_up);
        time_coarsen += get_wtime() - tic;
        if(verbose)
            cout_quality(mesh, "coarsen");

        tic = get_wtime();
        swapping.swap(0.1);
        if(verbose)
            cout_quality(mesh, "Swap");
        time_swap += get_wtime() - tic;

        tic = get_wtime();
        smooth.smart_laplacian(1);
        if(verbose)
            cout_quality(mesh, "Smooth");
        time_smooth += get_wtime()-tic;
    }

    double time_defrag = get_wtime();
    mesh->defragment();
    time_defrag = get_wtime()-time_defrag;

    if(verbose) {
        mesh->verify();

        VTKTools<double>::export_vtu("../data/test_adapt_3d-basic", mesh);
    }

    tic = get_wtime();
    smooth.smart_laplacian(20);
    smooth.optimisation_linf(20);
    time_smooth += get_wtime()-tic;

    if(verbose)
        cout_quality(mesh, "Final smooth");

    time_adapt = get_wtime()-time_adapt;

    if(verbose) {
        if(rank==0)
            std::cout<<"After optimisation based smoothing:\n";
        mesh->verify();
    }

    VTKTools<double>::export_vtu("../data/test_adapt_3d", mesh);

    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();

    long double volume = mesh->calculate_volume();
    long double area = mesh->calculate_area();

    delete mesh;

    if(rank==0) {
        std::cout<<"BENCHMARK: time_coarsen time_refine time_swap time_smooth time_defrag time_adapt time_other\n";
        double time_other = (time_adapt-(time_coarsen+time_refine+time_swap+time_smooth+time_defrag));
        std::cout<<"BENCHMARK: "
                 <<std::setw(12)<<time_coarsen<<" "
                 <<std::setw(11)<<time_refine<<" "
                 <<std::setw(9)<<time_swap<<" "
                 <<std::setw(11)<<time_smooth<<" "
                 <<std::setw(11)<<time_defrag<<" "
                 <<std::setw(10)<<time_adapt<<" "
                 <<std::setw(10)<<time_other<<"\n";

        std::cout<<"Expecting qmean>0.65, qmin>0.07: ";
        if((qmean>0.6)&&(qmin>0.07))
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail (qmean="<<qmean<<", qmin="<<qmin<<")"<<std::endl;

        long double volume_exact = 1;
        std::cout<<"Expecting volume == "<<volume_exact<<": ";
        if(std::abs(volume-volume_exact)/std::max(volume, volume_exact)<DBL_EPSILON)
            std::cout<<"pass (volume="<<volume<<", std::abs(volume-volume_exact)/std::max(volume, volume_exact)="<<std::abs(volume-volume_exact)/std::max(volume, volume_exact)<<", tol="<<DBL_EPSILON<<")"<<std::endl;
        else
            std::cout<<"fail (volume="<<volume<<", std::abs(volume-volume_exact)/std::max(volume, volume_exact)="<<std::abs(volume-volume_exact)/std::max(volume, volume_exact)<<", tol="<<DBL_EPSILON<<")"<<std::endl;

        long double area_exact = 6;
        std::cout<<"Expecting area == "<<area_exact<<": ";
        if(std::abs(area-area_exact)/std::max(area, area_exact)<DBL_EPSILON)
            std::cout<<"pass (area="<<area<<", std::abs(area-area_exact)/std::max(area, area_exact)="<<std::abs(area-area_exact)/std::max(area, area_exact)<<", tol="<<DBL_EPSILON<<")"<<std::endl;
        else
            std::cout<<"fail (area="<<area<<", std::abs(area-area_exact)/std::max(area, area_exact)="<<std::abs(area-area_exact)/std::max(area, area_exact)<<", tol="<<DBL_EPSILON<<")"<<std::endl;
    }
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
