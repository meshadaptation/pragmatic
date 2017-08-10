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

#include "Refine.h"
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
//    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box5x5x5.vtu");
    Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/cube.vtu");
    mesh->create_boundary();

    MetricField<double,3> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    double m[6] = {0};
    for(size_t i=0; i<NNodes; i++) {
        double lmax = 1/(0.05*0.05);
        m[0] = lmax;
        m[3] = lmax;
        m[5] = lmax;
        metric_field.set_metric(m, i);
    }
    metric_field.update_mesh();

    Refine<double,3> adapt(*mesh);

    double tic = get_wtime();
    for(int i=0; i<25; i++) {
        printf("DEBUG  ===== refine %d\n", i);
        double nsplits = adapt.refine_new(sqrt(2.0));
        char name[128];
        sprintf(name, "../data/refine.%d", i);
        VTKTools<double>::export_vtu(name, mesh);
        if (nsplits==0) break;
    }
    double toc = get_wtime();

    if(verbose)
        mesh->verify();

    mesh->defragment();

    VTKTools<double>::export_vtu("../data/test_refine_3d", mesh);

    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();
    int nelements = mesh->get_number_elements();

    if(verbose)
        std::cout<<"Refine loop time:    "<<toc-tic<<std::endl
                 <<"Number elements:     "<<nelements<<std::endl
                 <<"Quality mean:        "<<qmean<<std::endl
                 <<"Quality min:         "<<qmin<<std::endl;

    long double area = mesh->calculate_area();
    long double volume = mesh->calculate_volume();

    long double ideal_area(6), ideal_volume(1);
    std::cout<<"Checking area == 6: ";
    if(std::abs(area-ideal_area)/std::max(area, ideal_area)<DBL_EPSILON)
        std::cout<<"pass"<<std::endl;
    else
        std::cout<<"fail (area="<<area<<")"<<std::endl;

    std::cout<<"Checking volume == 1: ";
    if(std::abs(volume-ideal_volume)/std::max(volume, ideal_volume)<DBL_EPSILON)
        std::cout<<"pass"<<std::endl;
    else
        std::cout<<"fail (volume="<<volume<<")"<<std::endl;

    delete mesh;
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
