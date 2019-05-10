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
#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"
#include "Refine.h"
#include "ticker.h"
#include "cpragmatic.h"

#include <mpi.h>

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

#ifdef HAVE_LIBMESHB
    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh("../data/square50x50");
    pragmatic_init_light((void*)mesh);

    int * boundary = mesh->get_boundaryTags();

    int nbrElm = mesh->get_number_elements();
    std::vector<int> regions;
    regions.resize(nbrElm);
    for (int iElm = 0; iElm < nbrElm; ++iElm) {
        const int * m = mesh->get_element(iElm);
        double barycenter[2] ={0.,0.};
        for (int i=0; i<3;++i) {
            const double * coords = mesh->get_coords(m[i]);
            for (int j=0; j<2; ++j)
                barycenter[j] += coords[j];
        }
        if (barycenter[0] >= barycenter[1]) {
            regions[iElm] = 1;
        }
        else {
            if (barycenter[0] < 1.5)
                regions[iElm] = 2;
            else
                regions[iElm] = 3;
        }
    }
    mesh->set_regions(&regions[0]);
    mesh->set_internal_boundaries();

    MetricField<double,2> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();
    
    double m[6] = {0};
    for(size_t i=0; i<NNodes; i++) {
        double lmax = 1/(0.07*0.07);
        m[0] = lmax;
        m[2] = 0.07*lmax;
        metric_field.set_metric(m, i);
    }
    metric_field.update_mesh();

    GMFTools<double>::export_gmf_mesh("../data/test_int_regions_2d-initial", mesh);

    Refine<double,2> adapt(*mesh);

    pragmatic_adapt(0, 0);

    if(verbose)
        mesh->verify();

    mesh->defragment();

    GMFTools<double>::export_gmf_mesh("../data/test_int_regions_2d", mesh);
#ifdef HAVE_VTK
    VTKTools<double>::export_vtu("../data/test_int_regions_2d", mesh);
#else
    std::cerr<<"Warning: Pragmatic was configured without VTK support"<<std::endl;
#endif

    long double perimeter = mesh->calculate_perimeter();
    long double area = mesh->calculate_area();
    long double area1 = mesh->calculate_area(1);
    long double area2 = mesh->calculate_area(2);
    long double area3 = mesh->calculate_area(3);
    if(rank==0) {
        long double ideal_perimeter(4+2*sqrt(2)+1), ideal_area(1), ideal_area1(0.5), ideal_area2(0.375), ideal_area3(0.125); // the internal boundary is counted twice
        std::cout<<"Expecting perimeter == 4+2*sqrt(2)+1: ";
        if(std::abs(perimeter-ideal_perimeter)/std::max(perimeter, ideal_perimeter)<DBL_EPSILON)
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail"<<std::endl;

        std::cout<<"Expecting total area == 1:            ";
        if(std::abs(area-ideal_area)/std::max(area, ideal_area)<DBL_EPSILON)
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail"<<std::endl;
        std::cout<<"Expecting area of region 1 == 0.5:    ";
        if(std::abs(area1-ideal_area1)/std::max(area1, ideal_area1)<DBL_EPSILON)
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail"<<std::endl;
        std::cout<<"Expecting area of region 2 == 0.375:  ";
        if(std::abs(area2-ideal_area2)/std::max(area2, ideal_area2)<DBL_EPSILON)
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail"<<std::endl;
        std::cout<<"Expecting area of region 3 == 0.125:  ";
        if(std::abs(area3-ideal_area3)/std::max(area3, ideal_area3)<DBL_EPSILON)
            std::cout<<"pass"<<std::endl;
        else
            std::cout<<"fail"<<std::endl;
    }

    delete mesh;
#else
    std::cerr<<"Pragmatic was configured without libmeshb"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
