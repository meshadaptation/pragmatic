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





int main(int argc, char **argv)
{
    
    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef HAVE_LIBMESHB
    char filename_meshin[256];
    sprintf(filename_meshin, "/home/nbarral/calcul/UGAWG/solution-adapt-results/hemisphere-cylinder/sa-m06-a00-fun3d/hsc03");
    //sprintf(filename_meshin, "/Users/barral/calcul/UGAWG/solution-adapt-results/hemisphere-cylinder/sa-m06-a00-fun3d/hsc10");
    //sprintf(filename_meshin, "/Users/barral/calcul/UGAWG/solution-adapt-results/onera-m6/sa-m084-a306-fun3d/onera03");
    //sprintf(filename_meshin, "/Users/barral/calcul/UGAWG/solution-adapt-results/onera-m6/sa-m084-a306-fun3d/onera10");
    //sprintf(filename_meshin, "/home/nbarral/calcul/UGAWG/solution-adapt-results/onera-m6/viscous-m084-a306-ggns/oneram6_a3p06-rev-faceid");
    char filename_metin[256];
    sprintf(filename_metin, "/home/nbarral/calcul/UGAWG/solution-adapt-results/hemisphere-cylinder/sa-m06-a00-fun3d/hsc03-metric");
    //sprintf(filename_metin, "/Users/barral/calcul/UGAWG/solution-adapt-results/hemisphere-cylinder/sa-m06-a00-fun3d/hsc10-metric");
    //sprintf(filename_metin, "/Users/barral/calcul/UGAWG/solution-adapt-results/onera-m6/sa-m084-a306-fun3d/onera03-metric");
    //sprintf(filename_metin, "/Users/barral/calcul/UGAWG/solution-adapt-results/onera-m6/sa-m084-a306-fun3d/onera10-metric");
    //sprintf(filename_metin, "/home/nbarral/calcul/UGAWG/solution-adapt-results/onera-m6/viscous-m084-a306-ggns/oneram6_a3p06.metnode");

    Mesh<double> *mesh=GMFTools<double>::import_gmf_mesh(filename_meshin);
    pragmatic_init_light((void*)mesh);
    MetricField<double,3> *metric = GMFTools<double>::import_gmf_metric3d(filename_metin, *mesh);
    metric->update_mesh();
    
#ifdef HAVE_EGADS
    int res = mesh->analyzeCAD("/home/nbarral/calcul/UGAWG/solution-adapt-cases/hemisphere-cylinder/geometry/hemisph-cyl.egads");
    //int res = mesh->analyzeCAD("/Users/barral/calcul/UGAWG/solution-adapt-cases/onera-m6/geometry/onera-m6-sharp-te.egads");
    //int res = mesh->analyzeCAD("/home/nbarral/calcul/UGAWG/solution-adapt-results/onera-m6/viscous-m084-a306-ggns/oneram6_with_sharp_TE_boxff_cf.egads");
    if (!res) printf("pass\n");

    mesh->associate_CAD_with_Mesh();
    printf("pass\n");
#else
        printf("ERROR  test configured without EGADS\n");
#endif

#if 1
    // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
    double L_up = sqrt(2.0);
    double L_low = L_up/2;

    Coarsen<double, 3> coarsen(*mesh);
    Smooth<double, 3> smooth(*mesh);
    Refine<double, 3> refine(*mesh);
    Swapping<double, 3> swapping(*mesh);

    double L_max = mesh->maximal_edge_length();
    double alpha = sqrt(2.0)/2;
    double L_ref = std::max(alpha*L_max, L_up);

    printf("========================\n");
    printf("=== PHASE I: coarsen ===\n");
    printf("========================\n");
    for(int i=0; i<4; i++) {
        printf("---- coarsen %d ----\n", i);
        coarsen.coarsen(L_low, L_ref, false, false, false);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        printf("---- swap %d ----\n", i);
        swapping.swap(0.1);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        if (i==3){
            printf("---- smooth %d ----\n", i);
            smooth.optimisation_linf(3);
            mesh->print_quality_histo();
            mesh->print_edge_length_histo();
        }
    }

    printf("========================\n");
    printf("=== PHASE II: refine ===\n");
    printf("========================\n");
    for(int i=0; i<5; i++) {
        printf("---- refine %d ----\n", i);
        mesh->scale_metric(1.3333333333); 
        refine.refine_new(L_ref);
        mesh->scale_metric(0.75);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
    }

    printf("======================\n");
    printf("=== PHASE III: all ===\n");
    printf("======================\n");
    
    for(int i=0; i<20; i++) {

        printf("---- refine %d ----\n", i);
        if (i<5)
            mesh->scale_metric(1.3333333333);
        refine.refine_new(L_ref);
        int cntSplit = refine.refine_new(L_ref);
        if (i<5)
            mesh->scale_metric(0.75);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        printf("---- coarsen %d ----\n", i);
        int cntCoarsen = coarsen.coarsen(L_low, L_ref,false,false,i>5?true:false);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        printf("---- swap %d ----\n", i);
        swapping.swap(0.1);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        printf("---- smooth sl %d ----\n", i);
        smooth.smart_laplacian(1);
        mesh->print_quality_histo();
        mesh->print_edge_length_histo();
        if (!(i%5)) {
            printf("---- smooth ol %d ----\n", i);
            smooth.optimisation_linf(5, 0.15);
            mesh->print_quality_histo();
            mesh->print_edge_length_histo();
        }

        alpha = (1.0-1e-2*i*i)*sqrt(2.0)/2;
        L_max = mesh->maximal_edge_length();
        L_ref = std::max(alpha*L_max, L_up);

        if (cntSplit==0 && cntCoarsen == 0)
            break;

//        if(L_max>1.0 and (L_max-L_up)<0.01)
//            break;
    }

    mesh->defragment();

    printf("=================================\n");
    printf("=== PHASE IV: FINAL SMOOTHING ===\n");
    printf("=================================\n");
    printf("---- smooth sl ----\n");
    smooth.smart_laplacian(10);
    mesh->print_quality_histo();
    mesh->print_edge_length_histo();
    printf("---- smooth ol ----\n");
    smooth.optimisation_linf(10, 0.2);
    mesh->print_quality_histo();
    mesh->print_edge_length_histo();
#else
    pragmatic_adapt(0);
#endif

	char filename_out[256];
	sprintf(filename_out, "../data/test_ugawg2_hsc03_new");
	GMFTools<double>::export_gmf_mesh(filename_out, mesh);
    MetricField<double, 3> metric_final(*mesh);
    const double *met = mesh->get_metric(0);
    metric_final.set_metric(met);
    GMFTools<double>::export_gmf_metric3d(filename_out, &metric_final, mesh);

#ifdef HAVE_VTK
    VTKTools<double>::export_vtu("../data/test_ugawg_cube2", mesh);
#else
    std::cerr<<"Warning: Pragmatic was configured without VTK support"<<std::endl;
#endif


#else
    std::cerr<<"Pragmatic was configured without libMeshb support, cannot run this test"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
