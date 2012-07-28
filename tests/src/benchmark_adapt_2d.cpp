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

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#include <stdio.h>

#include <mpi.h>

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  // Benchmark times.
  double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0;

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box200x200.vtu");

  Surface<double, int> surface(*mesh);
  surface.find_surface(true);

  char filename[256];
  double eta=0.001;
  
  for(int t=0;t<100;t++){
    std::cout<<".";

    size_t NNodes = mesh->get_number_nodes();
    
    MetricField<double, int> metric_field(*mesh, surface);        
    vector<double> psi(NNodes);
    for(size_t i=0;i<NNodes;i++){
      double x = 2*mesh->get_coords(i)[0]-1;
      double y = 2*mesh->get_coords(i)[1]-1;
      
      psi[i] = atan2(-0.100000000000000, (double)(2*x - sin(t/5.0)*sin(5*y)));
      // psi[i] = atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
      
      // double dxx = -0.8*(2*x - sin(5*y))/pow(pow(2*x - sin(5*y), 2) + 0.01,2);
      // double dxy = 2.0*(2*x - sin(5*y))*cos(5*y)/pow(pow(2*x - sin(5*y),2) + 0.01,2);
      // double dyy = -5.0*(2*x - sin(5*y))*pow(cos(5*y),2)/pow(pow(2*x - sin(5*y), 2) + 0.01,2) + 2.5*sin(5*y)/(pow(2*x - sin(5*y),2) + 0.01);
      //double m[] =
      //  {dxx, dxy,
      //   dxy, dyy};
      //metric_field.set_metric(m, i);
    }
    
    metric_field.add_field(&(psi[0]), eta, 1);
    metric_field.apply_nelements(5000);
    metric_field.update_mesh();
    
    sprintf(filename, "../data/test_adapt_2d-init-%d", t);
    VTKTools<double, int>::export_vtu(&(filename[0]), mesh, &(psi[0]));
    
    double T1 = get_wtime();

    // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
    double L_up = sqrt(2.0);
    double L_low = L_up/2;
    
    Coarsen<double, int> coarsen(*mesh, surface);  
    Smooth<double, int> smooth(*mesh, surface);
    Refine<double, int> refine(*mesh, surface);
    Swapping<double, int> swapping(*mesh, surface);
  
    double tic = get_wtime();
    coarsen.coarsen(L_low, L_up);
    double toc = get_wtime();
    if(t>0) time_coarsen += (toc-tic);

    double L_max = mesh->maximal_edge_length();
    
    double alpha = sqrt(2.0)/2;  
        
    for(size_t i=0;i<100;i++){
      double L_ref = std::max(alpha*L_max, L_up);
      
      tic = get_wtime();
      refine.refine(L_ref);
      toc = get_wtime();
      if(t>0) time_refine += (toc-tic);

      tic = get_wtime();
      coarsen.coarsen(L_low, L_up);
      toc = get_wtime();
      if(t>0) time_coarsen += (toc-tic);

      L_max = mesh->maximal_edge_length();
      
      tic = get_wtime();
      swapping.swap(0.9);
      toc = get_wtime();
      if(t>0) time_swap += (toc-tic);
      
      if((L_max-L_up)<0.01)
        break;
    }

    std::map<int, int> active_vertex_map;
    mesh->defragment(&active_vertex_map);
    surface.defragment(&active_vertex_map);
    
    tic = get_wtime();
    smooth.smooth("optimisation Linf", 200);
    toc = get_wtime();
    if(t>0) time_smooth += (toc-tic);

    double T2 = get_wtime();
    if(t>0) time_adapt += (T2-T1);
    
    NNodes = mesh->get_number_nodes();
    psi.resize(NNodes);
    for(size_t i=0;i<NNodes;i++){
      double x = 2*mesh->get_coords(i)[0]-1;
      double y = 2*mesh->get_coords(i)[1]-1;

      // psi[i] = atan2(-0.100000000000000, (double)(2*x - sin(t/5.0)*sin(5*y)));
      psi[i] = atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
    }

    sprintf(filename, "../data/test_adapt_2d-%d", t);
    VTKTools<double, int>::export_vtu(&(filename[0]), mesh, &(psi[0]));
  }
  std::cout<<std::endl;

  std::cout<<"BENCHMARK: time_coarsen time_refine time_swap time_smooth time_adapt\n";
  std::cout<<"BENCHMARK: "
           <<std::setw(12)<<time_coarsen/99<<" "
           <<std::setw(11)<<time_refine/99<<" "
           <<std::setw(9)<<time_swap/99<<" "
           <<std::setw(11)<<time_smooth/99<<" "
           <<std::setw(10)<<time_adapt/99<<"\n";

  delete mesh;

  MPI::Finalize();

  return 0;
}
