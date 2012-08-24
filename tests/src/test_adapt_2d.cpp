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

#include <mpi.h>

int main(int argc, char **argv){
  MPI::Init_thread(argc,argv, MPI::THREAD_SERIALIZED);

  bool verbose = false;
  if(argc>1){
    verbose = std::string(argv[1])=="-v";
  }
  
  // Benchmark times.
  double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0;

  int rank = MPI::COMM_WORLD.Get_rank();

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/smooth_2d.vtu");

  Surface2D<double, int> surface(*mesh);
  surface.find_surface(true);

  MetricField2D<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();
  double eta=0.0002;

  std::vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double x = 2*mesh->get_coords(i)[0]-1;
    double y = 2*mesh->get_coords(i)[1]-1;

    psi[i] = 0.100000000000000*sin(50*x) + atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
  }

  metric_field.add_field(&(psi[0]), eta, 1);
  metric_field.update_mesh();

  double qmean = mesh->get_qmean();
  double qrms = mesh->get_qrms();
  double qmin = mesh->get_qmin();
  
  if((rank==0)&&(verbose)) std::cout<<"Initial quality:\n"
                                    <<"Quality mean:  "<<qmean<<std::endl
                                    <<"Quality min:   "<<qmin<<std::endl
                                    <<"Quality RMS:   "<<qrms<<std::endl;
  VTKTools<double, int>::export_vtu("../data/test_adapt_2d-initial", mesh);

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = sqrt(2.0);
  double L_low = L_up/2;

  Coarsen2D<double, int> coarsen(*mesh, surface);  
  Smooth2D<double, int> smooth(*mesh, surface);
  Refine2D<double, int> refine(*mesh, surface);
  Swapping2D<double, int> swapping(*mesh, surface);

  time_adapt = get_wtime();

  double tic = get_wtime();
  coarsen.coarsen(L_low, L_up);
  time_coarsen += get_wtime()-tic;

  if(verbose)
    if(!mesh->verify()){
      std::vector<int> active_vertex_map;
      mesh->defragment(&active_vertex_map);
      surface.defragment(&active_vertex_map);
      
      VTKTools<double, int>::export_vtu("../data/test_adapt_2d-coarsen0", mesh);
      exit(-1);
    }

  double L_max = mesh->maximal_edge_length();
  
  double alpha = sqrt(2.0)/2;  
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);
    
    tic = get_wtime();
    refine.refine(L_ref);
    time_refine += get_wtime() - tic;
    
    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after refine.\n";
      
      if(!mesh->verify()){    
        std::vector<int> active_vertex_map;
        mesh->defragment(&active_vertex_map);
        surface.defragment(&active_vertex_map);
        
        VTKTools<double, int>::export_vtu("../data/test_adapt_2d-refine", mesh);
        exit(-1);
      }
    }

    tic = get_wtime();
    coarsen.coarsen(L_low, L_ref, 10);
    time_coarsen += get_wtime() - tic;

    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after coarsen.\n";
      
      if(!mesh->verify()){
        std::vector<int> active_vertex_map;
        mesh->defragment(&active_vertex_map);
        surface.defragment(&active_vertex_map);
        
        VTKTools<double, int>::export_vtu("../data/test_adapt_2d-coarsen", mesh);
        exit(-1);
      }
    }

    tic = get_wtime();
    swapping.swap(0.7);
    time_swap += get_wtime() - tic;

    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after swapping.\n";
      
      if(!mesh->verify()){
        std::vector<int> active_vertex_map;
        mesh->defragment(&active_vertex_map);
        surface.defragment(&active_vertex_map);
        
        VTKTools<double, int>::export_vtu("../data/test_adapt_2d-swapping", mesh);
        exit(-1);
      }
    }

    L_max = mesh->maximal_edge_length();
    
    if((L_max-L_up)<0.01)
      break;
  }
  
  std::vector<int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  if(verbose){
    if(rank==0)
      std::cout<<"Basic quality:\n";
    mesh->verify();
    
    VTKTools<double, int>::export_vtu("../data/test_adapt_2d-basic", mesh);
  }
  
  tic = get_wtime();
  smooth.smooth("optimisation Linf", 200);
  time_smooth += get_wtime()-tic;
  
  time_adapt = get_wtime()-time_adapt;

  if(verbose){
    if(rank==0)
      std::cout<<"After optimisation based smoothing:\n";
    mesh->verify();
  }

  NNodes = mesh->get_number_nodes();
  psi.resize(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double x = 2*mesh->get_coords(i)[0]-1;
    double y = 2*mesh->get_coords(i)[1]-1;
    
    psi[i] = 0.100000000000000*sin(50*x) + atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
  }

  VTKTools<double, int>::export_vtu("../data/test_adapt_2d", mesh, &(psi[0]));
  VTKTools<double, int>::export_vtu("../data/test_adapt_2d_surface", &surface);

  qmean = mesh->get_qmean();
  qrms = mesh->get_qrms();
  qmin = mesh->get_qmin();
  
  delete mesh;

  if(rank==0){
    std::cout<<"BENCHMARK: time_coarsen time_refine time_swap time_smooth time_adapt\n";
    std::cout<<"BENCHMARK: "
             <<std::setw(12)<<time_coarsen<<" "
             <<std::setw(11)<<time_refine<<" "
             <<std::setw(9)<<time_swap<<" "
             <<std::setw(11)<<time_smooth<<" "
             <<std::setw(10)<<time_adapt<<"\n";

    if((qmean>0.8)&&(qmin>0.4))
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail"<<std::endl;
  }

  MPI::Finalize();

  return 0;
}
