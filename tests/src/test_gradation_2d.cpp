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
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool verbose = false;
  if(argc>1){
    verbose = std::string(argv[1])=="-v";
  }
  
  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box50x50.vtu");
  mesh->create_boundary();

  size_t NNodes = mesh->get_number_nodes();

  // Set up field - use first touch policy
  std::vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++){
    if(mesh->get_coords(i)[0]>0.25 && mesh->get_coords(i)[0]<0.75 &&
       mesh->get_coords(i)[1]>0.25 && mesh->get_coords(i)[1]<0.75)
      psi[i] = 1.0;
    else
      psi[i] = -1.0;
  }
  
  MetricField<double,2> metric_field(*mesh);
  
  metric_field.add_field(&(psi[0]), 0.2);

  metric_field.apply_gradation(1.1);
  metric_field.apply_max_nelements(10000);

  metric_field.update_mesh();
  
  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = sqrt(2.0);
  double L_low = L_up/2;

  Coarsen<double, 2> coarsen(*mesh);
  Smooth<double, 2> smooth(*mesh);
  Refine<double, 2> refine(*mesh);
  Swapping2D<double> swapping(*mesh);

  coarsen.coarsen(L_low, L_up);

  if(verbose)
    if(!mesh->verify()){
      mesh->defragment();
      
      VTKTools<double>::export_vtu("../data/test_adapt_2d-coarsen0", mesh);
      exit(-1);
    }

  double L_max = mesh->maximal_edge_length();
  
  double alpha = sqrt(2.0)/2;  
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);
    
    refine.refine(L_ref);
    
    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after refine.\n";
      
      if(!mesh->verify()){
        std::cout<<"ERROR(rank="<<rank<<"): Verification failed after refinement.\n";

        mesh->defragment();
        
        VTKTools<double>::export_vtu("../data/test_adapt_2d-refine", mesh);
        exit(-1);
      }
    }

    coarsen.coarsen(L_low, L_ref);

    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after coarsen.\n";
      
      if(!mesh->verify()){
        std::cout<<"ERROR(rank="<<rank<<"): Verification failed after coarsening.\n";

        mesh->defragment();
        
        VTKTools<double>::export_vtu("../data/test_adapt_2d-coarsen", mesh);
        exit(-1);
      }
    }

    swapping.swap(0.7);

    if(verbose){
      if(rank==0)
        std::cout<<"INFO: Verify quality after swapping.\n";
      
      if(!mesh->verify()){
        std::cout<<"ERROR(rank="<<rank<<"): Verification failed after swapping.\n";

        mesh->defragment();
        
        VTKTools<double>::export_vtu("../data/test_adapt_2d-swapping", mesh);
        exit(-1);
      }
    }

    L_max = mesh->maximal_edge_length();
    
    if((L_max-L_up)<0.01)
      break;
  }

  mesh->defragment();

  if(verbose){
    if(rank==0)
      std::cout<<"Basic quality:\n";
    mesh->verify();
  }

  VTKTools<double>::export_vtu("../data/test_gradation_2d", mesh);

  std::cout<<"pass\n";
  
  delete mesh;

  MPI_Finalize();

  return 0;
}

