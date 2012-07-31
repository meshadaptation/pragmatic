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

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Swapping.h"
#include "ticker.h"

#include <mpi.h>

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc, argv);

  bool verbose = false;
  if(argc>1){
    verbose = std::string(argv[1])=="-v";
  }

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box10x10.vtu");

  Surface<double, int> surface(*mesh);
  surface.find_surface();

  MetricField2D<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 4) + pow(mesh->get_coords(i)[1], 4);
  
  metric_field.add_field(&(psi[0]), 0.001);
  metric_field.update_mesh();
  
  double qmean = mesh->get_qmean();
  double qrms = mesh->get_qrms();
  double qmin = mesh->get_qmin();

  std::cout<<"Initial quality:\n"
            <<"Quality mean:   "<<qmean<<std::endl
            <<"Quality min:    "<<qmin<<std::endl
            <<"Quality RMS:    "<<qrms<<std::endl;
  VTKTools<double, int>::export_vtu("../data/test_swap_2d-initial", mesh);

  double L_up = sqrt(2.0);
  double L_low = L_up/2;

  Coarsen<double, int> coarsen(*mesh, surface);
  Refine<double, int> refine(*mesh, surface);
  Swapping<double, int> swapping(*mesh, surface);

  coarsen.coarsen(L_low, L_up);

  double L_max = mesh->maximal_edge_length();

  double alpha = sqrt(2.0)/2;
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);

    refine.refine(L_ref);

    coarsen.coarsen(L_low, L_ref);

    L_max = mesh->maximal_edge_length();

    if((L_max-L_up)<0.01)
      break;
  }

  std::map<int, int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  std::cout<<"Basic quality:\n";
  mesh->verify();

  VTKTools<double, int>::export_vtu("../data/test_swap_2d-basic", mesh);

  double tic = get_wtime();
  swapping.swap(0.95);
  double toc = get_wtime();

  VTKTools<double, int>::export_vtu("../data/test_swap_2d", mesh);
  VTKTools<double, int>::export_vtu("../data/test_swap_2d_surface", &surface);
  
  qmean = mesh->get_qmean();
  qrms = mesh->get_qrms();
  qmin = mesh->get_qmin();
  if(verbose){
    std::cout<<"Swap loop time: "<<toc-tic<<std::endl
             <<"Quality mean:   "<<qmean<<std::endl
             <<"Quality min:    "<<qmin<<std::endl
             <<"Quality RMS:    "<<qrms<<std::endl;
  }

  delete mesh;

  MPI::Finalize();

  return 0;
}
