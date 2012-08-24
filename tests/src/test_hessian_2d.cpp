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

#include "pragmatic_config.h"

#include <iostream>
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box200x200.vtu");

  Surface2D<double, int> surface(*mesh);
  surface.find_surface(true);

  size_t NNodes = mesh->get_number_nodes();

  // Set up field - use first touch policy
  std::vector<double> psi(NNodes);
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for(size_t i=0;i<NNodes;i++)
      psi[i] = pow(mesh->get_coords(i)[0]+0.1, 2) + pow(mesh->get_coords(i)[1]+0.1, 2);
  }
  
  MetricField2D<double, int> metric_field(*mesh, surface);
  
  double tic = get_wtime();
  metric_field.add_field(&(psi[0]), 1.0);
  double toc = get_wtime();
  
  metric_field.update_mesh();
  
  std::vector<float> metric(NNodes*3);
  metric_field.get_metric(&(metric[0]));
  
  double rms[] = {0., 0., 0.};
  for(size_t i=0;i<NNodes;i++){
    rms[0] += pow(2.0-metric[i*3  ], 2);
    rms[1] += pow(    metric[i*3+1], 2);
    rms[2] += pow(2.0-metric[i*3+2], 2);
  }
  
  double max_rms = 0;
  for(size_t i=0;i<3;i++){
    rms[i] = sqrt(rms[i]/NNodes);
    max_rms = std::max(max_rms, rms[i]);
  }
  
  std::string vtu_filename("../data/test_hessian_2d");
  VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh, &(psi[0]));
  
  std::cout<<"Hessian :: loop time = "<<toc-tic<<std::endl
           <<"RMS = "<<rms[0]<<", "<<rms[1]<<", "<<rms[2]<<std::endl;
  if(max_rms>0.01)
    std::cout<<"fail\n";
  else
    std::cout<<"pass\n";

  delete mesh;

  return 0;
}

