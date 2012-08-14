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
#include <unistd.h>

#include <errno.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Refine.h"
#include "ticker.h"

using namespace std;

int main(int argc, char **argv){
#ifdef HAVE_MPI
  MPI::Init(argc,argv);

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box10x10.vtu");

  Surface<double, int> surface(*mesh);
  surface.find_surface();

  MetricField2D<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 4) + pow(mesh->get_coords(i)[1], 4);
  
  metric_field.add_field(&(psi[0]), 0.01);
  metric_field.update_mesh();
  
  Refine2D<double, int> adapt(*mesh, surface);

  double tic = get_wtime();
  adapt.refine(sqrt(2.0));
  double toc = get_wtime();

  std::vector<int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  VTKTools<double, int>::export_vtu("../data/test_mpi_refine_2d", mesh);
  VTKTools<double, int>::export_vtu("../data/test_mpi_refine_2d_surface", &surface);

  delete mesh;

  if(MPI::COMM_WORLD.Get_rank()==0){
    std::cout<<"Refine time = "<<toc-tic<<std::endl;
    std::cout<<"pass"<<std::endl;
  }
  
  MPI::Finalize();
#else
  std::cout<<"warning - no MPI compiled"<<std::endl;
#endif
  return 0;
}
