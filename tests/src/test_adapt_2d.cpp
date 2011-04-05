/* 
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a full list
 *    of copyright holders.
 *
 *    Gerard Gorman
 *    Applied Modelling and Computation Group
 *    Department of Earth Science and Engineering
 *    Imperial College London
 *
 *    amcgsoftware@imperial.ac.uk
 *    
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation,
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
 *    USA
 */
#include <iostream>
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "vtk_tools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"

using namespace std;

int main(int argc, char **argv){
  Mesh<double, int> *mesh=NULL;
  import_vtu("../data/box20x20.vtu", mesh);

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 3) + pow(mesh->get_coords(i)[1], 3);
  
  metric_field.add_field(&(psi[0]), 1.0);

  metric_field.apply_nelements(1000);
  metric_field.update_mesh();
  
  double start_tic = omp_get_wtime();
  Coarsen<double, int> coarsen(*mesh, surface);
  coarsen.coarsen(0.8);
  std::cout<<"Coarsen: "<<omp_get_wtime()-start_tic<<std::endl;

  start_tic = omp_get_wtime();
  Refine<double, int> refine(*mesh, surface);
  refine.refine(1.2);
  std::cout<<"Refine: "<<omp_get_wtime()-start_tic<<std::endl;

  start_tic = omp_get_wtime();
  std::map<int, int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);
  std::cout<<"Defragment: "<<omp_get_wtime()-start_tic<<std::endl;

  start_tic = omp_get_wtime();
  Smooth<double, int> smooth(*mesh, surface);
  int iter = smooth.smooth(1.0e-4, 500);
  std::cout<<"Smooth 1 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;

  start_tic = omp_get_wtime();
  iter = smooth.smooth(1.0e-5, 500, true);
  std::cout<<"Smooth 2 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;

  export_vtu("../data/test_adapt_2d.vtu", mesh, &(psi[0]));
  export_vtu("../data/test_adapt_2d_surface.vtu", &surface);

  delete mesh;

  std::cout<<"pass"<<std::endl;

  return 0;
}
