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

#include "Smooth.h"

using namespace std;

int main(int argc, char **argv){
  Mesh<double, int> *mesh=NULL;
  import_vtu("../data/box20x20x20.vtu", mesh);

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();
  
  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 3) +
      pow(mesh->get_coords(i)[1], 3) + pow(mesh->get_coords(i)[2], 3);

  metric_field.add_field(&(psi[0]), 0.6);

  size_t NElements = mesh->get_number_elements();
  
  metric_field.apply_nelements(NElements);
  metric_field.update_mesh();
  
  Smooth<double, int> smooth(*mesh, surface);
  
  double start_tic = omp_get_wtime();
  double prev_mean_quality = smooth.smooth();
  int iter=1;
  for(;iter<500;iter++){
    double mean_quality = smooth.smooth();
    double res = abs(mean_quality-prev_mean_quality)/prev_mean_quality;
    prev_mean_quality = mean_quality;
    if(res<1.0e-4)
      break;
  }
  std::cout<<"Smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 3) +
      pow(mesh->get_coords(i)[1], 3) + pow(mesh->get_coords(i)[2], 3);
  
  mesh->calc_edge_lengths();
  export_vtu("../data/test_smooth_simple_3d.vtu", mesh, &(psi[0]));
  delete mesh;

  if(iter<100)
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
