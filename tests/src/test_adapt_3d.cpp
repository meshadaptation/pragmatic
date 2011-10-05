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
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"

using namespace std;

int main(int argc, char **argv){
  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box10x10x10.vtu");

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();
  size_t NElements = mesh->get_number_elements();

  for(size_t i=0;i<NNodes;i++){
    double hx=0.025 + 0.09*mesh->get_coords(i)[0];
    double hy=0.025 + 0.09*mesh->get_coords(i)[1];
    double hz=0.025 + 0.09*mesh->get_coords(i)[2];
    double m[] =
      {1.0/pow(hx, 2), 0.0,            0.0,
       0.0,            1.0/pow(hy, 2), 0.0,
       0.0,            0.0,            1.0/pow(hz, 2)};
    metric_field.set_metric(m, i);
  }
  metric_field.apply_nelements(NElements);
  metric_field.update_mesh();

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = 1.0; // sqrt(2);
  double L_low = L_up/2;

  Coarsen<double, int> coarsen(*mesh, surface);
  Smooth<double, int> smooth(*mesh, surface);
  Refine<double, int> refine(*mesh, surface);
  Swapping<double, int> swapping(*mesh, surface);
  
  coarsen.coarsen(L_low, L_up);
  
  double L_max = mesh->maximal_edge_length();
  
  double alpha = sqrt(2)/2;
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);
    
    refine.refine(L_ref);
    coarsen.coarsen(L_low, L_ref);
    for(int j=0;j<2;j++)
      swapping.swap(0.1);
    
    L_max = mesh->maximal_edge_length();

    if((L_max-L_up)<0.01)
      break;
  }
  
  std::map<int, int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  smooth.smooth("smart Laplacian");
  mesh->verify();
    
  double qmean = mesh->get_qmean();
  double qrms = mesh->get_qrms();
  double qmin = mesh->get_qmin();

  std::cout<<"After adaptivity:\n"
           <<"Quality mean:  "<<qmean<<std::endl
           <<"Quality min:   "<<qmin<<std::endl
           <<"Quality RMS:   "<<qrms<<std::endl;
  
  VTKTools<double, int>::export_vtu("../data/test_adapt_3d", mesh);
  VTKTools<double, int>::export_vtu("../data/test_adapt_3d_surface", &surface);
  
  delete mesh;
  
  if((qmean>0.8)&&(qmin>0.0003))
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
