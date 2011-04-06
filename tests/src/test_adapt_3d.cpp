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
  import_vtu("../data/box10x10x10.vtu", mesh);

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = 
      pow(mesh->get_coords(i)[0]+0.1, 4) + 
      pow(mesh->get_coords(i)[1]+0.1, 4) +
      pow(mesh->get_coords(i)[2]+0.1, 4);
  
  metric_field.add_field(&(psi[0]), 0.2);
  metric_field.update_mesh();

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_low = 0.4;
  double L_up = sqrt(2);

  double start_tic = omp_get_wtime();
  Coarsen<double, int> coarsen(*mesh, surface);
  coarsen.coarsen(L_low, L_up);
  std::cout<<"Coarsen1: "<<omp_get_wtime()-start_tic<<std::endl;

  start_tic = omp_get_wtime();
  Smooth<double, int> smooth(*mesh, surface);
  int iter = smooth.smooth(1.0e-1, 100);
  std::cout<<"Smooth 1 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;

  double L_max = mesh->maximal_edge_length();

  int adapt_iter=0;
  double alpha = sqrt(2)/2;
  do{
    double L_ref = std::max(alpha*L_max, L_up);
    
    std::cout<<"#####################\nAdapt iteration "<<adapt_iter++<<std::endl
             <<"L_max = "<<L_max<<", "
             <<"L_ref = "<<L_ref<<", "
             <<"Num elements = "<<mesh->get_number_elements()<<std::endl;

    start_tic = omp_get_wtime();
    Refine<double, int> refine(*mesh, surface);
    refine.refine(L_ref);
    std::cout<<"Refine: "<<omp_get_wtime()-start_tic<<std::endl;
    
    start_tic = omp_get_wtime();
    coarsen.coarsen(L_low, L_max);
    std::cout<<"Coarsen2: "<<omp_get_wtime()-start_tic<<std::endl;
    
    start_tic = omp_get_wtime();
    iter = smooth.smooth(1.0e-3, 50);
    std::cout<<"Smooth 2 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;
    
    L_max = mesh->maximal_edge_length();
  }while((L_max>L_up)&&(adapt_iter<10));
  
  start_tic = omp_get_wtime();
  iter = smooth.smooth(1.0e-5, 100, true);
  std::cout<<"Smooth 3 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;
  
  export_vtu("../data/test_adapt_3d.vtu", mesh);
  export_vtu("../data/test_adapt_3d_surface.vtu", &surface);

  delete mesh;

  std::cout<<"pass"<<std::endl;

  return 0;
}
