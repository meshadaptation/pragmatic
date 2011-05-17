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
#include <errno.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Smooth.h"

using namespace std;

int main(int argc, char **argv){
  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box20x20.vtu");

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 3) + pow(mesh->get_coords(i)[1], 3);
  
  metric_field.add_field(&(psi[0]), 0.6);

  size_t NElements = mesh->get_number_elements();

  metric_field.apply_nelements(NElements);
  metric_field.update_mesh();
  
  Smooth<double, int> smooth(*mesh, surface);

  double tic = omp_get_wtime();
  smooth.smooth("optimisation");
  double toc = omp_get_wtime();
  
  double lmean = mesh->get_lmean();
  double lrms = mesh->get_lrms();

  double qmean = mesh->get_qmean();
  double qrms = mesh->get_qrms();
  double qmin = mesh->get_qmin();
  
  std::cout<<"Smooth loop time:     "<<toc-tic<<std::endl
           <<"Edge length mean:      "<<lmean<<std::endl
           <<"Edge length RMS:      "<<lrms<<std::endl
           <<"Quality mean:          "<<qmean<<std::endl
           <<"Quality min:          "<<qmin<<std::endl
           <<"Quality RMS:          "<<qrms<<std::endl;
  
  VTKTools<double, int>::export_vtu("../data/test_optimise_smooth_2d", mesh);
  delete mesh;

  if((lrms<0.4)&&(qrms<0.1))
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
