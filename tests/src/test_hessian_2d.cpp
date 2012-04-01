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

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box200x200.vtu");

  Surface<double, int> surface(*mesh, true);

  size_t NNodes = mesh->get_number_nodes();

  // Set up field - use first touch policy
  vector<double> psi(NNodes);
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for(size_t i=0;i<NNodes;i++)
      psi[i] = pow(mesh->get_coords(i)[0]+0.1, 2) + pow(mesh->get_coords(i)[1]+0.1, 2);
  }
  
  const char *methods[] = {"qls", "qls2"};
  for(int m=0;m<2;m++){
    const char *method = methods[m];
    
    MetricField<double, int> metric_field(*mesh, surface);
    metric_field.set_hessian_method(method); // default
    
    double tic = get_wtime();
    metric_field.add_field(&(psi[0]), 1.0);
    double toc = get_wtime();

    metric_field.update_mesh();
    
    vector<double> metric(NNodes*4);
    metric_field.get_metric(&(metric[0]));
    
    double rms[] = {0., 0., 0., 0.};
    for(size_t i=0;i<NNodes;i++){
      rms[0] += pow(2.0-metric[i*4  ], 2); rms[1] += pow(    metric[i*4+1], 2);
      rms[2] += pow(    metric[i*4+2], 2); rms[3] += pow(2.0-metric[i*4+3], 2);
    }
    
    double max_rms = 0;
    for(size_t i=0;i<4;i++){
      rms[i] = sqrt(rms[i]/NNodes);
      max_rms = std::max(max_rms, rms[i]);
    }

    string vtu_filename = string("../data/test_hessian_2d_")+string(method);
    VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh);
    
    std::cout<<"Hessian ("<<method<<") loop time = "<<toc-tic<<std::endl
             <<"Max RMS = "<<max_rms<<std::endl;
    if(max_rms>0.01)
      std::cout<<"fail\n";
    else
      std::cout<<"pass\n";
  }

  delete mesh;

  
  return 0;
}

