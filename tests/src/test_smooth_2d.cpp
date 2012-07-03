/* 
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a
 *    full list of copyright holders.
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
 *    License as published by the Free Software Foundation, version
 *    2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free
 *    Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *    MA 02111-1307 USA
 */
#include "pragmatic_config.h"

#include <iostream>
#include <vector>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"
#include "Smooth.h"
#include "ticker.h"

#include <mpi.h>

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  int rank = MPI::COMM_WORLD.Get_rank();

  const char *methods[] = {"Laplacian", "smart Laplacian", "smart Laplacian search", "optimisation Linf"};
  const double target_quality_mean[] = {0.4, 0.7, 0.7, 0.7};
  const double target_quality_min[]  = {0.0, 0.1, 0.2, 0.3};
  for(int m=0;m<4;m++){
    const char *method = methods[m];

    Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/smooth_2d.vtu");
    Surface<double, int> surface(*mesh);
    surface.find_surface(true);

    MetricField<double, int> metric_field(*mesh, surface);

    size_t NNodes = mesh->get_number_nodes();

    double eta=0.002;

    vector<double> psi(NNodes);
    for(size_t i=0;i<NNodes;i++){
      double x = 2*mesh->get_coords(i)[0]-1;
      double y = 2*mesh->get_coords(i)[1]-1;
    
      psi[i] = 0.100000000000000*sin(50*x) + atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
    }

    metric_field.add_field(&(psi[0]), eta, 1);
    metric_field.update_mesh();
    
    if(m==0){
      VTKTools<double, int>::export_vtu("../data/test_smooth_2d_init", mesh);
      double qmean = mesh->get_qmean();
      double qrms = mesh->get_qrms();
      double qmin = mesh->get_qmin();
      
      if(rank==0)
        std::cout<<"Initial quality:"<<std::endl
                 <<"Quality mean:    "<<qmean<<std::endl
                 <<"Quality min:     "<<qmin<<std::endl
                 <<"Quality RMS:     "<<qrms<<std::endl;
    }
    
    Smooth<double, int> smooth(*mesh, surface);
    
    int max_smooth_iter=1000;
    
    double tic = get_wtime();
    smooth.smooth(method, max_smooth_iter);
    double toc = get_wtime();
    
    double lmean = mesh->get_lmean();
    double lrms = mesh->get_lrms();
    
    double qmean = mesh->get_qmean();
    double qrms = mesh->get_qrms();
    double qmin = mesh->get_qmin();
    
    if(rank==0)
      std::cout<<"Smooth loop time ("<<method<<"):     "<<toc-tic<<std::endl
               <<"Edge length mean:      "<<lmean<<std::endl
               <<"Edge length RMS:      "<<lrms<<std::endl
               <<"Quality mean:          "<<qmean<<std::endl
               <<"Quality min:          "<<qmin<<std::endl
               <<"Quality RMS:          "<<qrms<<std::endl;
    
    string vtu_filename = string("../data/test_smooth_2d_")+string(method);
    VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh);
    delete mesh;
    
    if(rank==0){
      std::cout<<"Smooth - "<<methods[m]<<" ("<<qmean<<", "<<qmin<<"): ";
      if((qmean>target_quality_mean[m])&&(qmin>target_quality_min[m]))
        std::cout<<"pass"<<std::endl;
      else
        std::cout<<"fail"<<std::endl;
    }
  }
  
  MPI::Finalize();

  return 0;
}
