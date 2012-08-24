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

int main(int argc, char **argv){
  MPI::Init_thread(argc,argv, MPI::THREAD_SERIALIZED);

  int rank = MPI::COMM_WORLD.Get_rank();

  const char *methods[] = {"Laplacian", "smart Laplacian", "smart Laplacian search", "optimisation Linf"};
  const double target_quality_mean[] = {0.4, 0.7, 0.7, 0.7};
  const double target_quality_min[]  = {0.0, 0.1, 0.2, 0.3};
  for(int m=0;m<4;m++){
    const char *method = methods[m];

    Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/smooth_2d.vtu");
    Surface2D<double, int> surface(*mesh);
    surface.find_surface(true);

    MetricField2D<double, int> metric_field(*mesh, surface);

    size_t NNodes = mesh->get_number_nodes();

    double eta=0.002;

    std::vector<double> psi(NNodes);
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
    
    Smooth2D<double, int> smooth(*mesh, surface);
    
    int max_smooth_iter=2;
    
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
    
    std::string vtu_filename = std::string("../data/test_smooth_2d_")+std::string(method);
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
