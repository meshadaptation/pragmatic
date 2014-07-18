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
#include <cfloat>

#include <omp.h>

#include "Mesh.h"
#include "VTKTools.h"
#include "MetricField.h"
#include "Smooth.h"
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const char *methods[] = {"smart Laplacian", "optimisation Linf"};
  const double target_quality_mean[] = {0.7, 0.7};
  const double target_quality_min[]  = {0.05, 0.1};

  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/smooth_2d.vtu");
  mesh->create_boundary();

  for(int m=0;m<2;m++){
    const char *method = methods[m];

    MetricField2D<double> metric_field(*mesh);

    size_t NNodes = mesh->get_number_nodes();

    double eta=0.002;

    std::vector<double> psi(NNodes);
    for(size_t i=0;i<NNodes;i++){
      double x = 2*mesh->get_coords(i)[0]-1;
      double y = 2*mesh->get_coords(i)[1]-1;
    
      psi[i] = 0.1*sin(50*x) + atan2(-0.1, (double)(2*x - sin(5*y)));
    }

    metric_field.add_field(&(psi[0]), eta, 1);
    metric_field.update_mesh();
    
    if(m==0){
      VTKTools<double>::export_vtu("../data/test_smooth_2d_init", mesh);
      double qmean = mesh->get_qmean();
      double qmin = mesh->get_qmin();
      
      if(rank==0)
        std::cout<<"Initial quality:"<<std::endl
                 <<"Quality mean:    "<<qmean<<std::endl
                 <<"Quality min:     "<<qmin<<std::endl;
    }
    
    Smooth2D<double> smooth(*mesh);
    
    double tic = get_wtime();
    smooth.smooth(method, 100);
    double toc = get_wtime();
    
    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();
    
    long double perimeter = mesh->calculate_perimeter();
    long double area = mesh->calculate_area();

    if(rank==0)
      std::cout<<"Smooth loop time ("<<method<<"):     "<<toc-tic<<std::endl
               <<"Quality mean:     "<<qmean<<std::endl
               <<"Quality min:      "<<qmin<<std::endl;
    
    std::string vtu_filename = std::string("../data/test_smooth_2d_")+std::string(method);
    VTKTools<double>::export_vtu(vtu_filename.c_str(), mesh);
    
    if(rank==0){
      std::cout<<"Checking quality between bounds - "<<methods[m]<<" (mean>"<<target_quality_mean[m]<<", min>"<<target_quality_min[m]<<"): ";
      if((qmean>target_quality_mean[m])&&(qmin>target_quality_min[m]))
        std::cout<<"pass"<<std::endl;
      else
        std::cout<<"fail"<<std::endl;
      
      std::cout<<"Checking perimeter == 4: ";
      if(fabs(perimeter-4)<DBL_EPSILON)
        std::cout<<"pass"<<std::endl;
      else
        std::cout<<"fail ("<<perimeter<<")"<<std::endl;

      std::cout<<"Checking area == 1: ";
      if(fabs(area-1)<DBL_EPSILON)
        std::cout<<"pass"<<std::endl;
      else
        std::cout<<"fail ("<<area<<")"<<std::endl;
    }
  }
  
  MPI_Finalize();

  return 0;
}
