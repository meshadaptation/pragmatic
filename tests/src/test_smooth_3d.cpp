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

  const double target_quality_mean = 0.3;
  const double target_quality_min = 0.01;

  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box20x20x20.vtu");
  mesh->create_boundary();
  
  double pi = 3.14159265358979323846;
  
  MetricField<double,3> metric_field(*mesh);
  
  size_t NNodes = mesh->get_number_nodes();
  
  double dx = 1.0/20;
  for(size_t i=0;i<NNodes;i++){
    double x = 2*mesh->get_coords(i)[0]-1;
    double y = 2*mesh->get_coords(i)[1]-1;
    double z = 2*mesh->get_coords(i)[2]-1;
    
    double l = dx + 0.9*dx*(sin(3*x*pi) + sin(3*y*pi) + sin(3*z*pi))/3;
    double invl2 = 1.0/(l*l);
    double m[] = {invl2, 0.0, 0.0, invl2, 0.0, invl2};
    
    metric_field.set_metric(m, i);
  }
  
  metric_field.update_mesh();
  
  VTKTools<double>::export_vtu("../data/test_smooth_3d_init", mesh);
  double qmean = mesh->get_qmean();
  double qmin = mesh->get_qmin();
  
  if(rank==0)
    std::cout<<"Initial quality:"<<std::endl
	     <<"Quality mean:    "<<qmean<<std::endl
	     <<"Quality min:     "<<qmin<<std::endl;
  
  Smooth<double, 3> smooth(*mesh);
  
  double tic = get_wtime();
  smooth.smooth(100);
  double toc = get_wtime();
  
  qmean = mesh->get_qmean();
  qmin = mesh->get_qmin();
  
  long double area = mesh->calculate_area();
  long double volume = mesh->calculate_volume();
  
  if(rank==0)
    std::cout<<"Smooth loop time: "<<toc-tic<<std::endl
	     <<"Quality mean:     "<<qmean<<std::endl
	     <<"Quality min:      "<<qmin<<std::endl;
  
  std::string vtu_filename = std::string("../data/test_smooth_3d_");
  VTKTools<double>::export_vtu(vtu_filename.c_str(), mesh);
  
  if(rank==0){
    std::cout<<"Checking quality between bounds - "<<" (mean>"<<target_quality_mean<<", min>"<<target_quality_min<<"): ";
    if((qmean>target_quality_mean)&&(qmin>target_quality_min))
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail"<<std::endl;
    
    std::cout<<"Checking area == 6: ";
    if(fabs(area-6)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
    
    std::cout<<"Checking volume == 1: ";
    if(fabs(volume-1)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (volume="<<volume<<")"<<std::endl;
  }
  
  MPI_Finalize();

  return 0;
}
