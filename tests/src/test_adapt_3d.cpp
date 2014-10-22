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

#include <cmath>
#include <iostream>
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"
#include "Swapping.h"

#include <mpi.h>

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);

  bool verbose = false;
  if(argc>1){
    verbose = std::string(argv[1])=="-v";
  }

  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box10x10x10.vtu");
  mesh->create_boundary();

  MetricField<double,3> metric_field(*mesh);

  size_t NNodes = mesh->get_number_nodes();
  size_t NElements = mesh->get_number_elements();

  for(size_t i=0;i<NNodes;i++){
    double hx=0.025 + 0.09*mesh->get_coords(i)[0];
    double hy=0.025 + 0.09*mesh->get_coords(i)[1];
    double hz=0.025 + 0.09*mesh->get_coords(i)[2];
    double m[] =
      {1.0f/powf(hx, 2),
       0.0f,           
       0.0f,
       1.0f/powf(hy, 2),
       0.0f,
       1.0f/powf(hz, 2)};

    metric_field.set_metric(m, i);
  }
  metric_field.apply_nelements(NElements);
  metric_field.update_mesh();

  double qmean = mesh->get_qmean();
  double qmin = mesh->get_qmin();

  std::cout<<"Initial quality:\n"
           <<"Quality mean:  "<<qmean<<std::endl
           <<"Quality min:   "<<qmin<<std::endl;
  VTKTools<double>::export_vtu("../data/test_adapt_3d-initial", mesh);

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = 1.0; // sqrt(2);
  double L_low = L_up/2;

  Coarsen<double, 3> coarsen(*mesh);
  Smooth<double, 3> smooth(*mesh);
  Refine<double, 3> refine(*mesh);
  Swapping<double, 3> swapping(*mesh);
  
  coarsen.coarsen(L_low, L_up);
  std::cout<<"INFO: Verify quality after initial coarsen.\n";
  assert(mesh->verify());
  std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
  long double area = mesh->calculate_area();
  long double volume = mesh->calculate_volume();
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
  
  double L_max = mesh->maximal_edge_length();
  
  double alpha = sqrt(2.0)/2;
  for(size_t i=0;i<10;i++){
    double L_ref = std::max(alpha*L_max, L_up);
    
    refine.refine(L_ref);
    std::cout<<"INFO: Verify quality after refine; but before coarsen.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    area = mesh->calculate_area();
    volume = mesh->calculate_volume();
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

    coarsen.coarsen(L_low, L_ref);

    std::cout<<"INFO: Verify quality after coarsen; but before swapping.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    area = mesh->calculate_area();
    volume = mesh->calculate_volume();
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

    for(int j=0;j<10;j++)
      swapping.swap(0.5);

    std::cout<<"INFO: Verify quality after swapping.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    area = mesh->calculate_area();
    volume = mesh->calculate_volume();
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
    
    L_max = mesh->maximal_edge_length();

    if((L_max-L_up)<0.01)
      break;
  }
  
  mesh->defragment();

  smooth.smooth();
  
  qmean = mesh->get_qmean();
  qmin = mesh->get_qmin();
  
  std::cout<<"After adaptivity:\n";
  mesh->verify();
  area = mesh->calculate_area();
  volume = mesh->calculate_volume();
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
  
  VTKTools<double>::export_vtu("../data/test_adapt_3d", mesh);
  
  delete mesh;
  
  if((qmean>0.3)&&(qmin>0.0002))
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  MPI_Finalize();

  return 0;
}
