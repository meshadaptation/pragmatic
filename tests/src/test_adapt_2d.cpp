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
#include "ticker.h"

#include <mpi.h>

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool verbose = false;
  if(argc>1){
    verbose = std::string(argv[1])=="-v";
  }

  // Benchmark times.
  double time_coarsen=0, time_refine=0, time_swap=0, time_smooth=0, time_adapt=0, tic;

  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box200x200.vtu");
  mesh->create_boundary();

  MetricField<double,2> metric_field(*mesh);

  size_t NNodes = mesh->get_number_nodes();
  double eta=0.001;

  std::vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double x = 2*mesh->get_coords(i)[0]-1;
    double y = 2*mesh->get_coords(i)[1]-1;

    psi[i] = 0.1*sin(50*x) + atan2(-0.1, (double)(2*x - sin(5*y)));
  }

  metric_field.add_field(&(psi[0]), eta, 2);
  metric_field.update_mesh();

  double qmean = mesh->get_qmean();
  double qmin = mesh->get_qmin();

  if((rank==0)&&(verbose)) std::cout<<"Initial quality:\n"
                                    <<"Quality mean:  "<<qmean<<std::endl
                                    <<"Quality min:   "<<qmin<<std::endl;
  //VTKTools<double>::export_vtu("../data/test_adapt_2d-initial", mesh);

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = sqrt(2.0);
  double L_low = L_up/2;

  Coarsen<double, 2> coarsen(*mesh);
  Smooth<double, 2> smooth(*mesh);
  Refine<double, 2> refine(*mesh);
  Swapping<double, 2> swapping(*mesh);

  time_adapt = get_wtime();

  double L_max = mesh->maximal_edge_length();
  double alpha = sqrt(2.0)/2;
  for(size_t i=0;i<20;i++){
    double L_ref = std::max(alpha*L_max, L_up);
    
    tic = get_wtime();
    coarsen.coarsen(L_low, L_ref);
    std::cout<<"INFO: Verify quality after coarsen.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    long double perimeter = mesh->calculate_perimeter();
    long double area = mesh->calculate_area();
    std::cout<<"Expecting area == 1: ";
    if(fabs(area-1)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
    std::cout<<"Expecting perimeter == 4: ";
    if(fabs(perimeter-4)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (perimeter="<<perimeter<<")"<<std::endl;
    time_coarsen += get_wtime() - tic;

    tic = get_wtime();
    swapping.swap(0.7);
    std::cout<<"INFO: Verify quality after swapping.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    perimeter = mesh->calculate_perimeter();
    area = mesh->calculate_area();
    std::cout<<"Expecting area == 1: ";
    if(fabs(area-1)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
    std::cout<<"Expecting perimeter == 4: ";
    if(fabs(perimeter-4)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (perimeter="<<perimeter<<")"<<std::endl;
    time_swap += get_wtime() - tic;

    tic = get_wtime();
    refine.refine(L_ref);
    std::cout<<"INFO: Verify quality after refinement.\n";
    assert(mesh->verify());
    std::cout<<"Number elements: "<<mesh->get_number_elements()<<std::endl;
    perimeter = mesh->calculate_perimeter();
    area = mesh->calculate_area();
    std::cout<<"Expecting area == 1: ";
    if(fabs(area-1)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
    std::cout<<"Expecting perimeter == 4: ";
    if(fabs(perimeter-4)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (perimeter="<<perimeter<<")"<<std::endl;
    time_refine += get_wtime() - tic;
    
    L_max = mesh->maximal_edge_length();
    
    if((L_max-L_up)<0.01)
      break;
  }

  double time_defrag = get_wtime();
  mesh->defragment();
  time_defrag = get_wtime()-time_defrag;

  if(verbose){
    if(rank==0)
      std::cout<<"Basic quality:\n";
    mesh->verify();

    VTKTools<double>::export_vtu("../data/test_adapt_2d-basic", mesh);
  }

  tic = get_wtime();
  smooth.smooth(20);
  time_smooth += get_wtime()-tic;

  time_adapt = get_wtime()-time_adapt;

  if(verbose){
    if(rank==0)
      std::cout<<"After optimisation based smoothing:\n";
    mesh->verify();
  }

  NNodes = mesh->get_number_nodes();
  psi.resize(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double x = 2*mesh->get_coords(i)[0]-1;
    double y = 2*mesh->get_coords(i)[1]-1;

    psi[i] = 0.100000000000000*sin(50*x) + atan2(-0.100000000000000, (double)(2*x - sin(5*y)));
  }

  VTKTools<double>::export_vtu("../data/test_adapt_2d", mesh, &(psi[0]));

  qmean = mesh->get_qmean();
  qmin = mesh->get_qmin();
  
  long double perimeter = mesh->calculate_perimeter();
  long double area = mesh->calculate_area();

  delete mesh;

  if(rank==0){
    std::cout<<"BENCHMARK: time_coarsen time_refine time_swap time_smooth time_defrag time_adapt time_other\n";
    double time_other = (time_adapt-(time_coarsen+time_refine+time_swap+time_smooth+time_defrag));
    std::cout<<"BENCHMARK: "
             <<std::setw(12)<<time_coarsen<<" "
             <<std::setw(11)<<time_refine<<" "
             <<std::setw(9)<<time_swap<<" "
             <<std::setw(11)<<time_smooth<<" "
             <<std::setw(11)<<time_defrag<<" "
             <<std::setw(10)<<time_adapt<<" "
             <<std::setw(10)<<time_other<<"\n";

    std::cout<<"Expecting qmean>0.8, qmin>0.2: ";
    if((qmean>0.8)&&(qmin>0.1))
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (qmean="<<qmean<<", qmin="<<qmin<<")"<<std::endl;

    std::cout<<"Expecting perimeter == 4: ";
    if(fabs(perimeter-4)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (perimeter="<<perimeter<<")"<<std::endl;

    std::cout<<"Expecting area == 1: ";
    if(fabs(area-1)<DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
  }

  MPI_Finalize();

  return 0;
}
