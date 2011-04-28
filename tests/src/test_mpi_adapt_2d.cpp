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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Refine.h"
#include "Smooth.h"

using namespace std;

int main(int argc, char **argv){
#ifdef HAVE_MPI
  MPI::Init(argc,argv);
  
  // Undo some MPI init shenanigans.
  if(chdir(getenv("PWD"))){
    perror("choked on MPI init shenanigans");
    exit(-1);
  }

  int rank = 0;
  int nprocs = 1;
  if(MPI::Is_initialized()){
    rank = MPI::COMM_WORLD.Get_rank();
    nprocs = MPI::COMM_WORLD.Get_size();
  }
  
  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box20x20.vtu");

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 4) + pow(mesh->get_coords(i)[1], 4);
  
  metric_field.add_field(&(psi[0]), 0.01);
  metric_field.update_mesh();

  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_low = 0.4;
  double L_up = 1.0;

  double start_tic = omp_get_wtime();
  Coarsen<double, int> coarsen(*mesh, surface);
  coarsen.coarsen(L_low, L_up);
  if(MPI::COMM_WORLD.Get_rank()==0)
    std::cout<<"Coarsen1: "<<omp_get_wtime()-start_tic<<std::endl;
  
  start_tic = omp_get_wtime();
  Smooth<double, int> smooth(*mesh, surface);
  int iter = smooth.smooth(1.0e-1, 100);
  if(MPI::COMM_WORLD.Get_rank()==0)
    std::cout<<"Smooth 1 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;
  
  double L_max = mesh->maximal_edge_length();
  int long_edges;
  if(nprocs>1)
    MPI_Allreduce(&long_edges, &long_edges, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  int adapt_iter=0;
  double alpha = sqrt(2)/2;
  Refine<double, int> refine(*mesh, surface);
  do{
    double L_ref = std::max(alpha*L_max, L_up);
    if(rank==0)
      std::cout<<"#####################\nAdapt iteration "<<adapt_iter<<std::endl
               <<"L_max = "<<L_max<<", "
               <<"L_ref = "<<L_ref<<", "
               <<"Num elements = "<<mesh->get_number_elements()<<std::endl;
    
    start_tic = omp_get_wtime();
    refine.refine(L_ref);
    if(rank==0)
      std::cout<<"Refine: "<<omp_get_wtime()-start_tic<<std::endl;
    
    start_tic = omp_get_wtime();
    coarsen.coarsen(L_low, L_max);
    if(rank==0)
      std::cout<<"Coarsen2: "<<omp_get_wtime()-start_tic<<std::endl;
    
    start_tic = omp_get_wtime();
    iter = smooth.smooth(1.0e-3, 50);
    if(rank==0)
      std::cout<<"Smooth 2 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;
    
    L_max = mesh->maximal_edge_length();
    long_edges=(L_max>L_up)?1:0;
    if(nprocs>1)
      MPI_Allreduce(&long_edges, &long_edges, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
  }while((long_edges>0)&&(adapt_iter++<10));
  
  start_tic = omp_get_wtime();
  iter = smooth.smooth(1.0e-5, 100, true);
  if(rank==0)
    std::cout<<"Smooth 3 (Iterations="<<iter<<"): "<<omp_get_wtime()-start_tic<<std::endl;
  
  std::map<int, int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);
  
  VTKTools<double, int>::export_vtu("../data/test_mpi_adapt_2d", mesh);
  VTKTools<double, int>::export_vtu("../data/test_mpi_adapt_2d_surface", &surface);
  
  delete mesh;
  
  if(MPI::COMM_WORLD.Get_rank()==0)
    std::cout<<"pass"<<std::endl;
  
  MPI::Finalize();
#else
  std::cout<<"warning - no MPI compiled"<<std::endl;
#endif
  return 0;
}
