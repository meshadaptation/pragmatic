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
#include <unistd.h>

#include <errno.h>
#include <stdlib.h>

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

#include "Refine.h"
#include "ticker.h"

using namespace std;

int main(int argc, char **argv){
#ifdef HAVE_MPI
  MPI::Init(argc,argv);
  
  // Undo some MPI init shenanigans.
  if(chdir(getenv("PWD"))){
    perror("");
    exit(-1);
  }

  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box10x10x10.vtu");

  Surface<double, int> surface(*mesh);

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] =
      pow(mesh->get_coords(i)[0], 4) + 
      pow(mesh->get_coords(i)[1], 4) + 
      pow(mesh->get_coords(i)[2], 4);
  
  metric_field.add_field(&(psi[0]), 0.01);
  metric_field.update_mesh();
  
  Refine<double, int> adapt(*mesh, surface);

  double tic = get_wtime();
  adapt.refine(sqrt(2.0));
  double toc = get_wtime();

  std::map<int, int> active_vertex_map;
  mesh->defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  VTKTools<double, int>::export_vtu("../data/test_mpi_refine_3d", mesh);
  VTKTools<double, int>::export_vtu("../data/test_mpi_refine_3d_surface", &surface);

  delete mesh;

  if(MPI::COMM_WORLD.Get_rank()==0){
    std::cout<<"Refine time = "<<toc-tic<<std::endl;
    std::cout<<"pass"<<std::endl;
  }
  
  MPI::Finalize();
#else
  std::cout<<"warning - no MPI compiled"<<std::endl;
#endif
  return 0;
}
