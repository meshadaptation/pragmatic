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

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <vector>
#include <set>
#include <iostream>

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkCell.h>

#include "zoltan_colour.h"

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  int rank = 0;
  if(MPI::Is_initialized()){
    rank = MPI::COMM_WORLD.Get_rank();
  }
  assert(MPI::COMM_WORLD.Get_size()==1);

  const char filename[]="../data/box10x10x10.vtu";

  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName(filename);
  reader->Update();
  
  vtkUnstructuredGrid *ug = reader->GetOutput();
  
  size_t NElements = ug->GetNumberOfCells();  
  std::vector<int> ENList(NElements*4);
  for(size_t i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    assert(cell->GetNumberOfPoints()==4);
    for(int j=0;j<4;j++){
      ENList[i*4+j] = cell->GetPointId(j);
    }
  }

  size_t NNodes = ug->GetNumberOfPoints();
  reader->Delete();
   
  // Graph
  std::vector< std::set<int> > graph_ragged(NNodes);
  for(size_t i=0;i<NElements;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
        graph_ragged[ENList[i*4+j]].insert(ENList[i*4+k]);
      }
    }
  }

  // Colour.
  zoltan_colour_graph_t graph;
  
  std::vector<int> colour(NNodes);
  graph.colour = &(colour[0]);
  graph.rank = rank; 
  graph.nnodes = NNodes;
  graph.npnodes = NNodes;
  
  std::vector<size_t> nedges(NNodes);
  size_t sum = 0;
  for(size_t i=0;i<NNodes;i++){
    size_t cnt = 0;
    cnt = graph_ragged[i].size();
    nedges[i] = cnt;
    sum+=cnt;
  }
  graph.nedges = &(nedges[0]);
  
  std::vector<size_t> csr_edges(sum);
  sum=0;
  for(size_t i=0;i<NNodes;i++){
    for(std::set<int>::iterator it=graph_ragged[i].begin();it!=graph_ragged[i].end();++it){
      csr_edges[sum++] = *it;
    }
  }
  graph.csr_edges = &(csr_edges[0]);
        
  std::vector<int> lnn2gnn(NNodes);
  std::vector<size_t> owner(NNodes, 0);
  for(size_t i=0;i<NNodes;i++){
    lnn2gnn[i] = i;
  }

  graph.gid = &(lnn2gnn[0]);
  graph.owner = &(owner[0]);
  
  zoltan_colour(&graph, 1, MPI_COMM_WORLD);
  zoltan_colour(&graph, 1, MPI_COMM_WORLD);

  MPI::Finalize();

  std::cout<<"pass\n";

  return 0;
}
