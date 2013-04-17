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

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#include <vector>
#include <set>
#include <iostream>

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkCell.h>

#include "zoltan_tools.h"

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
  zoltan_graph_t graph;

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
  std::vector<int> owner(NNodes, 0);
  for(size_t i=0;i<NNodes;i++){
    lnn2gnn[i] = i;
  }

  graph.gid = &(lnn2gnn[0]);
  graph.owner = &(owner[0]);
  
  zoltan_colour(&graph, 1, MPI_COMM_WORLD);
  zoltan_colour(&graph, 1, MPI_COMM_WORLD);

  MPI_Finalize();

  std::cout<<"pass\n";

  return 0;
}
