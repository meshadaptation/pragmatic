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
#include <algorithm>

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkCell.h>

#include "ticker.h"

#include "Colour.h"

void colour_stats(std::vector< std::vector<index_t> > &graph, const char *colour, int NNodes){
  std::map<int, int> colours;
  int max_var=0, mean_var=0; 
  bool valid=true;
  for(int i=0;i<NNodes;i++){
    max_var = std::max(max_var, (int)graph[i].size());
    mean_var += graph[i].size();
    if(colours.count(colour[i])){
      colours[colour[i]]++;
    }else{
      colours[colour[i]]=1;
    }
    for(std::vector<index_t>::const_iterator it=graph[i].begin();it!=graph[i].end();++it){
      valid = valid && (colour[i]!=colour[*it]);
      if(colour[i]==colour[*it]){
        std::cout<<"invalid colour "<<i<<", "<<*it<<" "<<colour[i]<<std::endl;
      }
    }

  }
  mean_var/=NNodes;
  std::cout<<"Valid colouring: ";
  if(valid)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";
  std::cout<<"Chromatic number: "<<colours.size()<<std::endl;
  for(std::map<int, int>::const_iterator it=colours.begin();it!=colours.end();++it)
    std::cout<<it->first<<"\t";
  std::cout<<std::endl;
   for(std::map<int, int>::const_iterator it=colours.begin();it!=colours.end();++it)
    std::cout<<it->second<<"\t";
  std::cout<<std::endl;

  std::cout<<"Max variance: "<<max_var<<std::endl;
  std::cout<<"Mean variance: "<<mean_var<<std::endl;
}

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const char filename[]="../data/box200x200.vtu";

  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName(filename);
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();

  size_t NElements = ug->GetNumberOfCells();  
  std::vector<int> ENList(NElements*3);
  for(size_t i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    assert(cell->GetNumberOfPoints()==3);
    for(int j=0;j<3;j++){
      ENList[i*3+j] = cell->GetPointId(j);
    }
  }

  size_t NNodes = ug->GetNumberOfPoints();
  reader->Delete();

  // Graph
  std::vector< std::vector<int> > graph(NNodes);
  for(size_t i=0;i<NElements;i++){
    for(int j=0;j<3;j++){
      for(int k=j+1;k<3;k++){
        graph[ENList[i*3+j]].push_back(ENList[i*3+k]);
        graph[ENList[i*3+k]].push_back(ENList[i*3+j]);
      }
    }
  }
  for(size_t i=0;i<NNodes;i++){
    std::sort(graph[i].begin(), graph[i].end());
    graph[i].erase(std::unique(graph[i].begin(), graph[i].end()), graph[i].end() );
  }

  double tic, toc;
  // Colour.
  std::cout<<"################\nGreedy colouring\n";
  std::vector<char> colour0(NNodes);
  tic = get_wtime();
  Colour::greedy(NNodes, graph, colour0);
  toc = get_wtime();
  std::cout<<"Wall time "<<toc-tic<<std::endl;
  colour_stats(graph, &(colour0[0]), NNodes);

  std::cout<<"################\nGebremedhin-Manne colouring\n";
  std::vector<char> colour1(NNodes);
  tic = get_wtime();
  Colour::GebremedhinManne(NNodes, graph, colour1);
  toc = get_wtime();
  std::cout<<"Wall time "<<toc-tic<<std::endl;
  colour_stats(graph, &(colour1[0]), NNodes);

  for(int i=0;i<5;i++){
    std::cout<<"################\nRepair colouring "<<i<<"\n";
    tic = get_wtime();
    Colour::repair(NNodes, graph, colour1);
    toc = get_wtime();
    std::cout<<"Wall time "<<toc-tic<<std::endl;
    colour_stats(graph, &(colour1[0]), NNodes);
  }

  MPI_Finalize();

  return 0;
}
