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

#ifndef GRAPH_PARTITIONING_H
#define GRAPH_PARTITIONING_H

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <vector>
#include <set>

extern "C" {
#ifdef HAVE_METIS_H
#include <metis.h>
#else
#ifdef HAVE_METIS_METIS_H
#include <metis/metis.h>
#endif
#endif
}

/*! \brief Useful set of partitioners
 */
namespace pragmatic{
  
  void partition_fine(std::vector< std::vector<int> > &NNList, int *active_vertex, int nparts, int *part){
    int NNodes = NNList.size();
    
    if(nparts==1){
      for(int i=0;i<NNodes;i++){
        part[i] = 0;
      }
      return;
    }
    
    int wgtflag=0;
    int numflag=0;
    int options[] = {1, 2, 1, 1, 0};
    int edgecut;
    
    // Compress graph
    std::vector<idxtype> xadj, adjncy, vwgt, adjwgt;
    xadj.reserve(NNodes+1);
    vwgt.reserve(NNodes);
    
    adjncy.reserve(NNodes*5);
    adjwgt.reserve(NNodes*5);
    
    xadj.push_back(0);
    for(int i=0;i<NNodes;i++){
      for(typename std::vector<int>::const_iterator jt=NNList[i].begin();jt!=NNList[i].end();jt++){
        adjncy.push_back(*jt);
        if(std::max(active_vertex[i], active_vertex[*jt])>=0){
          adjwgt.push_back(1);
        }else{
          adjwgt.push_back(0);
        }
      }

      xadj.push_back(*(xadj.rbegin()) + NNList[i].size());
      if(active_vertex[i]>=0){
        vwgt.push_back(1);
      }else{
        vwgt.push_back(0);
      }
    }

    if(nparts>8){
      METIS_PartGraphKway(&NNodes, &(xadj[0]), &(adjncy[0]),
                          &(vwgt[0]), &(adjwgt[0]), &wgtflag,
                          &numflag, &nparts, options, &edgecut, part);
    }else{
      METIS_PartGraphRecursive(&NNodes, &(xadj[0]), &(adjncy[0]),
                               &(vwgt[0]), &(adjwgt[0]), &wgtflag,
                               &numflag, &nparts, options, &edgecut, part);
    }
  }
  
  void partition_fast(std::vector< std::vector<int> > &NNList, int *active_vertex, int nparts, int *part){
    int NNodes = NNList.size();
    
    if(nparts==1){
    	memset(part, 0, NNodes*sizeof(int));
      return;
    }
    
    // Graph partitioning using breadth-first traversal
    int active_cnt=0;
    for(int i=0;i<NNodes;i++){
      part[i] = -1;
      if(active_vertex[i]>=0)
        active_cnt++;
    }

    const int target = active_cnt/nparts;
    int j=0;
    for(int p=0;p<nparts;p++){
      for(;j<NNodes;j++){
        if((part[j]<0)&&(active_vertex[j]>=0))
          break;
      }
      if(j==NNodes)
        break;

      part[j] = p;
      int cnt = 1;
      std::deque<int> front;
      front.push_back(j);

      while(!front.empty() && cnt<target){
      	int v = *front.begin();
      	front.pop_front();

				for(std::vector<int>::const_iterator it=NNList[v].begin();it!=NNList[v].end();++it){
					if((part[*it]<0)&&(active_vertex[*it]>=0)){
						front.push_back(*it);
						part[*it] = p;
						cnt++;
					}
				}
      }
    }
  }
}
#endif
