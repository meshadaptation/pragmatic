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

#ifndef METIS_H
#define METIS_H

#include "pragmatic_config.h"

#include <cassert>
#include <cstdlib>
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

typedef int idxtype;
}

/*! \brief Class provides a specialised interface to some METIS
 *   functionality.
 */
namespace metis{
  /*! Calculate a node renumbering.
   * @param graph is the undirected graph to be partitioned.
   * @param norder is an array storing the partition each node in the graph is assigned to.
   */
  void reorder(const std::vector< std::set<int> > &graph, std::vector<int> &norder){
    int nnodes = graph.size();
    
    // Compress graph
    std::vector<idxtype> xadj(nnodes+1), adjncy;
    int pos=0;
    xadj[0]=0;
    for(int i=0;i<nnodes;i++){
      for(typename std::set<int>::const_iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
        assert((*jt)>=0);
        assert((*jt)<nnodes);
        adjncy.push_back(*jt);
        pos++;
      }
      xadj[i+1] = pos;
    }
    
    norder.resize(nnodes);
    std::vector<int> inorder(nnodes);
    
    int ierr = METIS_NodeND(&nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &(norder[0]), &(inorder[0]));
  }
};
#endif
