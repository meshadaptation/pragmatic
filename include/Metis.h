/*  Copyright (C) 2010 Imperial College London and others.
    
    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Gerard Gorman
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    g.gorman@imperial.ac.uk
    
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/
#ifndef METIS_H
#define METIS_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <set>

extern "C" {
  // Declarations needed from METIS
  typedef int idxtype;
  void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
  void METIS_PartMeshNodal(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart);
}

/*! \brief Class provides a specialised interface to some METIS
 *   functionality.
 */
template<typename index_t>
class Metis{
 public:
  /*! Calculate a node renumbering.
   * @param graph is the undirected graph to be partitioned.
   * @param decomp is an array storing the partition each node in the graph is assigned to.
   */
  static int reorder(const std::vector< std::set<index_t> > &graph, std::vector<int> &norder){
    int nnodes = graph.size();
    
    // Compress graph
    std::vector<idxtype> xadj(nnodes+1), adjncy;
    int pos=0;
    xadj[0]=0;
    for(int i=0;i<nnodes;i++){
      for(std::set<int>::iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
        adjncy.push_back(*jt);
        pos++;
      }
      xadj[i+1] = pos;
    }
    
    norder.resize(nnodes);
    std::vector<int> inorder(nnodes);
    int numflag=0, options[] = {0};
    
    METIS_NodeND(&nnodes, &(xadj[0]), &(adjncy[0]), &numflag, options, &(norder[0]), &(inorder[0]));
    
    return 0;
  }
};
#endif
