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

#include "partition.h"

using namespace std;
  
int partition(const vector< set<idxtype> > &graph, int npartitions, int partition_method, vector<int> &decomp){
  // If no partitioning method is set, choose a default.
  if(partition_method<0){
    if(npartitions<=8)
      partition_method = 0; // METIS PartGraphRecursive
    else
      partition_method = 1; // METIS PartGraphKway
  }
  
  int nnodes = graph.size();
  
  // Compress graph    
  vector<idxtype> xadj(nnodes+1), adjncy;
  int pos=0;
  xadj[0]=0;
  for(int i=0;i<nnodes;i++){
    for(set<int>::iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
      adjncy.push_back(*jt);
      pos++;
    }
    xadj[i+1] = pos;
  }
  
  // Partition graph
  decomp.resize(nnodes);
  int wgtflag=0, numflag=0, options[] = {0}, edgecut=0;
  
  if(partition_method){
    METIS_PartGraphKway(&nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &wgtflag, 
                        &numflag, &npartitions, options, &edgecut, &(decomp[0]));
  }else{
    METIS_PartGraphRecursive(&nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &wgtflag, 
                             &numflag, &npartitions, options, &edgecut, &(decomp[0]));
  }
  
  return edgecut;
}

int reorder(const vector< set<idxtype> > &graph, vector<int> &norder){
  int nnodes = graph.size();
  
  // Compress graph
  vector<idxtype> xadj(nnodes+1), adjncy;
  int pos=0;
  xadj[0]=0;
  for(int i=0;i<nnodes;i++){
    for(set<int>::iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
      adjncy.push_back(*jt);
      pos++;
    }
    xadj[i+1] = pos;
  }
  
  // Partition graph
  norder.resize(nnodes);
  vector<int> inorder(nnodes);
  int numflag=0, options[] = {0};
  
  METIS_NodeND(&nnodes, &(xadj[0]), &(adjncy[0]), &numflag, options, &(norder[0]), &(inorder[0]));
  
  return 0;
}
