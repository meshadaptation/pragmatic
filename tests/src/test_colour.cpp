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
#include <map>

#include "zoltan_colour.h"

using namespace std;

int main(int argc, char **argv){
  
  /* 0: 3, 1
     1: 0, 2
     2: 1, 3
     3: 2, 0
  */
  zoltan_colour_graph_t graph;

  graph.rank = 0;

  /* Number of nodes in the graph assigned to the local process.
   */
  graph.npnodes = 4;
  
  /* Total number of nodes on local process.
   */
  graph.nnodes = 4;
  
  /* Array storing the number of edges connected to each node.
   */
  size_t nedges[] = {2, 2, 2, 2};
  graph.nedges = nedges;

  /* Array storing the edges in compressed row storage format.
   */
  size_t csr_edges[] = {3, 1, 0, 2, 1, 3, 2, 0};
  graph.csr_edges = csr_edges;
  
  /* Mapping from local node numbers to global node numbers.
   */
  int gid[] = {0, 1, 2, 3};
  graph.gid = gid;
  
  /* Process owner of each node.
   */
  size_t owner[] = {0, 0, 0, 0};
  graph.owner = owner;

  /* Graph colouring.
   */
  int colour[] = {0, 0, 0, 0};
  graph.colour = colour;

  zoltan_colour(&graph);
  
  if((graph.colour[0]==1)&&(graph.colour[1]==2)&&
     (graph.colour[2]==1)&&(graph.colour[3]==2))
    cout<<"pass\n";
  else{
    cout<<"Colouring = ";
    for(int i=0;i<4;i++)
      cout<<graph.colour[i]<<" ";
    cout<<endl;
    cout<<"fail\n";
  }
  return 0;
}
