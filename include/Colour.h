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

#ifndef COLOUR_H
#define COLOUR_H

#include <map>
#include <set>
#include <vector>
#include "confdefs.h"
#ifdef HAVE_ZOLTAN
#include "zoltan_cpp.h"
#endif

/*! \brief Class contains various methods for colouring undirected graphs.
 */
template<typename index_t>
class Colour{
 public:
  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(std::vector< std::deque<index_t> > &NNList, index_t *colour){
    size_t NNodes = NNList.size();

    // Colour first active node.
    size_t node;
    for(node=0;node<NNodes;node++){
      if(NNList[node].size()>0){
        colour[node] = 0;
        break;
      }
    }
    
    // Colour remaining active nodes.
    for(;node<NNodes;node++){
      if(NNList[node].size()==0)
        continue;
      
      std::set<index_t> used_colours;
      for(typename std::deque<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it)
        if(*it<(int)node)
          used_colours.insert(colour[*it]);
      
      for(index_t i=0;;i++)
        if(used_colours.count(i)==0){
          colour[node] = i;
          break;
        }
    }
  }

  /*! This routine colours a undirected graph using the greedy colouring algorithm.
   * @param NNList Node-Node-adjancy-List, i.e. the undirected graph to be coloured.
   * @param active indicates which nodes are turned on in the graph.
   * @param colour array that the node colouring is copied into.
   */
  static void greedy(std::vector< std::deque<index_t> > &NNList, std::vector<bool> &active, index_t *colour){
    size_t NNodes = NNList.size();
    
    // Colour first active node.
    size_t node;
    for(node=0;node<NNodes;node++){
      if(active[node]){
        colour[node] = 0;
        break;
      }
    }
    
    // Colour remaining active nodes.
    for(;node<NNodes;node++){
      if(!active[node])
        continue;
      
      std::set<index_t> used_colours;
      for(typename std::deque<index_t>::const_iterator it=NNList[node].begin();it!=NNList[node].end();++it)
        if(*it<(int)node)
          used_colours.insert(colour[*it]);
      
      for(index_t i=0;;i++)
        if(used_colours.count(i)==0){
          colour[node] = i;
          break;
        }
    }
  }

#ifdef HAVE_ZOLTAN
  static void colour_zoltan(std::vector< std::set<index_t> > &NNList, ZOLTAN_ID_PTR global_ids, int nprivate_nodes, int *colour){
    int ierr;
    float ver;

    ierr = Zoltan_Initialize(-1, NULL, &ver);
    Zoltan *zz = new Zoltan(MPI::COMM_WORLD);

    // Delete this for production.
    zz->Set_Param("CHECK_GRAPH", "2");

    // Register the callbacks.
    zz->Set_Fn(ZOLTAN_NUM_OBJ_FN_TYPE, (void) (*)() num_obj_fn, (void*) &graph);
    zz->Set_Fn(ZOLTAN_OBJ_LIST_FN_TYPE, (void) (*)() obj_list_fn, (void*) &graph);
    zz->Set_Fn(ZOLTAN_NUM_EDGES_MULTI_FN_TYPE, (void) (*)() num_edges_multi_fn, (void*) &graph);
    zz->Set_Fn(ZOLTAN_EDGE_LIST_MULTI_FN_TYPE, (void) (*)() edge_list_multi_fn, (void*) &graph);

    ierr = zz->Color(1, NNList.size(), global_ids, colour);
    delete zz;
  }

  static int num_obj_fn(void* data, int* ierr)
  {
    *ierr = ZOLTAN_OK;
    // data is a pointer to a Graph class.
    Graph* graph = (Graph*) data;
    return graph->get_nowned_nodes();
  }

  static void obj_list_fn(void* data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids,
                          ZOLTAN_ID_PTR local_ids, int wgt_dim, float* obj_wgts, int* ierr)
  {
    *ierr = ZOLTAN_OK; 
    // data is a pointer to a Graph class.
    Graph* graph = (Graph*) data;

    for (int i = 0; i < graph->get_nowned_nodes(); i++)
    {
      global_ids[i] = graph->get_global_node_number(i);
      local_ids[i] = i;
    }
  }

  static void num_edges_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                                 ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, int* ierr)
  {
    *ierr = ZOLTAN_OK; 
    // data is a pointer to a Graph class.
    Graph* graph = (Graph*) data;

    for (int i = 0; i < num_obj; i++)
    {
      num_edges[i] = graph->get_num_edges(i);
    }
  }

  static void edge_list_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                                 ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, ZOLTAN_ID_PTR nbor_global_ids,
                                 int* nbor_procs, int wgt_dim, float *ewgts, int* ierr)
  {
    *ierr = ZOLTAN_OK;
    // data is a pointer to a Graph class.
    Graph* graph = (Graph*) data;

    int sum = 0;
    for (int i = 0; i < num_obj; i++)
    {
      for (int j = 0; j < num_edges[i]; j++)
      {
        nbor_global_ids[sum] = graph->get_global_number(i, j);
        nbor_procs[sum] = graph->get_owner(i, j);
        sum++;
      }
    }

  }
#endif
 private:
};
#endif
