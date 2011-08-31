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

#include "pragmatic_config.h"

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
  void colour_zoltan(std::vector< std::vector<index_t> > &graph, std::vector<index_t> &gid, std::map<index_t, int> &owner, int *colour){
    _graph = &graph;
    _gid = &gid;
    _owner = &owner;

    int ierr;
    float ver;
    
    ierr = Zoltan_Initialize(-1, NULL, &ver);
    Zoltan *zz = new Zoltan(MPI::COMM_WORLD);

    // Global ID's.
    ZOLTAN_ID_PTR global_ids = (ZOLTAN_ID_PTR) ZOLTAN_MALLOC(gid.size()*sizeof(ZOLTAN_ID_TYPE));
    for(int i=0;i<gid.size();i++)
      global_ids[i] = gid[i];

    // Delete this for production.
    zz->Set_Param("CHECK_GRAPH", "2");
    
    // Register the callbacks.
    
    /* A ZOLTAN_NUM_OBJ_FN query function returns the number of
       objects that are currently assigned to the processor. */
    zz->Set_Fn(ZOLTAN_NUM_OBJ_FN_TYPE, &Colour<index_t>::num_obj_fn, NULL);

    /* A ZOLTAN_OBJ_LIST_FN query function fills two (three if weights
       are used) arrays with information about the objects currently
       assigned to the processor. Both arrays are allocated (and
       subsequently freed) by Zoltan; their size is determined by a
       call to a ZOLTAN_NUM_OBJ_FN query function to get the array
       size. For many algorithms, either a ZOLTAN_OBJ_LIST_FN query
       function or a ZOLTAN_FIRST_OBJ_FN/ZOLTAN_NEXT_OBJ_FN
       query-function pair must be registered; however, both query
       options need not be provided. The ZOLTAN_OBJ_LIST_FN is
       preferred for efficiency. */
    zz->Set_Fn(ZOLTAN_OBJ_LIST_FN_TYPE, &Colour<index_t>::obj_list_fn, NULL);

    /* A ZOLTAN_NUM_EDGES_MULTI_FN query function returns the number
      of edges in the communication graph of the application for each
      object in a list of objects. That is, for each object in the
      global_ids/local_ids arrays, the number of objects with which
      the given object must share information is returned. */
    zz->Set_Fn(ZOLTAN_NUM_EDGES_MULTI_FN_TYPE, Colour<index_t>::num_edges_multi_fn, NULL);

    /* A ZOLTAN_EDGE_LIST_MULTI_FN query function returns lists of
    global IDs, processor IDs, and optionally edge weights for objects
    sharing edges with objects specified in the global_ids input
    array; objects share edges when they must share information with
    other objects. The arrays for the returned neighbor lists are
    allocated by Zoltan; their size is determined by a calls to
    ZOLTAN_NUM_EDGES_MULTI_FN or ZOLTAN_NUM_EDGES_FN query
    functions. */
    zz->Set_Fn(ZOLTAN_EDGE_LIST_MULTI_FN_TYPE, Colour<index_t>::edge_list_multi_fn, NULL);

    ierr = zz->Color(1, graph.size(), global_ids, colour);
    free(global_ids);
    delete zz;
  }
  
  int num_obj_fn(void* data, int* ierr){
    *ierr = ZOLTAN_OK;
    
    // data is being ignored. Using class member instead.
    // std::vector< std::vector<index_t> >* graph = (std::vector< std::vector<index_t> > *) data;
    return _graph->size(); // This should be the number of owned nodes.
  }

  void obj_list_fn(void* data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids,
                   ZOLTAN_ID_PTR local_ids, int wgt_dim, float* obj_wgts, int* ierr){
    *ierr = ZOLTAN_OK; 
    
    // data is a pointer to a Graph class.
    // std::vector<index_t>* gid = (std::vector<index_t> *)data;
    // std::cout<<"sizes = "<<gid->size()<<", "<<num_gid_entries<<", "<<num_lid_entries<<std::endl;
    
    for (int i=0;i<_gid->size();i++){
      global_ids[i] = _gid[i];
      local_ids[i] = i;
    }
  }

  void num_edges_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, int* ierr){
    *ierr = ZOLTAN_OK; 

    // data is a pointer to a Graph class.
    // std::vector< std::vector<index_t> > *graph = (std::vector< std::vector<index_t> > *) data;

    for(int i=0;i< num_obj;i++){
      num_edges[i] = (*_graph)[i].size();
    }
  }

  void edge_list_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                          ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, ZOLTAN_ID_PTR nbor_global_ids,
                          int* nbor_procs, int wgt_dim, float *ewgts, int* ierr){
    *ierr = ZOLTAN_OK;

    // data is a pointer to a Graph class.
    // std::vector< std::vector<index_t> > *graph = (std::vector< std::vector<index_t> > *) data;

    int sum = 0;
    for(int i=0; i < num_obj; i++){
      for(int j = 0; j < num_edges[i]; j++){
        index_t gid = _gid[_graph[i][j]];
        nbor_global_ids[sum] = gid;
        nbor_procs[sum] = _owner[gid];
        sum++;
      }
    }

  }
#endif
 private:
  std::vector< std::vector<index_t> > *_graph;
  std::vector<index_t> *_gid;
  std::map<index_t, int> *_owner;
};
#endif
