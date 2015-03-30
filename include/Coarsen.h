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

#ifndef COARSEN_H
#define COARSEN_H

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#include "ElementProperty.h"
#include "Lock.h"
#include "Mesh.h"
#include "Worklist.h"

/*! \brief Performs 2D/3D mesh coarsening.
 *
 */

template<typename real_t, int dim> class Coarsen{
 public:
  /// Default constructor.
  Coarsen(Mesh<real_t> &mesh){
    _mesh = &mesh;

    property = NULL;
    size_t NElements = _mesh->get_number_elements();
    for(size_t i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      if(dim==2)
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]));
      else
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]),
                                               _mesh->get_coords(n[3]));

      break;
    }

    nnodes_reserve = 0;
    delete_slivers = false;

    nthreads = omp_get_max_threads();
    current_worklist.resize(nthreads);
    current_worklist.resize(nthreads);
  }

  /// Default destructor.
  ~Coarsen(){
    if(property!=NULL)
      delete property;
  }

  /*! Perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen(real_t L_low, real_t L_max, bool enable_sliver_deletion=false){
    size_t NNodes = _mesh->get_number_nodes();

    _L_low = L_low;
    _L_max = L_max;
    delete_slivers = enable_sliver_deletion;

    if(nnodes_reserve<NNodes){
      nnodes_reserve = NNodes;

      dynamic_vertex.resize(NNodes, -1);

      vLocks.resize(NNodes);
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      // Vector "retry" is used to store aborted vertices.
      // Vector "round" is used to store propagated vertices.
      std::vector<index_t> retry, next_retry;
      std::vector<index_t> next_round;
      std::vector<index_t> locks_held;
#pragma omp for schedule(static) nowait
      for(index_t node=0; node<NNodes; ++node){
        bool abort = false;

        if(!vLocks[node].try_lock()){
          retry.push_back(node);
          continue;
        }
        locks_held.push_back(node);

        for(auto& it : _mesh->NNList[node]){
          if(!vLocks[it].try_lock()){
            abort = true;
            break;
          }
          locks_held.push_back(it);
        }

        if(!abort){
          index_t target = coarsen_identify_kernel(node, L_low, L_max);
          if(target>=0){
            for(auto& it : _mesh->NNList[node]){
              next_round.push_back(it);
              dynamic_vertex[it] = -2;
            }
            coarsen_kernel(node, target);
          }
          dynamic_vertex[node] = -1;
        }
        else
          retry.push_back(node);

        for(auto& it : locks_held){
          vLocks[it].unlock();
        }
        locks_held.clear();
      }

      while(retry.size()>0){
        next_retry.clear();

        for(auto& node : retry){
          if(dynamic_vertex[node] == -1)
            continue;

          bool abort = false;

          if(!vLocks[node].try_lock()){
            next_retry.push_back(node);
            continue;
          }
          locks_held.push_back(node);

          for(auto& it : _mesh->NNList[node]){
            if(!vLocks[it].try_lock()){
              abort = true;
              break;
            }
            locks_held.push_back(it);
          }

          if(!abort){
            index_t target = coarsen_identify_kernel(node, L_low, L_max);
            if(target>=0){
              for(auto& it : _mesh->NNList[node]){
                next_round.push_back(it);
                dynamic_vertex[it] = -2;
              }
              coarsen_kernel(node, target);
            }
            dynamic_vertex[node] = -1;
          }
          else
            next_retry.push_back(node);

          for(auto& it : locks_held){
            vLocks[it].unlock();
          }
          locks_held.clear();
        }

        retry.swap(next_retry);
      }

      while(!next_round.empty()){
        current_worklist[tid].replace(next_round);
        next_round.clear();

        current_worklist[tid].init_traversal();
        while(current_worklist[tid].is_valid()){
          index_t node = current_worklist[tid].get_next();
          if(dynamic_vertex[node] == -1)
            continue;

          bool abort = false;

          if(!vLocks[node].try_lock()){
            retry.push_back(node);
            continue;
          }
          locks_held.push_back(node);

          for(auto& it : _mesh->NNList[node]){
            if(!vLocks[it].try_lock()){
              abort = true;
              break;
            }
            locks_held.push_back(it);
          }

          if(!abort){
            index_t target = coarsen_identify_kernel(node, L_low, L_max);
            if(target>=0){
              for(auto& it : _mesh->NNList[node]){
                next_round.push_back(it);
                dynamic_vertex[it] = -2;
              }
              coarsen_kernel(node, target);
            }
            dynamic_vertex[node] = -1;
          }
          else
            retry.push_back(node);

          for(auto& it : locks_held){
            vLocks[it].unlock();
          }
          locks_held.clear();
        }

        while(retry.size()>0){
          next_retry.clear();

          for(auto& node : retry){
            if(dynamic_vertex[node] == -1)
              continue;

            bool abort = false;

            if(!vLocks[node].try_lock()){
              next_retry.push_back(node);
              continue;
            }
            locks_held.push_back(node);

            for(auto& it : _mesh->NNList[node]){
              if(!vLocks[it].try_lock()){
                abort = true;
                break;
              }
              locks_held.push_back(it);
            }

            if(!abort){
              index_t target = coarsen_identify_kernel(node, L_low, L_max);
              if(target>=0){
                for(auto& it : _mesh->NNList[node]){
                  next_round.push_back(it);
                  dynamic_vertex[it] = -2;
                }
                coarsen_kernel(node, target);
              }
              dynamic_vertex[node] = -1;
            }
            else
              next_retry.push_back(node);

            for(auto& it : locks_held){
              vLocks[it].unlock();
            }
            locks_held.clear();
          }

          retry.swap(next_retry);
        }

        if(next_round.empty()){
          // Try to steal work
          for(int t=(tid+1)%nthreads; t!=tid; t=(t+1)%nthreads){
            if(current_worklist[t].steal_work(next_round))
              break;
          }
        }
      }
    }
  }

 private:

  /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
   */
  inline int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const{
    // Cannot delete if already deleted.
    if(_mesh->NNList[rm_vertex].empty())
      return -1;

    // If this is not owned then return -1.
    // if(!_mesh->is_owned_node(rm_vertex))
    //   return -1;

    // For now, lock the halo
    if(_mesh->is_halo_node(rm_vertex))
      return -1;

    //
    bool delete_with_extreme_prejudice = false;
    if(delete_slivers && dim==3){
      std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();
      double q_linf = _mesh->quality[*ee];
      ++ee;

      for(;ee!=_mesh->NEList[rm_vertex].end();++ee)
        q_linf = std::min(q_linf, _mesh->quality[*ee]);

      if(q_linf<1.0e-6)
        delete_with_extreme_prejudice = true;
    }

    /* Sort the edges according to length. We want to collapse the
       shortest. If it is not possible to collapse the edge then move
       onto the next shortest.*/
    std::multimap<real_t, index_t> short_edges;
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      double length = _mesh->calc_edge_length(rm_vertex, *nn);
      if(length<L_low || delete_with_extreme_prejudice)
        short_edges.insert(std::pair<real_t, index_t>(length, *nn));
    }

    bool reject_collapse = false;
    index_t target_vertex=-1;
    while(short_edges.size()){
      // Get the next shortest edge.
      target_vertex = short_edges.begin()->second;
      short_edges.erase(short_edges.begin());

      // Assume the best.
      reject_collapse=false;

      /* Check the properties of new elements. If the
         new properties are not acceptable then continue. */

      // Check area/volume of new elements.
      long double total_old_av=0;
      long double total_new_av=0;
      bool better=true;
      for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
        const int *old_n=_mesh->get_element(*ee);

        double q_linf = _mesh->quality[*ee];
        double old_av;
        if(dim==2)
          old_av = property->area(_mesh->get_coords(old_n[0]),
                                  _mesh->get_coords(old_n[1]),
                                  _mesh->get_coords(old_n[2]));
        else
          old_av = property->volume(_mesh->get_coords(old_n[0]),
                                    _mesh->get_coords(old_n[1]),
                                    _mesh->get_coords(old_n[2]),
                                    _mesh->get_coords(old_n[3]));

        total_old_av+=old_av;

        // Skip if this element would be deleted under the operation.
        if(_mesh->NEList[target_vertex].find(*ee)!=_mesh->NEList[target_vertex].end())
          continue;

        // Create a copy of the proposed element
        std::vector<int> n(nloc);
        for(size_t i=0;i<nloc;i++){
          int nid = old_n[i];
          if(nid==rm_vertex)
            n[i] = target_vertex;
          else
            n[i] = nid;
        }

        // Check the area/volume of this new element.
        double new_av;
        if(dim==2)
          new_av = property->area(_mesh->get_coords(n[0]),
                                  _mesh->get_coords(n[1]),
                                  _mesh->get_coords(n[2]));
        else{
          new_av = property->volume(_mesh->get_coords(n[0]),
                                    _mesh->get_coords(n[1]),
                                    _mesh->get_coords(n[2]),
                                    _mesh->get_coords(n[3]));
          double new_q = property->lipnikov(_mesh->get_coords(n[0]),
                                            _mesh->get_coords(n[1]),
                                            _mesh->get_coords(n[2]),
                                            _mesh->get_coords(n[3]),
                                            _mesh->get_metric(n[0]),
                                            _mesh->get_metric(n[1]),
                                            _mesh->get_metric(n[2]),
                                            _mesh->get_metric(n[3]));
          if(new_q<q_linf)
            better=false;
        }
        total_new_av+=new_av;

        // Reject inverted elements.
        if(new_av<DBL_EPSILON){
          reject_collapse=true;
          break;
        }
      }

      // Check we are not removing surface features.
      if(!reject_collapse && fabs(total_new_av-total_old_av)>DBL_EPSILON){
        reject_collapse=true;
      }

      /*
      // Check if any of the new edges are longer than L_max.
      if(!reject_collapse && !delete_with_extreme_prejudice){
        for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
          if(target_vertex==*nn)
            continue;

          if(_mesh->calc_edge_length(target_vertex, *nn)>L_max){
            reject_collapse=true;
            break;
          }
        }
      }
      */
      if(!better)
        reject_collapse=true;

      // If this edge is ok to collapse then jump out.
      if(!reject_collapse)
        break;
    }

    // If we've checked all edges and none are collapsible then return.
    if(reject_collapse)
      return -2;

    if(delete_with_extreme_prejudice)
      std::cerr<<"-";

    return target_vertex;
  }

  /*! Kernel for performing coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  inline void coarsen_kernel(index_t rm_vertex, index_t target_vertex){
    std::set<index_t> deleted_elements;
    std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                     _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                     std::inserter(deleted_elements, deleted_elements.begin()));

    // This is the set of vertices which are common neighbours between rm_vertex and target_vertex.
    std::set<index_t> common_patch;

    // Remove deleted elements from node-element adjacency list and from element-node list.
    for(typename std::set<index_t>::const_iterator de=deleted_elements.begin(); de!=deleted_elements.end();++de){
      index_t eid = *de;

      // Remove element from NEList[rm_vertex].
      _mesh->NEList[rm_vertex].erase(eid);

      // Remove element from NEList of the other two vertices.
      size_t lrm_vertex;
      std::vector<index_t> other_vertex;
      for(size_t i=0; i<nloc; ++i){
        index_t vid = _mesh->_ENList[eid*nloc+i];
        if(vid==rm_vertex){
          lrm_vertex = i;
        }else{
          _mesh->NEList[vid].erase(eid);

          // If this vertex is neither rm_vertex nor target_vertex, it is one of the common neighbours.
          if(vid != target_vertex){
            other_vertex.push_back(vid);
            common_patch.insert(vid);
          }
        }
      }

      // Handle vertex collapsing onto boundary.
      if(_mesh->boundary[eid*nloc+lrm_vertex]>0){
        // Find element whose internal edge will be pulled into an external edge.
        std::set<index_t> otherNE;
        if(dim==2){
          assert(other_vertex.size()==1);
          otherNE = _mesh->NEList[other_vertex[0]];
        }else{
          assert(other_vertex.size()==2);
          std::set_intersection(_mesh->NEList[other_vertex[0]].begin(), _mesh->NEList[other_vertex[0]].end(),
              _mesh->NEList[other_vertex[1]].begin(), _mesh->NEList[other_vertex[1]].end(),
              std::inserter(otherNE, otherNE.begin()));
        }
        std::set<index_t> new_boundary_eid;
        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
            otherNE.begin(), otherNE.end(), std::inserter(new_boundary_eid, new_boundary_eid.begin()));

        if(!new_boundary_eid.empty()){
          // eid has been removed from NEList[rm_vertex],
          // so new_boundary_eid contains only the other element.
          assert(new_boundary_eid.size()==1);
          index_t target_eid = *new_boundary_eid.begin();
          for(int i=0;i<nloc;i++){
            int nid=_mesh->_ENList[target_eid*nloc+i];
            if(dim==2){
              if(nid!=rm_vertex && nid!=other_vertex[0]){
                _mesh->boundary[target_eid*nloc+i] = _mesh->boundary[eid*nloc+lrm_vertex];
                break;
              }
            }else{
              if(nid!=rm_vertex && nid!=other_vertex[0] && nid!=other_vertex[1]){
                _mesh->boundary[target_eid*nloc+i] = _mesh->boundary[eid*nloc+lrm_vertex];
                break;
              }
            }
          }
        }
      }

      // Remove element from mesh.
      _mesh->_ENList[eid*nloc] = -1;
    }

    assert((dim==2 && common_patch.size() == deleted_elements.size()) || (dim==3));

    // For all adjacent elements, replace rm_vertex with target_vertex in ENList and update quality.
    for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
      for(size_t i=0;i<nloc;i++){
        if(_mesh->_ENList[nloc*(*ee)+i]==rm_vertex){
          _mesh->_ENList[nloc*(*ee)+i] = target_vertex;
          break;
        }
      }

      _mesh->template update_quality<dim>(*ee);

      // Add element to target_vertex's NEList.
      _mesh->NEList[target_vertex].insert(*ee);
    }

    // Update surrounding NNList.
    common_patch.insert(target_vertex);
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      typename std::vector<index_t>::iterator it = std::find(_mesh->NNList[*nn].begin(), _mesh->NNList[*nn].end(), rm_vertex);
      _mesh->NNList[*nn].erase(it);

      // Find all entries pointing back to rm_vertex and update them to target_vertex.
      if(common_patch.count(*nn)==0){
        _mesh->NNList[*nn].push_back(target_vertex);
        _mesh->NNList[target_vertex].push_back(*nn);
      }
    }

    _mesh->erase_vertex(rm_vertex);
  }

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  size_t nnodes_reserve;
  std::vector<index_t> dynamic_vertex;
  std::vector<Lock> vLocks;
  std::vector< Worklist<index_t> > current_worklist, next_worklist;

  real_t _L_low, _L_max;
  bool delete_slivers;

  const static size_t ndims=dim;
  const static size_t nloc=dim+1;
  const static size_t msize=(dim==2?3:6);

  int nthreads;
};

#endif
