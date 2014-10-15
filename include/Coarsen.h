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

#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#include "DeferredOperations.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D mesh coarsening.
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

      property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                             _mesh->get_coords(n[1]),
                                             _mesh->get_coords(n[2]));

      break;
    }

    nnodes_reserve = 0;
    nthreads = pragmatic_nthreads();

    dynamic_vertex = NULL;

    for(int i=0; i<3; ++i){
      worklist[i] = NULL;
    }

    node_colour = NULL;

    for(int i=0; i<3; ++i)
      ind_set_size[i].resize(max_colour, 0);
    ind_sets.resize(nthreads, std::vector< std::vector<index_t> >(max_colour));
    range_indexer.resize(nthreads, std::vector< std::pair<size_t,size_t> >(max_colour, std::pair<size_t,size_t>(0,0)));

    def_ops = new DeferredOperations<real_t>(_mesh, nthreads, defOp_scaling_factor);
  }

  /// Default destructor.
  ~Coarsen(){
    if(property!=NULL)
      delete property;

    if(dynamic_vertex!=NULL)
      delete[] dynamic_vertex;

    if(node_colour!=NULL)
      delete[] node_colour;

    for(int i=0; i<3; ++i){
      if(worklist[i] != NULL)
        delete[] worklist[i];
    }

    delete def_ops;
  }

  /*! Perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen(real_t L_low, real_t L_max){
    size_t NNodes = _mesh->get_number_nodes();

    _L_low = L_low;
    _L_max = L_max;

    if(nnodes_reserve<NNodes){
      nnodes_reserve = NNodes;

      if(dynamic_vertex!=NULL)
        delete [] dynamic_vertex;

      dynamic_vertex = new index_t[NNodes];

      GlobalActiveSet.resize(NNodes);

      for(int i=0; i<3; ++i){
        if(worklist[i] != NULL)
          delete[] worklist[i];
        worklist[i] = new index_t[NNodes];
      }

      if(node_colour!=NULL)
        delete[] node_colour;

      node_colour = new int[NNodes];
    }

#pragma omp parallel
    {
      const int tid = pragmatic_thread_id();

      // Thread-private array of forbidden colours
      std::vector<index_t> forbiddenColours(max_colour, std::numeric_limits<index_t>::max());

      /* dynamic_vertex[i] >= 0 :: target to collapse node i
       * dynamic_vertex[i] = -1 :: node inactive (deleted/locked)
       * dynamic_vertex[i] = -2 :: recalculate collapse - this is how propagation is implemented
       */

#pragma omp single nowait
      memset(node_colour, 0, NNodes * sizeof(int));

#pragma omp single nowait
      {
        for(int i=0; i<max_colour; ++i)
          ind_set_size[0][i] = 0;
        GlobalActiveSet_size[0] = 0;
      }

      // Mark all vertices for evaluation.
#pragma omp for schedule(guided)
      for(size_t i=0; i<NNodes; ++i){
        dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);
      }

      // Variable for accessing GlobalActiveSet_size[rnd] and ind_set_size[rnd]
      int rnd = 2;

      bool first_time = true;
      do{
        // Switch to the next round
        rnd = (rnd+1)%3;

        // Prepare worklists for conflict resolution.
        // Reset GlobalActiveSet_size and ind_set_size for next (not this!) round.
#pragma omp single nowait
        {
          for(int i=0; i<3; ++i)
            worklist_size[i] = 0;

          int next_rnd = (rnd+1)%3;
          for(int i=0; i<max_colour; ++i)
            ind_set_size[next_rnd][i] = 0;
          GlobalActiveSet_size[next_rnd] = 0;
        }

        if(!first_time){
#pragma omp for schedule(guided)
          for(size_t i=0; i<NNodes; ++i){
            if(dynamic_vertex[i] == -2){
              dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);
            }
            node_colour[i] = 0;
          }
        }else
          first_time = false;

#pragma omp barrier
        // Colour the active sub-mesh
        std::vector<index_t> local_coloured;
#pragma omp for schedule(guided)
        for(size_t i=0; i<NNodes; ++i){
          if(dynamic_vertex[i]>=0){
            /*
             * Create subNNList for vertex i and also execute the first parallel
             * loop of RokosGorman colouring. This way, two time-consuming barriers,
             * the one at the end of the aforementioned loop and the one a few lines
             * below this comment, are merged into one.
             */
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[i].begin(); jt!=_mesh->NNList[i].end(); ++jt){
              if(dynamic_vertex[*jt]>=0){
                forbiddenColours[node_colour[*jt]] = (index_t) i;
              }

              for(size_t j=0; j<forbiddenColours.size(); ++j){
                if(forbiddenColours[j] != (index_t) i){
                  node_colour[i] = (int) j;
                  break;
                }
              }
            }

            local_coloured.push_back(i);
          }
        }

        if(local_coloured.size()>0){
          size_t pos;
          pos = pragmatic_omp_atomic_capture(&GlobalActiveSet_size[rnd], local_coloured.size());
          memcpy(&GlobalActiveSet[pos], &local_coloured[0], local_coloured.size() * sizeof(index_t));
        }

#pragma omp barrier
        if(GlobalActiveSet_size[rnd]>0){
          for(int set_no=0; set_no<max_colour; ++set_no){
            ind_sets[tid][set_no].clear();
            range_indexer[tid][set_no].first = std::numeric_limits<size_t>::infinity();
            range_indexer[tid][set_no].second = std::numeric_limits<size_t>::infinity();
          }

          // Continue colouring and coarsening
          std::vector<size_t> conflicts;

#pragma omp for schedule(guided)
          for(size_t i=0; i<GlobalActiveSet_size[rnd]; ++i){
            bool defective = false;
            index_t n = GlobalActiveSet[i];
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
              if(dynamic_vertex[*jt]>=0){
                if(node_colour[n] == node_colour[*jt]){
                  // No need to mark both vertices as defectively coloured.
                  // Just mark the one with the lesser ID.
                  if(n < *jt){
                    defective = true;
                    break;
                  }
                }
              }
            }

            if(defective){
              conflicts.push_back(i);

              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                if(dynamic_vertex[*jt]>=0){
                  int c = node_colour[*jt];
                  forbiddenColours[c] = n;
                }
              }

              for(size_t j=0; j<forbiddenColours.size(); j++){
                if(forbiddenColours[j] != n){
                  node_colour[n] = (int) j;
                  break;
                }
              }
            }else{
              ind_sets[tid][node_colour[n]].push_back(n);
            }
          }

          size_t pos;
          pos = pragmatic_omp_atomic_capture(&worklist_size[0], conflicts.size());

          memcpy(&worklist[0][pos], &conflicts[0], conflicts.size() * sizeof(index_t));

          conflicts.clear();
#pragma omp barrier

          int wl = 0;

          while(worklist_size[wl]){
#pragma omp for schedule(guided)
            for(size_t item=0; item<worklist_size[wl]; ++item){
              index_t n = worklist[wl][item];
              bool defective = false;
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                if(dynamic_vertex[*jt]>=0){
                  if(node_colour[n] == node_colour[*jt]){
                    // No need to mark both vertices as defectively coloured.
                    // Just mark the one with the lesser ID.
                    if(n < *jt){
                      defective = true;
                      break;
                    }
                  }
                }
              }

              if(defective){
                conflicts.push_back(n);

                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                  if(dynamic_vertex[*jt]>=0){
                    int c = node_colour[*jt];
                    forbiddenColours[c] = n;
                  }
                }

                for(size_t j=0; j<forbiddenColours.size(); j++){
                  if(forbiddenColours[j] != n){
                    node_colour[n] = j;
                    break;
                  }
                }
              }else{
                ind_sets[tid][node_colour[n]].push_back(n);
              }
            }

            // Switch worklist
            wl = (wl+1)%3;

            size_t pos = pragmatic_omp_atomic_capture(&worklist_size[wl], conflicts.size());

            memcpy(&worklist[wl][pos], &conflicts[0], conflicts.size() * sizeof(size_t));

            conflicts.clear();

            // Clear the next worklist
#pragma omp single
            {
              worklist_size[(wl+1)%3] = 0;
            }
          }

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_sets[tid][set_no].size()>0){
              range_indexer[tid][set_no].first = pragmatic_omp_atomic_capture(&ind_set_size[rnd][set_no], ind_sets[tid][set_no].size());
              range_indexer[tid][set_no].second = range_indexer[tid][set_no].first + ind_sets[tid][set_no].size();
            }
          }

#pragma omp barrier
          /* Start processing independent sets. After processing each set, colouring
           * might be invalid. More precisely, it's the target vertices whose colours
           * might clash with their neighbours' colours. To avoid hazards, we just
           * un-colour these vertices if their colour clashes with the colours of
           * their new neighbours. Being a neighbour of a removed vertex, the target
           * vertex will be marked for re-evaluation during coarsening of rm_vertex,
           * i.e. dynamic_vertex[target_target] == -2, so it will get a chance to be
           * processed at the next iteration of the do...while loop, when a new
           * colouring of the active sub-mesh will have been established. This is an
           * optimised approach compared to only processing the maximal independent
           * set and then discarding the other colours and looping all over again -
           * at least we make use of the existing colouring as much as possible.
           */

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_set_size[rnd][set_no] == 0)
              continue;

            // Sort range indexer
            std::vector<range_element> range;
            for(int t=0; t<nthreads; ++t){
              if(range_indexer[t][set_no].first != range_indexer[t][set_no].second)
                range.push_back(range_element(range_indexer[t][set_no], t));
            }
            std::sort(range.begin(), range.end(), pragmatic_range_element_comparator);


#pragma omp for schedule(guided)
            for(size_t idx=0; idx<ind_set_size[rnd][set_no]; ++idx){
              // Find which vertex corresponds to idx.
              index_t rm_vertex = -1;
              std::vector<range_element>::iterator ele = std::lower_bound(range.begin(), range.end(),
                  range_element(std::pair<size_t,size_t> (idx,idx), 0), pragmatic_range_element_finder);
              assert(ele != range.end());
              assert(idx >= range_indexer[ele->second][set_no].first && idx < range_indexer[ele->second][set_no].second);
              rm_vertex = ind_sets[ele->second][set_no][idx - range_indexer[ele->second][set_no].first];
              assert(rm_vertex>=0);

              // If the node has been un-coloured, skip it.
              if(node_colour[rm_vertex] != set_no)
                continue;

              /* If this rm_vertex is marked for re-evaluation, it means that the
               * local neighbourhood has changed since coarsen_identify_kernel was
               * called for this vertex. Call coarsen_identify_kernel again.
               * Obviously, this check is redundant for set_no=0.
               */

              if(dynamic_vertex[rm_vertex] == -2)
                dynamic_vertex[rm_vertex] = coarsen_identify_kernel(rm_vertex, L_low, L_max);

              if(dynamic_vertex[rm_vertex] < 0)
                continue;

              index_t target_vertex = dynamic_vertex[rm_vertex];

              // Mark neighbours for re-evaluation.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt)
                def_ops->propagate_coarsening(*jt, tid);

              // Un-colour target_vertex if its colour clashes with any of its new neighbours.
              if(node_colour[target_vertex] >= 0){
                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  if(*jt != target_vertex){
                    if(node_colour[*jt] == node_colour[target_vertex]){
                      def_ops->reset_colour(target_vertex, tid);
                      break;
                    }
                  }
                }
              }

              // Mark rm_vertex as inactive.
              dynamic_vertex[rm_vertex] = -1;

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }

#pragma omp for schedule(guided)
            for(size_t vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
              for(int i=0; i<nthreads; ++i){
                def_ops->commit_remNN(i, vtid);
                def_ops->commit_addNN(i, vtid);
                def_ops->commit_remNE(i, vtid);
                def_ops->commit_addNE(i, vtid);
                def_ops->commit_repEN(i, vtid);
                def_ops->commit_coarsening_propagation(dynamic_vertex, i, vtid);
                def_ops->commit_colour_reset(node_colour, i, vtid);
              }
            }
          }
        }
      }while(GlobalActiveSet_size[rnd]>0);
    }
  }

 private:

  /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
   */
  int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const{
    // Cannot delete if already deleted.
    if(_mesh->NNList[rm_vertex].empty())
      return -1;

    // If this is not owned then return -1.
    if(_mesh->is_owned_node(rm_vertex))
      return -1;

    /* Sort the edges according to length. We want to collapse the
       shortest. If it is not possible to collapse the edge then move
       onto the next shortest.*/
    std::multimap<real_t, index_t> short_edges;
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      double length = _mesh->calc_edge_length(rm_vertex, *nn);
      if(length<L_low)
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

      // Check volume/area of new elements.
      double total_old_av=0;
      double total_new_av=0;
      for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
        const int *old_n=_mesh->get_element(*ee);

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
        else
          new_av = property->volume(_mesh->get_coords(n[0]),
                                    _mesh->get_coords(n[1]),
                                    _mesh->get_coords(n[2]),
                                    _mesh->get_coords(n[3]));

        total_new_av+=new_av;

        // Reject inverted elements.
        if(new_av<DBL_EPSILON){
          reject_collapse=true;
          break;
        }
      }

      if(fabs(total_new_av-total_old_av)>DBL_EPSILON)
        reject_collapse=true;

      // Check if any of the new edges are longer than L_max.
      if(!reject_collapse){
        for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
          if(target_vertex==*nn)
            continue;

          if(_mesh->calc_edge_length(target_vertex, *nn)>L_max){
            reject_collapse=true;
            break;
          }
        }
      }

      // If this edge is ok to collapse then jump out.
      if(!reject_collapse)
        break;
    }

    // If we've checked all edges and none is collapsible then return.
    if(reject_collapse)
      return -2;

    return target_vertex;
  }

  /*! Kernel for perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen_kernel(index_t rm_vertex, index_t target_vertex, size_t tid){
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
      index_t other_vertex;
      for(size_t i=0; i<nloc; ++i){
        index_t vid = _mesh->_ENList[eid*nloc+i];
        if(vid==rm_vertex){
          lrm_vertex = i;
        }else{
          def_ops->remNE(vid, eid, tid);

          // If this vertex is neither rm_vertex nor target_vertex, it is one of the common neighbours.
          if(vid != target_vertex){
            other_vertex = vid;
            common_patch.insert(vid);
          }
        }
      }

      // Handle vertex collapsing onto boundary.
      if(_mesh->boundary[eid*nloc+lrm_vertex]>0){
        // Find element whose internal edge will be pulled into an external edge.
        std::set<index_t> new_boundary_eid;
        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
            _mesh->NEList[other_vertex].begin(), _mesh->NEList[other_vertex].end(),
            std::inserter(new_boundary_eid, new_boundary_eid.begin()));

        if(!new_boundary_eid.empty()){
          assert(new_boundary_eid.size()==1);
          index_t target_eid = *new_boundary_eid.begin();
          for(int i=0;i<3;i++){
            int nid=_mesh->_ENList[target_eid*nloc+i];
            if(nid!=rm_vertex && nid!=other_vertex){
              _mesh->boundary[target_eid*nloc+i] = _mesh->boundary[eid*nloc+lrm_vertex];
              break;
            }
          }
        }
      }

      // Remove element from mesh.
      _mesh->_ENList[eid*nloc] = -1;
    }

    assert(common_patch.size() == deleted_elements.size());

    // For all adjacent elements, replace rm_vertex with target_vertex in ENList.
    for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
      for(size_t i=0;i<nloc;i++){
        if(_mesh->_ENList[nloc*(*ee)+i]==rm_vertex){
          def_ops->repEN(nloc*(*ee)+i, target_vertex, tid);
          break;
        }
      }

      // Add element to target_vertex's NEList.
      def_ops->addNE(target_vertex, *ee, tid);
    }

    // Update surrounding NNList.
    common_patch.insert(target_vertex);
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      def_ops->remNN(*nn, rm_vertex, tid);

      // Find all entries pointing back to rm_vertex and update them to target_vertex.
      if(common_patch.count(*nn)==0){
        def_ops->addNN(*nn, target_vertex, tid);
        def_ops->addNN(target_vertex, *nn, tid);
      }
    }

    _mesh->erase_vertex(rm_vertex);
  }

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  size_t nnodes_reserve;
  index_t *dynamic_vertex;

  // Colouring
  int *node_colour;
  static const int max_colour = (dim==2?16:64);
  std::vector<index_t> GlobalActiveSet;
  size_t GlobalActiveSet_size[3];
  std::vector<size_t> ind_set_size[3];
  std::vector< std::vector< std::vector<index_t> > > ind_sets;
  std::vector< std::vector< std::pair<size_t,size_t> > > range_indexer;

  // Used for iterative colouring
  index_t* worklist[3];
  size_t worklist_size[3];

  DeferredOperations<real_t>* def_ops;
  static const int defOp_scaling_factor = 4;

  real_t _L_low, _L_max;

  const static size_t ndims=dim;
  const static size_t nloc=dim+1;
  const static size_t msize=(dim==2?3:6);

  int nthreads;
};

#endif
