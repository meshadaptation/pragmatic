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

#ifndef COARSEN2D_H
#define COARSEN2D_H

#include <algorithm>
#include <set>
#include <vector>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#include "AdaptiveAlgorithm.h"
#include "Colouring.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D mesh coarsening.
 *
 */

template<typename real_t> class Coarsen2D : public AdaptiveAlgorithm<real_t>{
 public:
  /// Default constructor.
  Coarsen2D(Mesh<real_t> &mesh, Surface2D<real_t> &surface){
    _mesh = &mesh;
    _surface = &surface;
    
    nprocs = pragmatic_nprocesses(_mesh->get_mpi_comm());    
    
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
    dynamic_vertex = NULL;
    colouring = NULL;
  }
  
  /// Default destructor.
  virtual ~Coarsen2D(){
    if(property!=NULL)
      delete property;

    if(dynamic_vertex!=NULL)
      delete[] dynamic_vertex;

    if(colouring!=NULL)
      delete colouring;
  }

  /*! Perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen(real_t L_low, real_t L_max){
    size_t NNodes= _mesh->get_number_nodes();

    _L_low = L_low;
    _L_max = L_max;

    if(nnodes_reserve<1.5*NNodes){
      nnodes_reserve = 2*NNodes;
      
      if(dynamic_vertex!=NULL){
        delete [] dynamic_vertex;
      }
      
      dynamic_vertex = new index_t[nnodes_reserve];

      if(colouring==NULL)
        colouring = new Colouring<real_t>(_mesh, this, nnodes_reserve);
      else
        colouring->resize(nnodes_reserve);
    }

#pragma omp parallel
    {
      const int tid = pragmatic_thread_id();

      /* dynamic_vertex[i] >= 0 :: target to collapse node i
       * dynamic_vertex[i] = -1 :: node inactive (deleted/locked)
       * dynamic_vertex[i] = -2 :: recalculate collapse - this is how propagation is implemented
       */

      // Mark all vertices for evaluation.
      if(nprocs==1){
#pragma omp for schedule(static)
        for(size_t i=0;i<NNodes;i++){
          dynamic_vertex[i] = -2;
          colouring->node_colour[i] = -1;
        }
      }else{
#pragma omp for schedule(static, 32)
        for(size_t i=0;i<NNodes;i++){
          if(_mesh->is_halo_node(i))
            dynamic_vertex[i] = -1; // lock halo
          else
            dynamic_vertex[i] = -2;
          colouring->node_colour[i] = -1;
        }
      }

      do{
        /* Create the active sub-mesh. This is the mesh consisting of all dynamic
         * vertices and all edges connecting two dynamic vertices, i.e. sub_NNList[i]
         * of active sub-mesh contains all vertices of _mesh->NNList[i] which are
         * dynamic. The goal is to prevent two adjacent vertices from collapsing
         * at the same time, therefore avoiding structural hazards. A nice side-effect
         * is that we also enforce the "every other vertex" rule. Safe parallel
         * updating of _mesh adjacency lists can be achieved later using the deferred
         * updates mechanism.
         */

        // Start by finding which vertices comprise the active sub-mesh.
        std::vector<index_t> active_set;

        /* Initialise list of vertices to be coarsened. A dynamic schedule is used as
         * previous coarsening may have introduced significant gaps in the node list.
         * This could lead to significant load imbalance if a static schedule was used.
         */
#pragma omp for schedule(dynamic, 8) nowait
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(dynamic_vertex[i] == -2){
            dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);
	    
            if(dynamic_vertex[i]>=0){
              active_set.push_back(i);
            }
          }
        }

        size_t pos;
        pragmatic_omp_atomic_capture()
        {
          pos = colouring->GlobalActiveSet_size;
          colouring->GlobalActiveSet_size += active_set.size();
        }

        if(active_set.size()>0)
          memcpy(&colouring->GlobalActiveSet[pos], &active_set[0], active_set.size() * sizeof(index_t));

#pragma omp barrier

        if(colouring->GlobalActiveSet_size == 0)
          break;
	  
        colouring->multiHashJonesPlassmann();

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
	  
        for(int set_no=0; set_no<colouring->nsets; ++set_no){
          if(((double) colouring->ind_set_size[set_no]/colouring->GlobalActiveSet_size < 0.1))
            continue;

#pragma omp for schedule(dynamic)
          for(size_t i=0; i<colouring->ind_set_size[set_no]; ++i){
            index_t rm_vertex = colouring->independent_sets[set_no][i];
            assert((size_t) rm_vertex < NNodes);
	      
            // If the node has been un-coloured, skip it.
            if(colouring->node_colour[rm_vertex] < 0)
              continue;
	      
            assert(colouring->node_colour[rm_vertex] == set_no);
            colouring->node_colour[rm_vertex] = -1;

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
              _mesh->deferred_propagate_coarsening(*jt, tid);

            // Un-colour target_vertex if its colour clashes with any of its new neighbours.
            if(colouring->node_colour[target_vertex] >= 0){
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                if(*jt != target_vertex){
                  if(colouring->node_colour[*jt] == colouring->node_colour[target_vertex]){
                    _mesh->deferred_reset_colour(target_vertex, tid);
                    break;
                  }
                }
              }
            }

            // Mark rm_vertex as non-active.
            dynamic_vertex[rm_vertex] = -1;

            // Coarsen the edge.
            coarsen_kernel(rm_vertex, target_vertex, tid);
          }

          _mesh->commit_deferred(tid);
          _mesh->commit_coarsening_propagation(dynamic_vertex, tid);
          _mesh->commit_colour_reset(colouring->node_colour, tid);
          _surface->commit_deferred(tid);
#pragma omp barrier
        }

        colouring->destroy();
#pragma omp barrier
      }while(true);
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
    if(_mesh->is_halo_node(rm_vertex))
      return -1;

    /* Sort the edges according to length. We want to collapse the
       shortest. If it is not possible to collapse the edge then move
       onto the next shortest.*/
    std::multimap<real_t, index_t> short_edges;
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      // First check if this edge can be collapsed
      if(!_surface->is_collapsible(rm_vertex, *nn))
        continue;

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
      for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
        if(_mesh->NEList[target_vertex].find(*ee)!=_mesh->NEList[target_vertex].end())
          continue;

        // Create a copy of the proposed element
        std::vector<int> n(nloc);
        const int *orig_n=_mesh->get_element(*ee);
        for(size_t i=0;i<nloc;i++){
          int nid = orig_n[i];
          if(nid==rm_vertex)
            n[i] = target_vertex;
          else
            n[i] = nid;
        }

        // Check the area of this new element.
        double orig_area = property->area(_mesh->get_coords(orig_n[0]),
                                          _mesh->get_coords(orig_n[1]),
                                          _mesh->get_coords(orig_n[2]));

        double area = property->area(_mesh->get_coords(n[0]),
                                     _mesh->get_coords(n[1]),
                                     _mesh->get_coords(n[2]));

        // Not very satisfactory - requires more thought.
        if(area/orig_area<=1.0e-3){
          reject_collapse=true;
          break;
        }
      }

      // Check of any of the new edges are longer than L_max.
      for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
        if(target_vertex==*nn)
          continue;

        if(_mesh->calc_edge_length(target_vertex, *nn)>L_max){
          reject_collapse=true;
          break;
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
   * Returns the node ID that rm_vertex collapses onto, negative if the operation is not performed.
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
      for(size_t i=0; i<nloc; ++i){
        index_t vid = _mesh->_ENList[eid*nloc+i];
        if(vid != rm_vertex){
          _mesh->deferred_remNE(vid, eid, tid);

          // If this vertex is neither rm_vertex nor target_vertex, it is one of the 2 common neighbours.
          if(vid != target_vertex)
            common_patch.insert(vid);
        }
      }

      // Remove element from mesh.
      _mesh->_ENList[eid*nloc] = -1;
    }

    assert(common_patch.size() == deleted_elements.size());
    assert(common_patch.size() == 2 || (common_patch.size()==1 && _surface->contains_node(rm_vertex)));

    // For all adjacent elements, replace rm_vertex with target_vertex in ENList.
    for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
      for(size_t i=0;i<nloc;i++){
        if(_mesh->_ENList[nloc*(*ee)+i]==rm_vertex){
          _mesh->_ENList[nloc*(*ee)+i]=target_vertex;
          break;
        }
      }
      
      // Add element to target_vertex's NEList.
      _mesh->deferred_addNE(target_vertex, *ee, tid);
    }
    
    // Update surrounding NNList.
    std::set<index_t> additional_patch;
    for(typename std::vector<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
      if(*nn==target_vertex)
        continue;

      // Find all entries pointing back to rm_vertex and update them to target_vertex.
      if(common_patch.count(*nn))
        _mesh->deferred_remNN(*nn, rm_vertex, tid);
      else{
        typename std::vector<index_t>::iterator back_reference = std::find(_mesh->NNList[*nn].begin(), _mesh->NNList[*nn].end(), rm_vertex);
        assert(back_reference!=_mesh->NNList[*nn].end());
        *back_reference = target_vertex;
        additional_patch.insert(*nn);
      }
    }

    // Update NNList for target_vertex
    _mesh->deferred_remNN(target_vertex, rm_vertex, tid);
    for(typename std::set<index_t>::const_iterator it=additional_patch.begin();it!=additional_patch.end();++it){
      _mesh->deferred_addNN(target_vertex, *it, tid);
    }
    
    // Perform coarsening on surface if necessary.
    if(_surface->contains_node(rm_vertex)){
      assert(_surface->contains_node(target_vertex));
      _surface->collapse(rm_vertex, target_vertex, tid);
    }

    _mesh->erase_vertex(rm_vertex);
  }

  virtual bool is_dynamic(index_t vid){
    return (dynamic_vertex[vid]>=0);
  }

  Mesh<real_t> *_mesh;
  Surface2D<real_t> *_surface;
  ElementProperty<real_t> *property;
  Colouring<real_t> *colouring;

  size_t nnodes_reserve;
  index_t *dynamic_vertex;

  real_t _L_low, _L_max;

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;

  const static size_t node_package_int_size = 1 + (sizeof(index_t) +
                                                   ndims*sizeof(real_t) +
                                                   msize*sizeof(double)) / sizeof(int);
  const static size_t idx_owner = sizeof(index_t) / sizeof(int);
  const static size_t idx_coords = idx_owner + 1;
  const static size_t idx_metric = idx_coords + ndims*sizeof(real_t) / sizeof(int);

  int nprocs;
};

#endif
