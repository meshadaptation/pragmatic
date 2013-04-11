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

#ifndef SWAPPING2D_H
#define SWAPPING2D_H

#include <algorithm>
#include <list>
#include <set>
#include <vector>

#include "AdaptiveAlgorithm.h"
#include "Colouring.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, typename index_t> class Swapping2D : public AdaptiveAlgorithm<real_t, index_t>{
 public:
  /// Default constructor.
  Swapping2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;
    
    size_t NElements = _mesh->get_number_elements();
    
    // Set the orientation of elements.
    property = NULL;
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
    colouring = NULL;

#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#else
    nthreads=1;
#endif
  }
  
  /// Default destructor.
  ~Swapping2D(){
    if(property!=NULL)
      delete property;

    if(colouring!=NULL)
      delete colouring;
  }
  
  void swap(real_t Q_min){
    min_Q = Q_min;
    size_t NNodes = _mesh->get_number_nodes();
    size_t NElements = _mesh->get_number_elements();

    // Total number of active vertices. Used to break the infinite loop.
    size_t total_active;

    if(nnodes_reserve<NNodes){
      nnodes_reserve = 1.5*NNodes;

      if(colouring==NULL)
        colouring = new Colouring<real_t, index_t>(_mesh, this, nnodes_reserve);
      else
        colouring->resize(nnodes_reserve);
    }

    quality.clear();
    quality.resize(NElements);
    
    marked_edges.clear();
    marked_edges.resize(nnodes_reserve);
    
#pragma omp parallel
    {
      int tid = get_tid();

      // Cache the element quality's.
#pragma omp for schedule(static, 32)
      for(size_t i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0){
          quality[i] = 0.0;
          continue;
        }
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        
        quality[i] = property->lipnikov(x0, x1, x2,
                                        _mesh->get_metric(n[0]),
                                        _mesh->get_metric(n[1]),
                                        _mesh->get_metric(n[2]));
      }
      
      // Initialise list of dynamic edges.
#pragma omp for schedule(static, 32)
      for(size_t i=0;i<NNodes;i++){
        colouring->node_colour[i] = -1;

        for(size_t it=0; it<_mesh->NNList[i].size(); ++it){
          if(i < (size_t) _mesh->NNList[i][it])
            marked_edges[i].insert(_mesh->NNList[i][it]);
        }
      }

      do{
        // Finding which vertices comprise the active sub-mesh.
        std::vector<index_t> active_set;

#pragma omp single
        {
          total_active = 0;
        }

#pragma omp for schedule(dynamic, 32) reduction(+:total_active)
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(marked_edges[i].size()>0){
            active_set.push_back(i);
            ++total_active;
          }
        }

        if(total_active == 0)
          break;

        size_t pos;
#pragma omp atomic capture
        {
          pos = colouring->GlobalActiveSet_size;
          colouring->GlobalActiveSet_size += active_set.size();
        }

        memcpy(&colouring->GlobalActiveSet[pos], &active_set[0], active_set.size() * sizeof(index_t));

#pragma omp barrier

        colouring->multiHashJonesPlassmann();

        // Start processing independent sets
        for(int set_no=0; set_no<colouring->nsets; ++set_no){
          do{
            active_set.clear();

#pragma omp for schedule(dynamic, 16) reduction(+:total_active)
            for(size_t idx=0; idx<colouring->ind_set_size[set_no]; ++idx){
              index_t i = colouring->independent_sets[set_no][idx];
              assert(i < NNodes);

              // If the node has been un-coloured, skip it.
              if(colouring->node_colour[i] < 0)
                continue;

              assert(colouring->node_colour[i] == set_no);

              bool swapped = false;

              while(!swapped){
                if(marked_edges[i].size()>0){
                  index_t j = *marked_edges[i].begin();

                  // Mark edge as processed, i.e. remove it from the set of marked edges
                  marked_edges[i].erase(marked_edges[i].begin());

                  Edge<index_t> edge(i, j);
                  swap_kernel(edge, tid);

                  // If edge was swapped
                  if(edge.edge.first != i){
                    swapped = true;
                    index_t k = edge.edge.first;
                    index_t l = edge.edge.second;
                    // Uncolour one of the lateral vertices if their colours clash.
                    // There is a race condition here, but it doesn't do any harm.
                    if(colouring->node_colour[k] == colouring->node_colour[l])
                      _mesh->deferred_reset_colour(l, tid);

                    Edge<index_t> lateralEdges[] = {
                        Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                    // Propagate the operation
                    for(size_t ee=0; ee<4; ++ee)
                      _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                  }
                }else
                  break;
              }

              // If all marked edges adjacent to i have been processed, reset i's colour.
              if(marked_edges[i].empty())
                colouring->node_colour[i] = -1;
              else
                active_set.push_back(i);
            }

#pragma omp single
            {
              colouring->ind_set_size[set_no] = 0;
            }

            _mesh->commit_deferred(tid);
            _mesh->commit_swapping_propagation(marked_edges, tid);
            _mesh->commit_colour_reset(colouring->node_colour, tid);

#pragma omp atomic capture
            {
              pos = colouring->ind_set_size[set_no];
              colouring->ind_set_size[set_no] += active_set.size();
            }

            memcpy(&colouring->independent_sets[set_no][pos], &active_set[0], active_set.size() * sizeof(index_t));
#pragma omp barrier
          } while(colouring->ind_set_size[set_no]>0);
        }

        colouring->destroy();
      }while(true);
    }

    return;
  }

 private:

  void swap_kernel(Edge<index_t>& edge, size_t tid){
    index_t i = edge.edge.first;
    index_t j = edge.edge.second;

    // Find the two elements sharing this edge
    std::set<index_t> intersection;
    std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
        _mesh->NEList[j].begin(), _mesh->NEList[j].end(),
        std::inserter(intersection, intersection.begin()));

    // If this is a surface edge, it cannot be swapped, so un-mark it.
    if(intersection.size()!=2)
      return;

    index_t eid0 = *intersection.begin();
    index_t eid1 = *intersection.rbegin();

    const index_t *n = _mesh->get_element(eid0);
    int n_off=-1;
    for(size_t k=0;k<3;k++){
      if((n[k]!=i) && (n[k]!=j)){
        n_off = k;
        break;
      }
    }
    assert(n[n_off]>=0);

    const index_t *m = _mesh->get_element(eid1);
    int m_off=-1;
    for(size_t k=0;k<3;k++){
      if((m[k]!=i) && (m[k]!=j)){
        m_off = k;
        break;
      }
    }
    assert(m[m_off]>=0);

    assert(n[(n_off+2)%3]==m[(m_off+1)%3] && n[(n_off+1)%3]==m[(m_off+2)%3]);

    real_t worst_q = std::min(quality[eid0], quality[eid1]);
    /*
    if(worst_q>min_Q)
      return;
    */

    index_t k = n[n_off];
    index_t l = m[m_off];

    int n_swap[] = {n[n_off], m[m_off],       n[(n_off+2)%3]}; // new eid0
    int m_swap[] = {n[n_off], n[(n_off+1)%3], m[m_off]};       // new eid1

    real_t q0 = property->lipnikov(_mesh->get_coords(n_swap[0]),
                                   _mesh->get_coords(n_swap[1]),
                                   _mesh->get_coords(n_swap[2]),
                                   _mesh->get_metric(n_swap[0]),
                                   _mesh->get_metric(n_swap[1]),
                                   _mesh->get_metric(n_swap[2]));
    real_t q1 = property->lipnikov(_mesh->get_coords(m_swap[0]),
                                   _mesh->get_coords(m_swap[1]),
                                   _mesh->get_coords(m_swap[2]),
                                   _mesh->get_metric(m_swap[0]),
                                   _mesh->get_metric(m_swap[1]),
                                   _mesh->get_metric(m_swap[2]));
    real_t new_worst_q = std::min(q0, q1);

    if(new_worst_q>worst_q){
      // Cache new quality measures.
      quality[eid0] = q0;
      quality[eid1] = q1;

      // Update NNList
      typename std::vector<index_t>::iterator it;
      it = std::find(_mesh->NNList[i].begin(), _mesh->NNList[i].end(), j);
      _mesh->NNList[i].erase(it);
      _mesh->deferred_remNN(j, i, tid);
      _mesh->deferred_addNN(k, l, tid);
      _mesh->deferred_addNN(l, k, tid);

      // Update node-element list.
      _mesh->deferred_remNE(n_swap[2], eid1, tid);
      _mesh->deferred_remNE(m_swap[1], eid0, tid);
      _mesh->deferred_addNE(n_swap[0], eid1, tid);
      _mesh->deferred_addNE(n_swap[1], eid0, tid);

      // Update element-node list for this element.
      for(size_t cnt=0;cnt<nloc;cnt++){
        _mesh->_ENList[eid0*nloc+cnt] = n_swap[cnt];
        _mesh->_ENList[eid1*nloc+cnt] = m_swap[cnt];
      }

      edge.edge.first = std::min(k, l);
      edge.edge.second = std::max(k, l);
    }

    return;
  }

  inline virtual index_t is_dynamic(index_t vid){
    return (index_t) marked_edges[vid].size();
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  Colouring<real_t, index_t> *colouring;

  size_t nnodes_reserve;

  static const size_t ndims=2;
  static const size_t nloc=3;
  int nthreads;
  std::vector< std::set<index_t> > marked_edges;
  std::vector<real_t> quality;
  real_t min_Q;
};

#endif
