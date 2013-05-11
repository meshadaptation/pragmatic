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
    
    nprocs = 1;
    rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
    MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
#endif

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

    if(nnodes_reserve<1.5*NNodes){
      nnodes_reserve = 2*NNodes;

      if(colouring==NULL)
        colouring = new Colouring<real_t, index_t>(_mesh, this, nnodes_reserve);
      else
        colouring->resize(nnodes_reserve);

      quality.resize(nnodes_reserve * 2, 0.0);
      marked_edges.resize(nnodes_reserve, std::set<index_t>());
    }

#ifdef HAVE_MPI
    // Global arrays used to separate all vertices of an
    // independent set into interior vertices and halo vertices.
    index_t *global_interior = new index_t[nnodes_reserve];
    index_t *global_halo = new index_t[nnodes_reserve];
    size_t global_interior_size = 0;
    size_t global_halo_size = 0;

    std::vector< std::vector<int> > send_buffer(nprocs);
    std::vector<int> send_buffer_size(nprocs, 0);
    std::vector< std::vector<index_t> > sent_vertices_vec(nprocs);
    std::vector<size_t> sent_vertices_vec_size(nprocs, 0);
#endif

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
      if(nprocs>1){
#pragma omp for schedule(static, 32)
        for(index_t i=0; i<(index_t)_mesh->NNodes;i++){
          colouring->node_colour[i] = -1;
          marked_edges[i].clear();

          if(_mesh->node_owner[i] != rank)
            continue;

          for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[i].begin(); it!=_mesh->NNList[i].end(); ++it){
            if(_mesh->lnn2gnn[i] < _mesh->lnn2gnn[*it])
              marked_edges[i].insert(*it);
          }
        }
      }else{
#pragma omp for schedule(static, 32)
        for(index_t i=0;i<(index_t)NNodes;i++){
          colouring->node_colour[i] = -1;
          marked_edges[i].clear();

          for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[i].begin(); it!=_mesh->NNList[i].end(); ++it){
            if(i < *it)
              marked_edges[i].insert(*it);
          }
        }
      }

      do{
        // Find which vertices comprise the active sub-mesh.
        std::vector<index_t> active_set;

#pragma omp single
        {
          total_active = 0;
        }

#pragma omp for schedule(dynamic, 32) reduction(+:total_active)
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(marked_edges[i].size()>0){
            assert(_mesh->node_owner[i]==rank);
            active_set.push_back(i);
            ++total_active;
          }
        }

#ifdef HAVE_MPI
        if(nprocs>1){
#pragma omp single
          {
            MPI_Allreduce(MPI_IN_PLACE, &total_active, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
          }
        }
#endif

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
        if(nprocs>1){
#ifdef HAVE_MPI
          for(int set_no=0; set_no<colouring->global_nsets; ++set_no){
            std::vector<index_t> interior_vertices;
            std::vector<index_t> halo_vertices;
            std::vector< std::vector<int> > local_buffers(nprocs);
            std::vector< std::vector<index_t> > local_sent(nprocs);

#pragma omp for schedule(dynamic, 32) nowait
            for(size_t idx=0; idx<colouring->ind_set_size[set_no]; ++idx){
              index_t i = colouring->independent_sets[set_no][idx];
              assert(i < (index_t) NNodes);

              // If the node has been un-coloured, skip it.
              if(colouring->node_colour[i] < 0)
                continue;

              assert(colouring->node_colour[i] == set_no);

              // Separate interior vertices from halo vertices.
              assert(_mesh->recv_halo.count(i)==0);
              if(_mesh->send_halo.count(i) > 0)
                halo_vertices.push_back(i);
              else
                interior_vertices.push_back(i);
            }

            size_t int_pos, halo_pos;

#pragma omp atomic capture
            {
              int_pos = global_interior_size;
              global_interior_size += interior_vertices.size();
            }

#pragma omp atomic capture
            {
              halo_pos = global_halo_size;
              global_halo_size += halo_vertices.size();
            }

            memcpy(&global_interior[int_pos], &interior_vertices[0], interior_vertices.size() * sizeof(index_t));
            memcpy(&global_halo[halo_pos], &halo_vertices[0], halo_vertices.size() * sizeof(index_t));

#pragma omp barrier

            do{
              active_set.clear();

              // Process all halo vertices and marshal all necessary data which
              // need to be communicated to other processes. Also, mark the order
              // in which edges adjacent to each vertex were swapped.

#pragma omp for schedule(dynamic, 4)
              for(size_t idx=0; idx<global_halo_size; ++idx){
                index_t i = global_halo[idx];
                assert(_mesh->node_owner[i] == rank);
                assert(_mesh->send_halo.count(i) > 0);

                // Set of elements in this cavity which were modified since the last commit of deferred operations.
                std::set<index_t> modified_elements;
                std::set<index_t> marked_edges_copy = marked_edges[i];

                for(typename std::set<index_t>::const_iterator vid=marked_edges_copy.begin(); vid!=marked_edges_copy.end(); ++vid){
                  index_t j = *vid;

                  // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
                  std::vector<index_t> intersection;
                  std::set_intersection(modified_elements.begin(), modified_elements.end(),
                      _mesh->NEList[j].begin(), _mesh->NEList[j].end(), std::inserter(intersection, intersection.begin()));
                  if(intersection.size()>0)
                    continue;

                  // Mark edge as processed, i.e. remove it from the set of marked edges
                  marked_edges[i].erase(j);

                  Edge<index_t> edge(i, j);
                  std::vector<index_t> ele0(3), ele1(3);
                  swap_kernel(edge, modified_elements, &ele0, &ele1, tid);

                  // If edge was swapped
                  if(edge.edge.first != i && edge.edge.first != j){ // The edge was marked for swapping using gnn's
                    index_t k = edge.edge.first;
                    index_t l = edge.edge.second;
                    // Uncolour one of the lateral vertices if their colours
                    // clash and both of them are owned by *this* process.
                    if(_mesh->node_owner[k] == rank && _mesh->node_owner[l] == rank &&
                        colouring->node_colour[k] == colouring->node_colour[l])
                      _mesh->deferred_reset_colour(l, tid);

                    Edge<index_t> lateralEdges[] = {
                        Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                    // Propagate the operation
                    for(size_t ee=0; ee<4; ++ee){
                      // Swap first and second vertices of the edge if necessary.
                      if(_mesh->lnn2gnn[lateralEdges[ee].edge.first] > _mesh->lnn2gnn[lateralEdges[ee].edge.second]){
                        index_t tmp = lateralEdges[ee].edge.first;
                        lateralEdges[ee].edge.first = lateralEdges[ee].edge.second;
                        lateralEdges[ee].edge.second = tmp;
                      }

                      if(_mesh->node_owner[lateralEdges[ee].edge.first] != rank)
                        continue;

                      _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                    }

                    // If the operation is visible by other processes, marshal necessary data.
                    marshal_data(i, j, k, l, ele0, ele1, local_buffers, local_sent);
                  }
                }

                // If all marked edges adjacent to i have been processed, reset i's colour.
                if(marked_edges[i].empty())
                  colouring->node_colour[i] = -1;
                else
                  active_set.push_back(i);
              }

#pragma omp single
              {
                global_halo_size = 0;
              }

              _mesh->commit_deferred(tid);
              _mesh->commit_swapping_propagation(marked_edges, tid);
              _mesh->commit_colour_reset(colouring->node_colour, tid);

#pragma omp atomic capture
              {
                pos = global_halo_size;
                global_halo_size += active_set.size();
              }

              memcpy(&global_halo[pos], &active_set[0], active_set.size() * sizeof(index_t));
#pragma omp barrier
            }while(global_halo_size>0);

            // All processing on the halo for this independent set has been done.
            // Let's create MPI messages and communicate them.

            size_t *send_pos = new size_t[nprocs];
            size_t *sent_vert_pos = new size_t[nprocs];

            for(int i=0; i<nprocs; ++i){
              if(local_buffers[i].size() == 0)
                continue;

#pragma omp atomic capture
              {
                send_pos[i] = send_buffer_size[i];
                send_buffer_size[i] += local_buffers[i].size();
              }

#pragma omp atomic capture
              {
                sent_vert_pos[i] = sent_vertices_vec_size[i];
                sent_vertices_vec_size[i] += local_sent[i].size();
              }
            }

            // Wait for all threads to increment send_buffer_size[i] and then allocate
            // memory for each send_buffer[i]. Same for sent_vertices_vec.
#pragma omp barrier
#pragma omp for schedule(static,1)
            for(int i=0; i<nprocs; ++i){
              if(send_buffer_size[i]>0){
                // Allocate one extra int to store the length of the original message.
                // The receiver will know that everything beyond that size is part of the extension.
                send_buffer[i].resize(++send_buffer_size[i]);
                send_buffer[i][0] = send_buffer_size[i];

                sent_vertices_vec[i].resize(sent_vertices_vec_size[i]);
              }
            }

            for(int i=0; i<nprocs; ++i){
              if(local_buffers[i].size() > 0){
                memcpy(&send_buffer[i][send_pos[i]+1], &local_buffers[i][0], local_buffers[i].size() * sizeof(int));
                memcpy(&sent_vertices_vec[i][sent_vert_pos[i]], &local_sent[i][0], local_sent[i].size() * sizeof(index_t));
              }
            }

            delete[] send_pos;
            delete[] sent_vert_pos;

#pragma omp barrier

            /*********************************************************
             * Let one OpenMP thread take care of all communication. *
             *********************************************************/
#pragma omp single nowait
            {
              process_recv_halo(send_buffer, send_buffer_size, sent_vertices_vec, sent_vertices_vec_size);

              for(int i=0; i<nprocs; ++i){
                send_buffer[i].clear();
                sent_vertices_vec[i].clear();
                send_buffer_size[i] = 0;
                sent_vertices_vec_size[i] = 0;
              }
            }

            // Meanwhile, the rest of the OpenMP threads
            // can start processing interior vertices.
            do{
              active_set.clear();

#pragma omp for schedule(dynamic, 4)
              for(size_t idx=0; idx<global_interior_size; ++idx){
                index_t i = global_interior[idx];
                assert(i < (index_t) NNodes);

                // If the node has been un-coloured, skip it.
                if(colouring->node_colour[i] < 0)
                  continue;

                if(_mesh->send_halo.count(i)>0){
                  colouring->node_colour[i] = -1;
                  continue;
                }

                assert(colouring->node_colour[i] == set_no);

                // Set of elements in this cavity which were modified since the last commit of deferred operations.
                std::set<index_t> modified_elements;
                std::set<index_t> marked_edges_copy = marked_edges[i];

                for(typename std::set<index_t>::const_iterator vid=marked_edges_copy.begin(); vid!=marked_edges_copy.end(); ++vid){
                  index_t j = *vid;

                  // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
                  std::vector<index_t> intersection;
                  std::set_intersection(modified_elements.begin(), modified_elements.end(),
                      _mesh->NEList[j].begin(), _mesh->NEList[j].end(), std::inserter(intersection, intersection.begin()));
                  if(intersection.size()>0)
                    continue;

                  // Mark edge as processed, i.e. remove it from the set of marked edges
                  marked_edges[i].erase(j);

                  Edge<index_t> edge(i, j);
                  swap_kernel(edge, modified_elements, NULL, NULL, tid);

                  // If edge was swapped
                  if(edge.edge.first != i && edge.edge.first != j){ // The edge was marked for swapping using gnn's
                    index_t k = edge.edge.first;
                    index_t l = edge.edge.second;
                    // Uncolour one of the lateral vertices if their colours clash.
                    if(colouring->node_colour[k] == colouring->node_colour[l])
                      _mesh->deferred_reset_colour(l, tid);

                    Edge<index_t> lateralEdges[] = {
                        Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                    // Propagate the operation
                    for(size_t ee=0; ee<4; ++ee){
                      // Swap first and second vertices of the edge if necessary.
                      if(_mesh->lnn2gnn[lateralEdges[ee].edge.first] > _mesh->lnn2gnn[lateralEdges[ee].edge.second]){
                        index_t tmp = lateralEdges[ee].edge.first;
                        lateralEdges[ee].edge.first = lateralEdges[ee].edge.second;
                        lateralEdges[ee].edge.second = tmp;
                      }

                      // Oh yes, this check is necessary.... Unbelievable!
                      if(_mesh->node_owner[lateralEdges[ee].edge.first]!=rank)
                        continue;

                      _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                    }
                  }
                }

                // If all marked edges adjacent to i have been processed, reset i's colour.
                if(marked_edges[i].empty())
                  colouring->node_colour[i] = -1;
                else
                  active_set.push_back(i);
              }

#pragma omp single
              {
                global_interior_size = 0;
              }

              _mesh->commit_deferred(tid);
              _mesh->commit_swapping_propagation(marked_edges, tid);
              _mesh->commit_colour_reset(colouring->node_colour, tid);

#pragma omp atomic capture
              {
                pos = global_interior_size;
                global_interior_size += active_set.size();
              }

              memcpy(&global_interior[pos], &active_set[0], active_set.size() * sizeof(index_t));
#pragma omp barrier
            }while(global_interior_size>0);
          }
#endif
        }else{
          for(int set_no=0; set_no<colouring->nsets; ++set_no){
            do{
              active_set.clear();

#pragma omp for schedule(dynamic, 4)
              for(size_t idx=0; idx<colouring->ind_set_size[set_no]; ++idx){
                index_t i = colouring->independent_sets[set_no][idx];
                assert(i < (index_t) NNodes);

                // If the node has been un-coloured, skip it.
                if(colouring->node_colour[i] < 0)
                  continue;

                assert(colouring->node_colour[i] == set_no);

                // Set of elements in this cavity which were modified since the last commit of deferred operations.
                std::set<index_t> modified_elements;
                std::set<index_t> marked_edges_copy = marked_edges[i];

                for(typename std::set<index_t>::const_iterator vid=marked_edges_copy.begin(); vid!=marked_edges_copy.end(); ++vid){
                  index_t j = *vid;

                  // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
                  std::vector<index_t> intersection;
                  std::set_intersection(modified_elements.begin(), modified_elements.end(),
                      _mesh->NEList[j].begin(), _mesh->NEList[j].end(), std::inserter(intersection, intersection.begin()));
                  if(intersection.size()>0)
                    continue;

                  // Mark edge as processed, i.e. remove it from the set of marked edges
                  marked_edges[i].erase(j);

                  Edge<index_t> edge(i, j);
                  swap_kernel(edge, modified_elements, NULL, NULL, tid);

                  // If edge was swapped
                  if(edge.edge.first != i){
                    index_t k = edge.edge.first;
                    index_t l = edge.edge.second;
                    // Uncolour one of the lateral vertices if their colours clash.
                    if(colouring->node_colour[k] == colouring->node_colour[l])
                      _mesh->deferred_reset_colour(l, tid);

                    Edge<index_t> lateralEdges[] = {
                        Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                    // Propagate the operation
                    for(size_t ee=0; ee<4; ++ee)
                      _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                  }
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
            }while(colouring->ind_set_size[set_no]>0);
          }
        }

#pragma omp barrier
        colouring->destroy();
      }while(true);
    }

#ifdef HAVE_MPI
    delete[] global_interior;
    delete[] global_halo;

    if(nprocs>1)
      _mesh->trim_halo();
#endif

    return;
  }

 private:

  void swap_kernel(Edge<index_t>& edge, std::set<index_t>& modified_elements,
      std::vector<index_t>* ele0, std::vector<index_t>* ele1, size_t tid){
    index_t i = edge.edge.first;
    index_t j = edge.edge.second;

    // Find the two elements sharing this edge
    std::set<index_t> intersection;
    std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
        _mesh->NEList[j].begin(), _mesh->NEList[j].end(),
        std::inserter(intersection, intersection.begin()));

    // If this is a surface edge, it cannot be swapped.
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

      // Cache old elements
      if(ele0 != NULL)
        for(size_t k=0; k<nloc; ++k){
          ele0->at(k) = n[k];
          ele1->at(k) = m[k];
        }

      // Update element-node list for this element.
      for(size_t cnt=0;cnt<nloc;cnt++){
        _mesh->_ENList[eid0*nloc+cnt] = n_swap[cnt];
        _mesh->_ENList[eid1*nloc+cnt] = m_swap[cnt];
      }

      edge.edge.first = std::min(k, l);
      edge.edge.second = std::max(k, l);
      modified_elements.insert(eid0);
      modified_elements.insert(eid1);
    }

    return;
  }

  void swap_kernel_single_thr(Edge<index_t>& edge){
    index_t i = edge.edge.first;
    index_t j = edge.edge.second;

    // Find the two elements sharing this edge
    std::set<index_t> intersection;
    std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
        _mesh->NEList[j].begin(), _mesh->NEList[j].end(),
        std::inserter(intersection, intersection.begin()));

    assert(intersection.size() == 2);

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

    // Cache new quality measures.
    quality[eid0] = q0;
    quality[eid1] = q1;

    // Update NNList
    typename std::vector<index_t>::iterator it;
    it = std::find(_mesh->NNList[i].begin(), _mesh->NNList[i].end(), j);
    _mesh->NNList[i].erase(it);
    it = std::find(_mesh->NNList[j].begin(), _mesh->NNList[j].end(), i);
    _mesh->NNList[j].erase(it);
    _mesh->NNList[k].push_back(l);
    _mesh->NNList[l].push_back(k);

    // Update node-element list.
    _mesh->NEList[n_swap[2]].erase(eid1);
    _mesh->NEList[m_swap[1]].erase(eid0);
    _mesh->NEList[n_swap[0]].insert(eid1);
    _mesh->NEList[n_swap[1]].insert(eid0);

    // Update element-node list for this element.
    for(size_t cnt=0;cnt<nloc;cnt++){
      _mesh->_ENList[eid0*nloc+cnt] = n_swap[cnt];
      _mesh->_ENList[eid1*nloc+cnt] = m_swap[cnt];
    }

    edge.edge.first = std::min(k, l);
    edge.edge.second = std::max(k, l);

    return;
  }

  inline virtual index_t is_dynamic(index_t vid){
    return (index_t) marked_edges[vid].size();
  }

  void marshal_data(index_t i, index_t j, index_t k, index_t l, std::vector<index_t>& ele0, std::vector<index_t>& ele1,
      std::vector< std::vector<int> >& local_buffers, std::vector< std::vector<index_t> >& sent){
    index_t lnns[] = {j, k, l};
    index_t gnns[] = {_mesh->lnn2gnn[j], _mesh->lnn2gnn[k], _mesh->lnn2gnn[l]};
    bool visible_by_proc[3];

    for(int proc=0; proc<nprocs; ++proc){
      if(proc == rank)
        continue;

      if(_mesh->send_map[proc].count(_mesh->lnn2gnn[i])==0)
        continue;

      /* A vertex is visible by proc either if it is owned by proc
       * or if it is in our send_map[proc]. There is a third case in
       * which proc knows about this vertex through another process.
       */
      for(size_t t=0; t<3; ++t){
        if(_mesh->node_owner[lnns[t]] == proc)
          visible_by_proc[t] = true;
        else if(_mesh->send_map[proc].count(gnns[t]) > 0)
          visible_by_proc[t] = true;
        else{
          visible_by_proc[t] = false;
          for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[lnns[t]].begin();it!=_mesh->NNList[lnns[t]].end();++it){
            if(_mesh->node_owner[*it]==proc){
              visible_by_proc[t] = true;
              break;
            }
          }
        }
      }

      if(visible_by_proc[0]){
        // The edge is possibly visible by proc, so we have to communicate the operation.
        local_buffers[proc].push_back(_mesh->lnn2gnn[i]);
        assert(_mesh->node_owner[i]==rank); // Owner of i is always the sender
        for(size_t t=0; t<3; ++t){
          local_buffers[proc].push_back(gnns[t]);
          local_buffers[proc].push_back(_mesh->node_owner[lnns[t]]);
        }

        // Check whether extra data needs to be sent.
        if(_mesh->node_owner[j]==proc){
          // If j is owned by proc, then proc sees both elements and all vertices involved.
          local_buffers[proc].push_back(0); // 0 vertices
          local_buffers[proc].push_back(0); // 0 elements
        }
        else{
          // Otherwise, send any data which may not be visible to proc.
          std::vector<int> vertex_idx;
          std::vector< std::vector<index_t> > elements;

          if((_mesh->node_owner[k] == proc) != (_mesh->node_owner[l] == proc)){
            // If proc owns one (but not both) of k,l it definitely sees one of
            // the two elements involved. Send the other one and the non-owned
            // vertex if it is not visible by proc.
            int idx = (_mesh->node_owner[k] == proc ? 2 : 1); // 2-->l, 1-->k
            if(!visible_by_proc[idx]){ // If it is really invisible
              vertex_idx.push_back(idx);
            }

            // Find which is the invisible element
            bool is_ele0 = false;
            for(size_t t=0; t<nloc; ++t)
              if(lnns[idx] == ele0[t]){
                is_ele0 = true;
                break;
              }

            if(is_ele0)
              elements.push_back(ele0);
            else
              elements.push_back(ele1);

          }else{
            // Send any invisible vertex and both elements just in case.
            for(size_t t=1; t<nloc; ++t)
              if(!visible_by_proc[t])
                vertex_idx.push_back(t);

            elements.push_back(ele0);
            elements.push_back(ele1);
          }

          local_buffers[proc].push_back(vertex_idx.size()); // We are sending so many vertices
          for(typename std::vector<int>::const_iterator it=vertex_idx.begin(); it!=vertex_idx.end(); ++it){
            sent[proc].push_back(lnns[*it]);

            std::vector<int> ivertex(node_package_int_size);

            index_t *rgnn = (index_t *) &ivertex[0];
            int *rowner = (int *) &ivertex[idx_owner];
            real_t *rcoords = (real_t *) &ivertex[idx_coords];
            float *rmetric = (float *) &ivertex[idx_metric];

            *rgnn = gnns[*it];
            *rowner = _mesh->node_owner[lnns[*it]];

            const real_t *x = _mesh->get_coords(lnns[*it]);
            rcoords[0] = x[0];
            rcoords[1] = x[1];

            const float *m = _mesh->get_metric(lnns[*it]);
            rmetric[0] = m[0];
            rmetric[1] = m[1];
            rmetric[2] = m[2];

            local_buffers[proc].insert(local_buffers[proc].end(), ivertex.begin(), ivertex.end());
          }

          local_buffers[proc].push_back(elements.size()); // We are sending so many elements
          for(typename std::vector< std::vector<index_t> >::const_iterator it=elements.begin(); it!=elements.end(); ++it){
            const std::vector<index_t>& ele = *it;

            for(size_t t=0; t<nloc; ++t){
              local_buffers[proc].push_back(_mesh->lnn2gnn[ele[t]]);
              local_buffers[proc].push_back(_mesh->node_owner[ele[t]]);
            }

            // Check for surface facets
            std::vector<int> lfacets;
            _surface->find_facets(&ele[0], lfacets);

            local_buffers[proc].push_back(lfacets.size());

            for(size_t f=0; f<lfacets.size(); ++f){
              // Push back surface vertices
              const int *sn = _surface->get_facet(lfacets[f]);
              for(size_t t=0; t<snloc; ++t){
                local_buffers[proc].push_back(_mesh->lnn2gnn[sn[t]]);
                local_buffers[proc].push_back(_mesh->node_owner[sn[t]]);
              }

              local_buffers[proc].push_back(_surface->get_boundary_id(lfacets[f]));
              local_buffers[proc].push_back(_surface->get_coplanar_id(lfacets[f]));
            }
          }
        }

        /* Take care of colouring. There are six cases:
         *
         * 1st case: *This* process owns both vertices k and l.
         *           Nothing needs to be sent, *this* process can decide alone
         *           what has to be done, no one else cares about the colours.
         *
         * 2nd case: Process *proc owns both vertices k and l.
         *           Nothing needs to be sent, *proc can decide alone what
         *           has to be done, no one else cares about the colours.
         *
         * 3rd case: None of k and l are owned by *this* process or *proc.
         *           Nothing needs to be sent, neither *this* process nor
         *           *proc care about the colours of those vertices.
         *
         * 4th case: *This* process owns one of k or l and *proc owns the other.
         *           Keep the colour of owned vertex and send it to *proc.
         *
         * 5th case: *This process owns one of k or l and a third process owns the other.
         *           Process *proc does not care about the colours of those vertices, so
         *           nothing needs to be sent.
         *
         * 6th case: Process *proc owns one of k or l and a third process owns the other. We don't
         *           know anything about the colour of the vertex owned by the third process,
         *           so tell *proc to uncolour the vertex owned by it for security reasons.
         */

        if((_mesh->node_owner[k] == rank && _mesh->node_owner[l] == proc) ||
            (_mesh->node_owner[k] == proc && _mesh->node_owner[l] == rank)){ // 4th case
          index_t owned = (_mesh->node_owner[k] == rank ? k : l);
          local_buffers[proc].push_back(1);
          local_buffers[proc].push_back(_mesh->lnn2gnn[owned]);
          local_buffers[proc].push_back(rank);
          local_buffers[proc].push_back(colouring->node_colour[owned]);
        }else if((_mesh->node_owner[k] == proc && _mesh->node_owner[l] != rank && _mesh->node_owner[l] != proc) ||
            (_mesh->node_owner[l] == proc && _mesh->node_owner[k] != rank && _mesh->node_owner[k] != proc)){ // 6th case
          index_t owned = (_mesh->node_owner[k] == proc ? k : l);
          local_buffers[proc].push_back(1);
          local_buffers[proc].push_back(_mesh->lnn2gnn[owned]);
          local_buffers[proc].push_back(proc);
          // -2 is a special value which indicates that we don't know the colour of the
          // other vertex. This is a message for the receiver to uncolour its owned vertex.
          local_buffers[proc].push_back(-2);
        }else
          local_buffers[proc].push_back(0); // Nothing to be sent
      }
    }
  }

  void process_recv_halo(std::vector< std::vector<int> >& send_buffer, std::vector<int>&  send_buffer_size,
      std::vector< std::vector<index_t> >& sent_vertices_vec, std::vector<size_t>& sent_vertices_vec_size){
    std::vector< std::vector<int> > recv_buffer(nprocs);
    std::vector<int> recv_buffer_size(nprocs, 0);
    std::vector<MPI_Request> request(2*nprocs);
    std::vector<MPI_Status> status(2*nprocs);

    /* For each process P_i, we create a map(gnn, lnn) containing all
     * vertices owned by *this* MPI process which will be sent to P_i,
     * no matter whether the vertex is sent by *this* process or by a third
     * process. Similarly, we create a map for vertices *this* process will
     * receive. Vertices will be sorted by their gnn's, so they will be
     * added later to _mesh->send[P_i] on *this* process in the same order
     * as they will be added to _mesh->recv[*this*] on P_i and vice versa.
     */
    std::vector< std::map<index_t, index_t> > sent_vertices(nprocs);
    std::vector< std::map<index_t, index_t> > recv_vertices(nprocs);

    // Set up sent_vertices and append appropriate data to the end of send
    // buffers. So far, every sent_vertices_vec[i] contains all vertices
    // sent to process i, no matter whether they are owned by us or not.
    std::vector< std::vector<int> > message_extensions(nprocs);

    for(int i=0; i<nprocs; ++i){
      for(typename std::vector<index_t>::const_iterator it = sent_vertices_vec[i].begin();
          it != sent_vertices_vec[i].end(); ++it){
        int owner = _mesh->node_owner[*it];
        index_t gnn = _mesh->lnn2gnn[*it];

        if(owner==rank){
          if(_mesh->send_map[i].count(gnn)==0){
            sent_vertices[i][gnn] = *it;
            _mesh->send_map[i][gnn] = *it;
          }
        }
        else{
          // Send message (gnn, proc): Tell the owner that we have sent vertex gnn to process proc.
          message_extensions[owner].push_back(gnn);
          message_extensions[owner].push_back(i);
        }
      }
    }

    for(int i=0; i<nprocs; ++i){
      // Append the extension.
      if(message_extensions[i].size()>0){
        // If there is no original part in the message, indicate
        // its zero length before pushing back the extension.
        if(send_buffer[i].size() == 0)
          send_buffer[i].push_back(0);

        send_buffer[i].insert(send_buffer[i].end(),
            message_extensions[i].begin(), message_extensions[i].end());
        send_buffer_size[i] = send_buffer[i].size();
      }
    }

    // First we need to communicate message sizes using MPI_Alltoall.
    MPI_Alltoall(&send_buffer_size[0], 1, MPI_INT, &recv_buffer_size[0], 1, MPI_INT, _mesh->get_mpi_comm());

    // Now that we know the size of all messages we are going to receive from
    // other MPI processes, we can set up asynchronous communication for the
    // exchange of the actual send_buffers. Also, allocate memory for the receive buffers.
    for(int i=0;i<nprocs;i++){
      if(recv_buffer_size[i]>0){
        recv_buffer[i].resize(recv_buffer_size[i]);
        MPI_Irecv(&recv_buffer[i][0], recv_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &request[i]);
      }
      else
        request[i] = MPI_REQUEST_NULL;

      if(send_buffer_size[i]>0)
        MPI_Isend(&send_buffer[i][0], send_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &request[nprocs+i]);
      else
        request[nprocs+i] = MPI_REQUEST_NULL;
    }

    // Wait for MPI transfers to complete.
    MPI_Waitall(2*nprocs, &request[0], &status[0]);

    // Unmarshal data and apply received operations.

    for(int proc=0;proc<nprocs;proc++){
      if(recv_buffer[proc].size()==0)
        continue;

      int *buffer = &recv_buffer[proc][0];
      size_t original_part_size = buffer[0];
      size_t loc=1;

      // Part 1: append new vertices, elements and facets to the mesh.
      while(loc < original_part_size){
        // Find vertices i and j
        index_t ignn = buffer[loc++];
        // Vertex i is always owned by the sender.
        assert(_mesh->recv_map[proc].count(ignn) > 0);
        index_t i = _mesh->recv_map[proc][ignn];

        index_t jgnn = buffer[loc++];
        int jowner = buffer[loc++];
        index_t j;
        if(jowner == rank){
          assert(_mesh->send_map[proc].count(jgnn) > 0);
          j = _mesh->send_map[proc][jgnn];
        }else{
          assert(_mesh->recv_map[jowner].count(jgnn) > 0);
          j = _mesh->recv_map[jowner][jgnn];
        }

        // k or l may not be known to us yet - we will resolve this later.
        index_t kgnn = buffer[loc++];
        int kowner = buffer[loc++];
        index_t lgnn = buffer[loc++];
        int lowner = buffer[loc++];

        // Parse data about invisible vertices.
        int nVertices = buffer[loc++];
        for(size_t cnt=0; cnt<(size_t)nVertices; ++cnt){
          index_t gnn = *((index_t *) &buffer[loc]);
          int owner = buffer[loc+idx_owner];

          // Only append this vertex to the mesh if we haven't received it before.
          if(_mesh->recv_map[owner].count(gnn) == 0){
            real_t *rcoords = (real_t *) &buffer[loc+idx_coords];
            float *rmetric = (float *) &buffer[loc+idx_metric];

            index_t new_lnn = _mesh->append_vertex(rcoords, rmetric);

            _mesh->lnn2gnn[new_lnn] = gnn;
            _mesh->node_owner[new_lnn] = owner;
            _mesh->recv_map[owner][gnn] = new_lnn;
            colouring->node_colour[new_lnn] = -1;
            marked_edges[new_lnn].clear();
            recv_vertices[owner][gnn] = new_lnn;
          }

          loc += node_package_int_size;
        }

        // Parse data about invisible elements.
        int nElements = buffer[loc++];
        for(size_t cnt=0; cnt<(size_t)nElements; ++cnt){
          index_t ele[] = {-1, -1, -1};
          for(size_t t=0; t<nloc; ++t){
            index_t gnn = buffer[loc++];
            int owner = buffer[loc++];
            if(owner == rank){
              assert(_mesh->send_map[proc].count(gnn)>0);
              ele[t] = _mesh->send_map[proc][gnn];
            }else{
              assert(_mesh->recv_map[owner].count(gnn)>0);
              ele[t] = _mesh->recv_map[owner][gnn];
            }
            assert(ele[t] >= 0);
          }

          std::set<index_t> intersection;
          std::set_intersection(_mesh->NEList[ele[0]].begin(), _mesh->NEList[ele[0]].end(),
              _mesh->NEList[ele[1]].begin(), _mesh->NEList[ele[1]].end(),
              std::inserter(intersection, intersection.begin()));
          std::set<index_t> common_element;
          std::set_intersection(_mesh->NEList[ele[2]].begin(), _mesh->NEList[ele[2]].end(),
              intersection.begin(), intersection.end(),
              std::inserter(common_element, common_element.begin()));

          if(common_element.empty()){
            index_t eid = _mesh->append_element(ele);

            /* Update NNList and NEList. Updates are thread-safe, because they pertain
             * to recv_halo vertices only, which are not touched by the rest of the
             * OpenMP threads that are processing the interior of this MPI partition.
             */
            for(size_t t=0; t<nloc; ++t){
              _mesh->NEList[ele[t]].insert(eid);

              for(size_t u=t+1; u<nloc; ++u){
                typename std::vector<index_t>::iterator it;
                it = std::find(_mesh->NNList[ele[t]].begin(), _mesh->NNList[ele[t]].end(), ele[u]);

                if(it == _mesh->NNList[ele[t]].end()){
                  _mesh->NNList[ele[t]].push_back(ele[u]);
                  _mesh->NNList[ele[u]].push_back(ele[t]);
                }
              }
            }
          }

          // Unpack any new facets which are part of this element.
          int nFacets = buffer[loc++];

          for(int i=0; i<nFacets; ++i){
            index_t facet[] = {-1, -1};

            for(size_t j=0; j<snloc; ++j){
              index_t gnn = buffer[loc++];
              int owner = buffer[loc++];
              assert(_mesh->recv_map[owner].count(gnn) > 0);
              facet[j] = _mesh->recv_map[owner][gnn];
              assert(facet[j] >= 0);
            }

            int boundary_id = buffer[loc++];
            int coplanar_id = buffer[loc++];

            // Updates to surface are thread-safe for the
            // same reason as updates to adjacency lists.
            _surface->append_facet(facet, boundary_id, coplanar_id, true);
          }
        }

        // By now, the receiver has all necessary information
        // to perform swapping on edge (i,j). Resolve k and l.
        index_t k;
        if(kowner == rank){
          assert(_mesh->send_map[proc].count(kgnn) > 0);
          k = _mesh->send_map[proc][kgnn];
        }else{
          assert(_mesh->recv_map[kowner].count(kgnn) > 0);
          k = _mesh->recv_map[kowner][kgnn];
        }

        index_t l;
        if(lowner == rank){
          assert(_mesh->send_map[proc].count(lgnn) > 0);
          l = _mesh->send_map[proc][lgnn];
        }else{
          assert(_mesh->recv_map[lowner].count(lgnn) > 0);
          l = _mesh->recv_map[lowner][lgnn];
        }

        int extra_colour = buffer[loc++];
        if(extra_colour == 1){
          index_t gnn = buffer[loc++];
          int owner = buffer[loc++];
          assert(owner == proc || owner == rank);

          index_t ref_vertex;
          if(owner == proc){ // This is the 4th case
            assert(_mesh->recv_map[proc].count(gnn) > 0);
            ref_vertex = _mesh->recv_map[proc][gnn];
          }else{ // This is the 6th case
            assert(_mesh->send_map[proc].count(gnn) > 0);
            ref_vertex = _mesh->send_map[proc][gnn];
            assert(buffer[loc] == -2);
          }
          assert(ref_vertex==k || ref_vertex==l);

          index_t other_vertex = (ref_vertex==k ? l : k);

          int colour = buffer[loc++];
          if(colour >= 0){
            // Check whether ref_vertex's colour (owned by the sender)
            // clashes with other_vertex's colour (owned by the receiver).
            if(colouring->node_colour[other_vertex] == colour)
              colouring->node_colour[other_vertex] = -1;
          }else
            colouring->node_colour[ref_vertex] = -1;
        }

        // Perform swapping for this edge.
        Edge<index_t> edge(i,j);
        swap_kernel_single_thr(edge);

        k = edge.edge.first;
        l = edge.edge.second;

        Edge<index_t> lateralEdges[] = {
            Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

        // Propagate the operation
        for(size_t ee=0; ee<4; ++ee){
          // Swap first and second vertices of the edge if necessary.
          if(_mesh->lnn2gnn[lateralEdges[ee].edge.first] > _mesh->lnn2gnn[lateralEdges[ee].edge.second]){
            index_t tmp = lateralEdges[ee].edge.first;
            lateralEdges[ee].edge.first = lateralEdges[ee].edge.second;
            lateralEdges[ee].edge.second = tmp;
          }

          if(_mesh->node_owner[lateralEdges[ee].edge.first] != rank)
            continue;

          marked_edges[lateralEdges[ee].edge.first].insert(lateralEdges[ee].edge.second);
        }
      }

      assert(loc == original_part_size);

      // Part 2: Look at the extensions. The extension contains pairs (gnn, receiver):
      // The sender process proc sent to process receiver information about vertex gnn.
      while(loc < recv_buffer[proc].size()){
        index_t gnn = buffer[loc++];
        assert(_mesh->send_map[proc].count(gnn) > 0);
        index_t vid = _mesh->send_map[proc][gnn];

        int receiver = buffer[loc++];

        // If the receiver didn't know about vertex vid before, now we know that this process
        // has appended vid to its halo, so we have to add vid to our _mesh->send[receiver].
        if(_mesh->send_map[receiver].count(gnn) == 0){
          _mesh->send_map[receiver][gnn] = vid;
          sent_vertices[receiver][gnn] = vid;
        }
      }
    }

    // Update _mesh->send and _mesh->recv.
    for(int i=0; i<nprocs; ++i){
      for(typename std::map<index_t, index_t>::const_iterator it=sent_vertices[i].begin(); it!=sent_vertices[i].end(); ++it){
        _mesh->send[i].push_back(it->second);
        _mesh->send_halo.insert(it->second);
      }

      for(typename std::map<index_t, index_t>::const_iterator it=recv_vertices[i].begin(); it!=recv_vertices[i].end(); ++it){
        _mesh->recv[i].push_back(it->second);
        _mesh->recv_halo.insert(it->second);
      }

      assert(_mesh->send[i].size() == _mesh->send_map[i].size());
      assert(_mesh->recv[i].size() == _mesh->recv_map[i].size());
    }
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  Colouring<real_t, index_t> *colouring;

  size_t nnodes_reserve;

  static const size_t ndims=2;
  static const size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;
  int nthreads;
  std::vector< std::set<index_t> > marked_edges;
  std::vector<real_t> quality;
  real_t min_Q;

  const static size_t node_package_int_size = 1 + (sizeof(index_t) +
                                                   ndims*sizeof(real_t) +
                                                   msize*sizeof(float)) / sizeof(int);
  const static size_t idx_owner = sizeof(index_t) / sizeof(int);
  const static size_t idx_coords = idx_owner + 1;
  const static size_t idx_metric = idx_coords + ndims*sizeof(real_t) / sizeof(int);

  int nprocs, rank;
};

#endif
