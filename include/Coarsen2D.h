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

template<typename real_t, typename index_t> class Coarsen2D : public AdaptiveAlgorithm<real_t, index_t>{
 public:
  /// Default constructor.
  Coarsen2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;

    nprocs = 1;
    rank = 0;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
      MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
    }
#endif

#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#else
    nthreads = 1;
#endif

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
  ~Coarsen2D(){
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
    _L_low = L_low;
    _L_max = L_max;
    
    size_t NNodes= _mesh->get_number_nodes();
    
    if(nnodes_reserve<NNodes){
      nnodes_reserve = 1.5*NNodes;
      
      if(dynamic_vertex!=NULL){
        delete [] dynamic_vertex;
      }
      
      dynamic_vertex = new index_t[nnodes_reserve];

      if(colouring==NULL)
        colouring = new Colouring<real_t, index_t>(_mesh, this, nnodes_reserve);
      else
        colouring->resize(nnodes_reserve);
    }

    /* dynamic_vertex[i] >= 0 :: target to collapse node i
       dynamic_vertex[i] = -1 :: node inactive (deleted/locked)
       dynamic_vertex[i] = -2 :: recalculate collapse - this is how propagation is implemented
    */

    // Total number of active vertices. Used to break the infinite loop.
    size_t total_active;

#ifdef HAVE_MPI
    // Global arrays used to separate all vertices of an
    // independent set into interior vertices and halo vertices.
    index_t *global_interior = new index_t[nnodes_reserve];
    index_t *global_halo = new index_t[nnodes_reserve];
    size_t global_interior_size = 0;
    size_t global_halo_size = 0;

    // Receive and send buffers for all MPI processes.
    std::vector< std::vector<int> > send_buffer(nprocs);
    std::vector< std::vector<int> > recv_buffer(nprocs);
    std::vector<int> send_buffer_size(nprocs, 0);
    std::vector<int> recv_buffer_size(nprocs, 0);
    std::vector<MPI_Request> request(2*nprocs);
    std::vector<MPI_Status> status(2*nprocs);

    /* For each process P_i, we create a map(gnn, lnn) containing all owned
     * vertices this MPI process will send to the other process. Vertices will
     * be sorted by their gnn's, so they will be added later to _mesh->send[]
     * in the same order as they will be added to _mesh->recv[] on P_i. The
     * vector version of this map is there so that OpenMP threads can concatenate
     * their local lists into a global one, so that a thread can set up the map.
     */
    std::vector< std::vector<index_t> > sent_vertices_vec(nprocs);
    std::vector<size_t> sent_vertices_vec_size(nprocs, 0);
#endif

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();

      // Mark all vertices for evaluation.
#pragma omp for schedule(static)
      for(size_t i=0;i<NNodes;i++){
        dynamic_vertex[i] = -2;
        colouring->node_colour[i] = -1;
      }

      do{
        /* Create the active sub-mesh. This is the mesh consisting of all dynamic
         * vertices and all edges connecting two dynamic vertices, i.e. sub_NNList[i]
         * of active sub-mesh contains all vertices of _mesh->NNList[i] which are
         * dynamic. The goal is to prevent two adjacent vertices from collapsing at the
         * same time, therefore avoiding structural hazards. A nice side-effect is that
         * we also enforce the "every other vertex" rule. Safe parallel updating of _mesh
         * adjacency lists can be achieved later using the deferred updates mechanism.
         */

        // Start by finding which vertices comprise the active sub-mesh.
        std::vector<index_t> active_set;

#pragma omp single
        {
          total_active = 0;
        }

        /* Initialise list of vertices to be coarsened. A dynamic
           schedule is used as previous coarsening may have introduced
           significant gaps in the node list. This could lead to
           significant load imbalance if a static schedule was used. */
#pragma omp for schedule(dynamic, 32) reduction(+:total_active)
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(dynamic_vertex[i] == -2){
            dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);

            if(dynamic_vertex[i]>=0){
              active_set.push_back(i);
              ++total_active;
            }
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

        /*
         * Start processing independent sets. After processing each set, colouring
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
        if(nprocs>1){
#ifdef HAVE_MPI
          for(int set_no=0; set_no<colouring->global_nsets; ++set_no){
            /* See which vertices in the independent set are actually going to collapse and
             * communicate related data to other MPI processes. We separate vertices into
             * interior and halo groups. The idea is that we can set up asynchronous MPI
             * communications for the halo group and proceed to coarsen interior vertices
             * while communication is in progress. Hopefully, if the interior group is
             * sufficiently large, computation will overlap communication completely!
             */
            std::vector<index_t> interior_vertices;
            std::vector<index_t> halo_vertices;

            // Local (per OpenMP thread) vectors for data marshalling.
            std::vector< std::vector<int> > local_buffers(nprocs);

            // Local vectors storing IDs of vertices which will be sent to other processes.
            std::vector< std::vector<index_t> > local_sent(nprocs);

#pragma omp for schedule(dynamic,32) nowait
            for(size_t i=0; i<colouring->ind_set_size[set_no]; ++i){
              index_t rm_vertex = colouring->independent_sets[set_no][i];
              assert((size_t) rm_vertex < _mesh->NNodes);

              // If the node has been un-coloured, skip it.
              if(colouring->node_colour[rm_vertex] < 0)
                continue;

              assert(colouring->node_colour[rm_vertex] == set_no);

              /* If this rm_vertex is marked for re-evaluation, it means that the
               * local neighbourhood has changed since coarsen_identify_kernel was
               * called for this vertex. Call coarsen_identify_kernel again.
               * Obviously, this check is redundant for set_no=0.
               */
              if(dynamic_vertex[rm_vertex] == -2)
                dynamic_vertex[rm_vertex] = coarsen_identify_kernel(rm_vertex, L_low, L_max);

              if(dynamic_vertex[rm_vertex] < 0){
                colouring->node_colour[rm_vertex] = -1;
                continue;
              }

              // Un-colour target_vertex if its colour clashes with any of its new neighbours.
              // There is a race condition here, but it doesn't do any harm.
              index_t target_vertex = dynamic_vertex[rm_vertex];
              if(colouring->node_colour[target_vertex] >= 0){
                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  if(*jt != target_vertex){
                    if(colouring->node_colour[*jt] == colouring->node_colour[target_vertex]){
                      colouring->node_colour[target_vertex] = -1;
                      break;
                    }
                  }
                }
              }

              // Separate interior vertices from halo vertices.
              assert(_mesh->recv_halo.count(rm_vertex)==0);
              if(_mesh->send_halo.count(rm_vertex) > 0){
                // If rm_vertex is a halo vertex, then marshal necessary data.
                halo_vertices.push_back(rm_vertex);
                marshal_data(rm_vertex, target_vertex, local_buffers, local_sent);
              }
              else
                interior_vertices.push_back(rm_vertex);
            }

            size_t int_pos, halo_pos;
            size_t *send_pos = new size_t[nprocs];
            size_t *sent_vert_pos = new size_t[nprocs];

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
                    sent_vertices[i][gnn] = *it;
                    _mesh->send_map[i][gnn] = *it;
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

              // Wait for MPI transfers to complete and unmarshal data.
              MPI_Waitall(2*nprocs, &request[0], &status[0]);

              unmarshal_data(recv_buffer, global_halo, global_halo_size, set_no, sent_vertices, recv_vertices);

              for(int i=0; i<nprocs; ++i){
                send_buffer[i].clear();
                recv_buffer[i].clear();
                sent_vertices_vec[i].clear();
                send_buffer_size[i] = 0;
                recv_buffer_size[i] = 0;
                sent_vertices_vec_size[i] = 0;
              }
            }

            // Meanwhile, the rest of the OpenMP threads
            // can start processing interior vertices.
#pragma omp for schedule(dynamic,32)
            for(size_t i=0; i<global_interior_size; ++i){
              index_t rm_vertex = global_interior[i];
              index_t target_vertex = dynamic_vertex[rm_vertex];

              // Mark neighbours for re-evaluation.
              // Two threads might be marking the same vertex at the same time.
              // This is a race condition which doesn't cause any trouble.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt)
                dynamic_vertex[*jt] = -2;

              // Mark rm_vertex as non-active.
              dynamic_vertex[rm_vertex] = -1;
              colouring->node_colour[rm_vertex] = -1;

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }

            // By now, both communication and processing of the interior are done.
            // Perform coarsening on the halo.
#pragma omp for schedule(dynamic)
            for(size_t i=0; i<global_halo_size; ++i){
              index_t rm_vertex = global_halo[i];
              index_t target_vertex = dynamic_vertex[rm_vertex];

              // Mark neighbours for re-evaluation.
              // Two threads might be marking the same vertex at the same time.
              // This is a race condition which doesn't cause any trouble.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt)
                dynamic_vertex[*jt] = -2;

              // Mark rm_vertex as non-active.
              dynamic_vertex[rm_vertex] = -1;
              colouring->node_colour[rm_vertex] = -1;

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }

#pragma omp single
            {
              global_interior_size = 0;
              global_halo_size = 0;
            }

            _mesh->commit_deferred(tid);
            _surface->commit_deferred(tid);
#pragma omp barrier
          }
#endif
        }else{
          for(int set_no=0; set_no<colouring->nsets; ++set_no){
#pragma omp for schedule(dynamic, 16)
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
        }

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
  
  /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
   */
  int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const{
    // Cannot delete if already deleted.
    if(_mesh->NNList[rm_vertex].empty())
      return -1;

    // If this is a corner-vertex then cannot collapse;
    if(_surface->is_corner_vertex(rm_vertex))
      return -1;

    // If this is not owned then return -1.
    if(!_mesh->is_owned_node(rm_vertex))
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

      // Find the elements what will be collapsed.
      std::set<index_t> collapsed_elements;
      std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                       _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                       std::inserter(collapsed_elements,collapsed_elements.begin()));
      
      // Check volume/area of new elements.
      for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
        if(collapsed_elements.count(*ee))
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

  struct msg_block{
    /* List of vertices (gnn) adjacent to rm_vertex but not visible by the other MPI process. For each vertex
     * we also send information about its coordinates, the metric at the vertex and the vertex's owner.
     */
    std::vector<index_t> gnn;
    std::vector<int> owner;
    std::vector<real_t> coords;
    std::vector<float> metric;

    //Each element eid={v0, v1, v2} is sent using pairs (gnn, owner).
    std::vector<index_t> elements;
    std::vector<int> elements_owner;

    // Same as elements: If any of the element's edges are on the surface, send the facet to the neighbour.
    std::vector<index_t> facets;
    std::vector<int> facets_owner;
    std::vector<int> boundary_ids;
    std::vector<int> coplanar_ids;
  };

  void marshal_data(index_t rm_vertex, index_t target_vertex, std::vector< std::vector<int> >& local_buffers,
      std::vector< std::vector<index_t> >& sent){
    std::set<int> seen_by;

    // Find who sees rm_vertex
    index_t rm_gnn = _mesh->lnn2gnn[rm_vertex];
    for(int i=0; i<nprocs; ++i)
      if(_mesh->send_map[i].count(rm_gnn) > 0)
        seen_by.insert(i);

    for(typename std::set<int>::const_iterator proc=seen_by.begin(); proc!=seen_by.end(); ++proc){
      msg_block msg;

      assert(_mesh->send_map[*proc].count(rm_gnn)>0);
      assert(_mesh->node_owner[rm_vertex] == rank);

      // Set up a vertex set and an element set of all neighbours
      // of rm_vertex which are invisible to *proc.
      std::set<index_t> inv_vertices, inv_elements;

      // Fill the set of invisible vertices with all neighbours which are not in _mesh->send[*proc].
      for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[rm_vertex].begin();
          neigh!=_mesh->NNList[rm_vertex].end(); ++neigh)
        if(_mesh->send_map[*proc].count(_mesh->lnn2gnn[*neigh]) == 0)
          inv_vertices.insert(*neigh);

      // Visit all adjacent elements. For each element, if one of its vertices
      // belongs to *proc, then the whole element is visible by *proc, so remove
      // any vertices which might be in the set.
      for(typename std::set<index_t>::const_iterator ele = _mesh->NEList[rm_vertex].begin();
          ele != _mesh->NEList[rm_vertex].end(); ++ele){
        const index_t *n = _mesh->get_element(*ele);

        bool visible = false;

        for(size_t i=0; i<nloc; ++i)
          if(_mesh->node_owner[n[i]] == *proc){
            visible = true;
            break;
          }

        // If the element is visible, then remove all 3 vertices from the set.
        if(visible)
          for(size_t i=0; i<nloc; ++i)
            inv_vertices.erase(n[i]);
        // Otherwise, add the element to the set of invisible elements.
        else
          inv_elements.insert(*ele);
      }

      // By now, the invisible set contains only vertices which are truly
      // invisible to MPI process *proc, so we will arrange to send them.
      for(typename std::set<index_t>::const_iterator v=inv_vertices.begin(); v!=inv_vertices.end(); ++v){
        msg.gnn.push_back(_mesh->lnn2gnn[*v]);
        msg.owner.push_back(_mesh->node_owner[*v]);

        const real_t *x = _mesh->get_coords(*v);
        msg.coords.push_back(x[0]);
        msg.coords.push_back(x[1]);

        const float *m = _mesh->get_metric(*v);
        msg.metric.push_back(m[0]);
        msg.metric.push_back(m[1]);
        msg.metric.push_back(m[2]);

        // Append this vertex to the local (per OpenMP thread) list of sent vertices
        sent[*proc].push_back(*v);
        assert(_mesh->send_map[*proc].count(_mesh->lnn2gnn[*v])==0);
      }

      // Pack invisible elements.
      for(typename std::set<index_t>::const_iterator ele=inv_elements.begin(); ele!=inv_elements.end(); ++ele){
        const index_t *n = _mesh->get_element(*ele);

        // Push back to the element list
        for(size_t i=0; i<nloc; ++i){
          msg.elements.push_back(_mesh->lnn2gnn[n[i]]);
          msg.elements_owner.push_back(_mesh->node_owner[n[i]]);
        }

        // And also check whether any of the element's edges are on the surface.
        std::vector<int> lfacets;
        _surface->find_facets(n, lfacets);

        for(size_t i=0; i<lfacets.size(); ++i){
          // Push back surface vertices
          const int *sn=_surface->get_facet(lfacets[i]);
          for(size_t j=0;j<snloc;j++){
            msg.facets.push_back(_mesh->lnn2gnn[sn[j]]);
            msg.facets_owner.push_back(_mesh->node_owner[sn[j]]);
          }

          msg.boundary_ids.push_back(_surface->get_boundary_id(lfacets[i]));
          msg.coplanar_ids.push_back(_surface->get_coplanar_id(lfacets[i]));
        }
      }

      // Serialise message and append it to local_buffer[*proc]

      // Append rm_vertex
      local_buffers[*proc].push_back(_mesh->lnn2gnn[rm_vertex]);
      // Append the number of invisible neighbours.
      local_buffers[*proc].push_back(msg.gnn.size());
      // Append gnn, owner, coords, metric for each invisible neighbour via int's
      for(size_t i=0; i<msg.gnn.size(); ++i){
        std::vector<int> ivertex(node_package_int_size);

        index_t *rgnn = (index_t *) &ivertex[0];
        int *rowner = (int *) &ivertex[idx_owner];
        real_t *rcoords = (real_t *) &ivertex[idx_coords];
        float *rmetric = (float *) &ivertex[idx_metric];

        *rgnn = msg.gnn[i];
        *rowner = msg.owner[i];

        rcoords[0] = msg.coords[2*i];
        rcoords[1] = msg.coords[2*i+1];

        rmetric[0] = msg.metric[3*i];
        rmetric[1] = msg.metric[3*i+1];
        rmetric[2] = msg.metric[3*i+2];

        local_buffers[*proc].insert(local_buffers[*proc].end(), ivertex.begin(), ivertex.end());
      }

      // Append target vertex, its owner and its colour.
      local_buffers[*proc].push_back(_mesh->lnn2gnn[target_vertex]);
      local_buffers[*proc].push_back(_mesh->node_owner[target_vertex]);
      local_buffers[*proc].push_back(colouring->node_colour[target_vertex]);
      assert(_mesh->node_owner[target_vertex] == rank || colouring->node_colour[target_vertex] == -1);

      // Append elements
      local_buffers[*proc].push_back(inv_elements.size());
      for(size_t i=0; i<nloc*inv_elements.size(); ++i){
        local_buffers[*proc].push_back(msg.elements[i]);
        local_buffers[*proc].push_back(msg.elements_owner[i]);
      }

      // Append facets
      local_buffers[*proc].push_back(msg.boundary_ids.size());
      for(size_t i=0; i<msg.boundary_ids.size(); ++i){
        for(size_t j=0;j<snloc;j++){
          local_buffers[*proc].push_back(msg.facets[2*i+j]);
          local_buffers[*proc].push_back(msg.facets_owner[2*i+j]);
        }

        local_buffers[*proc].push_back(msg.boundary_ids[i]);
        local_buffers[*proc].push_back(msg.coplanar_ids[i]);
      }
    }
  }

  void unmarshal_data(std::vector< std::vector<int> >& recv_buffer, index_t *halo, size_t& halo_size, int current_colour,
      std::vector< std::map<index_t, index_t> >& sent_vertices, std::vector< std::map<index_t, index_t> >& recv_vertices){
    for(int proc=0;proc<nprocs;proc++){
      if(recv_buffer[proc].size()==0)
        continue;

      int *buffer = &recv_buffer[proc][0];
      size_t original_part_size = buffer[0];
      size_t loc=1;

      // Part 1: append new vertices, elements and facets to the mesh.
      while(loc < original_part_size){
        // Find which vertex we are talking about and colour it
        index_t rm_gnn = buffer[loc++];
        assert(_mesh->recv_map[proc].count(rm_gnn) > 0);
        index_t rm_vertex = _mesh->recv_map[proc][rm_gnn];
        colouring->node_colour[rm_vertex] = current_colour;

        // Add rm_vertex to the global halo worklist
        halo[halo_size++] = rm_vertex;

        // Unpack new vertices
        int nInvisible = buffer[loc++];

        for(int i=0; i<nInvisible; ++i){
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
            recv_vertices[owner][gnn] = new_lnn;
          }

          loc += node_package_int_size;
        }

        // Find target vertex
        index_t target_gnn = buffer[loc++];
        int target_owner = buffer[loc++];
        index_t target_vertex = -1;
        if(target_owner == rank){
          assert(_mesh->send_map[proc].count(target_gnn) > 0);
          target_vertex = _mesh->send_map[proc][target_gnn];
        }else{
          assert(_mesh->recv_map[target_owner].count(target_gnn) > 0);
          target_vertex = _mesh->recv_map[target_owner][target_gnn];
        }
        assert(_mesh->node_owner[target_vertex] == target_owner);
        assert(target_vertex >= 0);
        dynamic_vertex[rm_vertex] = target_vertex;

        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[rm_vertex].begin(), _mesh->NNList[rm_vertex].end(), target_vertex);

        if(it == _mesh->NNList[rm_vertex].end()){
          _mesh->NNList[rm_vertex].push_back(target_vertex);
          _mesh->NNList[target_vertex].push_back(rm_vertex);
        }

        /* Have a look at its colour. If the sender is not the owner of target_vertex,
         * then he considers it to be uncoloured, so we must do the same (un-colour
         * it). If the sender is the owner of target_vertex, then this colour is
         * valid and if it clashes with the colour of any of our vertices, then
         * we must un-colour those vertices of ours.
         */
        int target_colour = buffer[loc++];
        if(_mesh->node_owner[target_vertex] != proc){
          assert(target_colour == -1);
          colouring->node_colour[target_vertex] = -1;
        }else if(target_colour >= 0){
          for(typename std::vector<index_t>::const_iterator it = _mesh->NNList[rm_vertex].begin();
              it != _mesh->NNList[rm_vertex].end(); ++it)
            if(_mesh->node_owner[*it] == rank)
              if(colouring->node_colour[*it] == target_colour)
                colouring->node_colour[*it] = -1;
        }

        // Unpack new elements
        int nElements = buffer[loc++];

        for(int i=0; i<nElements; ++i){
          index_t ele[] = {-1, -1, -1};

          for(size_t j=0; j<nloc; ++j){
            index_t gnn = buffer[loc++];
            int owner = buffer[loc++];
            assert(_mesh->recv_map[owner].count(gnn)>0);
            ele[j] = _mesh->recv_map[owner][gnn];
            assert(ele[j] >= 0);
          }

          /* Check whether this element is already visible. This case can occur if an
           * element used to cross the halo and after a vertex collapse it was constrained
           * exclusively to one partition. In this case, the sender believes that the
           * element is invisible to us, because all its vertices belong to the sender.
           */
          std::set<index_t> intersection;
          std::set_intersection(_mesh->NEList[ele[0]].begin(), _mesh->NEList[ele[0]].end(),
              _mesh->NEList[ele[1]].begin(), _mesh->NEList[ele[1]].end(),
              std::inserter(intersection, intersection.begin()));
          std::set<index_t> common_element;
          std::set_intersection(_mesh->NEList[ele[2]].begin(), _mesh->NEList[ele[2]].end(),
              intersection.begin(), intersection.end(),
              std::inserter(common_element, common_element.begin()));

          if(!common_element.empty())
            continue;

          index_t eid = _mesh->append_element(ele);

          /* Update NNList and NEList. Updates are thread-safe, because they pertain
           * to recv_halo vertices only, which are not touched by the rest of the
           * OpenMP threads that are processing the interior of this MPI partition.
           */
          for(size_t j=0; j<nloc; ++j){
            _mesh->NEList[ele[j]].insert(eid);

            for(size_t k=j+1; k<nloc; ++k){
              typename std::vector<index_t>::iterator it;
              it = std::find(_mesh->NNList[ele[j]].begin(), _mesh->NNList[ele[j]].end(), ele[k]);

              if(it == _mesh->NNList[ele[j]].end()){
                _mesh->NNList[ele[j]].push_back(ele[k]);
                _mesh->NNList[ele[k]].push_back(ele[j]);
              }
            }
          }
        }

        // Unpack new facets
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
    }
  }

  inline virtual index_t is_dynamic(index_t vid){
    return dynamic_vertex[vid];
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  Colouring<real_t, index_t> *colouring;
  
  size_t nnodes_reserve;
  index_t *dynamic_vertex;

  real_t _L_low, _L_max;

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;

  const static size_t node_package_int_size = 1 + (sizeof(index_t) +
                                                   ndims*sizeof(real_t) +
                                                   msize*sizeof(float)) / sizeof(int);
  const static size_t idx_owner = sizeof(index_t) / sizeof(int);
  const static size_t idx_coords = idx_owner + 1;
  const static size_t idx_metric = idx_coords + ndims*sizeof(real_t) / sizeof(int);

  int nprocs, rank, nthreads;
};

#endif
