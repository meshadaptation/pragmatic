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

#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D mesh coarsening.
 *
 */

template<typename real_t, typename index_t> class Coarsen2D{
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
    node_colour = NULL;
    node_hash = NULL;
  }
  
  /// Default destructor.
  ~Coarsen2D(){
    if(property!=NULL)
      delete property;

    if(dynamic_vertex!=NULL){
      delete[] dynamic_vertex;
      delete[] node_colour;
      delete[] node_hash;
    }
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
        delete[] node_colour;
        delete[] node_hash;
      }
      
      dynamic_vertex = new index_t[nnodes_reserve];
      node_colour = new int[nnodes_reserve];
      node_hash = new size_t[nnodes_reserve];
    }
    
    /* dynamic_vertex[i] >= 0 :: target to collapse node i
       dynamic_vertex[i] = -1 :: node inactive (deleted/locked)
       dynamic_vertex[i] = -2 :: recalculate collapse - this is how propagation is implemented
    */

    // Total number of active vertices. Used to break the infinite loop.
    size_t total_active;

    // Set of all dynamic vertices. We pre-allocate the maximum capacity
    // that may be needed. Accessed using active sub-mesh vertex IDs.
    index_t *GlobalActiveSet = new index_t[nnodes_reserve];
    size_t GlobalActiveSet_size;

    // Subset of GlobalActiveSet: it contains the vertices which the colouring
    // algorithm will try to colour in one round. UncolouredActiveSet[i] stores
    // the active sub-mesh ID of an uncoloured vertex, i.e. the vertex at
    // UncolouredActiveSet[i] has neighbours=subNNList[UncolouredActiveSet[i]]
    // and its regular ID is vid = GlobalActiveSet[UncolouredActiveSet[i]].
    index_t *UncolouredActiveSet = new index_t[nnodes_reserve];
    size_t UncolouredActiveSet_size;

    // Subset of UncolouredActiveSet: it contains the vertices which couldn't be
    // coloured in that round and are left for the next round. Like UncolouredActiveSet,
    // it stores active sub-mesh IDs. At the end of each colouring round,
    // the pointers to UncolouredActiveSet and NextRoundActive set are swapped.
    index_t *NextRoundActiveSet = new index_t[nnodes_reserve];
    size_t NextRoundActiveSet_size;

    // NNList of the active sub-mesh. Accessed using active sub-mesh vertex IDs.
    // Each vector contains regular vertex IDs.
    std::vector<index_t> **subNNList = new std::vector<index_t>* [nnodes_reserve];

    /* It's highly unlikely that more than a dozen colours will be needed, let's
     * allocate 256 just in case. They are just pointers, so there is no memory
     * footprint. The array for each independent set will be allocated on demand.
     * Independent sets contain regular vertex IDs.
     */
    index_t **independent_sets = new index_t* [max_colours];
    std::vector<size_t> ind_set_size(max_colours, 0);

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
        node_colour[i] = -1;
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
          GlobalActiveSet_size = 0;
          total_active = 0;
        }

        /* Initialise list of vertices to be coarsened. A dynamic
           schedule is used as previous coarsening may have introduced
           significant gaps in the node list. This could lead to
           significant load imbalance if a static schedule was used. */
#pragma omp for schedule(dynamic,32) reduction(+:total_active)
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(dynamic_vertex[i] == -2)
            dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);

          if(dynamic_vertex[i]>=0){
            active_set.push_back(i);
            ++total_active;
          }
        }

#ifdef HAVE_MPI
        if(nprocs>1)
          MPI_Allreduce(MPI_IN_PLACE, &total_active, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#endif

        if(total_active == 0)
          break;

        size_t pos;
#pragma omp atomic capture
        {
          pos = GlobalActiveSet_size;
          GlobalActiveSet_size += active_set.size();
        }

        memcpy(&GlobalActiveSet[pos], &active_set[0], active_set.size() * sizeof(index_t));

#ifdef HAVE_MPI
        if(nprocs>1){
          /* Each MPI process needs to know which vertices in recv_halo are marked
           * as dynamic by their owner in order to colour its own vertices accordingly.
           * We could do that by exchanging halo data in terms of dynamic_vertex,
           * i.e. call _mesh->halo_update(dynamic_vertex, 1). Instead, in order
           * to avoid one MPI communication, we can just assume that all vertices in
           * recv_halo are dynamic. The GlobalActiveSet has already been constructed,
           * so marking non-owned vertices as dynamic does not do any harm, as they
           * will not be put into any independent set, therefore they won't be processed.
           */
#pragma omp single nowait
          {
            for(typename std::set<index_t>::const_iterator it=_mesh->recv_halo.begin(); it!=_mesh->recv_halo.end(); ++it)
              // Assign an arbitrary value.
              dynamic_vertex[*it] = 0;
          }
        }
#endif

        // Construct the node adjacency list for the active sub-mesh.
#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
        for(size_t i=0; i<GlobalActiveSet_size; ++i){
          index_t vid = GlobalActiveSet[i];
          std::vector<index_t> *dynamic_NNList = new std::vector<index_t>;

          for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[vid].begin(); it!=_mesh->NNList[vid].end(); ++it)
            if(dynamic_vertex[*it]>=0){
              dynamic_NNList->push_back(*it);
            }

          subNNList[i] = dynamic_NNList;
          node_colour[vid] = -1; // Reset colour
        }

        /**********************************************
         * Active sub-mesh is ready, let's colour it. *
         **********************************************/

        // Initialise the uncoloured active set: it contains all vertices of the
        // GlobalActiveSet. The first thread to reach this point will do the copy.
#pragma omp single
        {
          UncolouredActiveSet_size = GlobalActiveSet_size;
          NextRoundActiveSet_size = 0;
        }

        // Reset vertex hashes, starting with global IDs - this is important for MPI!
        // Also, initialise UncolouredActiveSet.
#pragma omp for schedule(static)
        for(size_t i=0; i<GlobalActiveSet_size; ++i){
          index_t vid = GlobalActiveSet[i];
          node_hash[vid] = _mesh->lnn2gnn[vid];
          UncolouredActiveSet[i] = i;
        }

        // At the end of colouring, this variable will be equal to the total number of colours used.
        int colour = 0;

        // Local independent sets.
        std::vector< std::vector<index_t> > local_ind_sets(max_colours);
        // idx[i] stores the index in independent_sets[i] at which this thread
        // will copy the contents of its local independent set local_ind_sets[i].
        std::vector<size_t> idx(max_colours);

        // Local list of vertices which are left for the next round.
        std::vector<index_t> uncoloured;
        uncoloured.reserve(GlobalActiveSet_size/nthreads);

        while(UncolouredActiveSet_size > 0){
          // Calculate the n-th order vertex hashes (n being the iteration count of this while loop).
#pragma omp for schedule(static)
          for(size_t i=0; i<GlobalActiveSet_size; ++i){
            index_t vid = GlobalActiveSet[i];
            node_hash[vid] = _mesh->hash(node_hash[vid]);
          }

#pragma omp for schedule(dynamic,16) nowait
          for(size_t i=0; i<UncolouredActiveSet_size; ++i){
            index_t vid = GlobalActiveSet[UncolouredActiveSet[i]];
            assert(node_colour[vid] == -1);

            // A vertex is eligible to be coloured in this round only if it's got
            // the highest or the lowest hash among all uncoloured neighbours.

            // Check whether this is the highest hash.
            bool eligible = true;

            for(size_t j=0; j<subNNList[UncolouredActiveSet[i]]->size(); ++j){
              index_t neigh = subNNList[UncolouredActiveSet[i]]->at(j);
              assert(dynamic_vertex[neigh] >= 0);
              if(node_hash[vid] < node_hash[neigh]){
                eligible = false;
                break;
              }
            }

            if(eligible){
              node_colour[vid] = colour;
              local_ind_sets[colour].push_back(vid);
              continue;
            }

            // Check whether this is the lowest hash.
            eligible = true;

            for(size_t j=0; j<subNNList[UncolouredActiveSet[i]]->size(); ++j){
              index_t neigh = subNNList[UncolouredActiveSet[i]]->at(j);
              assert(dynamic_vertex[neigh] >= 0);
              if(node_hash[vid] > node_hash[neigh]){
                eligible = false;
                break;
              }
            }

            if(eligible){
              node_colour[vid] = colour+1;
              local_ind_sets[colour+1].push_back(vid);
              continue;
            }

            // If the vertex was not eligible for colouring
            // in this round, advance it to the next round.
            uncoloured.push_back(UncolouredActiveSet[i]);
          }

          // Copy uncoloured vertices into NextRoundActiveSet
          // (which will become the next round's UncolouredActiveSet).
          size_t pos;
#pragma omp atomic capture
          {
            pos = NextRoundActiveSet_size;
            NextRoundActiveSet_size += uncoloured.size();
          }

          memcpy(&NextRoundActiveSet[pos], &uncoloured[0], uncoloured.size() * sizeof(index_t));
          uncoloured.clear();

          // Capture and increment the index in independent_sets[colour] and independent_sets[colour+1]
          // at which the two local independent sets will be copied later, after memory for the global
          // independent sets will have been allocated.
#pragma omp atomic capture
          {
            idx[colour] = ind_set_size[colour];
            ind_set_size[colour] += local_ind_sets[colour].size();
          }

#pragma omp atomic capture
          {
            idx[colour+1] = ind_set_size[colour+1];
            ind_set_size[colour+1] += local_ind_sets[colour+1].size();
          }

          colour += 2;

          // Swap UncolouredActiveSet and NextRoundActiveSet
#pragma omp barrier
#pragma omp single
          {
            index_t *swap = NextRoundActiveSet;
            NextRoundActiveSet = UncolouredActiveSet;
            UncolouredActiveSet = swap;
            UncolouredActiveSet_size = NextRoundActiveSet_size;
            NextRoundActiveSet_size = 0;
          }
        }

        // Total number of independent sets = colour
        int nsets = colour;
        assert(nsets <= max_colours);

        // Allocate memory for the global independent sets.
#pragma omp for schedule(dynamic)
        for(int set_no=0; set_no<nsets; ++set_no)
          independent_sets[set_no] = new index_t[ind_set_size[set_no]];

        // Copy local independent sets into the global structure.
        for(int set_no=0; set_no<nsets; ++set_no)
          memcpy(&independent_sets[set_no][idx[set_no]], &local_ind_sets[set_no][0],
              local_ind_sets[set_no].size() * sizeof(index_t));

        // Thanks to multi-hash colouring, there is no need to exchange colours
        // with neighbouring MPI processes. Starting with global IDs, every vertex
        // is coloured without caring what the colour of neighbouring vertices is.

#pragma omp barrier

        /********************
         * End of colouring *
         ********************/

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
        for(int set_no=0; set_no<nsets; ++set_no){
          if(nprocs>1){
#ifdef HAVE_MPI
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
            for(size_t i=0; i<ind_set_size[set_no]; ++i){
              index_t rm_vertex = independent_sets[set_no][i];
              assert((size_t) rm_vertex < _mesh->NNodes);

              // If the node has been un-coloured, skip it.
              if(node_colour[rm_vertex] < 0)
                continue;

              assert(node_colour[rm_vertex] == set_no);

              /* If this rm_vertex is marked for re-evaluation, it means that the
               * local neighbourhood has changed since coarsen_identify_kernel was
               * called for this vertex. Call coarsen_identify_kernel again.
               * Obviously, this check is redundant for set_no=0.
               */
              if(dynamic_vertex[rm_vertex] == -2)
                dynamic_vertex[rm_vertex] = coarsen_identify_kernel(rm_vertex, L_low, L_max);

              if(dynamic_vertex[rm_vertex] < 0)
                continue;

              // Separate interior vertices from halo vertices.
              if(_mesh->is_halo_node(rm_vertex)){
                // If rm_vertex is a halo vertex, then marshal necessary data.
                halo_vertices.push_back(rm_vertex);
                marshal_data(rm_vertex, local_buffers, local_sent);
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

                  if(owner==rank)
                    sent_vertices[i][gnn] = *it;
                  else{
                    // Send message (idx, proc): Tell the owner that we have sent
                    // the vertex stored at _mesh->recv[owner][idx] to process proc.
                    int idx = _mesh->recv_map[owner][*it];
                    message_extensions[owner].push_back(idx);
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

              unmarshal_data(recv_buffer, global_halo, global_halo_size, sent_vertices, recv_vertices);

              for(int i=0; i<nprocs; ++i){
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

              // No matter whether rm_vertex will be deleted or not, its colour must be reset.
              node_colour[rm_vertex] = -1;

              // Mark neighbours for re-evaluation.
              // Two threads might be marking the same vertex at the same time.
              // This is a race condition which doesn't cause any trouble.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt)
                dynamic_vertex[*jt] = -2;

              // Mark rm_vertex as non-active.
              dynamic_vertex[rm_vertex] = -1;

              // Un-colour target_vertex if its colour clashes with any of its new neighbours.
              // There is race condition here, but it doesn't do any harm.
              if(node_colour[target_vertex] >= 0){
                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  if(*jt != target_vertex){
                    if(node_colour[*jt] == node_colour[target_vertex]){
                      node_colour[target_vertex] = -1;
                      break;
                    }
                  }
                }
              }

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }

            // By now, both communication and processing of the interior are done.
            // Perform coarsening on the halo.
#pragma omp for schedule(dynamic)
            for(size_t i=0; i<global_halo_size; ++i){
              index_t rm_vertex = global_halo[i];
              index_t target_vertex = dynamic_vertex[rm_vertex];

              // No matter whether rm_vertex will be deleted or not, its colour must be reset.
              node_colour[rm_vertex] = -1;

              // Mark neighbours for re-evaluation.
              // Two threads might be marking the same vertex at the same time.
              // This is a race condition which doesn't cause any trouble.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt)
                dynamic_vertex[*jt] = -2;

              // Mark rm_vertex as non-active.
              dynamic_vertex[rm_vertex] = -1;

              // Un-colour target_vertex if its colour clashes with any of its new neighbours.
              // There is race condition here, but it doesn't do any harm.
              if(node_colour[target_vertex] >= 0){
                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  if(*jt != target_vertex){
                    if(node_colour[*jt] == node_colour[target_vertex]){
                      node_colour[target_vertex] = -1;
                      break;
                    }
                  }
                }
              }

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }

#pragma omp single
            {
              global_interior_size = 0;
              global_halo_size = 0;
            }
#endif
          }else{
#pragma omp for schedule(dynamic,32)
            for(size_t i=0; i<ind_set_size[set_no]; ++i){
              index_t rm_vertex = independent_sets[set_no][i];
              assert((size_t) rm_vertex < NNodes);

              // If the node has been un-coloured, skip it.
              if(node_colour[rm_vertex] < 0)
                continue;

              assert(node_colour[rm_vertex] == set_no);

              // No matter whether rm_vertex will be deleted or not, its colour must be reset.
              node_colour[rm_vertex] = -1;

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
              // Two threads might be marking the same vertex at the same time.
              // This is a race condition which doesn't cause any trouble.
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  dynamic_vertex[*jt] = -2;
              }

              // Mark rm_vertex as non-active.
              dynamic_vertex[rm_vertex] = -1;

              // Un-colour target_vertex if its colour clashes with any of its new neighbours.
              // There is race condition here, but it doesn't do any harm.
              if(node_colour[target_vertex] >= 0){
                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
                  if(*jt != target_vertex){
                    if(node_colour[*jt] == node_colour[target_vertex]){
                      node_colour[target_vertex] = -1;
                      break;
                    }
                  }
                }
              }

              // Coarsen the edge.
              coarsen_kernel(rm_vertex, target_vertex, tid);
            }
          }

          _mesh->commit_deferred(tid);
          _surface->commit_deferred(tid);
#pragma omp barrier
        }

#pragma omp for schedule(static) nowait
        for(int set_no=0; set_no<nsets; ++set_no){
          delete[] independent_sets[set_no];
          ind_set_size[set_no] = 0;
        }

#pragma omp for schedule(static)
        for(size_t i=0; i<GlobalActiveSet_size; ++i)
          delete subNNList[i];

      }while(true);
    }

    delete[] GlobalActiveSet;
    delete[] UncolouredActiveSet;
    delete[] NextRoundActiveSet;
    delete[] subNNList;
    delete[] independent_sets;

#ifdef HAVE_MPI
    delete[] global_interior;
    delete[] global_halo;

    if(nprocs>1)
      _mesh->trim_halo();
#endif

    return;
  }

 private:
  
  /*! Kernel for identifying what if any vertex rm_vertex should collapse onto.
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
    
    // If we're checked all edges and none is collapsible then return.
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

    // For all adjacent elements, replace rm_vertex with target_vertex in ENList.
    for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
      for(size_t i=0;i<nloc;i++){
        if(_mesh->_ENList[nloc*(*ee)+i]==rm_vertex){
          _mesh->_ENList[nloc*(*ee)+i]=target_vertex;
          break;
        }
      }
      
      // Add element to target_vertex' NEList.
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
    if(_surface->contains_node(rm_vertex)&&_surface->contains_node(target_vertex)){
      _surface->collapse(rm_vertex, target_vertex, tid);
    }

    _mesh->erase_vertex(rm_vertex);
  }

  struct msg_block{
    // Index of rm_vertex in _mesh->send[receiver]/_mesh->recv[sender]
    int pos;

    /* List of vertices (gnn) adjacent to rm_vertex but not visible by the
     * other MPI process. For each vertex, we also send information about
     * its coordinates, the metric at the vertex and the vertex's owner.
     */
    std::vector<index_t> gnn;
    std::vector<int> owner;
    std::vector<int> colour;
    std::vector<real_t> coords;
    std::vector<float> metric;

    // Target vertex onto which rm_vertex will collapse.
    // We send target's gnn if target is invisible, position in _mesh->send[] otherwise.
    std::vector<index_t> target;

    /* Each element eid={v0, v1, v2} is sent in the following form:
     * For vertices which are not visible to the other MPI process, we send their
     * global IDs. Vertices which are visible are stored using their position in
     * _mesh->send[receiver]/_mesh->recv[sender]. To distinguish between global
     * IDs and positions, the latter are sent negated.
     */
    std::vector<index_t> elements;

    // Same as elements: If any of the element's edges are on the surface,
    // send the facet to the neighbour.
    std::vector<index_t> facets;
    std::vector<int> boundary_ids;
    std::vector<int> coplanar_ids;
  };

  void marshal_data(index_t rm_vertex, std::vector< std::vector<int> >& local_buffers,
      std::vector< std::vector<index_t> >& sent){
    std::set<int> seen_by;

    // Find who sees rm_vertex
    for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[rm_vertex].begin(); it!=_mesh->NNList[rm_vertex].end(); ++it)
      seen_by.insert(_mesh->node_owner[*it]);

    seen_by.erase(rank);

    for(typename std::set<int>::const_iterator proc=seen_by.begin(); proc!=seen_by.end(); ++proc){
      msg_block msg;

      // Find the position in _mesh->send[*proc] of rm_vertex. Using this position,
      // the receiver can find immediately in his _mesh->recv[i] which vertex we
      // are talking about. This saves us from having to maintain a gnn2lnn map.
      assert(_mesh->send_map[*proc].count(rm_vertex)>0);
      msg.pos = _mesh->send_map[*proc][rm_vertex];

      // Set up a vertex set and and element set of all neighbours
      // of rm_vertex which are invisible to *proc.
      std::set<index_t> inv_vertices, inv_elements;

      // Fill the set of invisible vertices with all neighbours.
      for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[rm_vertex].begin();
          neigh!=_mesh->NNList[rm_vertex].end(); ++neigh)
        if(_mesh->node_owner[*neigh] != *proc)
          inv_vertices.insert(*neigh);

      // Visit all adjacent elements. For each element, if one of its vertices
      // belongs to *proc, then the whole element is visible by *proc, so remove
      // any vertices which might be in the set.
      for(typename std::set<index_t>::const_iterator ele = _mesh->NEList[rm_vertex].begin();
          ele != _mesh->NEList[rm_vertex].end(); ++ele){
        const index_t *n = _mesh->get_element(*ele);

        bool visible =false;

        for(size_t i=0; i<nloc; ++i)
          if(_mesh->node_owner[n[i]] == *proc){
            visible= true;
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
        msg.colour.push_back(node_colour[*v]);

        const real_t *x = _mesh->get_coords(*v);
        msg.coords.push_back(x[0]);
        msg.coords.push_back(x[1]);

        const float *m = _mesh->get_metric(*v);
        msg.metric.push_back(m[0]);
        msg.metric.push_back(m[1]);
        msg.metric.push_back(m[2]);

        // Append this vertex to the local (per OpenMP thread) list of sent vertices
        sent[*proc].push_back(*v);
      }

      // Target vertex
      protocol_pack(dynamic_vertex[rm_vertex], *proc, msg.target);

      // Pack invisible elements.
      for(typename std::set<index_t>::const_iterator ele=inv_elements.begin(); ele!=inv_elements.end(); ++ele){
        const index_t *n = _mesh->get_element(*ele);

        // Push back to the element list using the vertex-encoding protocol
        for(size_t i=0; i<nloc; ++i)
          protocol_pack(n[i], *proc, msg.elements);

        // And also check whether any of the element's edges are on the surface.
        std::vector<int> lfacets;
        _surface->find_facets(n, lfacets);

        if(lfacets.size()>0){
          assert(lfacets.size()==1);

          const int *sn=_surface->get_facet(lfacets[0]);
          for(size_t i=0;i<snloc;i++){
            // Use the vertex-encoding protocol for surface vertices.
            protocol_pack(sn[i], *proc, msg.facets);
          }

          msg.boundary_ids.push_back(_surface->get_boundary_id(lfacets[0]));
          msg.coplanar_ids.push_back(_surface->get_coplanar_id(lfacets[0]));
        }
      }

      // Serialise message and append it to local_buffer[*proc]

      // Append the position of rm_vertex in _mesh->send[*proc]
      local_buffers[*proc].push_back(msg.pos);
      // Append the number of invisible neighbours.
      local_buffers[*proc].push_back(msg.gnn.size());
      // Append gnn, owner, coords, metric for each invisible neighbour via int's
      for(size_t i=0; i<msg.gnn.size(); ++i){
        std::vector<int> ivertex(node_package_int_size);

        index_t *rgnn = (index_t *) &ivertex[0];
        int *rowner = (int *) &ivertex[idx_owner];
        int *colour = (int *) &ivertex[idx_colour];
        real_t *rcoords = (real_t *) &ivertex[idx_coords];
        float *rmetric = (float *) &ivertex[idx_metric];

        *rgnn = msg.gnn[i];
        *rowner = msg.owner[i];
        *colour = msg.colour[i];

        rcoords[0] = msg.coords[2*i];
        rcoords[1] = msg.coords[2*i+1];

        rmetric[0] = msg.metric[3*i];
        rmetric[1] = msg.metric[3*i+1];
        rmetric[2] = msg.metric[3*i+2];

        local_buffers[*proc].insert(local_buffers[*proc].end(), ivertex.begin(), ivertex.end());
      }

      // Append target vertex
      local_buffers[*proc].insert(local_buffers[*proc].end(), msg.target.begin(), msg.target.end());

      // Append elements
      local_buffers[*proc].push_back(inv_elements.size());
      local_buffers[*proc].insert(local_buffers[*proc].end(), msg.elements.begin(), msg.elements.end());

      // Append facets
      local_buffers[*proc].push_back(msg.boundary_ids.size());
      for(size_t i=0, j=0; i<msg.boundary_ids.size(); ++i){
        int m = (msg.facets[j] != 3 ? 4 : 6);
        for(int k=0; k<m; ++k)
          local_buffers[*proc].push_back(msg.facets[j++]);

        local_buffers[*proc].push_back(msg.boundary_ids[i]);
        local_buffers[*proc].push_back(msg.coplanar_ids[i]);
      }
    }
  }

  void protocol_pack(index_t vertex, int proc, std::vector<index_t>& message){
    /* The target vertex, as well as vertices of invisible elements, will be sent to process
     * proc as a pair/triplet [flag, ID (, owner)] using the following protocol:
     * flag = 0: vertex is owned by *this* process (the sender) and it is visible by *proc.
     *           ID corresponds to the index in _mesh->send[*proc] at which the vertex is stored.
     * flag = 1: vertex is owned by *this* process (the sender) and it is NOT visible by *proc.
     *           ID corresponds to the global ID of this vertex.
     * flag = 2: vertex is owned by *proc (the receiver).
     *           ID corresponds to the index in _mesh->recv[*proc] at which the vertex is stored.
     * flag = 3: vertex is owned by a third process and we don't know whether it is visible by *proc.
     *           ID corresponds to the global ID of this vertex.
     *           owner corresponds to the owner of this vertex.
     */
    if(_mesh->node_owner[vertex] == rank){
      if(_mesh->send_map[proc].count(vertex) > 0){
        message.push_back(0);
        message.push_back(_mesh->send_map[proc][vertex]);
      }else{
        message.push_back(1);
        message.push_back(_mesh->lnn2gnn[vertex]);
      }
    }else if(_mesh->node_owner[vertex] == proc){
      message.push_back(2);
      message.push_back(_mesh->recv_map[proc][vertex]);
    }else{
      message.push_back(3);
      message.push_back(_mesh->lnn2gnn[vertex]);
      message.push_back(_mesh->node_owner[vertex]);
    }
  }

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
  void protocol_unpack(index_t& vertex, int proc, int *buffer, size_t& loc,
      boost::unordered_map<index_t, index_t>& gnn2lnn){
#else
  void protocol_unpack(index_t& vertex, int proc, int *buffer, size_t& loc,
      std::map<index_t, index_t>& gnn2lnn){
#endif

    int flag = buffer[loc++];
    index_t ID = buffer[loc++];
    int owner;

    switch(flag){
    case 0:
      vertex = _mesh->recv[proc][ID];
      break;
    case 1:
      assert(gnn2lnn.count(ID)>0);
      vertex = gnn2lnn[ID];
      break;
    case 2:
      vertex = _mesh->send[proc][ID];
      break;
    case 3:
      owner = buffer[loc++];
      if(gnn2lnn.count(ID)>0)
        vertex = gnn2lnn[ID];
      else{
        typename std::vector<index_t>::const_iterator it;

        for(it = _mesh->recv[owner].begin(); it != _mesh->recv[owner].end(); ++it)
          if(_mesh->lnn2gnn[*it] == ID){
            vertex = *it;
            break;
          }

        assert(it != _mesh->recv[owner].end());
      }
      break;
    default:
      std::cerr << "ERROR: corrupt vertex encoding in " << __FILE__ << std::endl;
      break;
    }
  }

  void unmarshal_data(std::vector< std::vector<int> >& recv_buffer,
      index_t *halo, size_t& halo_size,
      std::vector< std::map<index_t, index_t> >& sent_vertices,
      std::vector< std::map<index_t, index_t> >& recv_vertices){
    // Create a reverse lookup to map received gnn's back to lnn's.
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    boost::unordered_map<index_t, index_t> gnn2lnn;
#else
    std::map<index_t, index_t> gnn2lnn;
#endif

    for(int proc=0;proc<nprocs;proc++){
      if(recv_buffer[proc].size()==0)
        continue;

      int *buffer = &recv_buffer[proc][0];
      size_t original_part_size = buffer[0];
      size_t loc=1;

      // Part 1: append new vertices, elements and facets to the mesh.
      while(loc < original_part_size){
        // Find which vertex we are talking about
        index_t pos = buffer[loc++];
        index_t rm_vertex = _mesh->recv[proc][pos];

        // Add rm_vertex to the global halo worklist
        halo[halo_size++] = rm_vertex;

        // Unpack new vertices
        int nInvisible = buffer[loc++];

        for(int i=0; i<nInvisible; ++i){
          index_t gnn = *((index_t *) &buffer[loc]);

          // Only append this vertex to the mesh if we haven't received it before.
          if(gnn2lnn.count(gnn) == 0){
            int owner = buffer[loc+idx_owner];
            int colour = buffer[loc+idx_colour];
            real_t *rcoords = (real_t *) &buffer[loc+idx_coords];
            float *rmetric = (float *) &buffer[loc+idx_metric];

            index_t new_lnn = _mesh->append_vertex(rcoords, rmetric);

            gnn2lnn[gnn] = new_lnn;
            _mesh->lnn2gnn[new_lnn] = gnn;
            _mesh->node_owner[new_lnn] = owner;

            assert(owner < nprocs);

            recv_vertices[owner][gnn] = new_lnn;
            node_colour[new_lnn] = colour;
          }

          loc += node_package_int_size;
        }

        // Find target vertex
        index_t target_vertex = -1;
        protocol_unpack(target_vertex, proc, buffer, loc, gnn2lnn);
        assert(target_vertex >= 0);
        dynamic_vertex[rm_vertex] = target_vertex;

        // Unpack new elements
        // An element can only be sent once, so there is no need to check for duplicates
        int nElements = buffer[loc++];

        for(int i=0; i<nElements; ++i){
          index_t ele[] = {-1, -1, -1};

          for(size_t j=0; j<nloc; ++j)
            protocol_unpack(ele[j], proc, buffer, loc, gnn2lnn);

          index_t eid = _mesh->append_element(ele);

          /* Update NNList and NEList. Updates are thread-safe, because they pertain
           * to recv_halo vertices only, which are not touched by the rest of the
           * OpenMP threads who are processing the interior of this MPI partition.
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

          for(size_t j=0; j<snloc; ++j)
            protocol_unpack(facet[j], proc, buffer, loc, gnn2lnn);

          int boundary_id = buffer[loc++];
          int coplanar_id = buffer[loc++];

          // Updates to surface are thread-safe for the
          // same reason as updates to adjacency lists.
          _surface->append_facet(facet, boundary_id, coplanar_id, true);
        }
      }

      assert(loc == original_part_size);

      // Part 2: Look at the extensions. The extension contains pairs (idx, receiver):
      // The sender process proc sent to process receiver information about the vertex
      // which is stored at _mesh->send[proc][idx].
      while(loc < recv_buffer[proc].size()){
        int idx = buffer[loc++];
        int receiver = buffer[loc++];

        index_t vid = _mesh->send[proc][idx];
        index_t gnn = _mesh->lnn2gnn[vid];

        // Now we know that process receiver has appended vertex vid to its halo,
        // so we have to add vid to our _mesh->send[receiver].
        sent_vertices[receiver][gnn] = vid;
      }
    }

    // Update _mesh->send and _mesh->recv.
    for(int i=0; i<nprocs; ++i){
      for(typename std::map<index_t, index_t>::const_iterator it=sent_vertices[i].begin();
        it!=sent_vertices[i].end(); ++it){
        _mesh->send[i].push_back(it->second);
        _mesh->send_map[i][it->second] = _mesh->send[i].size() - 1;
        _mesh->send_halo.insert(it->second);
      }

      for(typename std::map<index_t, index_t>::const_iterator it=recv_vertices[i].begin();
        it!=recv_vertices[i].end(); ++it){
        _mesh->recv[i].push_back(it->second);
        _mesh->recv_map[i][it->second] = _mesh->recv[i].size() - 1;
        _mesh->recv_halo.insert(it->second);
      }
    }
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  
  size_t nnodes_reserve;
  index_t *dynamic_vertex;
  int *node_colour;
  size_t *node_hash;

  real_t _L_low, _L_max;

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;

  const static size_t max_colours = 256;
  const static size_t node_package_int_size = 2 + (sizeof(index_t) +
                                                   ndims*sizeof(real_t) +
                                                   msize*sizeof(float)) / sizeof(int);
  const static size_t idx_owner = sizeof(index_t) / sizeof(int);
  const static size_t idx_colour = idx_owner + 1;
  const static size_t idx_coords = idx_colour + 1;
  const static size_t idx_metric = idx_coords + ndims*sizeof(real_t) / sizeof(int);

  int nprocs, rank, nthreads;
};

#endif
