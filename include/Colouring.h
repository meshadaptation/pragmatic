/*  Copyright (C) 2013 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Georgios Rokos
 *  Software Performance Optimisation Group
 *  Department of Computing
 *  Imperial College London
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

#ifndef COLOURING_H_
#define COLOURING_H_

#include "AdaptiveAlgorithm.h"

template<typename real_t, typename index_t> class Colouring{
public:
  Colouring(Mesh<real_t, index_t> *mesh, AdaptiveAlgorithm<real_t, index_t> *algorithm, size_t nnodes_reserve){
    alg = algorithm;
    _mesh = mesh;

    // We pre-allocate the maximum capacity that may be needed.
    node_colour = new int[nnodes_reserve];

    nsets = 0;
    global_nsets = 0;

    GlobalActiveSet = new index_t[nnodes_reserve];
    GlobalActiveSet_size = 0;

    subNNList = new std::vector<index_t>* [nnodes_reserve];

    /* It's highly unlikely that more than a dozen colours will be needed, let's
     * allocate 256 just in case. They are just pointers, so there is no memory
     * footprint. The array for each independent set will be allocated on demand.
     * Independent sets contain regular vertex IDs.
     */
    independent_sets = new index_t* [max_colours];
    ind_set_size.resize(max_colours, 0);
  }

  ~Colouring(){
    delete[] node_colour;
    delete[] GlobalActiveSet;
    delete[] subNNList;
    delete[] independent_sets;
  }

  void resize(size_t nnodes_reserve){
    delete[] node_colour;
    delete[] GlobalActiveSet;
    delete[] subNNList;

    node_colour = new int[nnodes_reserve];
    GlobalActiveSet = new index_t[nnodes_reserve];
    subNNList = new std::vector<index_t>* [nnodes_reserve];
  }

  void multiHashJonesPlassmann(){
    // Construct the node adjacency list for the active sub-mesh.
#pragma omp for schedule(static)
    for(size_t i=0; i<GlobalActiveSet_size; ++i){
      index_t vid = GlobalActiveSet[i];
      std::vector<index_t> *dynamic_NNList = new std::vector<index_t>;

      for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[vid].begin(); it!=_mesh->NNList[vid].end(); ++it)
        /* Each MPI process needs to know which vertices in recv_halo are marked as
         * dynamic by their owner in order to colour its own vertices accordingly. We
         * could do that by exchanging halo data in terms of dynamic_vertex, i.e. call
         * _mesh->halo_update(dynamic_vertex, 1). Instead, in order to avoid one MPI
         * communication, we can just assume that all vertices in recv_halo are dynamic.
         */
        if(alg->is_dynamic(*it)>=0 || _mesh->node_owner[*it]!=_mesh->rank){
          dynamic_NNList->push_back(*it);
        }

      subNNList[i] = dynamic_NNList;
      node_colour[vid] = -1; // Reset colour
    }

    /**********************************************
     * Active sub-mesh is ready, let's colour it. *
     **********************************************/

    // Local independent sets.
    std::vector< std::vector<index_t> > local_ind_sets(max_colours);
    // idx[i] stores the index in independent_sets[i] at which this thread
    // will copy the contents of its local independent set local_ind_sets[i].
    std::vector<size_t> idx(max_colours);
    // Local max colour
    int max_colour = 0;

    uint32_t vid_hash;
    std::vector<uint32_t> hashes;

#pragma omp for schedule(dynamic) nowait
    for(size_t i=0; i<GlobalActiveSet_size; ++i){
      index_t vid = GlobalActiveSet[i];

      // Initialise hashes with the global IDs - this is important for MPI!
      vid_hash = (uint32_t) _mesh->lnn2gnn[vid];
      hashes.clear();
      for(typename std::vector<index_t>::const_iterator it=subNNList[i]->begin(); it!=subNNList[i]->end(); ++it){
        hashes.push_back((uint32_t) _mesh->lnn2gnn[*it]);
      }

      int colour = 0;

      while(true){
        // Calculate hashes for this round
        vid_hash = _mesh->hash(vid_hash);
        for(typename std::vector<uint32_t>::iterator it=hashes.begin(); it!=hashes.end(); ++it){
          *it = _mesh->hash(*it);
        }

        // A vertex is eligible to be coloured only if it's got
        // the highest or the lowest hash among all neighbours.

        // Check whether this is the highest hash.
        bool eligible = true;

        for(typename std::vector<uint32_t>::const_iterator it=hashes.begin(); it!=hashes.end(); ++it){
          if(vid_hash < *it){
            eligible = false;
            break;
          }
        }

        if(eligible){
          node_colour[vid] = colour;
          local_ind_sets[colour].push_back(vid);
          break;
        }else{
          // Check whether this is the lowest hash.
          eligible = true;
          ++colour;

          for(typename std::vector<uint32_t>::const_iterator it=hashes.begin(); it!=hashes.end(); ++it){
            if(vid_hash > *it){
              eligible = false;
              break;
            }
          }

          if(eligible){
            node_colour[vid] = colour;
            local_ind_sets[colour].push_back(vid);
            break;
          }else{
            ++colour;
          }
        }
      }

      if(colour > max_colour)
        max_colour = colour;

      assert(max_colour < max_colours);
    }

    ++max_colour;

    // Capture and increment the index in independent_sets[colour] at which the local independent
    // sets will be copied later, after memory for the global independent sets will have been allocated.
    for(int colour=0; colour < max_colour; ++colour){
#pragma omp atomic capture
      {
        idx[colour] = ind_set_size[colour];
        ind_set_size[colour] += local_ind_sets[colour].size();
      }
    }

    // Total number of independent sets
    int nsets_local;
#pragma omp atomic capture
    {
      nsets_local = nsets;
      nsets += (nsets_local < max_colours ? max_colours - nsets_local : 0);
    }

    // Allocate memory for the global independent sets.
#pragma omp barrier

#ifdef HAVE_MPI
    if(_mesh->num_processes>1){
#pragma omp single nowait
      {
        global_nsets = nsets;
        MPI_Allreduce(MPI_IN_PLACE, &global_nsets, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
      }
    }
#endif

#pragma omp for schedule(static)
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
  }

  void destroy(){
#pragma omp for schedule(static) nowait
    for(int set_no=0; set_no<nsets; ++set_no){
      delete[] independent_sets[set_no];
      ind_set_size[set_no] = 0;
    }

#pragma omp for schedule(static)
    for(size_t i=0; i<GlobalActiveSet_size; ++i)
      delete subNNList[i];

#pragma omp single
    {
      nsets = 0;
      GlobalActiveSet_size = 0;
    }
  }

private:
  template<typename _real_t, typename _index_t> friend class Coarsen2D;
  template<typename _real_t, typename _index_t> friend class Coarsen3D;
  template<typename _real_t, typename _index_t> friend class Smooth2D;
  template<typename _real_t, typename _index_t> friend class Smooth3D;
  template<typename _real_t, typename _index_t> friend class Swapping2D;
  template<typename _real_t, typename _index_t> friend class Swapping3D;

  int *node_colour;

  // Total number of independent sets for this MPI process and for the whole MPI_COMM_WORLD
  int nsets, global_nsets;

  // Set of all dynamic vertices.
  index_t *GlobalActiveSet;
  size_t GlobalActiveSet_size;

  // NNList of the active sub-mesh. Accessed using active sub-mesh vertex IDs. Each vector contains regular vertex IDs.
  std::vector<index_t> **subNNList;

  index_t **independent_sets;
  std::vector<size_t> ind_set_size;

  Mesh<real_t, index_t> *_mesh;

  AdaptiveAlgorithm<real_t, index_t> *alg;

  const static int max_colours = 256;
};


#endif
