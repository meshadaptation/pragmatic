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

#include "graph_partitioning.h"
#include "ElementProperty.h"
#include "zoltan_tools.h"
#include "Mesh.h"
#include "Colour.h"

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
  }
  
  /// Default destructor.
  ~Coarsen2D(){
    if(property!=NULL)
      delete property;

    if(dynamic_vertex!=NULL)
      delete [] dynamic_vertex;
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
      
      if(dynamic_vertex!=NULL)
        delete [] dynamic_vertex;
      
      dynamic_vertex = new index_t[nnodes_reserve];
    }
    
    /* dynamic_vertex[i] >= 0 :: target to collapse node i
       dynamic_vertex[i] = -1 :: node inactive (deleted/locked)
       dynamic_vertex[i] = -2 :: recalculate collapse - this is how propagation is implemented
    */

    // Total number of active vertices. Used to break the infinite loop.
    size_t total_active;

    // Set of all dynamic vertices. We pre-allocate the maximum capacity
    // that may be needed. Accessed using active sub-mesh vertex IDs.
    index_t *GlobalActiveSet = new index_t[NNodes];
    size_t GlobalActiveSet_size;

    // Subset of GlobalActiveSet: it contains the vertices which the colouring
    // algorithm will try to colour in one round. UncolouredActiveSet[i] stores
    // the active sub-mesh ID of an uncoloured vertex, i.e. the vertex at
    // UncolouredActiveSet[i] has neighbours=subNNList[UncolouredActiveSet[i]]
    // and its regular ID is vid = GlobalActiveSet[UncolouredActiveSet[i]].
    index_t *UncolouredActiveSet = new index_t[NNodes];
    size_t UncolouredActiveSet_size;

    // Subset of UncolouredActiveSet: it contains the vertices which couldn't be
    // coloured in that round and are left for the next round. Like UncolouredActiveSet,
    // it stores active sub-mesh IDs. At the end of each colouring round,
    // the pointers to UncolouredActiveSet and NextRoundActive set are swapped.
    index_t *NextRoundActiveSet = new index_t[NNodes];
    size_t NextRoundActiveSet_size;

    // NNList of the active sub-mesh. Accessed using active sub-mesh vertex IDs.
    // Each vector contains regular vertex IDs.
    std::vector<index_t> **subNNList = new std::vector<index_t>* [NNodes];

    // Accessed using regular vertex IDs.
    char *node_colour = new char[NNodes];
    size_t *node_hash = new size_t[NNodes];

    /* It's highly unlikely that more than a dozen colours will be needed, let's
     * allocate 32 just in case. They are just pointers, so there is no memory
     * footprint. The array for each independent set will be allocated on demand.
     * Independent sets contain regular vertex IDs.
     */
    index_t **independent_sets = new index_t* [max_colours];
    std::vector<size_t> ind_set_size(max_colours, 0);

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();

      // Mark all vertices for evaluation.
#pragma omp for schedule(static)
      for(size_t i=0;i<NNodes;i++)
        dynamic_vertex[i] = -2;

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
        for(size_t i=0;i<NNodes;i++){
          if(dynamic_vertex[i] == -2)
            dynamic_vertex[i] = coarsen_identify_kernel(i, L_low, L_max);

          if(dynamic_vertex[i]>=0){
            active_set.push_back(i);
            ++total_active;
          }
        }

        if(total_active == 0)
          break;

        size_t pos;
#pragma omp atomic capture
        {
          pos = GlobalActiveSet_size;
          GlobalActiveSet_size += active_set.size();
        }

        memcpy(&GlobalActiveSet[pos], &active_set[0], active_set.size() * sizeof(index_t));

#pragma omp barrier
        assert(GlobalActiveSet_size == total_active);

        // Construct the node adjacency list for the active sub-mesh.
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
        size_t colour = 0;

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
        size_t nsets = colour;
        assert(nsets <= max_colours);

        // Allocate memory for the global independent sets.
        // TODO: What happens if another MPI process has used more colours?
#pragma omp for schedule(dynamic)
        for(unsigned char set_no=0; set_no<nsets; ++set_no)
          independent_sets[set_no] = new index_t[ind_set_size[set_no]];

        // Copy local independent sets into the global structure.
        for(unsigned char set_no=0; set_no<nsets; ++set_no)
          memcpy(&independent_sets[set_no][idx[set_no]], &local_ind_sets[set_no][0],
              local_ind_sets[set_no].size() * sizeof(index_t));

        // Update MPI halo
#pragma omp barrier
        /********************
         * End of colouring *
         ********************/

        // TODO: Sort independent sets by size.

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
        for(unsigned char set_no=0; set_no<nsets; ++set_no){
#pragma omp for schedule(dynamic,32)
          for(size_t i=0; i<ind_set_size[set_no]; ++i){
            index_t rm_vertex = independent_sets[set_no][i];
            assert((size_t) rm_vertex < NNodes);

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

            index_t target_vertex = dynamic_vertex[rm_vertex];

            // Mark neighbours for re-evaluation.
            // Two threads might be marking the same vertex at the same time.
            // This is a race condition which doesn't cause any trouble.
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
              if(_mesh->is_owned_node(*jt))
                dynamic_vertex[*jt] = -2;
            }

            // Mark rm_vertex as non-active.
            dynamic_vertex[rm_vertex] = -1;

            // Un-colour target_vertex if its colour clashes with any of its new neighbours.
            // There is race condition here, but it doesn't do any harm.
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[rm_vertex].begin();jt!=_mesh->NNList[rm_vertex].end();++jt){
              if(_mesh->is_owned_node(*jt)) // TODO: What happens across MPI processes?
                if(node_colour[*jt] == node_colour[target_vertex])
                  if(*jt != target_vertex){
                    node_colour[target_vertex] = -1;
                    break;
                  }
            }

            // Coarsen the edge.
            coarsen_kernel(rm_vertex, target_vertex, tid);
          }

          _mesh->commit_deferred(tid);
          _surface->commit_deferred(tid);
#pragma omp barrier
        }

#pragma omp for schedule(static) nowait
        for(size_t set_no=0; set_no<nsets; ++set_no){
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
    delete[] node_colour;
    delete[] node_hash;
    delete[] independent_sets;

    // Phase 3 (halo)
#ifdef HAVE_MPI
    if(nprocs>1){
            
      // Loop until the maximum independent set is NULL.
      for(;;){
        /* Select an independent set of vertices that can be
           coarsened. */
        std::vector<bool> maximal_independent_set;
        select_max_independent_set(maximal_independent_set);
        
        /* Data may have been migrated for operations to complete so
           we have to recalculate the number of nodes. */
        NNodes = _mesh->get_number_nodes();
        
        int coarsen_cnt = 0;
        for(size_t rm_vertex=0;rm_vertex<NNodes;rm_vertex++){
          int target_vertex=dynamic_vertex[rm_vertex];
          if(target_vertex>=0){
            if(_mesh->is_halo_node(rm_vertex)){
              // If it's in the halo then only coarsen the independent set.
              if(!maximal_independent_set[rm_vertex])
                continue;
              
              coarsen_kernel(rm_vertex, target_vertex, 0);
              coarsen_cnt++;
            }else{
              // If it's not in the halo, then it's free to be coarsened.
              coarsen_kernel(rm_vertex, target_vertex, 0);
              coarsen_cnt++;
            }
          }
        }
        _mesh->trim_halo();
        
        MPI_Allreduce(MPI_IN_PLACE, &coarsen_cnt, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());

        // Break if there is nothing left to coarsen.
        if(coarsen_cnt==0){
          break;
        }
      }
    }
#endif
    return;
  }
  
  /*! Kernel for identifying what if any vertex rm_vertex should be collapsed onto.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   * Returns the node ID that rm_vertex should be collapsed onto, negative if no operation is to be performed.
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
      // For now impose the restriction that we will not coarsen across partition boundaries.
      if(_mesh->recv_halo.count(*nn))
        continue;
      
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
   * Returns the node ID that rm_vertex is collapsed onto, negative if the operation is not performed.
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

  void select_max_independent_set(std::vector<bool> &maximal_independent_set){
    int NNodes = _mesh->get_number_nodes();
    int NPNodes = NNodes - _mesh->recv_halo.size();

    // Create a reverse lookup to map received gnn's back to lnn's.
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    boost::unordered_map<int, int> gnn2lnn;
#else
    std::map<index_t, index_t> gnn2lnn;
#endif
    for(int i=0;i<NNodes;i++){
      assert(gnn2lnn.find(_mesh->lnn2gnn[i])==gnn2lnn.end());
      gnn2lnn[_mesh->lnn2gnn[i]] = i;
    }
    assert(gnn2lnn.size()==_mesh->lnn2gnn.size());

    // Use a bitmap to indicate the maximal independent set.
    assert(NNodes>=NPNodes);
    maximal_independent_set.resize(NNodes);
    std::fill(maximal_independent_set.begin(), maximal_independent_set.end(), false);
    
    /* Find the size of the local graph and create a new lnn2gnn for
       the compressed graph. */
    std::vector<size_t> nedges(NNodes, 0);
    std::vector<size_t> csr_edges;
    csr_edges.reserve(NNodes*5);
    for(int i=0;i<NNodes;i++){
      if(_mesh->is_halo_node(i)){
        for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[i].begin();it!=_mesh->NNList[i].end();++it){
          if(_mesh->is_halo_node(*it)){
            csr_edges.push_back(*it);
            nedges[i]++;
          }
        }
      }
    }
          
    // Colour.
    zoltan_graph_t graph;
    std::vector<int> colour(NNodes);
        
    graph.rank = rank;
    graph.nnodes = NNodes;
    graph.npnodes = NPNodes;
    graph.nedges = &(nedges[0]);
    graph.csr_edges = &(csr_edges[0]);
        
    graph.gid = &(_mesh->lnn2gnn[0]);
    graph.owner = &(_mesh->node_owner[0]);
        
    graph.colour = &(colour[0]);
        
    zoltan_colour(&graph, 1,  _mesh->get_mpi_comm());
          
    // Given a colouring, determine the maximum independent set.
          
    // Count number of active vertices of each colour.
    int max_colour = 0;
    for(int i=0;i<NPNodes;i++){
      max_colour = std::max(max_colour, colour[i]);
    }
    
    MPI_Allreduce(MPI_IN_PLACE, &max_colour, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
    
    std::vector<int> ncolours(max_colour+1, 0);
    for(int i=0;i<NPNodes;i++){
      if((colour[i]>=0)&&(dynamic_vertex[i]>=0)){
        ncolours[colour[i]]++;
      }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, &(ncolours[0]), max_colour+1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());

    // Find the colour of the largest active set.
    std::pair<int, int> MIS_colour(0, ncolours[0]);
    for(int i=1;i<=max_colour;i++){
      if(MIS_colour.second<ncolours[i]){
        MIS_colour.first = i;
        MIS_colour.second = ncolours[i];
      }
    }

    if(MIS_colour.second>=0){
      for(int i=0;i<NPNodes;i++){
        if((colour[i]==MIS_colour.first)&&(dynamic_vertex[i]>=0)){
          maximal_independent_set[i] = true;
        }
      }
    }

    // Cache who knows what.
    std::vector< std::set<int> > known_nodes(nprocs);
    for(int p=0;p<nprocs;p++){
      if(p==rank)
        continue;
              
      for(std::vector<int>::const_iterator it=_mesh->send[p].begin();it!=_mesh->send[p].end();++it)
        known_nodes[p].insert(*it);
              
      for(std::vector<int>::const_iterator it=_mesh->recv[p].begin();it!=_mesh->recv[p].end();++it)
        known_nodes[p].insert(*it);
    }

    // Communicate collapses.
    // Stuff in list of vertices that have to be communicated.
    std::vector< std::vector<int> > send_edges(nprocs);
    std::vector< std::set<int> > send_elements(nprocs), send_nodes(nprocs);
    for(int i=0;i<NPNodes;i++){
      if(!maximal_independent_set[i])
        continue;
              
      // Is the vertex being collapsed contained in the halo?
      if(_mesh->is_halo_node(i)){ 
        // Yes. Discover where we have to send this edge.
        for(int p=0;p<nprocs;p++){
          if(known_nodes[p].count(i)){
            send_edges[p].push_back(_mesh->lnn2gnn[i]);
            send_edges[p].push_back(_mesh->lnn2gnn[dynamic_vertex[i]]);
                    
            send_elements[p].insert(_mesh->NEList[i].begin(), _mesh->NEList[i].end());
          }
        }
      }
    }
            
    // Finalise list of additional elements and nodes to be sent.
    for(int p=0;p<nprocs;p++){
      for(std::set<int>::iterator it=send_elements[p].begin();it!=send_elements[p].end();){
        std::set<int>::iterator ele=it++;
        const int *n=_mesh->get_element(*ele);
        int cnt=0;
        for(size_t i=0;i<nloc;i++){
          if(known_nodes[p].count(n[i])==0){
            send_nodes[p].insert(n[i]);
          }
          if(_mesh->node_owner[n[i]]==(size_t)p)
            cnt++;
        }
        if(cnt){
          send_elements[p].erase(ele);
        }
      }
    }
            
    // Push data to be sent onto the send_buffer.
    std::vector< std::vector<int> > send_buffer(nprocs);
    size_t node_package_int_size = (ndims*sizeof(real_t)+msize*sizeof(float))/sizeof(int);
    for(int p=0;p<nprocs;p++){
      if(send_edges[p].size()==0)
        continue;
              
      // Push on the nodes that need to be communicated.
      send_buffer[p].push_back(send_nodes[p].size());
      for(std::set<int>::iterator it=send_nodes[p].begin();it!=send_nodes[p].end();++it){
        send_buffer[p].push_back(_mesh->lnn2gnn[*it]);
        send_buffer[p].push_back(_mesh->node_owner[*it]);
                
        // Stuff in coordinates and metric via int's.
        std::vector<int> ivertex(node_package_int_size);
        real_t *rcoords = (real_t *) &(ivertex[0]);
        float *rmetric = (float *) &(rcoords[ndims]);
        _mesh->get_coords(*it, rcoords);
        _mesh->get_metric(*it, rmetric);
                
        send_buffer[p].insert(send_buffer[p].end(), ivertex.begin(), ivertex.end());
      }
              
      // Push on edges that need to be sent.
      send_buffer[p].push_back(send_edges[p].size());
      send_buffer[p].insert(send_buffer[p].end(), send_edges[p].begin(), send_edges[p].end());
              
      // Push on elements that need to be communicated; record facets that need to be sent with these elements.
      send_buffer[p].push_back(send_elements[p].size());
      std::set<int> send_facets;
      for(std::set<int>::iterator it=send_elements[p].begin();it!=send_elements[p].end();++it){
        const int *n=_mesh->get_element(*it);
        for(size_t j=0;j<nloc;j++)
          send_buffer[p].push_back(_mesh->lnn2gnn[n[j]]);
                
        std::vector<int> lfacets;
        _surface->find_facets(n, lfacets);
        send_facets.insert(lfacets.begin(), lfacets.end());
      }
              
      // Push on facets that need to be communicated.
      send_buffer[p].push_back(send_facets.size());
      for(std::set<int>::iterator it=send_facets.begin();it!=send_facets.end();++it){
        const int *n=_surface->get_facet(*it);
        for(size_t i=0;i<snloc;i++)
          send_buffer[p].push_back(_mesh->lnn2gnn[n[i]]);
        send_buffer[p].push_back(_surface->get_boundary_id(*it));
        send_buffer[p].push_back(_surface->get_coplanar_id(*it));
      }
    }
            
    std::vector<int> send_buffer_size(nprocs), recv_buffer_size(nprocs);
    for(int p=0;p<nprocs;p++)
      send_buffer_size[p] = send_buffer[p].size();

    MPI_Alltoall(&(send_buffer_size[0]), 1, MPI_INT, &(recv_buffer_size[0]), 1, MPI_INT, _mesh->get_mpi_comm());

    // Setup non-blocking receives
    std::vector< std::vector<int> > recv_buffer(nprocs);
    std::vector<MPI_Request> request(nprocs*2);
    for(int i=0;i<nprocs;i++){
      if(recv_buffer_size[i]==0){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buffer[i].resize(recv_buffer_size[i]);
        MPI_Irecv(&(recv_buffer[i][0]), recv_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[i]));
      }
    }
            
    // Non-blocking sends.
    for(int i=0;i<nprocs;i++){
      if(send_buffer_size[i]==0){
        request[nprocs+i] =  MPI_REQUEST_NULL;
      }else{
        MPI_Isend(&(send_buffer[i][0]), send_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[nprocs+i]));
      }
    }
            
    // Wait for comms to finish.
    std::vector<MPI_Status> status(nprocs*2);
    MPI_Waitall(nprocs, &(request[0]), &(status[0]));
    MPI_Waitall(nprocs, &(request[nprocs]), &(status[nprocs]));
            
    // Unpack received data into dynamic_vertex
    std::vector< std::set<index_t> > extra_halo_receives(nprocs);
    for(int p=0;p<nprocs;p++){
      if(recv_buffer[p].empty())
        continue;
              
      int loc = 0;
              
      // Unpack additional nodes.
      int num_extra_nodes = recv_buffer[p][loc++];
      for(int i=0;i<num_extra_nodes;i++){
        int gnn = recv_buffer[p][loc++]; // think this through - can I get duplicates
        int lowner = recv_buffer[p][loc++];
                
        extra_halo_receives[lowner].insert(gnn);
                
        real_t *coords = (real_t *) &(recv_buffer[p][loc]);
        float *metric = (float *) &(coords[ndims]);
        loc+=node_package_int_size;
                
        // Add vertex+metric if we have not already received this data.
        if(gnn2lnn.find(gnn)==gnn2lnn.end()){
          // TODO: _mesh->append_vertex has been modified
          index_t lnn = _mesh->append_vertex(coords, metric);
                  
          _mesh->lnn2gnn.push_back(gnn);
          _mesh->node_owner.push_back(lowner);
          size_t nnodes_new = _mesh->node_owner.size();
          if(nnodes_reserve<nnodes_new){
            nnodes_reserve*=1.5;
            index_t *new_dynamic_vertex = new index_t[nnodes_reserve];
            for(size_t k=0;k<nnodes_new-1;k++)
              new_dynamic_vertex[k] = dynamic_vertex[k];
            std::swap(new_dynamic_vertex, dynamic_vertex);
            delete [] new_dynamic_vertex;
          }
          dynamic_vertex[nnodes_new-1] = -2;
          maximal_independent_set.push_back(false);     
          gnn2lnn[gnn] = lnn;
        }
      }
              
      // Unpack edges
      size_t edges_size=recv_buffer[p][loc++];
      for(size_t i=0;i<edges_size;i+=2){
        int rm_vertex = gnn2lnn[recv_buffer[p][loc++]];
        int target_vertex = gnn2lnn[recv_buffer[p][loc++]];
        assert(dynamic_vertex[rm_vertex]<0);
        assert(target_vertex>=0);
        dynamic_vertex[rm_vertex] = target_vertex;
        maximal_independent_set[rm_vertex] = true;
      }
              
      // Unpack elements.
      int num_extra_elements = recv_buffer[p][loc++];
      for(int i=0;i<num_extra_elements;i++){
        int element[nloc];
        for(size_t j=0;j<nloc;j++){
          element[j] = gnn2lnn[recv_buffer[p][loc++]];
        }
                
        // See if this is a new element.
        std::set<index_t> common_element01;
        std::set_intersection(_mesh->NEList[element[0]].begin(), _mesh->NEList[element[0]].end(),
                         _mesh->NEList[element[1]].begin(), _mesh->NEList[element[1]].end(),
                         std::inserter(common_element01, common_element01.begin()));
                
        std::set<index_t> common_element012;
        std::set_intersection(common_element01.begin(), common_element01.end(),
                         _mesh->NEList[element[2]].begin(), _mesh->NEList[element[2]].end(),
                         std::inserter(common_element012, common_element012.begin()));
        
        if(common_element012.empty()){
          // Add element
          // TODO: _mesh->append_element has been modified
          int eid = _mesh->append_element(element);
                  
          // Update NEList
          for(size_t l=0;l<nloc;l++){
            _mesh->NEList[element[l]].insert(eid);
          }

          // Update NNList
          for(size_t l=0;l<nloc;l++){
            for(size_t k=l+1;k<nloc;k++){
              std::vector<int>::iterator result0 = std::find(_mesh->NNList[element[l]].begin(), _mesh->NNList[element[l]].end(), element[k]);
              if(result0==_mesh->NNList[element[l]].end())
                _mesh->NNList[element[l]].push_back(element[k]);
                      
              std::vector<int>::iterator result1 = std::find(_mesh->NNList[element[k]].begin(), _mesh->NNList[element[k]].end(), element[l]);
              if(result1==_mesh->NNList[element[k]].end())
                _mesh->NNList[element[k]].push_back(element[l]);
            }
          }
        }
      }
              
      // Unpack facets.
      int num_extra_facets = recv_buffer[p][loc++];
      for(int i=0;i<num_extra_facets;i++){
        std::vector<int> facet(snloc);
        for(size_t j=0;j<snloc;j++){
          index_t gnn = recv_buffer[p][loc++];
          assert(gnn2lnn.find(gnn)!=gnn2lnn.end());
          facet[j] = gnn2lnn[gnn];
        }
                
        int boundary_id = recv_buffer[p][loc++];
        int coplanar_id = recv_buffer[p][loc++];
                
        _surface->append_facet(&(facet[0]), boundary_id, coplanar_id, true);
      }
    }
            
    assert(gnn2lnn.size()==_mesh->lnn2gnn.size());
            
    // Update halo.
    for(int p=0;p<nprocs;p++){
      send_buffer_size[p] = extra_halo_receives[p].size();
      send_buffer[p].clear();
      for(typename std::set<index_t>::const_iterator ht=extra_halo_receives[p].begin();ht!=extra_halo_receives[p].end();++ht)
        send_buffer[p].push_back(*ht);
              
    }

    MPI_Alltoall(&(send_buffer_size[0]), 1, MPI_INT, &(recv_buffer_size[0]), 1, MPI_INT, _mesh->get_mpi_comm());
            
    // Setup non-blocking receives
    for(int i=0;i<nprocs;i++){
      recv_buffer[i].clear();
      if(recv_buffer_size[i]==0){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buffer[i].resize(recv_buffer_size[i]);
        MPI_Irecv(&(recv_buffer[i][0]), recv_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[i]));
      }
    }
            
    // Non-blocking sends.
    for(int i=0;i<nprocs;i++){
      if(send_buffer_size[i]==0){
        request[nprocs+i] =  MPI_REQUEST_NULL;
      }else{
        MPI_Isend(&(send_buffer[i][0]), send_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[nprocs+i]));
      }
    }
            
    // Wait for comms to finish.
    MPI_Waitall(nprocs, &(request[0]), &(status[0]));
    MPI_Waitall(nprocs, &(request[nprocs]), &(status[nprocs]));
            
    // Use this data to update the halo information.
    for(int i=0;i<nprocs;i++){
      for(std::vector<int>::const_iterator it=recv_buffer[i].begin();it!=recv_buffer[i].end();++it){
        assert(gnn2lnn.find(*it)!=gnn2lnn.end());
        int lnn = gnn2lnn[*it];
        _mesh->send[i].push_back(lnn);
        _mesh->send_halo.insert(lnn);
      }
      for(std::vector<int>::const_iterator it=send_buffer[i].begin();it!=send_buffer[i].end();++it){
        assert(gnn2lnn.find(*it)!=gnn2lnn.end());
        int lnn = gnn2lnn[*it];
        _mesh->recv[i].push_back(lnn);
        _mesh->recv_halo.insert(lnn);
      }
    }
  }

 private:
  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  
  size_t nnodes_reserve;
  index_t *dynamic_vertex;

  real_t _L_low, _L_max;

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;

  const static size_t max_colours = 32;

  int nprocs, rank, nthreads;
};

#endif
