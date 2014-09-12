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

#ifndef REFINE_H
#define REFINE_H

#include <algorithm>
#include <set>
#include <vector>

#include <string.h>
#include <inttypes.h>

#include "DeferredOperations.h"
#include "Edge.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D mesh refinement.
 *
 */
template<typename real_t, int dim> class Refine{
 public:
  /// Default constructor.
  Refine(Mesh<real_t> &mesh){
    _mesh = &mesh;

    size_t NElements = _mesh->get_number_elements();

    // Set the orientation of elements.
    property = NULL;
    for(size_t i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      if(dim==2)
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
            _mesh->get_coords(n[1]), _mesh->get_coords(n[2]));
      else if(dim==3)
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
            _mesh->get_coords(n[1]), _mesh->get_coords(n[2]), _mesh->get_coords(n[3]));

      break;
    }

    MPI_Comm comm = _mesh->get_mpi_comm();

    nprocs = pragmatic_nprocesses(comm);
    rank = pragmatic_process_id(comm);

    nthreads = pragmatic_nthreads();

    newVertices.resize(nthreads);
    newElements.resize(nthreads);
    newBoundaries.resize(nthreads);
    newCoords.resize(nthreads);
    newMetric.resize(nthreads);

    // Pre-allocate the maximum size that might be required
    allNewVertices.resize(_mesh->_ENList.size());

    threadIdx.resize(nthreads);
    splitCnt.resize(nthreads);

    def_ops = new DeferredOperations<real_t>(_mesh, nthreads, defOp_scaling_factor);
  }

  /// Default destructor.
  ~Refine(){
    delete property;
    delete def_ops;
  }

  /*! Perform one level of refinement See Figure 25; X Li et al, Comp
   * Methods Appl Mech Engrg 194 (2005) 4915-4950. The actual
   * templates used for 3D refinement follows Rupak Biswas, Roger
   * C. Strawn, "A new procedure for dynamic adaption of
   * three-dimensional unstructured grids", Applied Numerical
   * Mathematics, Volume 13, Issue 6, February 1994, Pages 437-452.
   */
  void refine(real_t L_max){
    size_t origNElements = _mesh->get_number_elements();
    size_t origNNodes = _mesh->get_number_nodes();

    new_vertices_per_element.resize(nedge*origNElements);
    std::fill(new_vertices_per_element.begin(), new_vertices_per_element.end(), -1);

    int additional_split = 0;

#pragma omp parallel
    {
#pragma omp single nowait
      {
        if(dim==3)
          marked_edges.resize(origNNodes);
      }

      int tid = pragmatic_thread_id();
      splitCnt[tid] = 0;

      /*
       * Average vertex degree in 2D is ~6, so there
       * are approx. (6/2)*NNodes edges in the mesh.
       * In 3D, average vertex degree is ~12.
       */
      size_t reserve_size = nedge*origNNodes/nthreads;
      newVertices[tid].clear(); newVertices[tid].reserve(reserve_size);
      newCoords[tid].clear(); newCoords[tid].reserve(ndims*reserve_size);
      newMetric[tid].clear(); newMetric[tid].reserve(msize*reserve_size);

      /* Loop through all edges and select them for refinement if
         its length is greater than L_max in transformed space. */
#pragma omp for schedule(guided) nowait
      for(size_t i=0;i<origNNodes;++i){
        for(size_t it=0;it<_mesh->NNList[i].size();++it){
          index_t otherVertex = _mesh->NNList[i][it];
          assert(otherVertex>=0);

          /* Conditional statement ensures that the edge length is only calculated once.
           * By ordering the vertices according to their gnn, we ensure that all processes
           * calculate the same edge length when they fall on the halo.
           */
          if(_mesh->lnn2gnn[i] < _mesh->lnn2gnn[otherVertex]){
            double length = _mesh->calc_edge_length(i, otherVertex);
            if(length>L_max){
              ++splitCnt[tid];
              refine_edge(i, otherVertex, tid);

              // If in 3D, find which elements share this edge
              // and mark them as having one of their edges split.
              if(dim==3){
                std::set<index_t> intersection;
                std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
                                      _mesh->NEList[otherVertex].begin(), _mesh->NEList[otherVertex].end(),
                                      std::inserter(intersection, intersection.begin()));
                assert(intersection.size()>0);

                for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element){
                  index_t eid = *element;
                  size_t edgeOffset = edgeNumber(eid, i, otherVertex);
                  new_vertices_per_element[nedge*eid+edgeOffset] = 0;
                  // We don't know the actual ID of the new vertex,
                  // so let's use 0 for marking for the time being.
                }
              }
            }
          }
        }
      }

      /*
       * For 3D, given the set of refined edges, apply additional
       * edge-refinement to get a regular and conforming element
       * refinement throughout the domain.
       */
      if(dim==3){
#pragma omp barrier
        for(;;){
#pragma omp for schedule(guided) reduction(+:additional_split)
          for(size_t eid=0; eid<origNElements; ++eid){
            //If the element has been deleted, continue.
            const index_t *n = _mesh->get_element(eid);
            if(n[0] < 0)
              continue;

            // Find what edges have been split in this element.
            typename std::vector< Edge<index_t> > split_set;
            int pos=0;
            for(int i=0; i<nloc; ++i)
              for(int j=i+1; j<nloc; ++j){
                if(new_vertices_per_element[nedge*eid+pos]!=-1){
                  split_set.push_back(Edge<index_t>(n[i], n[j]));
                  // pos, n[i] and n[j] are related in line with
                  // the convention in function edgeNumber.
                }
                ++pos;
              }

            switch(split_set.size()){
            case 0: // No refinement
              break;
            case 1: // 1:2 refinement is ok.
              break;
            case 2:{
              /* Here there are two possibilities. Either the two split
               * edges share a vertex (case 1) or they are opposite edges
               * (case 2). Case 1 results in a 1:3 subdivision and a
               * possible mismatch on the surface. So we have to split an
               * additional edge. Case 2 results in a 1:4 with no issues
               * so it is left as is.*/

              int n0=split_set[0].connected(split_set[1]);
              if(n0>=0){
                // Case 1.
                int n1 = (n0 == split_set[0].edge.first) ? split_set[0].edge.second : split_set[0].edge.first;
                int n2 = (n0 == split_set[1].edge.first) ? split_set[1].edge.second : split_set[1].edge.first;

                Edge<index_t> extra(n1, n2);
                def_ops->propagate_refinement(extra.edge.first, extra.edge.second, tid);
                ++additional_split;
              }
              break;
            }
            case 3:{
              /* There are 3 cases that need to be considered. They can
               * be distinguished by the total number of nodes that are
               * common between any pair of edges. Only the case there
               * are 3 different nodes common between pairs of edges do
               * we get a 1:4 subdivision. Otherwise, we have to refine
               * the other edges.*/
              std::set<index_t> shared;
              for(int j=0;j<3;j++){
                for(int k=j+1;k<3;k++){
                  index_t nid = split_set[j].connected(split_set[k]);
                  if(nid>=0)
                    shared.insert(nid);
                }
              }
              size_t nshared = shared.size();

              if(nshared!=3){
                // Refine unsplit edges.
                for(int j=0;j<4;j++)
                  for(int k=j+1;k<4;k++){
                    Edge<index_t> test_edge(n[j], n[k]);
                    if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end()){
                      Edge<index_t> extra(n[j], n[k]);
                      def_ops->propagate_refinement(extra.edge.first, extra.edge.second, tid);
                      ++additional_split;
                    }
                  }
              }
              break;
            }
            case 4:{
              // Refine unsplit edges.
              for(int j=0;j<4;j++)
                for(int k=j+1;k<4;k++){
                  Edge<index_t> test_edge(n[j], n[k]);
                  if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end()){
                    Edge<index_t> extra(n[j], n[k]);
                    def_ops->propagate_refinement(extra.edge.first, extra.edge.second, tid);
                    ++additional_split;
                  }
                }
              break;
            }
            case 5:{
              // Refine unsplit edges.
              for(int j=0;j<4;j++)
                for(int k=j+1;k<4;k++){
                  Edge<index_t> test_edge(n[j], n[k]);
                  if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end()){
                    Edge<index_t> extra(n[j], n[k]);
                    def_ops->propagate_refinement(extra.edge.first, extra.edge.second, tid);
                    ++additional_split;
                  }
                }
              break;
            }
            case 6: // All edges split. Nothing to do.
              break;
            default:
              break;
            }
          }

          if(!additional_split)
            break;

#pragma omp for schedule(guided)
          for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
            for(int i=0; i<nthreads; ++i){
              def_ops->commit_refinement_propagation(marked_edges, i, vtid);
            }
          }

#pragma omp for schedule(guided) nowait
          for(size_t i=0;i<origNNodes;++i){
            for(std::set<index_t>::iterator it=marked_edges[i].begin();it!=marked_edges[i].end();++it){
              index_t otherVertex = *it;
              assert(otherVertex>=0);

              ++splitCnt[tid];
              refine_edge(i, otherVertex, tid);

              std::set<index_t> intersection;
              std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
                                    _mesh->NEList[otherVertex].begin(), _mesh->NEList[otherVertex].end(),
                                    std::inserter(intersection, intersection.begin()));

              for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element){
                index_t eid = *element;
                size_t edgeOffset = edgeNumber(eid, i, otherVertex);
                new_vertices_per_element[nedge*eid+edgeOffset] = 0;
              }
            }
            marked_edges[i].clear();
          }

#pragma omp single
          additional_split = 0;
        }
      }

      threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NNodes, splitCnt[tid]);
      assert(newVertices[tid].size()==splitCnt[tid]);

#pragma omp barrier

#pragma omp single
      {
        if(_mesh->_coords.size()<_mesh->NNodes*nloc){
          _mesh->_coords.resize(_mesh->NNodes*ndims);
          _mesh->metric.resize(_mesh->NNodes*msize);
          _mesh->NNList.resize(_mesh->NNodes);
          _mesh->NEList.resize(_mesh->NNodes);
          _mesh->node_owner.resize(_mesh->NNodes);
          _mesh->lnn2gnn.resize(_mesh->NNodes);
        }
      }

      // Append new coords and metric to the mesh.
      memcpy(&_mesh->_coords[ndims*threadIdx[tid]], &newCoords[tid][0], ndims*splitCnt[tid]*sizeof(real_t));
      memcpy(&_mesh->metric[msize*threadIdx[tid]], &newMetric[tid][0], msize*splitCnt[tid]*sizeof(double));

      // Fix IDs of new vertices
      assert(newVertices[tid].size()==splitCnt[tid]);
      for(size_t i=0;i<splitCnt[tid];i++){
        newVertices[tid][i].id = threadIdx[tid]+i;
      }

      // Accumulate all newVertices in a contiguous array
      memcpy(&allNewVertices[threadIdx[tid]-origNNodes], &newVertices[tid][0], newVertices[tid].size()*sizeof(DirectedEdge<index_t>));

      // Mark each element with its new vertices,
      // update NNList for all split edges.
#pragma omp barrier
#pragma omp for schedule(guided)
      for(size_t i=0; i<_mesh->NNodes-origNNodes; ++i){
        index_t vid = allNewVertices[i].id;
        index_t firstid = allNewVertices[i].edge.first;
        index_t secondid = allNewVertices[i].edge.second;

        // Find which elements share this edge and mark them with their new vertices.
        std::set<index_t> intersection;
        std::set_intersection(_mesh->NEList[firstid].begin(), _mesh->NEList[firstid].end(),
                              _mesh->NEList[secondid].begin(), _mesh->NEList[secondid].end(),
                              std::inserter(intersection, intersection.begin()));

        for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element){
          index_t eid = *element;
          size_t edgeOffset = edgeNumber(eid, firstid, secondid);
          new_vertices_per_element[nedge*eid+edgeOffset] = vid;
        }

        /*
         * Update NNList for newly created vertices. This has to be done here, it cannot be
         * done during element refinement, because a split edge is shared between two elements
         * and we run the risk that these updates will happen twice, once for each element.
         */
        _mesh->NNList[vid].push_back(firstid);
        _mesh->NNList[vid].push_back(secondid);

        def_ops->remNN(firstid, secondid, tid);
        def_ops->addNN(firstid, vid, tid);
        def_ops->remNN(secondid, firstid, tid);
        def_ops->addNN(secondid, vid, tid);

        // This branch is always taken or always not taken for every vertex,
        // so the branch predictor should have no problem handling it.
        if(nprocs==1){
          _mesh->node_owner[vid] = 0;
          _mesh->lnn2gnn[vid] = vid;
        }else{
          /*
           * Perhaps we should introduce a system of alternating min/max assignments,
           * i.e. one time the node is assigned to the min rank, one time to the max
           * rank and so on, so as to avoid having the min rank accumulate the majority
           * of newly created vertices and disturbing load balance among MPI processes.
           */
          int owner0 = _mesh->node_owner[firstid];
          int owner1 = _mesh->node_owner[secondid];
          int owner = std::min(owner0, owner1);
          _mesh->node_owner[vid] = owner;
          _mesh->lnn2gnn[vid] = vid;
        }
      }

      // Start element refinement.
      splitCnt[tid] = 0;
      newElements[tid].clear(); newBoundaries[tid].clear();
      newElements[tid].reserve(dim*dim*origNElements/nthreads);
      newBoundaries[tid].reserve(dim*dim*origNElements/nthreads);
#pragma omp for schedule(guided)
      for(size_t eid=0; eid<origNElements; ++eid){
        //If the element has been deleted, continue.
        const index_t *n = _mesh->get_element(eid);
        if(n[0] < 0)
          continue;

        for(size_t j=0; j<nedge; ++j)
          if(new_vertices_per_element[nedge*eid+j] != -1){
            refine_element(eid, tid);
            break;
          }
      }

      threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NElements, splitCnt[tid]);

      // Append new elements to the mesh and commit deferred operations
      memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
      memcpy(&_mesh->boundary[nloc*threadIdx[tid]], &newBoundaries[tid][0], nloc*splitCnt[tid]*sizeof(int));

      // Commit deferred operations.
#pragma omp for schedule(guided)
      for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
        for(int i=0; i<nthreads; ++i){
          def_ops->commit_remNN(i, vtid);
          def_ops->commit_addNN_unique(i, vtid);
          def_ops->commit_addNN(i, vtid);
          def_ops->commit_remNE(i, vtid);
          def_ops->commit_addNE(i, vtid);
          def_ops->commit_addNE_fix(threadIdx, i, vtid);
        }
      }

      if(nprocs==1){
        // If we update lnn2gnn and node_owner here, OMP performance suffers.
      }else{
#ifdef HAVE_MPI
#pragma omp for schedule(static)
        for(size_t i=0; i<_mesh->NNodes-origNNodes; ++i){
          DirectedEdge<index_t> *vert = &allNewVertices[i];
          /*
           * Perhaps we should introduce a system of alternating min/max assignments,
           * i.e. one time the node is assigned to the min rank, one time to the max
           * rank and so on, so as to avoid having the min rank accumulate the majority
           * of newly created vertices and disturbing load balance among MPI processes.
           */
          int owner0 = _mesh->node_owner[vert->edge.first];
          int owner1 = _mesh->node_owner[vert->edge.second];
          int owner = std::min(owner0, owner1);
          _mesh->node_owner[vert->id] = owner;
        }

        // TODO: This single section can be parallelised
#pragma omp single
        {
          // Once the owner for all new nodes has been set, it's time to amend the halo.
          std::vector< std::set< DirectedEdge<index_t> > > recv_additional(nprocs), send_additional(nprocs);
          std::vector<index_t> invisible_vertices;

          for(size_t i=0; i<_mesh->NNodes-origNNodes; ++i){
            DirectedEdge<index_t> *vert = &allNewVertices[i];

            if(_mesh->node_owner[vert->id] != rank){
              // Vertex is owned by another MPI process, so prepare to update recv and recv_halo.
              // Only update them if the vertex is actually visible by *this* MPI process,
              // i.e. if at least one of its neighbours is owned by *this* process.
              bool visible = false;
              for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[vert->id].begin(); neigh!=_mesh->NNList[vert->id].end(); ++neigh){
                if(_mesh->is_owned_node(*neigh)){
                  visible = true;
                  DirectedEdge<index_t> gnn_edge(_mesh->lnn2gnn[vert->edge.first], _mesh->lnn2gnn[vert->edge.second], vert->id);
                  recv_additional[_mesh->node_owner[vert->id]].insert(gnn_edge);
                  break;
                }
              }
              if(!visible)
                invisible_vertices.push_back(vert->id);
            }else{
              // Vertex is owned by *this* MPI process, so check whether it is visible by other MPI processes.
              // The latter is true only if both vertices of the original edge were halo vertices.
              if(_mesh->is_halo_node(vert->edge.first) && _mesh->is_halo_node(vert->edge.second)){
                // Find which processes see this vertex
                std::set<int> processes;
                for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[vert->id].begin(); neigh!=_mesh->NNList[vert->id].end(); ++neigh)
                  processes.insert(_mesh->node_owner[*neigh]);

                processes.erase(rank);

                for(typename std::set<int>::const_iterator proc=processes.begin(); proc!=processes.end(); ++proc){
                  DirectedEdge<index_t> gnn_edge(_mesh->lnn2gnn[vert->edge.first], _mesh->lnn2gnn[vert->edge.second], vert->id);
                  send_additional[*proc].insert(gnn_edge);
                }
              }
            }
          }

          // Append vertices in recv_additional and send_additional to recv and send.
          // Mark how many vertices are added to each of these vectors.
          std::vector<size_t> recv_cnt(nprocs, 0), send_cnt(nprocs, 0);

          for(int i=0;i<nprocs;++i){
            recv_cnt[i] = recv_additional[i].size();
            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=recv_additional[i].begin();it!=recv_additional[i].end();++it){
              _mesh->recv[i].push_back(it->id);
              _mesh->recv_halo.insert(it->id);
            }

            send_cnt[i] = send_additional[i].size();
            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=send_additional[i].begin();it!=send_additional[i].end();++it){
              _mesh->send[i].push_back(it->id);
              _mesh->send_halo.insert(it->id);
            }
          }

          // Update global numbering
          for(size_t i=origNNodes; i<_mesh->NNodes; ++i)
            if(_mesh->node_owner[i] == rank)
              _mesh->lnn2gnn[i] = _mesh->gnn_offset+i;

          _mesh->update_gappy_global_numbering(recv_cnt, send_cnt);

          // Now that the global numbering has been updated, update send_map and recv_map.
          for(int i=0;i<nprocs;++i){
            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=recv_additional[i].begin();it!=recv_additional[i].end();++it)
              _mesh->recv_map[i][_mesh->lnn2gnn[it->id]] = it->id;

            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=send_additional[i].begin();it!=send_additional[i].end();++it)
              _mesh->send_map[i][_mesh->lnn2gnn[it->id]] = it->id;
          }

          _mesh->clear_invisible(invisible_vertices);
          _mesh->trim_halo();
        }
#endif
      }

#if !defined NDEBUG
#pragma omp barrier
      // Fix orientations of new elements.
      size_t NElements = _mesh->get_number_elements();

      if(dim==2){
#pragma omp for schedule(guided)
        for(size_t i=0;i<NElements;i++){
          index_t n0 = _mesh->_ENList[i*nloc];
          if(n0<0)
            continue;

          index_t n1 = _mesh->_ENList[i*nloc + 1];
          index_t n2 = _mesh->_ENList[i*nloc + 2];

          const real_t *x0 = &_mesh->_coords[n0*ndims];
          const real_t *x1 = &_mesh->_coords[n1*ndims];
          const real_t *x2 = &_mesh->_coords[n2*ndims];

          real_t av = property->area(x0, x1, x2);

          if(av<=0){
#pragma omp critical
            std::cerr<<"ERROR: inverted element in refinement"<<std::endl
               <<"element = "<<n0<<", "<<n1<<", "<<n2<<std::endl;
            exit(-1);
          }
        }
      }else if(dim==3){
         int new_NElements = _mesh->get_number_elements();
         int new_cnt = new_NElements - NElements;

         index_t *tENList = &(_mesh->_ENList[NElements*nloc]);
         real_t *tcoords = &(_mesh->_coords[0]);

#pragma omp for schedule(guided)
         for(int i=0;i<new_cnt;i++){
           index_t n0 = tENList[i*nloc];
           index_t n1 = tENList[i*nloc + 1];
           index_t n2 = tENList[i*nloc + 2];
           index_t n3 = tENList[i*nloc + 3];

           const real_t *x0 = tcoords + n0*ndims;
           const real_t *x1 = tcoords + n1*ndims;
           const real_t *x2 = tcoords + n2*ndims;
           const real_t *x3 = tcoords + n3*ndims;

           real_t av = property->volume(x0, x1, x2, x3);

           if(av<0){
             // Flip element
             tENList[i*nloc] = n1;
             tENList[i*nloc+1] = n0;
           }
         }
      }
#endif
    }
  }

 private:

  void refine_edge(index_t n0, index_t n1, size_t tid){
    if(_mesh->lnn2gnn[n0] > _mesh->lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }
    newVertices[tid].push_back(DirectedEdge<index_t>(n0, n1));

    // Calculate the position of the new point. From equation 16 in
    // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
    real_t x, m;
    const real_t *x0 = _mesh->get_coords(n0);
    const double *m0 = _mesh->get_metric(n0);

    const real_t *x1 = _mesh->get_coords(n1);
    const double *m1 = _mesh->get_metric(n1);

    real_t weight = 1.0/(1.0 + sqrt(property->length<dim>(x0, x1, m0)/
        property->length<dim>(x0, x1, m1)));

    // Calculate position of new vertex and append it to OMP thread's temp storage
    for(size_t i=0;i<ndims;i++){
      x = x0[i]+weight*(x1[i] - x0[i]);
      newCoords[tid].push_back(x);
    }

    // Interpolate new metric and append it to OMP thread's temp storage
    for(size_t i=0;i<msize;i++){
      m = m0[i]+weight*(m1[i] - m0[i]);
      newMetric[tid].push_back(m);
      if(pragmatic_isnan(m))
        std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                 <<"m0[i] = "<<m0[i]<<std::endl
                 <<"m1[i] = "<<m1[i]<<std::endl
                 <<"property->length(x0, x1, m0) = "<<property->length<dim>(x0, x1, m0)<<std::endl
                 <<"property->length(x0, x1, m1) = "<<property->length<dim>(x0, x1, m1)<<std::endl
                 <<"weight = "<<weight<<std::endl;
    }
  }

  void refine_element(index_t eid, size_t tid){
    const int *n=_mesh->get_element(eid);
    const int *boundary=&(_mesh->boundary[eid*nloc]);

    if(dim==2){
      /*
       * 2D Element Refinement
       */

      // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
      index_t newVertex[3] = {-1, -1, -1};
      newVertex[0] = new_vertices_per_element[nedge*eid];
      newVertex[1] = new_vertices_per_element[nedge*eid+1];
      newVertex[2] = new_vertices_per_element[nedge*eid+2];

      int refine_cnt=0;
      for(size_t i=0; i<3; ++i)
        if(newVertex[i]!=-1)
          ++refine_cnt;

      if(refine_cnt==1){
        // Single edge split.
        int rotated_ele[3];
        int rotated_boundary[3];
        index_t vertexID = -1;
        for(int j=0;j<3;j++)
          if(newVertex[j] >= 0){
            vertexID = newVertex[j];

            rotated_ele[0] = n[j];
            rotated_ele[1] = n[(j+1)%3];
            rotated_ele[2] = n[(j+2)%3];

            rotated_boundary[0] = boundary[j];
            rotated_boundary[1] = boundary[(j+1)%3];
            rotated_boundary[2] = boundary[(j+2)%3];

            break;
          }
        assert(vertexID!=-1);

        const index_t ele0[] = {rotated_ele[0], rotated_ele[1], vertexID};
        const index_t ele1[] = {rotated_ele[0], vertexID, rotated_ele[2]};

        const index_t ele0_boundary[] = {rotated_boundary[0], 0, rotated_boundary[2]};
        const index_t ele1_boundary[] = {rotated_boundary[0], rotated_boundary[1], 0};

        index_t ele1ID;
        ele1ID = splitCnt[tid];

        // Add rotated_ele[0] to vertexID's NNList
        def_ops->addNN(vertexID, rotated_ele[0], tid);
        // Add vertexID to rotated_ele[0]'s NNList
        def_ops->addNN(rotated_ele[0], vertexID, tid);

        // ele1ID is a new ID which isn't correct yet, it has to be
        // updated once each thread has calculated how many new elements
        // it created, so put ele1ID into addNE_fix instead of addNE.
        // Put ele1 in rotated_ele[0]'s NEList
        def_ops->addNE_fix(rotated_ele[0], ele1ID, tid);

        // Put eid and ele1 in vertexID's NEList
        def_ops->addNE(vertexID, eid, tid);
        def_ops->addNE_fix(vertexID, ele1ID, tid);

        // Replace eid with ele1 in rotated_ele[2]'s NEList
        def_ops->remNE(rotated_ele[2], eid, tid);
        def_ops->addNE_fix(rotated_ele[2], ele1ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        splitCnt[tid] += 1;
      }else if(refine_cnt==2){
        int rotated_ele[3];
        int rotated_boundary[3];
        index_t vertexID[2];
        for(int j=0;j<3;j++){
          if(newVertex[j] < 0){
            vertexID[0] = newVertex[(j+1)%3];
            vertexID[1] = newVertex[(j+2)%3];

            rotated_ele[0] = n[j];
            rotated_ele[1] = n[(j+1)%3];
            rotated_ele[2] = n[(j+2)%3];

            rotated_boundary[0] = boundary[j];
            rotated_boundary[1] = boundary[(j+1)%3];
            rotated_boundary[2] = boundary[(j+2)%3];

            break;
          }
        }

        real_t ldiag0 = _mesh->calc_edge_length(rotated_ele[1], vertexID[0]);
        real_t ldiag1 = _mesh->calc_edge_length(rotated_ele[2], vertexID[1]);

        const int offset = ldiag0 < ldiag1 ? 0 : 1;

        const index_t ele0[] = {rotated_ele[0], vertexID[1], vertexID[0]};
        const index_t ele1[] = {vertexID[offset], rotated_ele[1], rotated_ele[2]};
        const index_t ele2[] = {vertexID[0], vertexID[1], rotated_ele[offset+1]};

        const index_t ele0_boundary[] = {0, rotated_boundary[1], rotated_boundary[2]};
        const index_t ele1_boundary[] = {rotated_boundary[0], (offset==0)?rotated_boundary[1]:0, (offset==0)?0:rotated_boundary[2]};
        const index_t ele2_boundary[] = {(offset==0)?rotated_boundary[2]:0, (offset==0)?0:rotated_boundary[1], 0};

        index_t ele0ID, ele2ID;
        ele0ID = splitCnt[tid];
        ele2ID = ele0ID+1;


        // NNList: Connect vertexID[0] and vertexID[1] with each other
        def_ops->addNN(vertexID[0], vertexID[1], tid);
        def_ops->addNN(vertexID[1], vertexID[0], tid);

        // vertexID[offset] and rotated_ele[offset+1] are the vertices on the diagonal
        def_ops->addNN(vertexID[offset], rotated_ele[offset+1], tid);
        def_ops->addNN(rotated_ele[offset+1], vertexID[offset], tid);

        // rotated_ele[offset+1] is the old vertex which is on the diagonal
        // Add ele2 in rotated_ele[offset+1]'s NEList
        def_ops->addNE_fix(rotated_ele[offset+1], ele2ID, tid);

        // Replace eid with ele0 in NEList[rotated_ele[0]]
        def_ops->remNE(rotated_ele[0], eid, tid);
        def_ops->addNE_fix(rotated_ele[0], ele0ID, tid);

        // Put ele0, ele1 and ele2 in vertexID[offset]'s NEList
        def_ops->addNE(vertexID[offset], eid, tid);
        def_ops->addNE_fix(vertexID[offset], ele0ID, tid);
        def_ops->addNE_fix(vertexID[offset], ele2ID, tid);

        // vertexID[(offset+1)%2] is the new vertex which is not on the diagonal
        // Put ele0 and ele2 in vertexID[(offset+1)%2]'s NEList
        def_ops->addNE_fix(vertexID[(offset+1)%2], ele0ID, tid);
        def_ops->addNE_fix(vertexID[(offset+1)%2], ele2ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
        assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);

        replace_element(eid, ele1, ele1_boundary);
        append_element(ele0, ele0_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        splitCnt[tid] += 2;
      }else{ // refine_cnt==3
        const index_t ele0[] = {n[0], newVertex[2], newVertex[1]};
        const index_t ele1[] = {n[1], newVertex[0], newVertex[2]};
        const index_t ele2[] = {n[2], newVertex[1], newVertex[0]};
        const index_t ele3[] = {newVertex[0], newVertex[1], newVertex[2]};

        const int ele0_boundary[] = {0, boundary[1], boundary[2]};
        const int ele1_boundary[] = {0, boundary[2], boundary[0]};
        const int ele2_boundary[] = {0, boundary[0], boundary[1]};
        const int ele3_boundary[] = {0, 0, 0};

        index_t ele1ID, ele2ID, ele3ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;

        // Update NNList
        def_ops->addNN(newVertex[0], newVertex[1], tid);
        def_ops->addNN(newVertex[0], newVertex[2], tid);
        def_ops->addNN(newVertex[1], newVertex[0], tid);
        def_ops->addNN(newVertex[1], newVertex[2], tid);
        def_ops->addNN(newVertex[2], newVertex[0], tid);
        def_ops->addNN(newVertex[2], newVertex[1], tid);

        // Update NEList
        def_ops->remNE(n[1], eid, tid);
        def_ops->addNE_fix(n[1], ele1ID, tid);
        def_ops->remNE(n[2], eid, tid);
        def_ops->addNE_fix(n[2], ele2ID, tid);

        def_ops->addNE_fix(newVertex[0], ele1ID, tid);
        def_ops->addNE_fix(newVertex[0], ele2ID, tid);
        def_ops->addNE_fix(newVertex[0], ele3ID, tid);

        def_ops->addNE(newVertex[1], eid, tid);
        def_ops->addNE_fix(newVertex[1], ele2ID, tid);
        def_ops->addNE_fix(newVertex[1], ele3ID, tid);

        def_ops->addNE(newVertex[2], eid, tid);
        def_ops->addNE_fix(newVertex[2], ele1ID, tid);
        def_ops->addNE_fix(newVertex[2], ele3ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
        assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);
        assert(ele3[0]>=0 && ele3[1]>=0 && ele3[2]>=0);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        splitCnt[tid] += 3;
      }
    }else{
      /*
       * 3D Element Refinement
       */

      int refine_cnt;
      std::vector<index_t> newVertex;
      std::vector< DirectedEdge<index_t> > splitEdges;
      std::vector<int> jk;
      for(int j=0, pos=0; j<4; j++)
        for(int k=j+1; k<4; k++){
          index_t vertexID = new_vertices_per_element[nedge*eid+pos];
          if(vertexID >= 0){
            newVertex.push_back(vertexID);
            splitEdges.push_back(DirectedEdge<index_t>(n[j], n[k]));
            jk.push_back(j);
            jk.push_back(k);
          }
          ++pos;
        }
      refine_cnt=newVertex.size();

      if(refine_cnt==0){
        // No refinement - continue to next element.
      }else if(refine_cnt==1){
        // Find the opposite edge
        index_t oe[2];
        for(int j=0, pos=0;j<4;j++)
          if(!splitEdges[0].contains(n[j])){
            oe[pos++] = n[j];
            jk.push_back(j);
          }

        // Form and add two new edges.
        const int ele0[] = {splitEdges[0].edge.first, newVertex[0], oe[0], oe[1]};
        const int ele1[] = {splitEdges[0].edge.second, newVertex[0], oe[0], oe[1]};

        const int ele0_boundary[] = {0, boundary[jk[1]], boundary[jk[2]], boundary[jk[3]]};
        const int ele1_boundary[] = {0, boundary[jk[0]], boundary[jk[2]], boundary[jk[3]]};

        index_t ele1ID;
        ele1ID = splitCnt[tid];

        // Add oe[0] and oe[1] to newVertex[0]'s NNList
        def_ops->addNN_unique(newVertex[0], oe[0], tid);
        def_ops->addNN_unique(newVertex[0], oe[1], tid);
        // Add newVertex[0] to oe[0] and oe[1]'s NNList
        def_ops->addNN_unique(oe[0], newVertex[0], tid);
        def_ops->addNN_unique(oe[1], newVertex[0], tid);

        // ele1ID is a new ID which isn't correct yet, it has to be
        // updated once each thread has calculated how many new elements
        // it created, so put ele1ID into addNE_fix instead of addNE.
        // Put ele1 in oe[0] and oe[1]'s NEList
        def_ops->addNE_fix(oe[0], ele1ID, tid);
        def_ops->addNE_fix(oe[0], ele1ID, tid);

        // Put eid and ele1 in newVertex[0]'s NEList
        def_ops->addNE(newVertex[0], eid, tid);
        def_ops->addNE_fix(newVertex[0], ele1ID, tid);

        // Replace eid with ele1 in splitEdges[0].edge.second's NEList
        def_ops->remNE(splitEdges[0].edge.second, eid, tid);
        def_ops->addNE_fix(splitEdges[0].edge.second, ele1ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        splitCnt[tid] += 1;
      }else if(refine_cnt==2){
        const int ele0[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
        const int ele1[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.second, newVertex[1]};
        const int ele2[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
        const int ele3[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.second, newVertex[1]};

        const int ele0_boundary[] = {0, boundary[jk[1]], 0, boundary[jk[3]]};
        const int ele1_boundary[] = {0, boundary[jk[1]], 0, boundary[jk[2]]};
        const int ele2_boundary[] = {0, boundary[jk[0]], 0, boundary[jk[3]]};
        const int ele3_boundary[] = {0, boundary[jk[0]], 0, boundary[jk[2]]};

        index_t ele1ID, ele2ID, ele3ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;

        def_ops->addNN_unique(newVertex[0], splitEdges[1].edge.first, tid);
        def_ops->addNN_unique(newVertex[0], splitEdges[1].edge.second, tid);
        def_ops->addNN_unique(splitEdges[1].edge.first, newVertex[0], tid);
        def_ops->addNN_unique(splitEdges[1].edge.second, newVertex[0], tid);
        def_ops->addNN_unique(newVertex[1], splitEdges[0].edge.first, tid);
        def_ops->addNN_unique(newVertex[1], splitEdges[0].edge.second, tid);
        def_ops->addNN_unique(splitEdges[0].edge.first, newVertex[1], tid);
        def_ops->addNN_unique(splitEdges[0].edge.second, newVertex[1], tid);

        def_ops->addNE(newVertex[0], eid, tid);
        def_ops->addNE_fix(newVertex[0], ele1ID, tid);
        def_ops->addNE_fix(newVertex[0], ele2ID, tid);
        def_ops->addNE_fix(newVertex[0], ele3ID, tid);

        def_ops->addNE(newVertex[1], eid, tid);
        def_ops->addNE_fix(newVertex[1], ele1ID, tid);
        def_ops->addNE_fix(newVertex[1], ele2ID, tid);
        def_ops->addNE_fix(newVertex[1], ele3ID, tid);

        def_ops->addNE_fix(splitEdges[0].edge.first, ele1ID, tid);

        def_ops->remNE(splitEdges[0].edge.second, eid, tid);
        def_ops->addNE_fix(splitEdges[0].edge.second, ele2ID, tid);
        def_ops->addNE_fix(splitEdges[0].edge.second, ele3ID, tid);

        def_ops->addNE_fix(splitEdges[1].edge.first, ele2ID, tid);

        def_ops->remNE(splitEdges[1].edge.second, eid, tid);
        def_ops->addNE_fix(splitEdges[1].edge.second, ele1ID, tid);
        def_ops->addNE_fix(splitEdges[1].edge.second, ele3ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        splitCnt[tid] += 3;
      }else if(refine_cnt==3){
        index_t m[] = {-1, -1, -1, -1, -1, -1, -1};
        int b[] = {-1, -1, -1, -1}; // boundary opposite of m[0] --> boundary[b[0]],
                                    //                      m[2] --> boundary[b[1]],
                                    //                      m[4] --> boundary[b[2]],
                                    //                      m[6] --> boundary[b[3]].
        m[0] = splitEdges[0].edge.first; b[0] = jk[0];
        m[1] = newVertex[0];
        m[2] = splitEdges[0].edge.second; b[1] = jk[1];
        if(splitEdges[1].contains(m[2])){
          m[3] = newVertex[1];
          if(splitEdges[1].edge.first!=m[2])
            {m[4] = splitEdges[1].edge.first; b[2] = jk[2];}
          else
            {m[4] = splitEdges[1].edge.second; b[2] = jk[3];}
          m[5] = newVertex[2];
        }else{
          m[3] = newVertex[2];
          if(splitEdges[2].edge.first!=m[2])
            {m[4] = splitEdges[2].edge.first; b[2] = jk[4];}
          else
            {m[4] = splitEdges[2].edge.second; b[2] = jk[5];}
          m[5] = newVertex[1];
        }
        for(int j=0;j<4;j++){
          if((n[j]!=m[0])&&(n[j]!=m[2])&&(n[j]!=m[4])){
            m[6] = n[j]; b[3] = j;
            break;
          }
        }

        const int ele0[] = {m[0], m[1], m[5], m[6]};
        const int ele1[] = {m[1], m[2], m[3], m[6]};
        const int ele2[] = {m[5], m[3], m[4], m[6]};
        const int ele3[] = {m[1], m[3], m[5], m[6]};

        const int ele0_boundary[] = {0, boundary[b[1]], boundary[b[2]], boundary[b[3]]};
        const int ele1_boundary[] = {boundary[b[0]], 0, boundary[b[2]], boundary[b[3]]};
        const int ele2_boundary[] = {boundary[b[0]], boundary[b[1]], 0, boundary[b[3]]};
        const int ele3_boundary[] = {0, 0, 0, boundary[b[3]]};

        index_t ele1ID, ele2ID, ele3ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;

        def_ops->addNN_unique(m[1], m[3], tid);
        def_ops->addNN_unique(m[1], m[5], tid);
        def_ops->addNN_unique(m[1], m[6], tid);
        def_ops->addNN_unique(m[3], m[1], tid);
        def_ops->addNN_unique(m[3], m[5], tid);
        def_ops->addNN_unique(m[3], m[6], tid);
        def_ops->addNN_unique(m[5], m[1], tid);
        def_ops->addNN_unique(m[5], m[3], tid);
        def_ops->addNN_unique(m[5], m[6], tid);
        def_ops->addNN_unique(m[6], m[1], tid);
        def_ops->addNN_unique(m[6], m[3], tid);
        def_ops->addNN_unique(m[6], m[5], tid);

        def_ops->addNE(m[1], eid, tid);
        def_ops->addNE_fix(m[1], ele1ID, tid);
        def_ops->addNE_fix(m[1], ele3ID, tid);

        def_ops->addNE(m[5], eid, tid);
        def_ops->addNE_fix(m[5], ele2ID, tid);
        def_ops->addNE_fix(m[5], ele3ID, tid);

        def_ops->addNE_fix(m[3], ele1ID, tid);
        def_ops->addNE_fix(m[3], ele2ID, tid);
        def_ops->addNE_fix(m[3], ele3ID, tid);

        def_ops->addNE_fix(m[6], ele1ID, tid);
        def_ops->addNE_fix(m[6], ele2ID, tid);
        def_ops->addNE_fix(m[6], ele3ID, tid);

        def_ops->remNE(m[2], eid, tid);
        def_ops->addNE_fix(m[2], ele1ID, tid);

        def_ops->remNE(m[4], eid, tid);
        def_ops->addNE_fix(m[4], ele2ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        splitCnt[tid] += 3;
      }else if(refine_cnt==6){
        const int ele0[] = {n[0], newVertex[0], newVertex[1], newVertex[2]};
        const int ele1[] = {n[1], newVertex[3], newVertex[0], newVertex[4]};
        const int ele2[] = {n[2], newVertex[1], newVertex[3], newVertex[5]};
        const int ele3[] = {newVertex[0], newVertex[3], newVertex[1], newVertex[4]};
        const int ele4[] = {newVertex[0], newVertex[4], newVertex[1], newVertex[2]};
        const int ele5[] = {newVertex[1], newVertex[3], newVertex[5], newVertex[4]};
        const int ele6[] = {newVertex[1], newVertex[4], newVertex[5], newVertex[2]};
        const int ele7[] = {newVertex[2], newVertex[4], newVertex[5], n[3]};

        const int ele0_boundary[] = {0, boundary[1], boundary[2], boundary[3]};
        const int ele1_boundary[] = {0, boundary[2], boundary[0], boundary[3]};
        const int ele2_boundary[] = {0, boundary[0], boundary[1], boundary[3]};
        const int ele3_boundary[] = {0, 0, 0, boundary[3]};
        const int ele4_boundary[] = {0, 0, boundary[2], 0};
        const int ele5_boundary[] = {boundary[0], 0, 0, 0};
        const int ele6_boundary[] = {0, boundary[1], 0, 0};
        const int ele7_boundary[] = {boundary[0], boundary[1], boundary[2], 0};

        index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID, ele6ID, ele7ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;
        ele4ID = ele1ID+3;
        ele5ID = ele1ID+4;
        ele6ID = ele1ID+5;
        ele7ID = ele1ID+6;

        def_ops->addNN_unique(newVertex[0], newVertex[1], tid);
        def_ops->addNN_unique(newVertex[1], newVertex[0], tid);
        def_ops->addNN_unique(newVertex[0], newVertex[2], tid);
        def_ops->addNN_unique(newVertex[2], newVertex[0], tid);
        def_ops->addNN_unique(newVertex[1], newVertex[2], tid);
        def_ops->addNN_unique(newVertex[2], newVertex[1], tid);
        def_ops->addNN_unique(newVertex[0], newVertex[3], tid);
        def_ops->addNN_unique(newVertex[3], newVertex[0], tid);
        def_ops->addNN_unique(newVertex[0], newVertex[4], tid);
        def_ops->addNN_unique(newVertex[4], newVertex[0], tid);
        def_ops->addNN_unique(newVertex[3], newVertex[4], tid);
        def_ops->addNN_unique(newVertex[4], newVertex[3], tid);
        def_ops->addNN_unique(newVertex[1], newVertex[3], tid);
        def_ops->addNN_unique(newVertex[3], newVertex[1], tid);
        def_ops->addNN_unique(newVertex[1], newVertex[5], tid);
        def_ops->addNN_unique(newVertex[5], newVertex[1], tid);
        def_ops->addNN_unique(newVertex[3], newVertex[5], tid);
        def_ops->addNN_unique(newVertex[5], newVertex[3], tid);
        def_ops->addNN_unique(newVertex[2], newVertex[4], tid);
        def_ops->addNN_unique(newVertex[4], newVertex[2], tid);
        def_ops->addNN_unique(newVertex[2], newVertex[5], tid);
        def_ops->addNN_unique(newVertex[5], newVertex[2], tid);
        def_ops->addNN_unique(newVertex[4], newVertex[5], tid);
        def_ops->addNN_unique(newVertex[5], newVertex[4], tid);

        def_ops->remNE(n[1], eid, tid);
        def_ops->addNE_fix(n[1], ele1ID, tid);

        def_ops->remNE(n[2], eid, tid);
        def_ops->addNE_fix(n[2], ele2ID, tid);

        def_ops->remNE(n[3], eid, tid);
        def_ops->addNE_fix(n[3], ele7ID, tid);

        def_ops->addNE(newVertex[0], eid, tid);
        def_ops->addNE_fix(newVertex[0], ele1ID, tid);
        def_ops->addNE_fix(newVertex[0], ele3ID, tid);
        def_ops->addNE_fix(newVertex[0], ele4ID, tid);

        def_ops->addNE(newVertex[1], eid, tid);
        def_ops->addNE_fix(newVertex[1], ele2ID, tid);
        def_ops->addNE_fix(newVertex[1], ele3ID, tid);
        def_ops->addNE_fix(newVertex[1], ele4ID, tid);
        def_ops->addNE_fix(newVertex[1], ele5ID, tid);
        def_ops->addNE_fix(newVertex[1], ele6ID, tid);

        def_ops->addNE(newVertex[2], eid, tid);
        def_ops->addNE_fix(newVertex[2], ele4ID, tid);
        def_ops->addNE_fix(newVertex[2], ele6ID, tid);
        def_ops->addNE_fix(newVertex[2], ele7ID, tid);

        def_ops->addNE_fix(newVertex[3], ele1ID, tid);
        def_ops->addNE_fix(newVertex[3], ele2ID, tid);
        def_ops->addNE_fix(newVertex[3], ele3ID, tid);
        def_ops->addNE_fix(newVertex[3], ele5ID, tid);

        def_ops->addNE_fix(newVertex[4], ele1ID, tid);
        def_ops->addNE_fix(newVertex[4], ele3ID, tid);
        def_ops->addNE_fix(newVertex[4], ele4ID, tid);
        def_ops->addNE_fix(newVertex[4], ele5ID, tid);
        def_ops->addNE_fix(newVertex[4], ele6ID, tid);
        def_ops->addNE_fix(newVertex[4], ele7ID, tid);

        def_ops->addNE_fix(newVertex[5], ele2ID, tid);
        def_ops->addNE_fix(newVertex[5], ele5ID, tid);
        def_ops->addNE_fix(newVertex[5], ele6ID, tid);
        def_ops->addNE_fix(newVertex[5], ele7ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        append_element(ele4, ele4_boundary, tid);
        append_element(ele5, ele5_boundary, tid);
        append_element(ele6, ele6_boundary, tid);
        append_element(ele7, ele7_boundary, tid);
        splitCnt[tid] += 7;
      }
    }
  }

  inline void append_element(const index_t *elem, const int *boundary, const size_t tid){
    for(size_t i=0; i<nloc; ++i){
      newElements[tid].push_back(elem[i]);
      newBoundaries[tid].push_back(boundary[i]);
    }
  }

  inline void replace_element(const index_t eid, const index_t *n, const int *boundary){
    for(size_t i=0;i<nloc;i++){
      _mesh->_ENList[eid*nloc+i]=n[i];
      _mesh->boundary[eid*nloc+i]=boundary[i];
    }
  }

  inline size_t edgeNumber(index_t eid, index_t v1, index_t v2) const{
    const int *n=_mesh->get_element(eid);

    if(dim==2){
      /* In 2D:
       * Edge 0 is the edge (n[1],n[2]).
       * Edge 1 is the edge (n[0],n[2]).
       * Edge 2 is the edge (n[0],n[1]).
       */
      if(n[1]==v1 || n[1]==v2){
        if(n[2]==v1 || n[2]==v2)
          return 0;
        else
          return 2;
      }
      else
        return 1;
    }else{ //if(dim=3)
      /*
       * In 3D:
       * Edge 0 is the edge (n[0],n[1]).
       * Edge 1 is the edge (n[0],n[2]).
       * Edge 2 is the edge (n[0],n[3]).
       * Edge 3 is the edge (n[1],n[2]).
       * Edge 4 is the edge (n[1],n[3]).
       * Edge 5 is the edge (n[2],n[3]).
       */
      if(n[0]==v1 || n[0]==v2){
        if(n[1]==v1 || n[1]==v2)
          return 0;
        else if(n[2]==v1 || n[2]==v2)
          return 1;
        else
          return 2;
      }else if(n[1]==v1 || n[1]==v2){
        if(n[2]==v1 || n[2]==v2)
          return 3;
        else
          return 4;
      }else
        return 5;
    }
  }

  std::vector< std::vector< DirectedEdge<index_t> > > newVertices;
  std::vector< std::vector<real_t> > newCoords;
  std::vector< std::vector<double> > newMetric;
  std::vector< std::vector<index_t> > newElements;
  std::vector< std::vector<int> > newBoundaries;
  std::vector<index_t> new_vertices_per_element;

  std::vector<size_t> threadIdx, splitCnt;
  std::vector< DirectedEdge<index_t> > allNewVertices;

  DeferredOperations<real_t>* def_ops;
  static const int defOp_scaling_factor = 32;

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  std::vector< std::set<index_t> > marked_edges;

  static const size_t ndims=dim, nloc=(dim+1), msize=(dim==2?3:6), nedge=(dim==2?3:6);
  int nprocs, rank, nthreads;
};


#endif
