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

#pragma omp parallel
    {
#pragma omp single nowait
      {
        new_vertices_per_element.resize(nedge*origNElements);
        std::fill(new_vertices_per_element.begin(), new_vertices_per_element.end(), -1);
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
            }
          }
        }
      }

      threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NNodes, splitCnt[tid]);
      assert(newVertices[tid].size()==splitCnt[tid]);

#pragma omp barrier

#pragma omp single
      {
        if(_mesh->_coords.size()<_mesh->NNodes*ndims){
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
        }
        _mesh->lnn2gnn[vid] = vid;
      }

      if(dim==3){
        // If in 3D, we need to refine facets first.
#pragma omp for schedule(guided)
        for(index_t eid=0; eid<origNElements; ++eid){
          // Find the 4 facets comprising the element
          const index_t *n = _mesh->get_element(eid);
          if(n[0] < 0)
            continue;

          const index_t facets[4][3] = {{n[0], n[1], n[2]},
                                        {n[0], n[1], n[3]},
                                        {n[0], n[2], n[3]},
                                        {n[1], n[2], n[3]}};

          for(int j=0; j<4; ++j){
            // Find which elements share this facet j
            const index_t *facet = facets[j];
            std::set<index_t> intersection01, EE;
            std::set_intersection(_mesh->NEList[facet[0]].begin(), _mesh->NEList[facet[0]].end(),
                                  _mesh->NEList[facet[1]].begin(), _mesh->NEList[facet[1]].end(),
                                  std::inserter(intersection01, intersection01.begin()));
            std::set_intersection(_mesh->NEList[facet[2]].begin(), _mesh->NEList[facet[2]].end(),
                                  intersection01.begin(), intersection01.end(),
                                  std::inserter(EE, EE.begin()));

            assert(EE.size() <= 2 );
            assert(EE.count(eid) == 1);

            // Prevent facet from being refined twice:
            // Only refine it if this is the element with the highest ID.
            if(eid == *EE.rbegin())
              for(size_t k=0; k<3; ++k)
                if(new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[k], facet[(k+1)%3])] != -1){
                  refine_facet(eid, facet, tid);
                  break;
                }
          }
        }

#pragma omp for schedule(guided)
        for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
          for(int i=0; i<nthreads; ++i){
            def_ops->commit_remNN(i, vtid);
            def_ops->commit_addNN(i, vtid);
          }
        }
      }

      // Update halo - we need to update the global node numbering here
      // for those cases in 3D where centroidal vertices are introduced.
#pragma omp single
      {
        if(nprocs==1){
          // If we update lnn2gnn and node_owner here, OMP performance suffers.
        }else{
#ifdef HAVE_MPI
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
#endif
        }
      }

      // Start element refinement.
      splitCnt[tid] = 0;
      newElements[tid].clear(); newBoundaries[tid].clear();
      newElements[tid].reserve(dim*dim*origNElements/nthreads);
      newBoundaries[tid].reserve(dim*dim*origNElements/nthreads);

#pragma omp for schedule(guided) nowait
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

#pragma omp barrier
#pragma omp single
      {
        if(_mesh->_ENList.size()<_mesh->NElements*nloc){
          _mesh->_ENList.resize(_mesh->NElements*nloc);
          _mesh->boundary.resize(_mesh->NElements*nloc);
        }
      }

      // Commit deferred operations.
#pragma omp for schedule(guided)
      for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
        for(int i=0; i<nthreads; ++i){
          def_ops->commit_remNN(i, vtid);
          def_ops->commit_addNN(i, vtid);
          def_ops->commit_remNE(i, vtid);
          def_ops->commit_addNE(i, vtid);
          def_ops->commit_addNE_fix(threadIdx, i, vtid);
        }
      }

      if(dim==2){
#if !defined NDEBUG
#pragma omp barrier
      // Fix orientations of new elements.
      size_t NElements = _mesh->get_number_elements();

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
#endif
      }else if(dim==3){
#pragma omp barrier
        // Fix orientations of new elements.
        size_t NElements = _mesh->get_number_elements();

        real_t *tcoords = &(_mesh->_coords[0]);

#pragma omp for schedule(guided)
        for(int i=0; i<NElements; i++){
          index_t n0 = _mesh->_ENList[i*nloc];
          index_t n1 = _mesh->_ENList[i*nloc + 1];
          index_t n2 = _mesh->_ENList[i*nloc + 2];
          index_t n3 = _mesh->_ENList[i*nloc + 3];

          const real_t *x0 = tcoords + n0*ndims;
          const real_t *x1 = tcoords + n1*ndims;
          const real_t *x2 = tcoords + n2*ndims;
          const real_t *x3 = tcoords + n3*ndims;

          real_t av = property->volume(x0, x1, x2, x3);

          if(av<0){
            // Flip element
            _mesh->_ENList[i*nloc] = n1;
            _mesh->_ENList[i*nloc+1] = n0;

            // and boundary
            int b0 = _mesh->boundary[i*nloc];
            _mesh->boundary[i*nloc] = _mesh->boundary[i*nloc + 1];
            _mesh->boundary[i*nloc + 1] = b0;
          }
        }
      }
    }
  }

 private:

  void refine_edge(index_t n0, index_t n1, int tid){
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

  void refine_facet(index_t eid, const index_t *facet, int tid){
    const index_t *n=_mesh->get_element(eid);

    index_t newVertex[3] = {-1, -1, -1};
    newVertex[0] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[1], facet[2])];
    newVertex[1] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[0], facet[2])];
    newVertex[2] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[0], facet[1])];

    int refine_cnt=0;
    for(size_t i=0; i<3; ++i)
      if(newVertex[i]!=-1)
        ++refine_cnt;

    switch(refine_cnt){
    case 0:
      // Do nothing
      break;
    case 1:
      // 1:2 facet bisection
      for(int j=0; j<3; j++)
        if(newVertex[j] >= 0){
          def_ops->addNN(newVertex[j], facet[j], tid);
          def_ops->addNN(facet[j], newVertex[j], tid);
          break;
        }
      break;
    case 2:
      // 1:3 refinement with trapezoid split
      for(int j=0; j<3; j++){
        if(newVertex[j] < 0){
          def_ops->addNN(newVertex[(j+1)%3], newVertex[(j+2)%3], tid);
          def_ops->addNN(newVertex[(j+2)%3], newVertex[(j+1)%3], tid);

          real_t ldiag1 = _mesh->calc_edge_length(newVertex[(j+1)%3], facet[(j+1)%3]);
          real_t ldiag2 = _mesh->calc_edge_length(newVertex[(j+2)%3], facet[(j+2)%3]);
          const int offset = ldiag1 < ldiag2 ? (j+1)%3 : (j+2)%3;

          def_ops->addNN(newVertex[offset], facet[offset], tid);
          def_ops->addNN(facet[offset], newVertex[offset], tid);

          break;
        }
      }
      break;
    case 3:
      // 1:4 regular refinement
      for(int j=0; j<3; j++){
        def_ops->addNN(newVertex[j], newVertex[(j+1)%3], tid);
        def_ops->addNN(newVertex[(j+1)%3], newVertex[j], tid);
      }
      break;
    default:
      break;
    }
  }

  void refine_element(size_t eid, int tid){
    const int *n=_mesh->get_element(eid);
    const int *boundary=&(_mesh->boundary[eid*nloc]);

    if(dim==2){
      /*
       *************************
       * 2D Element Refinement *
       *************************
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
       *************************
       * 3D Element Refinement *
       *************************
       */
      int refine_cnt;
      std::vector<index_t> newVertex;
      std::vector< DirectedEdge<index_t> > splitEdges;
      for(int j=0, pos=0; j<4; j++)
        for(int k=j+1; k<4; k++){
          index_t vertexID = new_vertices_per_element[nedge*eid+pos];
          if(vertexID >= 0){
            newVertex.push_back(vertexID);
            splitEdges.push_back(DirectedEdge<index_t>(n[j], n[k]));
          }
          ++pos;
        }
      refine_cnt=newVertex.size();

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
  boost::unordered_map<index_t, int> b;
#else
  std::map<index_t, int> b;
#endif

      for(int j=0; j<nloc; ++j)
        b[n[j]] = boundary[j];

      if(refine_cnt==0){
        // No refinement - continue to next element.
      }else if(refine_cnt==1){
        // Find the opposite edge
        index_t oe[2];
        for(int j=0, pos=0;j<4;j++)
          if(!splitEdges[0].contains(n[j]))
            oe[pos++] = n[j];

        // Form and add two new edges.
        const int ele0[] = {splitEdges[0].edge.first, newVertex[0], oe[0], oe[1]};
        const int ele1[] = {splitEdges[0].edge.second, newVertex[0], oe[0], oe[1]};

        const int ele0_boundary[] = {0, b[splitEdges[0].edge.second], b[oe[0]], b[oe[1]]};
        const int ele1_boundary[] = {0, b[splitEdges[0].edge.first], b[oe[0]], b[oe[1]]};

        index_t ele1ID;
        ele1ID = splitCnt[tid];

        // ele1ID is a new ID which isn't correct yet, it has to be
        // updated once each thread has calculated how many new elements
        // it created, so put ele1ID into addNE_fix instead of addNE.
        // Put ele1 in oe[0] and oe[1]'s NEList
        def_ops->addNE_fix(oe[0], ele1ID, tid);
        def_ops->addNE_fix(oe[1], ele1ID, tid);

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
        /* Here there are two possibilities. Either the two split
         * edges share a vertex (case 2(a)) or they are opposite edges
         * (case 2(b)). Case 2(a) results in a 1:3 subdivision, case 2(b)
         * results in a 1:4.
         */

        int n0=splitEdges[0].connected(splitEdges[1]);
        if(n0>=0){
          // Case 2(a).
          int n1 = (n0 == splitEdges[0].edge.first) ? splitEdges[0].edge.second : splitEdges[0].edge.first;
          int n2 = (n0 == splitEdges[1].edge.first) ? splitEdges[1].edge.second : splitEdges[1].edge.first;

          // Opposite vertex
          int n3;
          for(int j=0; j<nloc; ++j)
            if(n[j] != n0 && n[j] != n1 && n[j] != n2){
              n3 = n[j];
              break;
            }

          // Find the diagonal which has bisected the trapezoid.
          DirectedEdge<index_t> diagonal, offdiagonal;
          std::vector<index_t>::const_iterator p = std::find(_mesh->NNList[newVertex[0]].begin(),
              _mesh->NNList[newVertex[0]].end(), n2);
          if(p != _mesh->NNList[newVertex[0]].end()){
            diagonal.edge.first = newVertex[0];
            diagonal.edge.second = n[2];
            offdiagonal.edge.first = newVertex[1];
            offdiagonal.edge.second = n[1];
          }else{
            diagonal.edge.first = newVertex[1];
            diagonal.edge.second = n[1];
            offdiagonal.edge.first = newVertex[0];
            offdiagonal.edge.second = n[0];
          }

          const int ele0[] = {n0, newVertex[0], newVertex[1], n3};
          const int ele1[] = {diagonal.edge.first, offdiagonal.edge.first, diagonal.edge.second, n3};
          const int ele2[] = {diagonal.edge.first, diagonal.edge.second, offdiagonal.edge.second, n3};

          const int ele0_boundary[] = {0, b[n1], b[n2], b[n3]};
          const int ele1_boundary[] = {b[offdiagonal.edge.second], 0, 0, b[n3]};
          const int ele2_boundary[] = {b[n0], b[diagonal.edge.second], 0, b[n3]};

          index_t ele1ID, ele2ID;
          ele1ID = splitCnt[tid];
          ele2ID = ele1ID+1;

          def_ops->addNE(diagonal.edge.first, eid, tid);
          def_ops->addNE_fix(diagonal.edge.first, ele1ID, tid);
          def_ops->addNE_fix(diagonal.edge.first, ele2ID, tid);

          def_ops->remNE(diagonal.edge.second, eid, tid);
          def_ops->addNE_fix(diagonal.edge.first, ele1ID, tid);
          def_ops->addNE_fix(diagonal.edge.first, ele2ID, tid);

          def_ops->addNE(offdiagonal.edge.first, eid, tid);
          def_ops->addNE_fix(offdiagonal.edge.first, ele1ID, tid);

          def_ops->remNE(offdiagonal.edge.second, eid, tid);
          def_ops->addNE_fix(offdiagonal.edge.second, ele2ID, tid);

          def_ops->remNE(n3, eid, tid);
          def_ops->addNE_fix(n3, ele1ID, tid);
          def_ops->addNE_fix(n3, ele2ID, tid);

          replace_element(eid, ele0, ele0_boundary);
          append_element(ele1, ele1_boundary, tid);
          append_element(ele2, ele2_boundary, tid);
          splitCnt[tid] += 2;
        }else{
          // Case 2(b).
          const int ele0[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
          const int ele1[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.second, newVertex[1]};
          const int ele2[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
          const int ele3[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.second, newVertex[1]};

          const int ele0_boundary[] = {0, b[splitEdges[0].edge.second], 0, b[splitEdges[1].edge.second]};
          const int ele1_boundary[] = {0, b[splitEdges[0].edge.second], 0, b[splitEdges[1].edge.first]};
          const int ele2_boundary[] = {0, b[splitEdges[0].edge.first], 0, b[splitEdges[1].edge.second]};
          const int ele3_boundary[] = {0, b[splitEdges[0].edge.first], 0, b[splitEdges[1].edge.first]};

          index_t ele1ID, ele2ID, ele3ID;
          ele1ID = splitCnt[tid];
          ele2ID = ele1ID+1;
          ele3ID = ele1ID+2;

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
        }
      }else if(refine_cnt==3){
        /* There are 3 cases that need to be considered. They can
         * be distinguished by the total number of nodes that are
         * common between any pair of edges.
         * Case 3(a): there are 3 different nodes common between pairs
         * of split edges, i.e the three new vertices are on the
         * same triangle.
         * Case 3(b): The three new vertices are around the same
         * original vertex.
         * Case 3(c): There are 2 different nodes common between pairs
         * of split edges.
         */
        std::set<index_t> shared;
        for(int j=0;j<3;j++){
          for(int k=j+1;k<3;k++){
            index_t nid = splitEdges[j].connected(splitEdges[k]);
            if(nid>=0)
              shared.insert(nid);
          }
        }
        size_t nshared = shared.size();

        if(nshared==3){
          // Case 3(a).
          index_t m[] = {-1, -1, -1, -1, -1, -1, -1};

          m[0] = splitEdges[0].edge.first;
          m[1] = newVertex[0];
          m[2] = splitEdges[0].edge.second;
          if(splitEdges[1].contains(m[2])){
            m[3] = newVertex[1];
            if(splitEdges[1].edge.first!=m[2])
              m[4] = splitEdges[1].edge.first;
            else
              m[4] = splitEdges[1].edge.second;
            m[5] = newVertex[2];
          }else{
            m[3] = newVertex[2];
            if(splitEdges[2].edge.first!=m[2])
              m[4] = splitEdges[2].edge.first;
            else
              m[4] = splitEdges[2].edge.second;
            m[5] = newVertex[1];
          }
          for(int j=0;j<4;j++){
            if((n[j]!=m[0])&&(n[j]!=m[2])&&(n[j]!=m[4])){
              m[6] = n[j];
              break;
            }
          }

          const int ele0[] = {m[0], m[1], m[5], m[6]};
          const int ele1[] = {m[1], m[2], m[3], m[6]};
          const int ele2[] = {m[5], m[3], m[4], m[6]};
          const int ele3[] = {m[1], m[3], m[5], m[6]};

          const int ele0_boundary[] = {0, b[m[2]], b[m[4]], b[m[6]]};
          const int ele1_boundary[] = {b[m[0]], 0, b[m[4]], b[m[6]]};
          const int ele2_boundary[] = {b[m[0]], b[m[2]], 0, b[m[6]]};
          const int ele3_boundary[] = {0, 0, 0, b[m[6]]};

          index_t ele1ID, ele2ID, ele3ID;
          ele1ID = splitCnt[tid];
          ele2ID = ele1ID+1;
          ele3ID = ele1ID+2;

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
        }else if(nshared==1){
          // Case 3(b).

          // Find the three bottom vertices, i.e. vertices of
          // the original elements which are part of the wedge.
          index_t top_vertex = *shared.begin();
          index_t bottom_vertex[3];
          for(int j=0; j<3; ++n){
            assert(splitEdges[j].id == newVertex[j]);
            if(splitEdges[j].edge.first != top_vertex){
              bottom_vertex[j] = splitEdges[j].edge.first;
            }else{
              bottom_vertex[j] = splitEdges[j].edge.second;
            }
          }

          // For each quadrilateral side of the wedge find
          // the diagonal which has bisected the wedge side.
          // Side0: bottom[0] - bottom[1] - new[1] - new[0]
          // Side1: bottom[1] - bottom[2] - new[2] - new[1]
          // Side2: bottom[2] - bottom[0] - new[0] - new[2]
          std::vector< DirectedEdge<index_t> > diagonals, ghostDiagonals;
          for(int j=0; j<3; ++j){
            std::vector<index_t>::const_iterator p = std::find(_mesh->NNList[bottom_vertex[j]].begin(),
                _mesh->NNList[bottom_vertex[j]].end(), newVertex[(j+1)%3]);
            if(p != _mesh->NNList[bottom_vertex[j]].end()){
              diagonals.push_back(DirectedEdge<index_t>(bottom_vertex[j], newVertex[(j+1)%3]));
              ghostDiagonals.push_back(DirectedEdge<index_t>(bottom_vertex[(j+1)%3], newVertex[j]));
            }else{
              diagonals.push_back(DirectedEdge<index_t>(bottom_vertex[(j+1)%3], newVertex[j]));
              ghostDiagonals.push_back(DirectedEdge<index_t>(bottom_vertex[j], newVertex[(j+1)%3]));
            }
          }

          // Determine how the wedge will be split
          std::vector<index_t> diag_shared;
          for(int j=0;j<3;j++){
            index_t nid = diagonals[j].connected(diagonals[(j+1)%3]);
            if(nid>=0)
              diag_shared.push_back(nid);
          }

          if(!diag_shared.empty()){
            assert(diag_shared.size() == 2);
            // Here we can subdivide the wedge into 3 tetrahedra.

            // Find the "middle" diagonal, i.e. the one which
            // consists of the two vertices in diag_shared.
            int middle;
            index_t non_shared_top=-1, non_shared_bottom=-1;
            for(int j=0; j<3; ++j){
              if(diagonals[j].contains(diag_shared[0]) && diagonals[j].contains(diag_shared[1])){
                middle = j;
                for(int k=0; k<2; ++k){
                  if(diagonals[(j+k+1)%3].edge.first != diag_shared[0] && diagonals[(j+k+1)%3].edge.first != diag_shared[1])
                    non_shared_bottom = diagonals[(j+k+1)%3].edge.first;
                  else
                    non_shared_top = diagonals[(j+k+1)%3].edge.second;
                }
                break;
              }
            }
            assert(non_shared_top >= 0 && non_shared_bottom >= 0);

            /*
             * 2 elements are formed by the three vertices of two connected
             * diagonals plus a fourth vertex which can be found via the
             * intersection on NNList of those three vertices. Any vertex which
             * is part of a diagonal must be removed from the intersection and
             * then the intersection will be left with one and only vertex.
             *
             * 1 element is formed by the four vertices of two disjoint diagonals.
             */
            index_t v_top, v_bottom;
            std::set<index_t> intersection0, intersection1;
            std::set_intersection(_mesh->NNList[diagonals[middle].edge.first].begin(),
                _mesh->NNList[diagonals[middle].edge.first].end(), _mesh->NNList[diagonals[middle].edge.second].begin(),
                _mesh->NNList[diagonals[middle].edge.second].end(), std::inserter(intersection0, intersection0.begin()));
            std::set_intersection(_mesh->NNList[non_shared_top].begin(), _mesh->NNList[non_shared_top].end(),
                intersection0.begin(), intersection0.end(), std::inserter(intersection1, intersection1.begin()));
            intersection1.erase(non_shared_bottom);
            assert(intersection1.size()==1);
            v_top = *intersection1.begin();

            intersection1.clear();
            std::set_intersection(_mesh->NNList[non_shared_bottom].begin(), _mesh->NNList[non_shared_bottom].end(),
                intersection0.begin(), intersection0.end(), std::inserter(intersection1, intersection1.begin()));
            intersection1.erase(non_shared_top);
            assert(intersection1.size()==1);
            v_bottom = *intersection1.begin();

            // diagonals[middle].edge.first is always one of the original element vertices (bottom)
            // diagonals[middle].edge.second is always one of the new vertices (top)
            const int ele0[] = {top_vertex, newVertex[0], newVertex[1], newVertex[2]};
            const int ele1[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, non_shared_top, v_top};
            const int ele2[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, non_shared_bottom, v_bottom};
            const int ele3[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, non_shared_top, non_shared_bottom};

            const int ele0_boundary[] = {0, b[bottom_vertex[0]], b[bottom_vertex[1]], b[bottom_vertex[2]]};
            const int ele1_boundary[] = {0, b[v_bottom], b[non_shared_bottom], 0};
            const int ele2_boundary[] = {b[diagonals[middle].edge.first], b[top_vertex], b[non_shared_bottom], 0};
            const int ele3_boundary[] = {b[diagonals[middle].edge.first], b[v_bottom], 0, 0};

            index_t ele1ID, ele2ID, ele3ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;

            def_ops->remNE(diagonals[middle].edge.first, eid, tid);
            def_ops->addNE_fix(diagonals[middle].edge.first, ele1ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.first, ele2ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.first, ele3ID, tid);

            def_ops->addNE(diagonals[middle].edge.second, eid, tid);
            def_ops->addNE_fix(diagonals[middle].edge.second, ele1ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.second, ele2ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.second, ele3ID, tid);

            def_ops->remNE(non_shared_bottom, eid, tid);
            def_ops->addNE_fix(non_shared_bottom, ele2ID, tid);
            def_ops->addNE_fix(non_shared_bottom, ele3ID, tid);

            def_ops->addNE(non_shared_top, eid, tid);
            def_ops->addNE_fix(non_shared_top, ele1ID, tid);
            def_ops->addNE_fix(non_shared_top, ele3ID, tid);

            def_ops->remNE(v_bottom, eid, tid);
            def_ops->addNE_fix(v_bottom, ele2ID, tid);

            def_ops->addNE(v_top, eid, tid);
            def_ops->addNE_fix(v_top, ele1ID, tid);

            replace_element(eid, ele0, ele0_boundary);
            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            splitCnt[tid] += 3;
          }else{
            /*
             * The wedge must by split into 8 tetrahedra with the introduction
             * of a new centroidal vertex. Each tetrahedron is formed by the
             * three vertices of a triangular facet (there are 8 triangular
             * facets: 6 are formed via the bisection of the 3 quadrilaterals
             * of the wedge, 2 are the original triangular facets of the wedge)
             * and the centroidal vertex.
             */

            // Allocate space for the centroid vertex
            index_t cid = pragmatic_omp_atomic_capture(&_mesh->NNodes, 1);

            const int ele0[] = {top_vertex, newVertex[0], newVertex[1], newVertex[2]};
            const int ele1[] = {diagonals[0].edge.first, ghostDiagonals[0].edge.first, diagonals[0].edge.second, cid};
            const int ele2[] = {diagonals[0].edge.first, diagonals[0].edge.second, ghostDiagonals[0].edge.second, cid};
            const int ele3[] = {diagonals[1].edge.first, ghostDiagonals[1].edge.first, diagonals[1].edge.second, cid};
            const int ele4[] = {diagonals[1].edge.first, diagonals[1].edge.second, ghostDiagonals[1].edge.second, cid};
            const int ele5[] = {diagonals[2].edge.first, ghostDiagonals[2].edge.first, diagonals[2].edge.second, cid};
            const int ele6[] = {diagonals[2].edge.first, diagonals[2].edge.second, ghostDiagonals[2].edge.second, cid};
            const int ele7[] = {newVertex[0], newVertex[1], newVertex[2], cid};
            const int ele8[] = {cid, bottom_vertex[0], bottom_vertex[2], bottom_vertex[1]};

            const int ele0_boundary[] = {0, b[bottom_vertex[0]], b[bottom_vertex[1]], b[bottom_vertex[2]]};
            const int ele1_boundary[] = {0, 0, 0, b[bottom_vertex[2]]};
            const int ele2_boundary[] = {0, 0, 0, b[bottom_vertex[2]]};
            const int ele3_boundary[] = {0, 0, 0, b[bottom_vertex[0]]};
            const int ele4_boundary[] = {0, 0, 0, b[bottom_vertex[0]]};
            const int ele5_boundary[] = {0, 0, 0, b[bottom_vertex[1]]};
            const int ele6_boundary[] = {0, 0, 0, b[bottom_vertex[1]]};
            const int ele7_boundary[] = {0, 0, 0, 0};
            const int ele8_boundary[] = {b[top_vertex], 0, 0, 0};

            index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID, ele6ID, ele7ID, ele8ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;
            ele4ID = ele1ID+3;
            ele5ID = ele1ID+4;
            ele6ID = ele1ID+5;
            ele7ID = ele1ID+6;
            ele8ID = ele1ID+7;

            for(int j=0; j<3; ++j){
              _mesh->NNList[cid].push_back(newVertex[j]);
              _mesh->NNList[cid].push_back(bottom_vertex[j]);
            }

            def_ops->addNE_fix(cid, ele1ID, tid);
            def_ops->addNE_fix(cid, ele2ID, tid);
            def_ops->addNE_fix(cid, ele3ID, tid);
            def_ops->addNE_fix(cid, ele4ID, tid);
            def_ops->addNE_fix(cid, ele5ID, tid);
            def_ops->addNE_fix(cid, ele6ID, tid);
            def_ops->addNE_fix(cid, ele7ID, tid);
            def_ops->addNE_fix(cid, ele8ID, tid);

            def_ops->remNE(bottom_vertex[0], eid, tid);
            def_ops->remNE(bottom_vertex[1], eid, tid);
            def_ops->remNE(bottom_vertex[2], eid, tid);
            def_ops->addNE_fix(bottom_vertex[0], ele8ID, tid);
            def_ops->addNE_fix(bottom_vertex[1], ele8ID, tid);
            def_ops->addNE_fix(bottom_vertex[2], ele8ID, tid);

            def_ops->addNE(newVertex[0], eid, tid);
            def_ops->addNE(newVertex[1], eid, tid);
            def_ops->addNE(newVertex[2], eid, tid);
            def_ops->addNE_fix(newVertex[0], ele7ID, tid);
            def_ops->addNE_fix(newVertex[1], ele7ID, tid);
            def_ops->addNE_fix(newVertex[2], ele7ID, tid);

            def_ops->addNE_fix(diagonals[0].edge.first, ele1ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.first, ele2ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.second, ele1ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.second, ele2ID, tid);
            def_ops->addNE_fix(ghostDiagonals[0].edge.first, ele1ID, tid);
            def_ops->addNE_fix(ghostDiagonals[0].edge.second, ele2ID, tid);

            def_ops->addNE_fix(diagonals[1].edge.first, ele3ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.first, ele4ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.second, ele3ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.second, ele4ID, tid);
            def_ops->addNE_fix(ghostDiagonals[1].edge.first, ele3ID, tid);
            def_ops->addNE_fix(ghostDiagonals[1].edge.second, ele4ID, tid);

            def_ops->addNE_fix(diagonals[2].edge.first, ele5ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.first, ele6ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.second, ele5ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.second, ele6ID, tid);
            def_ops->addNE_fix(ghostDiagonals[2].edge.first, ele5ID, tid);
            def_ops->addNE_fix(ghostDiagonals[2].edge.second, ele6ID, tid);

            replace_element(eid, ele0, ele0_boundary);
            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            append_element(ele4, ele4_boundary, tid);
            append_element(ele5, ele5_boundary, tid);
            append_element(ele6, ele6_boundary, tid);
            append_element(ele7, ele7_boundary, tid);
            append_element(ele8, ele8_boundary, tid);
            splitCnt[tid] += 8;

            // Sort all 6 vertices of the wedge by ascending gnn
            // Need to do so to enforce consistency across MPI processes
            std::map<index_t, index_t> gnn2lnn;
            for(int j=0; j<3; ++j){
              gnn2lnn[_mesh->lnn2gnn[bottom_vertex[j]]] = bottom_vertex[j];
              gnn2lnn[_mesh->lnn2gnn[newVertex[j]]] = newVertex[j];
            }

            // Calculate the coordinates of the centroidal vertex
            // Start with a temporary location at one of the wedge's corners, e.g. *gnn2lnn.begin().
            real_t nc[ndims]; // new coords
            double nm[msize]; // new metric
            _mesh->get_coords(gnn2lnn.begin()->second, nc);
            _mesh->get_metric(gnn2lnn.begin()->second, nm);

            // Use the 3D laplacian smoothing kernel to find the barycentre of the wedge.
            Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A =
                Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 3);
            Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);

            for(std::map<index_t, index_t>::const_iterator it=gnn2lnn.begin(); it!=gnn2lnn.end(); ++it){
              const real_t *x = _mesh->get_coords(it->second);

              q[0] += nm[0]*x[0] + nm[1]*x[1] + nm[2]*x[2];
              q[1] += nm[1]*x[0] + nm[3]*x[1] + nm[4]*x[2];
              q[2] += nm[2]*x[0] + nm[4]*x[1] + nm[6]*x[2];

              A[0] += nm[0]; A[1] += nm[1]; A[2] += nm[2];
              A[4] += nm[3]; A[5] += nm[4];
              A[8] += nm[6];
            }
            A[3] = A[1];
            A[6] = A[2];
            A[7] = A[5];

            // Want to solve the system Ap=q to find the new position, p.
            Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
            A.svd().solve(q, &b);

            for(int i=0;i<3;i++){
              nc[i] += b[i];
              _mesh->_coords[cid*3+i] = nc[i];
            }

            // Interpolate metric at new location
            real_t l[]={-1, -1, -1, -1};
            int best_e=-1;
            real_t tol=-1;

            const index_t *welements[] = {ele1, ele2, ele3, ele4, ele5, ele6, ele7, ele8};
            for(int i=0; i<8; ++i){
              const real_t *x0 = _mesh->get_coords(welements[i][0]);
              const real_t *x1 = _mesh->get_coords(welements[i][1]);
              const real_t *x2 = _mesh->get_coords(welements[i][2]);
              const real_t *x3 = _mesh->get_coords(welements[i][3]);

              real_t L = property->volume(x0, x1, x2, x3);

              real_t ll[4];
              ll[0] = property->volume(nc, x1, x2, x3)/L;
              ll[1] = property->volume(x0, nc, x2, x3)/L;
              ll[2] = property->volume(x0, x1, nc, x3)/L;
              ll[3] = property->volume(x0, x1, x2, nc)/L;

              real_t min_l = std::min(std::min(ll[0], ll[1]), std::min(ll[2], ll[3]));
              if(best_e==-1){
                tol = min_l;
                best_e = i;
                for(int j=0;j<nloc;++j)
                  l[j] = ll[j];
              }else{
                if(min_l>tol){
                  tol = min_l;
                  best_e = i;
                  for(int j=0;j<nloc;++j)
                    l[j] = ll[j];
                }
              }
            }

            const index_t *best_nodes = welements[best_e];
            assert(n[0]>=0);

            for(size_t i=0;i<msize;i++)
              _mesh->metric[cid*6+i] = l[0]*_mesh->metric[best_nodes[0]*msize+i]+
                                       l[1]*_mesh->metric[best_nodes[1]*msize+i]+
                                       l[2]*_mesh->metric[best_nodes[2]*msize+i]+
                                       l[3]*_mesh->metric[best_nodes[3]*msize+i];
          }
        }else{
          // Case 3(c).
          std::cerr << "Case 3(c) encountered" << std::endl;
          exit(-1);
        }
      }else if(refine_cnt==4){
        std::cerr << "Case 4 encountered" << std::endl;
        exit(-1);
      }else if(refine_cnt==5){
        std::cerr << "Case 5 encountered" << std::endl;
        exit(-1);
      }else if(refine_cnt==6){
        /*
         * There is an internal edge in this case. We choose the shortest among:
         * a) newVertex[0] - newVertex[5]
         * b) newVertex[1] - newVertex[4]
         * c) newVertex[2] - newVertex[3]
         */

        real_t ldiag0 = _mesh->calc_edge_length(newVertex[0], newVertex[5]);
        real_t ldiag1 = _mesh->calc_edge_length(newVertex[1], newVertex[4]);
        real_t ldiag2 = _mesh->calc_edge_length(newVertex[2], newVertex[3]);

        std::vector<index_t> internal(2);
        std::vector<index_t> opposite(4);
        std::vector<int> bndr(4);
        if(ldiag0 < ldiag1 && ldiag0 < ldiag2){
          // 0-5
          internal[0] = newVertex[5];
          internal[1] = newVertex[0];
          opposite[0] = newVertex[3];
          opposite[1] = newVertex[4];
          opposite[2] = newVertex[2];
          opposite[3] = newVertex[1];
          bndr[0] = boundary[0];
          bndr[1] = boundary[2];
          bndr[2] = boundary[1];
          bndr[3] = boundary[3];
        }else if(ldiag1 < ldiag2){
          // 1-4
          internal[0] = newVertex[1];
          internal[1] = newVertex[4];
          opposite[0] = newVertex[0];
          opposite[1] = newVertex[3];
          opposite[2] = newVertex[5];
          opposite[3] = newVertex[2];
          bndr[0] = boundary[3];
          bndr[1] = boundary[0];
          bndr[2] = boundary[1];
          bndr[3] = boundary[2];
        }else{
          // 2-3
          internal[0] = newVertex[3];
          internal[1] = newVertex[2];
          opposite[0] = newVertex[4];
          opposite[1] = newVertex[5];
          opposite[2] = newVertex[1];
          opposite[3] = newVertex[0];
          bndr[0] = boundary[0];
          bndr[1] = boundary[1];
          bndr[2] = boundary[3];
          bndr[3] = boundary[2];
        }

        const int ele0[] = {n[0], newVertex[0], newVertex[1], newVertex[2]};
        const int ele1[] = {n[1], newVertex[3], newVertex[0], newVertex[4]};
        const int ele2[] = {n[2], newVertex[1], newVertex[3], newVertex[5]};
        const int ele3[] = {n[3], newVertex[2], newVertex[4], newVertex[5]};
        const int ele4[] = {internal[0], opposite[0], opposite[1], internal[1]};
        const int ele5[] = {internal[0], opposite[1], opposite[2], internal[1]};
        const int ele6[] = {internal[0], opposite[2], opposite[3], internal[1]};
        const int ele7[] = {internal[0], opposite[3], opposite[0], internal[1]};

        const int ele0_boundary[] = {0, boundary[1], boundary[2], boundary[3]};
        const int ele1_boundary[] = {0, boundary[2], boundary[0], boundary[3]};
        const int ele2_boundary[] = {0, boundary[0], boundary[1], boundary[3]};
        const int ele3_boundary[] = {0, boundary[0], boundary[1], boundary[2]};
        const int ele4_boundary[] = {0, 0, 0, bndr[0]};
        const int ele5_boundary[] = {bndr[1], 0, 0, 0};
        const int ele6_boundary[] = {0, 0, 0, bndr[2]};
        const int ele7_boundary[] = {bndr[3], 0, 0, 0};

        index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID, ele6ID, ele7ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;
        ele4ID = ele1ID+3;
        ele5ID = ele1ID+4;
        ele6ID = ele1ID+5;
        ele7ID = ele1ID+6;

        def_ops->addNN(internal[0], internal[1], tid);
        def_ops->addNN(internal[1], internal[0], tid);

        def_ops->remNE(n[1], eid, tid);
        def_ops->addNE_fix(n[1], ele1ID, tid);
        def_ops->remNE(n[2], eid, tid);
        def_ops->addNE_fix(n[2], ele2ID, tid);
        def_ops->remNE(n[3], eid, tid);
        def_ops->addNE_fix(n[3], ele3ID, tid);

        def_ops->addNE(newVertex[0], eid, tid);
        def_ops->addNE_fix(newVertex[0], ele1ID, tid);

        def_ops->addNE(newVertex[1], eid, tid);
        def_ops->addNE_fix(newVertex[1], ele2ID, tid);

        def_ops->addNE(newVertex[2], eid, tid);
        def_ops->addNE_fix(newVertex[2], ele3ID, tid);

        def_ops->addNE_fix(newVertex[3], ele1ID, tid);
        def_ops->addNE_fix(newVertex[3], ele2ID, tid);

        def_ops->addNE_fix(newVertex[4], ele1ID, tid);
        def_ops->addNE_fix(newVertex[4], ele3ID, tid);

        def_ops->addNE_fix(newVertex[5], ele2ID, tid);
        def_ops->addNE_fix(newVertex[5], ele3ID, tid);

        def_ops->addNE_fix(internal[0], ele4ID, tid);
        def_ops->addNE_fix(internal[0], ele5ID, tid);
        def_ops->addNE_fix(internal[0], ele6ID, tid);
        def_ops->addNE_fix(internal[0], ele7ID, tid);
        def_ops->addNE_fix(internal[1], ele4ID, tid);
        def_ops->addNE_fix(internal[1], ele5ID, tid);
        def_ops->addNE_fix(internal[1], ele6ID, tid);
        def_ops->addNE_fix(internal[1], ele7ID, tid);

        def_ops->addNE_fix(opposite[0], ele4ID, tid);
        def_ops->addNE_fix(opposite[1], ele4ID, tid);
        def_ops->addNE_fix(opposite[1], ele5ID, tid);
        def_ops->addNE_fix(opposite[2], ele5ID, tid);
        def_ops->addNE_fix(opposite[2], ele6ID, tid);
        def_ops->addNE_fix(opposite[3], ele6ID, tid);
        def_ops->addNE_fix(opposite[3], ele7ID, tid);
        def_ops->addNE_fix(opposite[0], ele7ID, tid);

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

  static const size_t ndims=dim, nloc=(dim+1), msize=(dim==2?3:6), nedge=(dim==2?3:6);
  int nprocs, rank, nthreads;
};


#endif
