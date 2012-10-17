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

#ifndef REFINE2D_H
#define REFINE2D_H

#include <algorithm>
#include <deque>
#include <set>
#include <vector>
#include <limits>

#include <string.h>
#include <inttypes.h>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#include "graph_partitioning.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D mesh refinement.
 *
 */
template<typename real_t, typename index_t> class Refine2D{
 public:
  /// Default constructor.
  Refine2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;

    size_t NElements = _mesh->get_number_elements();

    lnn2gnn = NULL;

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

    rank = 0;
    nprocs = 1;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
      MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
    }
#endif

#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#else
    nthreads=1;
#endif

    newVertices.resize(nthreads);
    newElements.resize(nthreads);
    newCoords.resize(nthreads);
    newMetric.resize(nthreads);
  }

  /// Default destructor.
  ~Refine2D(){
    if(lnn2gnn!=NULL)
      delete [] lnn2gnn;

    delete property;
  }

  /*! Perform one level of refinement See Figure 25; X Li et al, Comp
   * Methods Appl Mech Engrg 194 (2005) 4915-4950. The actual
   * templates used for 3D refinement follows Rupak Biswas, Roger
   * C. Strawn, "A new procedure for dynamic adaption of
   * three-dimensional unstructured grids", Applied Numerical
   * Mathematics, Volume 13, Issue 6, February 1994, Pages 437-452.
   */
  void refine(real_t L_max){
    size_t NNodes = _mesh->get_number_nodes();
    size_t origNNodes = NNodes;

    size_t NElements = _mesh->get_number_elements();
    size_t origNElements = NElements;

    // Establish global node numbering.
    int gnn_offset=0;
#ifdef HAVE_MPI
    if(nprocs>1){
      // Calculate the global numbering offset for this partition.
      MPI_Scan(&NNodes, &gnn_offset, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
      gnn_offset-=NNodes;
    }
#endif

    // Initialise the lnn2gnn numbering.
    if(lnn2gnn!=NULL)
      delete [] lnn2gnn;
    lnn2gnn = new index_t[NNodes];

    for(size_t i=0;i<NNodes;i++)
      lnn2gnn[i] = gnn_offset+i;

    // Update halo values.
    _mesh->halo_update(lnn2gnn, 1);

    for(size_t i=0;i<NNodes;i++)
      gnn2lnn[lnn2gnn[i]] = i;

    // Calculate node ownership.
    node_owner.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      node_owner[i] = rank;

    if(nprocs>1){
      for(int i=0;i<nprocs;i++){
        for(std::vector<int>::const_iterator it=_mesh->recv[i].begin();
            it!=_mesh->recv[i].end();++it){
          node_owner[*it] = i;
        }
      }
    }

    const split_edge initial = {-1, -1};
    split_edges_per_element.clear();
    split_edges_per_element.resize(3*NElements, initial);
    dirtyElements.clear();
    dirtyElements.resize(NElements, 0);

    std::vector<size_t> threadIdx(nthreads), splitCnt(nthreads, 0);
    std::vector< std::vector<DirectedEdge<index_t> > > surfaceEdges(nthreads);

    int *tpartition = new int[NNodes];
    index_t *dynamic_vertex = new index_t[NNodes];
    memset(dynamic_vertex, 0, NNodes*sizeof(index_t));

    std::vector<char> n_marked_edges_per_element(NElements, 0);

#pragma omp parallel
    {
      int tid = get_tid();

      /*
       * Average vertex degree is ~6, so there
       * are approx. (6/2)*NNodes edges in the mesh.
       */
      size_t reserve_size = 3*NNodes/nthreads;
      newVertices[tid].clear();
      newVertices[tid].reserve(reserve_size);
      newCoords[tid].clear();
      newCoords[tid].reserve(ndims*reserve_size);
      newMetric[tid].clear();
      newMetric[tid].reserve(msize*reserve_size);

      /* Loop through all edges and select them for refinement if
         its length is greater than L_max in transformed space. */
#pragma omp for schedule(dynamic,100)
      for(size_t i=0;i<NNodes;++i){
        for(size_t it=0;it<_mesh->NNList[i].size();++it){
          index_t otherVertex = _mesh->NNList[i][it];
          assert(otherVertex>=0);

          /* Conditional statement ensures that the edge length is
             only calculated once.

             By ordering the vertices according to their gnn, we
             ensure that all processes calculate the same edge
             length when they fall on the halo. */
          if(lnn2gnn[i]<lnn2gnn[otherVertex]){
            double length = _mesh->calc_edge_length(i, otherVertex);
            if(length>L_max){ /* Here is why length must be calculated
                                 exactly the same way across all
                                 processes - need to ensure all
                                 processes that have this edge will
                                 decide to refine it. */

              std::set<index_t> intersection;
              set_intersection( _mesh->NEList[i].begin(), _mesh->NEList[i].end(),
                  _mesh->NEList[otherVertex].begin(), _mesh->NEList[otherVertex].end(), inserter(intersection, intersection.begin()));

              for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element){
                index_t eid = *element;
                size_t edgeOffset = edgeNumber(eid, i, otherVertex);
                split_edges_per_element[3*eid+edgeOffset].newVertex = splitCnt[tid];
                split_edges_per_element[3*eid+edgeOffset].thread = tid;
              }

              ++splitCnt[tid];

              refine_edge(i, otherVertex, tid);

              // dynamic_vertex[otherVertex] might be accessed by many threads at a time.
              // This is not a problem, since all threads will write the same value.
              dynamic_vertex[i] = 1;
              dynamic_vertex[otherVertex] = 1;
            }
          }
        }
      }

      // Perform prefix sum to find (for each OMP thread) the starting position
      // in mesh._coords and mesh.metric at which new coords and metric should be appended.
      threadIdx[tid] = 0;
      for(int id=0; id<tid; ++id)
        threadIdx[tid] += splitCnt[id];

      threadIdx[tid] += NNodes;

      // Resize mesh containers. The above must have completed first, thus the barrier.
#pragma omp barrier
#pragma omp single
      {
        NNodes = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
      }
#pragma omp single nowait
      {
        if(nthreads>1)
          pragmatic::partition_fast(_mesh->NNList, dynamic_vertex, nthreads, tpartition);

        _mesh->NNList.resize(NNodes);
      }
#pragma omp single
      {
        _mesh->NEList.resize(NNodes);
        _mesh->_coords.resize(ndims*NNodes);
        _mesh->metric.resize(msize*NNodes);
      }

      // Append new coords and metric to the mesh.
      memcpy(&_mesh->_coords[ndims*threadIdx[tid]], &newCoords[tid][0], ndims*splitCnt[tid]*sizeof(real_t));
      memcpy(&_mesh->metric[msize*threadIdx[tid]], &newMetric[tid][0], msize*splitCnt[tid]*sizeof(float));

      assert(newVertices[tid].size()==splitCnt[tid]);
      for(size_t i=0;i<splitCnt[tid];i++){
        newVertices[tid][i].id = threadIdx[tid]+i;

        // Check if surface edge
        if(_surface->contains_node(gnn2lnn[newVertices[tid][i].edge.first]) &&
            _surface->contains_node(gnn2lnn[newVertices[tid][i].edge.second])){

          DirectedEdge<index_t> sEdge(gnn2lnn[newVertices[tid][i].edge.first],
              gnn2lnn[newVertices[tid][i].edge.second], newVertices[tid][i].id);

          surfaceEdges[tid].push_back(sEdge);
        }
      }

      // Fix IDs of new vertices
#pragma omp for schedule(static)
      for(size_t eid=0; eid<NElements; ++eid){
      	//If the element has been deleted, continue.
      	const index_t *n = _mesh->get_element(eid);
      	if(n[0] < 0)
      		continue;

        for(size_t j=0; j<3; ++j)
          if(split_edges_per_element[3*eid+j].newVertex != -1){
            split_edges_per_element[3*eid+j].newVertex += threadIdx[split_edges_per_element[3*eid+j].thread];
            ++n_marked_edges_per_element[eid];
        }
      }

      // Perform element refinement.
      splitCnt[tid] = 0;
      newElements[tid].clear();
      newElements[tid].reserve(4*NElements/nthreads);

      // Phase 1
      if(nthreads>1){
        /*
         * Each thread creates a list of elements it is responsible
         * for refining. It also pre-calculates the number of new
         * elements - we need this to assign correct element IDs.
         */
        std::vector<index_t> *tdynamic_element = new std::vector<index_t>;
        tdynamic_element->reserve(NElements/nthreads);

        for(size_t eid=0;eid<NElements;eid++){
          if(n_marked_edges_per_element[eid]>0){
          	const index_t *n = _mesh->get_element(eid);

          	if((tpartition[n[0]]==tid)&&(tpartition[n[1]]==tid)&&(tpartition[n[2]]==tid)){
              tdynamic_element->push_back(eid);
              assert(n_marked_edges_per_element[eid] >= 0 && n_marked_edges_per_element[eid] <= 3);
              splitCnt[tid] += n_marked_edges_per_element[eid];
            }
          }
        }

#pragma omp barrier

        threadIdx[tid] = 0;
        for(int id=0; id<tid; ++id)
         threadIdx[tid] += splitCnt[id];

        threadIdx[tid] += NElements;

        index_t newEID = threadIdx[tid];
        for(typename std::vector<index_t>::const_iterator it=tdynamic_element->begin(); it!=tdynamic_element->end(); ++it){
          newEID += refine_element(*it, newEID, n_marked_edges_per_element[*it], tid);

          // Mark eid as processed
          n_marked_edges_per_element[*it] = 0;
        }

        delete tdynamic_element;
      }

      // Phase 2
#pragma omp barrier
#pragma omp single
      {
        index_t newEID;
        if(nthreads > 1)
          newEID = threadIdx[nthreads-1] + splitCnt[nthreads-1];
        else
          newEID = NElements;

        for(index_t eid=0;eid<(index_t)NElements;eid++){
          if(n_marked_edges_per_element[eid]>0){
            newEID += refine_element(eid, newEID, n_marked_edges_per_element[eid], nthreads-1);
          }
        }

        if(nthreads > 1)
          splitCnt[nthreads-1] += newEID - (threadIdx[nthreads-1] + splitCnt[nthreads-1]);
        else
          splitCnt[nthreads-1] += newEID - NElements;
      }

      // Perform prefix sum to find (for each OMP thread) the starting position
      // in mesh._ENList at which new elements should be appended.
      threadIdx[tid] = 0;
      for(int id=0; id<tid; ++id)
       threadIdx[tid] += splitCnt[id];

      threadIdx[tid] += NElements;

#pragma omp barrier

      // Resize mesh containers
#pragma omp single
      {
        NElements = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];

        _mesh->_ENList.resize(nloc*NElements);
      }

      // Append new elements to the mesh
      memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));

#ifdef HAVE_MPI
      if(nprocs>1){
#pragma omp master
        {
          // Time to amend halo.
        	node_owner.resize(NNodes, -1);

          std::map<index_t, DirectedEdge<index_t> > lut_newVertices;
          for(int i=0;i<nthreads;i++){
            for(typename std::vector< DirectedEdge<index_t> >::const_iterator vert=newVertices[i].begin();vert!=newVertices[i].end();++vert){
              assert(lut_newVertices.find(vert->id)==lut_newVertices.end());
              lut_newVertices[vert->id] = *vert;

              int owner0 = node_owner[gnn2lnn[vert->edge.first]];
              int owner1 = node_owner[gnn2lnn[vert->edge.second]];

              int owner = std::min(owner0, owner1);
              node_owner[vert->id] = owner;
            }
          }

          dirtyElements.resize(NElements, 1);
          typename std::vector< std::set< DirectedEdge<index_t> > > send_additional(nprocs), recv_additional(nprocs);

          for(size_t i=0;i<NElements;i++){
          	if(dirtyElements[i]==0)
          		continue;

            const int *n=_mesh->get_element(i);
            if(n[0]<0)
              continue;

            std::set<int> processes;
            for(size_t j=0;j<nloc;j++){
              processes.insert(node_owner[n[j]]);
            }
            assert(processes.count(-1)==0);

            // Element has no local vertices so we can erase it.
            if(processes.count(rank)==0){
              _mesh->erase_element(i);
              continue;
            }

            if(processes.size()==1)
              continue;

            // If we get this far it means that the element strides a halo.
            for(size_t j=0;j<nloc;j++){
              // Check if this is an old vertex.
              if(n[j]<(int)origNNodes)
                continue;

              if(node_owner[n[j]]==rank){
                // Send.
                for(std::set<int>::const_iterator ip=processes.begin(); ip!=processes.end();++ip){
                  if(*ip==rank)
                    continue;

                  send_additional[*ip].insert(lut_newVertices[n[j]]);
                }
              }else{
                // Receive.
                recv_additional[node_owner[n[j]]].insert(lut_newVertices[n[j]]);
              }
            }
          }

          for(int i=0;i<nprocs;i++){
            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=send_additional[i].begin();it!=send_additional[i].end();++it){
              _mesh->send[i].push_back(it->id);
              _mesh->send_halo.insert(it->id);
            }
          }

          for(int i=0;i<nprocs;i++){
            for(typename std::set< DirectedEdge<index_t> >::const_iterator it=recv_additional[i].begin();it!=recv_additional[i].end();++it){
              _mesh->recv[i].push_back(it->id);
              _mesh->recv_halo.insert(it->id);
            }
          }
        }
      }
#endif

      // Fix orientations of new elements.
      int new_NElements = _mesh->get_number_elements();
      int new_cnt = new_NElements - NElements;
      index_t *tENList = &(_mesh->_ENList[NElements*nloc]);
      real_t *tcoords = &(_mesh->_coords[0]);

#pragma omp for schedule(dynamic)
      for(int i=0;i<new_cnt;i++){
        index_t n0 = tENList[i*nloc];
        index_t n1 = tENList[i*nloc + 1];
        index_t n2 = tENList[i*nloc + 2];

        const real_t *x0 = tcoords + n0*ndims;
        const real_t *x1 = tcoords + n1*ndims;
        const real_t *x2 = tcoords + n2*ndims;

        real_t av = property->area(x0, x1, x2);

        if(av<0){
          // Flip element
          tENList[i*nloc] = n1;
          tENList[i*nloc+1] = n0;
        }
      }
    }

    // Refine surface
    _surface->refine(surfaceEdges);

    delete[] dynamic_vertex;
    delete[] tpartition;

    return;
  }

 private:

  void refine_edge(index_t n0, index_t n1, size_t tid){
    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }
    newVertices[tid].push_back(DirectedEdge<index_t>(lnn2gnn[n0], lnn2gnn[n1]));

    // Calculate the position of the new point. From equation 16 in
    // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
    real_t x, m;
    const real_t *x0 = _mesh->get_coords(n0);
    const float *m0 = _mesh->get_metric(n0);

    const real_t *x1 = _mesh->get_coords(n1);
    const float *m1 = _mesh->get_metric(n1);

    real_t weight = 1.0/(1.0 + sqrt(property->length(x0, x1, m0)/
                                    property->length(x0, x1, m1)));

    // Calculate position of new vertex and append it to OMP thread's temp storage
    for(size_t i=0;i<ndims;i++){
      x = x0[i]+weight*(x1[i] - x0[i]);
      newCoords[tid].push_back(x);
    }

    // Interpolate new metric and append it to OMP thread's temp storage
    for(size_t i=0;i<msize;i++){
      m = m0[i]+weight*(m1[i] - m0[i]);
      newMetric[tid].push_back(m);
      if(isnan(m))
        std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                 <<"m0[i] = "<<m0[i]<<std::endl
                 <<"m1[i] = "<<m1[i]<<std::endl
                 <<"weight = "<<weight<<std::endl;
    }
  }

  int refine_element(index_t eid, index_t newEID, char refine_cnt, size_t tid){
    // Check if this element has been erased - if so continue to next element.
    const int *n=_mesh->get_element(eid);

    // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
    index_t newVertex[3] = {-1, -1, -1};
    newVertex[0] = split_edges_per_element[3*eid].newVertex;
    newVertex[1] = split_edges_per_element[3*eid+1].newVertex;
    newVertex[2] = split_edges_per_element[3*eid+2].newVertex;

    if(refine_cnt==1){
      // Single edge split.
      int rotated_ele[3] = {-1, -1, -1};
      index_t vertexID=-1;
      for(int j=0;j<3;j++)
        if(newVertex[j] >= 0){
          vertexID = newVertex[j];

          // Loop hand unrolled because compiler could not vectorise.
          rotated_ele[0] = n[j];
          rotated_ele[1] = n[(j+1)%3];
          rotated_ele[2] = n[(j+2)%3];

          break;
        }
      assert(vertexID!=-1);

      const index_t ele0[] = {rotated_ele[0], rotated_ele[1], vertexID};
      const index_t ele1[] = {rotated_ele[0], vertexID, rotated_ele[2]};

      index_t ele1ID = newEID;

      // If the edge hosting the new vertex has not been processed before as part of the
      // adjacent element, connect the new vertex to rotated_ele[1] and rotated_ele[2].
      if(_mesh->NNList[vertexID].empty()){
        _mesh->NNList[vertexID].push_back(rotated_ele[1]);
        _mesh->NNList[vertexID].push_back(rotated_ele[2]);

        // Replace rotated_ele[1]'s adjacency to rotated_ele[2] with the new vertex.
        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[rotated_ele[1]].begin(),
            _mesh->NNList[rotated_ele[1]].end(), rotated_ele[2]);
        *it = vertexID;
        it = std::find(_mesh->NNList[rotated_ele[2]].begin(),
            _mesh->NNList[rotated_ele[2]].end(), rotated_ele[1]);
        *it = vertexID;
      }

      _mesh->NNList[vertexID].push_back(rotated_ele[0]);
      _mesh->NNList[rotated_ele[0]].push_back(vertexID);

      // Put ele1 in rotated_ele[0]'s NEList
      _mesh->NEList[rotated_ele[0]].insert(ele1ID);

      // Put eid and ele1 in vertexID's NEList
      _mesh->NEList[vertexID].insert(eid);
      _mesh->NEList[vertexID].insert(ele1ID);

      // Replace eid with ele1 in rotated_ele[2]'s NEList
      _mesh->NEList[rotated_ele[2]].erase(eid);
      _mesh->NEList[rotated_ele[2]].insert(ele1ID);

      assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
      assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);

      replace_element(eid, ele0, tid);
      append_element(ele1, tid);

      return 1;
    }else if(refine_cnt==2){
      int rotated_ele[3] = {-1, -1, -1};
      index_t vertexID[2];
      for(int j=0;j<3;j++){
        if(newVertex[j] < 0){
          vertexID[0] = newVertex[(j+1)%3];
          vertexID[1] = newVertex[(j+2)%3];

          rotated_ele[0] = n[j];
          rotated_ele[1] = n[(j+1)%3];
          rotated_ele[2] = n[(j+2)%3];

          break;
        }
      }

      real_t ldiag0 = _mesh->calc_edge_length(rotated_ele[1], vertexID[0]);
      real_t ldiag1 = _mesh->calc_edge_length(rotated_ele[2], vertexID[1]);

      const int offset = ldiag0 < ldiag1 ? 0 : 1;

      const index_t ele0[] = {rotated_ele[0], vertexID[1], vertexID[0]};
      const index_t ele1[] = {vertexID[offset], rotated_ele[1], rotated_ele[2]};
      const index_t ele2[] = {vertexID[0], vertexID[1], rotated_ele[offset+1]};

      index_t ele0ID = newEID;
      index_t ele2ID = newEID+1;

      // If the edge hosting vertexID[0] has not been processed before as part of the
      // adjacent element, connect vertexID[0] to rotated_ele[0] and rotated_ele[2].
      if(_mesh->NNList[vertexID[0]].empty()){
        _mesh->NNList[vertexID[0]].push_back(rotated_ele[0]);
        _mesh->NNList[vertexID[0]].push_back(rotated_ele[2]);

        // Replace rotated_ele[0]'s adjacency to rotated_ele[2] with vertexID[0].
        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[rotated_ele[0]].begin(),
            _mesh->NNList[rotated_ele[0]].end(), rotated_ele[2]);
        *it = vertexID[0];
        it = std::find(_mesh->NNList[rotated_ele[2]].begin(),
            _mesh->NNList[rotated_ele[2]].end(), rotated_ele[0]);
        *it = (vertexID[0]);
      }

      // Similarly for the edge hosting vertexID[1].
      if(_mesh->NNList[vertexID[1]].empty()){
        _mesh->NNList[vertexID[1]].push_back(rotated_ele[0]);
        _mesh->NNList[vertexID[1]].push_back(rotated_ele[1]);

        // Replace rotated_ele[0]'s adjacency to rotated_ele[1] with vertexID[1].
        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[rotated_ele[0]].begin(),
            _mesh->NNList[rotated_ele[0]].end(), rotated_ele[1]);
        *it = (vertexID[1]);
        it = std::find(_mesh->NNList[rotated_ele[1]].begin(),
            _mesh->NNList[rotated_ele[1]].end(), rotated_ele[0]);
        *it = (vertexID[1]);
      }

      // NNList: Connect vertexID[0] and vertexID[1] with each other
      _mesh->NNList[vertexID[0]].push_back(vertexID[1]);
      _mesh->NNList[vertexID[1]].push_back(vertexID[0]);

      // vertexID[offset] and rotated_ele[offset+1] are the vertices on the diagonal
      _mesh->NNList[vertexID[offset]].push_back(rotated_ele[offset+1]);
      _mesh->NNList[rotated_ele[offset+1]].push_back(vertexID[offset]);

      // rotated_ele[offset+1] is the old vertex which is on the diagonal
      // Add ele2 in rotated_ele[offset+1]'s NEList
      _mesh->NEList[rotated_ele[offset+1]].insert(ele2ID);

      // Replace eid with ele0 in NEList[rotated_ele[0]]
      _mesh->NEList[rotated_ele[0]].erase(eid);
      _mesh->NEList[rotated_ele[0]].insert(ele0ID);

      // Put ele0, ele1 and ele2 in vertexID[offset]'s NEList
      _mesh->NEList[vertexID[offset]].insert(eid);
      _mesh->NEList[vertexID[offset]].insert(ele0ID);
      _mesh->NEList[vertexID[offset]].insert(ele2ID);

      // vertexID[(offset+1)%2] is the new vertex which is not on the diagonal
      // Put ele0 and ele2 in vertexID[(offset+1)%2]'s NEList
      _mesh->NEList[vertexID[(offset+1)%2]].insert(ele0ID);
      _mesh->NEList[vertexID[(offset+1)%2]].insert(ele2ID);

      assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
      assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
      assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);

      replace_element(eid, ele1, tid);
      append_element(ele0, tid);
      append_element(ele2, tid);

      return 2;
    }else{ // refine_cnt==3
      const index_t ele0[] = {n[0], newVertex[2], newVertex[1]};
      const index_t ele1[] = {n[1], newVertex[0], newVertex[2]};
      const index_t ele2[] = {n[2], newVertex[1], newVertex[0]};
      const index_t ele3[] = {newVertex[0], newVertex[1], newVertex[2]};

      index_t ele1ID = newEID;
      index_t ele2ID = newEID+1;
      index_t ele3ID = newEID+2;

      // Update NNList

      // If the edge hosting newVertex[0] has not been processed before as
      // part of the adjacent element, connect newVertex[0] to n[1] and n[2].
      if(_mesh->NNList[newVertex[0]].empty()){
        _mesh->NNList[newVertex[0]].push_back(n[1]);
        _mesh->NNList[newVertex[0]].push_back(n[2]);

        // Replace n[1]'s adjacency to n[2] with newVertex[0].
        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[n[1]].begin(), _mesh->NNList[n[1]].end(), n[2]);
        *it = newVertex[0];
        it = std::find(_mesh->NNList[n[2]].begin(), _mesh->NNList[n[2]].end(), n[1]);
        *it = newVertex[0];
      }

      // Similarly for the edge hosting newVertex[1].
      if(_mesh->NNList[newVertex[1]].empty()){
        _mesh->NNList[newVertex[1]].push_back(n[0]);
        _mesh->NNList[newVertex[1]].push_back(n[2]);

        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[n[0]].begin(), _mesh->NNList[n[0]].end(), n[2]);
        *it = newVertex[1];
        it = std::find(_mesh->NNList[n[2]].begin(), _mesh->NNList[n[2]].end(), n[0]);
        *it = newVertex[1];
      }

      // Similarly for the edge hosting newVertex[2].
      if(_mesh->NNList[newVertex[2]].empty()){
        _mesh->NNList[newVertex[2]].push_back(n[0]);
        _mesh->NNList[newVertex[2]].push_back(n[1]);

        typename std::vector<index_t>::iterator it;
        it = std::find(_mesh->NNList[n[0]].begin(), _mesh->NNList[n[0]].end(), n[1]);
        *it = newVertex[2];
        it = std::find(_mesh->NNList[n[1]].begin(), _mesh->NNList[n[1]].end(), n[0]);
        *it = newVertex[2];
      }

      _mesh->NNList[newVertex[0]].push_back(newVertex[1]);
      _mesh->NNList[newVertex[0]].push_back(newVertex[2]);
      _mesh->NNList[newVertex[1]].push_back(newVertex[0]);
      _mesh->NNList[newVertex[1]].push_back(newVertex[2]);
      _mesh->NNList[newVertex[2]].push_back(newVertex[0]);
      _mesh->NNList[newVertex[2]].push_back(newVertex[1]);

      // Update NEList
      _mesh->NEList[n[1]].erase(eid);
      _mesh->NEList[n[1]].insert(ele1ID);
      _mesh->NEList[n[2]].erase(eid);
      _mesh->NEList[n[2]].insert(ele2ID);

      _mesh->NEList[newVertex[0]].insert(ele1ID);
      _mesh->NEList[newVertex[0]].insert(ele2ID);
      _mesh->NEList[newVertex[0]].insert(ele3ID);

      _mesh->NEList[newVertex[1]].insert(eid);
      _mesh->NEList[newVertex[1]].insert(ele2ID);
      _mesh->NEList[newVertex[1]].insert(ele3ID);

      _mesh->NEList[newVertex[2]].insert(eid);
      _mesh->NEList[newVertex[2]].insert(ele1ID);
      _mesh->NEList[newVertex[2]].insert(ele3ID);

      assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
      assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
      assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);
      assert(ele3[0]>=0 && ele3[1]>=0 && ele3[2]>=0);

      replace_element(eid, ele0, tid);
      append_element(ele1, tid);
      append_element(ele2, tid);
      append_element(ele3, tid);

      return 3;
    }
  }

  inline void append_element(const index_t *elem, size_t tid){
    for(size_t i=0; i<nloc; ++i)
      newElements[tid].push_back(elem[i]);
  }

  inline void replace_element(const index_t eid, const index_t *n, size_t tid){
    for(size_t i=0;i<nloc;i++)
      _mesh->_ENList[eid*nloc+i]=n[i];

    dirtyElements[eid]=1;
  }

  inline size_t edgeNumber(index_t eid, index_t v1, index_t v2) const{
    /*
     * Edge 0 is the edge (n[1],n[2]).
     * Edge 1 is the edge (n[0],n[2]).
     * Edge 2 is the edge (n[0],n[1]).
     */
    const int *n=_mesh->get_element(eid);
    if(n[1]==v1 || n[1]==v2){
      if(n[2]==v1 || n[2]==v2)
        return 0;
      else
        return 2;
    }
    else
      return 1;
  }

  struct split_edge{
    index_t newVertex;
    index_t thread;
  };

  std::vector<split_edge> split_edges_per_element;
  std::vector< std::vector< DirectedEdge<index_t> > > newVertices;
  std::vector< std::vector<real_t> > newCoords;
  std::vector< std::vector<float> > newMetric;
  std::vector< std::vector<index_t> > newElements;
  std::vector<char> dirtyElements;

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;

  index_t *lnn2gnn;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    boost::unordered_map<int, int> gnn2lnn;
#else
    std::map<index_t, index_t> gnn2lnn;
#endif
  std::vector<index_t> node_owner;

  size_t nnodes_reserve;
  index_t *dynamic_vertex;

  static const size_t ndims=2, nloc=3, msize=3;
  int nprocs, rank, nthreads;
};

#endif
