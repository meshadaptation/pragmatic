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
  }

  /// Default destructor.
  ~Refine2D(){
    if(lnn2gnn==NULL)
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

    NNList.clear();
    NNList.resize(NNodes);
    NEList.clear();
    NEList.resize(NNodes);
    refined_edges.clear();
    refined_edges.resize(NNodes);
    createdByThread.clear();
    createdByThread.resize(NNodes);

    std::vector< std::vector< DirectedEdge<index_t> > > newVertices(nthreads);
    std::vector< std::vector<real_t> > newCoords(nthreads);
    std::vector< std::vector<float> > newMetric(nthreads);
    std::vector< std::vector<index_t> > newElements(nthreads);
    std::vector<size_t> threadIdx(nthreads), splitCnt(nthreads, 0);

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

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      /* Loop through all edges and select them for refinement if
         its length is greater than L_max in transformed space. */
#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;++i){
        /*
         * Convert _mesh->NEList from std::vector<std::set> to std::vector<std::vector>
         * and create a local copy of _mesh->NNList. This copy of NNList is needed by
         * _mesh->get_new_vertex() - since _mesh->NNList is updated in-place and newly
         * created vertices delete the original adjacency information between the two vertices
         * of the split edge, we need to keep a copy of the original adjacency info.
         */
        size_t size = _mesh->NNList[i].size();
        NNList[i].resize(size);
        std::copy(_mesh->NNList[i].begin(), _mesh->NNList[i].end(), NNList[i].begin());

        size = _mesh->NEList[i].size();
        NEList[i].resize(size);
        std::copy(_mesh->NEList[i].begin(), _mesh->NEList[i].end(), NEList[i].begin());

        /*
         * Space must be allocated for refined_edges[i] in any case, no matter
         * whether any of the edges adjacent to vertex i will be refined or not.
         * This is because function mark_edge(...) assumes that space has already
         * been allocated. Allocating space for refined_edges[i] on demand, i.e.
         * inside mark_edge(...), is not possible, since mark_edge(...) may be
         * called for the same vertex i by two threads at the same time.
         */
        refined_edges[i].resize(_mesh->NNList[i].size(), -1);
        createdByThread[i].resize(_mesh->NNList[i].size());

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

              refined_edges[i][it]   = splitCnt[tid]++;
              createdByThread[i][it] = tid;

              refine_edge(i, otherVertex, newVertices[tid], newCoords[tid], newMetric[tid]);
            }
          }
        }
      }

      //
      // Insert new vertices into mesh.
      //

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

        _mesh->NNList.resize(NNodes);
        _mesh->NEList.resize(NNodes);
        NEList.resize(NNodes);
        createdByThread.resize(NNodes);
        _mesh->_coords.resize(ndims*NNodes);
        _mesh->metric.resize(msize*NNodes);
        node_owner.resize(NNodes, -1);
      }

      // Append new coords and metric to the mesh.
      memcpy(&_mesh->_coords[ndims*threadIdx[tid]], &newCoords[tid][0], ndims*splitCnt[tid]*sizeof(real_t));
      memcpy(&_mesh->metric[msize*threadIdx[tid]], &newMetric[tid][0], msize*splitCnt[tid]*sizeof(float));

      assert(newVertices[tid].size()==splitCnt[tid]);
      for(size_t i=0;i<splitCnt[tid];i++){
        newVertices[tid][i].id = threadIdx[tid]+i;
      }

      /*
       * TODO: a possible improvement for the following loop
       * Instead of visiting ALL mesh vertices and examining whether an adjacent
       * edge has been refined, we can have a per-thread set of refined edges,
       * which will contain those edges which were refined by that particular
       * thread. This way, every thread will fix the IDs of the new vertices it
       * created and only the edges which were actually refined will be visited.
       * This approach seems better only for subsequent calls to refine(), where
       * only a few edges are expected to be refined.
       */

       // Fix IDs of new vertices in refined_edges and update NNList.
#pragma omp for schedule(dynamic)
      for(index_t i=0; i<(int)refined_edges.size(); ++i){
        for(size_t j=0; j < refined_edges[i].size(); ++j)
          if(refined_edges[i][j] != -1){
            refined_edges[i][j] += threadIdx[createdByThread[i][j]];

            /*
             * i is the lesser ID vertex
             * oppositeVertex is the greater ID vertex
             * middleVertex is the newly created vertex
             */
            index_t middleVertex = refined_edges[i][j];
            index_t oppositeVertex = _mesh->NNList[i][j];

            // Add i in middleVertex's list
            _mesh->NNList[middleVertex].push_back(i);

            // Add oppositeVertex in middleVertex's list
            _mesh->NNList[middleVertex].push_back(oppositeVertex);

            /*
             * Add middleVertex in i's list, replacing oppositeVertex
             * which is no longer one of i's neighbours
             */
            assert(_mesh->NNList[i][j] == oppositeVertex);
            _mesh->NNList[i][j] = middleVertex;

            /*
             * Add middleVertex in oppositeVertex's list, replacing i
             * which is no longer one of oppositeVertex's neighbours
             */
            size_t pos = _mesh->indexOf(i, _mesh->NNList[oppositeVertex]);
            assert(_mesh->NNList[oppositeVertex][pos] == i);
            _mesh->NNList[oppositeVertex][pos] = middleVertex;
          }
      }

      //Resize NNList and NEList
#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;i++){
        if(_mesh->NNList[i].empty())
          continue;

        if(i < origNNodes){
          size_t size = _mesh->NNList[i].size();

          /*
           * An original vertex can only be connected to the newly created vertex
           * on the opposite edge of the element. This can happen for each one of
           * the elements of the cavity defined by the original vertex. There are
           * NNList[original_vertex].size() elements the original vertex is part
           * of, so NNList[original_vertex] needs to be doubled, i.e. create one
           * empty slot for each element. If the element containing the new vertex
           * is at index idx in NEList[original_vertex], then the new vertex will
           * be added to NNList[original_vertex][original_size+idx].
           */
          _mesh->NNList[i].resize(2*size, (index_t) -1);

          size = NEList[i].size();

          /*
           * An original vertex can become a member of at most 2 new elements for
           * each adjacent old element (i.e. in case the element is bisected), so
           * we need to reserve one additional slot in every NEList[i] (one of the
           * new elements will be put into the old slot, so only one extra slot is
           * needed). If the original element's index in NEList[original_vertex]
           * is idx, the new elements will be added at NEList[original_vertex][idx]
           * and NEList[original_vertex][original_size+idx]. As was the case with
           * refined_edges, we need to keep track of which thread created the new elements.
           */
          NEList[i].resize(2*size, (index_t) -1);
          createdByThread[i].clear();
          createdByThread[i].resize(2*size, std::numeric_limits<size_t>::max());
        }else{
          /*
           * A newly created vertex is by now connected to the two original vertices
           * which used to define the split edge. It is also part of the two elements
           * eid0 and eid1 which share the split edge. For each element, the new
           * vertex can be connected:
           * (1:2 case) --> to the opposite vertex
           * (1:3 case) --> to the opposite vertex and one newly created vertex
           * (1:4 case) --> to the two other newly created vertices
           * So, for each element the new vertex can be connected to at most 2 other
           * vertices, so we need to reserve 4 empty slots (2 for each element), so
           * the total size of NNList[new_vertex] is 6 (4 + the 2 existing neighbours).
           * The thread processing the lesser ID element (between eid0 and eid1) will
           * write to NNList[new_vertex][2..3], the other thread to NNList[new_vertex][4..5].
           * In order to find eid0 and eid1 we will use _mesh->NEList instead of the local
           * NEList, because eid0 and eid1 are original elements (i.e. not new elements created
           * by refinement) so we need the pre-refinement view of the mesh.
           */
          _mesh->NNList[i].resize(6, (index_t) -1);

          /*
           * A newly created vertex can become a member of at most 3 new elements
           * for each of the two old elements which share the edge on which the
           * new vertex was created. So we need to reserve 6 slots. As was the case
           * with NNList for new vertices, the thread processing the original
           * element with the lesser ID will write into NEList[0..2], the other
           * thread into NEList[3..5].
           */
          NEList[i].resize(6, (index_t) -1);
          createdByThread[i].resize(6, std::numeric_limits<size_t>::max());
        }
      }

      // Perform element refinement.
      splitCnt[tid] = 0;

#pragma omp for schedule(dynamic)
      for(index_t eid=0;eid<(int)NElements;eid++){
        // Check if this element has been erased - if so continue to next element.
        const int *n=_mesh->get_element(eid);
        if(n[0]<0)
          continue;

        // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
        index_t newVertex[3];
        newVertex[0] = _mesh->get_new_vertex(n[1], n[2], refined_edges, NNList, lnn2gnn);
        newVertex[1] = _mesh->get_new_vertex(n[2], n[0], refined_edges, NNList, lnn2gnn);
        newVertex[2] = _mesh->get_new_vertex(n[0], n[1], refined_edges, NNList, lnn2gnn);

        int refine_cnt=0;
        for(int j=0;j<3;j++)
          if(newVertex[j] >= 0)
            refine_cnt++;

        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
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

          const int ele0[] = {rotated_ele[0], rotated_ele[1], vertexID};
          const int ele1[] = {rotated_ele[0], vertexID, rotated_ele[2]};

          index_t ele0ID = splitCnt[tid]++;
          index_t ele1ID = splitCnt[tid]++;

          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);

          // Put vertexID in rotated_ele[0]'s NNList
          size_t idx = indexInNEList(eid, rotated_ele[0]);
          size_t originalSize = _mesh->NNList[rotated_ele[0]].size() / 2;
          _mesh->NNList[rotated_ele[0]][originalSize+idx] = vertexID;
          // Put ele0 and ele1 in rotated_ele[0]'s NEList and remove eid
          originalSize = NEList[rotated_ele[0]].size() / 2;
          addToNEList(rotated_ele[0], idx, ele0ID, tid);
          addToNEList(rotated_ele[0], originalSize+idx, ele1ID, tid);

          // Put rotated_ele[0] in vertexID's NNList
          // Put ele0 and ele1 in vertexID's NEList
          std::set<index_t> intersection;
          set_intersection(_mesh->NEList[rotated_ele[1]].begin(),
              _mesh->NEList[rotated_ele[1]].end(), _mesh->NEList[rotated_ele[2]].begin(),
              _mesh->NEList[rotated_ele[2]].end(), inserter(intersection, intersection.begin()));
          // If eid is the lesser ID element (or the only common element shared between
          // rotated_ele[1] and rotated_ele[2] in case we are on the mesh surface)
          if(eid == *intersection.begin()){
            _mesh->NNList[vertexID][2] = rotated_ele[0];
            addToNEList(vertexID, 0, ele0ID, tid);
            addToNEList(vertexID, 1, ele1ID, tid);
          }
          else{
            _mesh->NNList[vertexID][4] = rotated_ele[0];
            addToNEList(vertexID, 3, ele0ID, tid);
            addToNEList(vertexID, 4, ele1ID, tid);
          }

          // Replace eid with ele0 in rotated_ele[1]'s NEList
          idx = indexInNEList(eid, rotated_ele[1]);
          addToNEList(rotated_ele[1], idx, ele0ID, tid);

          // Replace eid with ele1 in rotated_ele[2]'s NEList
          idx = indexInNEList(eid, rotated_ele[2]);
          addToNEList(rotated_ele[2], idx, ele1ID, tid);
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

          const int ele0[] = {rotated_ele[0], vertexID[1], vertexID[0]};
          const int ele1[] = {vertexID[offset], rotated_ele[1], rotated_ele[2]};
          const int ele2[] = {vertexID[0], vertexID[1], rotated_ele[offset+1]};

          index_t ele0ID = splitCnt[tid]++;
          index_t ele1ID = splitCnt[tid]++;
          index_t ele2ID = splitCnt[tid]++;

          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);

          /*
           * Find the offset in NNList[vertexID[0]] at which neighbours of
           * vertexID[0] should be appended (3 or 5, depending on whether the
           * element we are processing has the lesser or the greater ID between
           * the two elements sharing the edge on which vertexID[0] was created).
           * We need the original node-element adjacency view, so we use
           * _mesh->NEList instead of the local NEList.
           */
          char list_offset[2];

          std::set<index_t> intersection;
          set_intersection(_mesh->NEList[rotated_ele[0]].begin(),
              _mesh->NEList[rotated_ele[0]].end(), _mesh->NEList[rotated_ele[2]].begin(),
              _mesh->NEList[rotated_ele[2]].end(), inserter(intersection, intersection.begin()));
          list_offset[0] = (eid == *intersection.begin() ? 2 : 4);

          // Same for the offset in NNList[vertexID[1]]
          intersection.clear();
          set_intersection(_mesh->NEList[rotated_ele[0]].begin(),
              _mesh->NEList[rotated_ele[0]].end(), _mesh->NEList[rotated_ele[1]].begin(),
              _mesh->NEList[rotated_ele[1]].end(), inserter(intersection, intersection.begin()));
          list_offset[1] = (eid == *intersection.begin() ? 2 : 4);

          // Put vertexID[1] in vertexID[0]'s NNList
          _mesh->NNList[vertexID[0]][list_offset[0]] = vertexID[1];
          // Put vertexID[0] in vertexID[1]'s NNList
          _mesh->NNList[vertexID[1]][list_offset[1]] = vertexID[0];

          // vertexID[offset] and rotated_ele[offset+1] are the vertices on the diagonal
          // Put rotated_ele[offset+1] in vertexID[offset]'s NNList
          _mesh->NNList[vertexID[offset]][list_offset[offset]+1] = rotated_ele[offset+1];

          // Put vertexID[offset] in rotated_ele[offset+1]'s NNList
          size_t idx = indexInNEList(eid, rotated_ele[offset+1]);
          size_t originalSize = _mesh->NNList[rotated_ele[offset+1]].size() / 2;
          _mesh->NNList[rotated_ele[offset+1]][originalSize+idx] = vertexID[offset];

          // rotated_ele[offset+1] is the old vertex which is on the diagonal
          // Replace eid with ele1 and ele2 in rotated_ele[offset+1]'s NEList
          originalSize = NEList[rotated_ele[offset+1]].size() / 2;
          addToNEList(rotated_ele[offset+1], idx, ele1ID, tid);
          addToNEList(rotated_ele[offset+1], originalSize+idx, ele2ID, tid);

          // rotated_ele[(offset+1)%2+1] is the old vertex which is not on the diagonal
          // Replace eid with ele1 in rotated_ele[(offset+1)%2+1]'s NEList
          size_t otherIdx = (offset+1)%2 + 1;
          idx = indexInNEList(eid, rotated_ele[otherIdx]);
          addToNEList(rotated_ele[otherIdx], idx, ele1ID, tid);

          // Replace eid with ele0 in NEList[rotated_ele[0]]
          idx = indexInNEList(eid, rotated_ele[0]);
          addToNEList(rotated_ele[0], idx, ele0ID, tid);

          // Put ele0, ele1 and ele2 in vertexID[offset]'s NEList
          if(list_offset[offset] == 2){
            addToNEList(vertexID[offset], 0, ele0ID, tid);
            addToNEList(vertexID[offset], 1, ele1ID, tid);
            addToNEList(vertexID[offset], 2, ele2ID, tid);
          }
          else{
            addToNEList(vertexID[offset], 3, ele0ID, tid);
            addToNEList(vertexID[offset], 4, ele1ID, tid);
            addToNEList(vertexID[offset], 5, ele2ID, tid);
          }

          // vertexID[(offset+1)%2] is the new vertex which is not on the diagonal
          // Put ele0 and ele2 in vertexID[(offset+1)%2]'s NEList
          otherIdx = (offset+1)%2;
          if(list_offset[otherIdx] == 2){
            addToNEList(vertexID[otherIdx], 0, ele0ID, tid);
            addToNEList(vertexID[otherIdx], 1, ele2ID, tid);
          }
          else{
            addToNEList(vertexID[otherIdx], 3, ele0ID, tid);
            addToNEList(vertexID[otherIdx], 4, ele2ID, tid);
          }
        }else if(refine_cnt==3){
          const int ele0[] = {n[0], newVertex[2], newVertex[1]};
          const int ele1[] = {n[1], newVertex[0], newVertex[2]};
          const int ele2[] = {n[2], newVertex[1], newVertex[0]};
          const int ele3[] = {newVertex[0], newVertex[1], newVertex[2]};

          index_t ele0ID = splitCnt[tid]++;
          index_t ele1ID = splitCnt[tid]++;
          index_t ele2ID = splitCnt[tid]++;
          index_t ele3ID = splitCnt[tid]++;

          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          append_element(ele3, newElements[tid]);

          // Find offsets in NNList for newVertex[0], newVertex[1] and [newVertex[2].
          char list_offset[3];

          std::set<index_t> intersection;
          set_intersection(_mesh->NEList[n[1]].begin(),
              _mesh->NEList[n[1]].end(), _mesh->NEList[n[2]].begin(),
              _mesh->NEList[n[2]].end(), inserter(intersection, intersection.begin()));
          list_offset[0] = (eid == *intersection.begin() ? 2 : 4);

          intersection.clear();
          set_intersection(_mesh->NEList[n[0]].begin(),
              _mesh->NEList[n[0]].end(), _mesh->NEList[n[2]].begin(),
              _mesh->NEList[n[2]].end(), inserter(intersection, intersection.begin()));
          list_offset[1] = (eid == *intersection.begin() ? 2 : 4);

          intersection.clear();
          set_intersection(_mesh->NEList[n[0]].begin(),
              _mesh->NEList[n[0]].end(), _mesh->NEList[n[1]].begin(),
              _mesh->NEList[n[1]].end(), inserter(intersection, intersection.begin()));
          list_offset[2] = (eid == *intersection.begin() ? 2 : 4);

          // Append newVertex[1] and newVertex[2] in newVertex[0]'s NNList
          _mesh->NNList[newVertex[0]][list_offset[0]] = newVertex[1];
          _mesh->NNList[newVertex[0]][list_offset[0]+1] = newVertex[2];

          // Append newVertex[0] and newVertex[2] in newVertex[1]'s NNList
          _mesh->NNList[newVertex[1]][list_offset[1]] = newVertex[0];
          _mesh->NNList[newVertex[1]][list_offset[1]+1] = newVertex[2];

          // Append newVertex[0] and newVertex[1] in newVertex[2]'s NNList
          _mesh->NNList[newVertex[2]][list_offset[2]] = newVertex[0];
          _mesh->NNList[newVertex[2]][list_offset[2]+1] = newVertex[1];

          // Update NEList
          size_t idx = indexInNEList(eid, n[0]);
          addToNEList(n[0], idx, ele0ID, tid);
          idx = indexInNEList(eid, n[1]);
          addToNEList(n[1], idx, ele1ID, tid);
          idx = indexInNEList(eid, n[2]);
          addToNEList(n[2], idx, ele2ID, tid);

          if(list_offset[0] == 2){
            addToNEList(newVertex[0], 0, ele1ID, tid);
            addToNEList(newVertex[0], 1, ele2ID, tid);
            addToNEList(newVertex[0], 2, ele3ID, tid);
          }
          else{
            addToNEList(newVertex[0], 3, ele1ID, tid);
            addToNEList(newVertex[0], 4, ele2ID, tid);
            addToNEList(newVertex[0], 5, ele3ID, tid);
          }

          if(list_offset[1] == 2){
            addToNEList(newVertex[1], 0, ele0ID, tid);
            addToNEList(newVertex[1], 1, ele2ID, tid);
            addToNEList(newVertex[1], 2, ele3ID, tid);
          }
          else{
            addToNEList(newVertex[1], 3, ele0ID, tid);
            addToNEList(newVertex[1], 4, ele2ID, tid);
            addToNEList(newVertex[1], 5, ele3ID, tid);
          }

          if(list_offset[2] == 2){
            addToNEList(newVertex[2], 0, ele0ID, tid);
            addToNEList(newVertex[2], 1, ele1ID, tid);
            addToNEList(newVertex[2], 2, ele3ID, tid);
          }
          else{
            addToNEList(newVertex[2], 3, ele0ID, tid);
            addToNEList(newVertex[2], 4, ele1ID, tid);
            addToNEList(newVertex[2], 5, ele3ID, tid);
          }
        }

        // Remove parent element.
        _mesh->erase_element(eid);
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

      // Fix IDs of new elements in NEList, compact _mesh->NNList update _mesh->NEList
#pragma omp for schedule(dynamic)
     for(index_t i=0; i<(int)NNodes; ++i){
       if(_mesh->NNList[i].empty())
         continue;

       // Fix IDs of new elements in NEList
       for(size_t j=0; j < NEList[i].size(); ++j)
         if(createdByThread[i][j] != std::numeric_limits<size_t>::max())
           NEList[i][j] += threadIdx[createdByThread[i][j]];

       // Compact _mesh->NNList
       size_t forward, backward;
       if(i < (int)origNNodes){
         /*
          * The first NNList[i].size()/2 slots of the list are occupied
          * for sure, so we can search only in the newly allocated slots.
          */
         forward = _mesh->NNList[i].size() / 2;
         backward = _mesh->NNList[i].size() - 1;
       }else{
         /*
          * The first 2 slots of the list are occupied for
          * sure, so we can search only in the rest 4 slots.
          */
         forward = 2;
         backward = 5;
       }

       while(forward < backward){
         while(_mesh->NNList[i][forward] != -1){
           ++forward;
           if(forward == backward)
             break;
         }
         while(_mesh->NNList[i][backward] == -1){
           --backward;
           if(forward > backward)
             break;
         }

         if(forward < backward){
           _mesh->NNList[i][forward++] = _mesh->NNList[i][backward];
           _mesh->NNList[i][backward--] = -1;
         }
         else
           break;
       }
       if(_mesh->NNList[i][forward] != -1)
         ++forward;

       _mesh->NNList[i].resize(forward);

       if(i < (int)origNNodes){
         /*
          * The first NEList[i].size()/2 slots of the list are occupied
          * for sure, so we can search only in the newly allocated slots.
          */
         forward = NEList[i].size() / 2;
         backward = NEList[i].size() - 1;
       }else{
         /*
          * The first 2 slots of the list are occupied for
          * sure, so we can search only in the rest 4 slots.
          */
         forward = 2;
         backward = 5;
       }

       while(forward < backward){
         while(NEList[i][forward] != -1){
           ++forward;
           if(forward==backward)
             break;
         }
         while(NEList[i][backward] == -1){
           --backward;
           if(forward>backward)
             break;
         }

         if(forward < backward){
           NEList[i][forward++] = NEList[i][backward];
           NEList[i][backward--] = -1;
         }
         else
           break;
       }
       if(NEList[i][forward] != -1)
         ++forward;

       // Update _mesh->NEList
       if(i < (int)origNNodes)
         _mesh->NEList[i].clear();
       std::copy(&NEList[i][0], &NEList[i][forward],
           std::inserter(_mesh->NEList[i], _mesh->NEList[i].begin()));
     }

#ifdef HAVE_MPI
      if(nprocs>1){
#pragma omp master
        {
          // Time to amend halo.
          assert(node_owner.size()==NNodes);

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

          typename std::vector< std::set< DirectedEdge<index_t> > > send_additional(nprocs), recv_additional(nprocs);
          for(size_t i=origNElements;i<NElements;i++){
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
    _surface->refine(refined_edges, NNList, lnn2gnn);

    return;
  }

 private:

  void refine_edge(index_t n0, index_t n1, std::vector< DirectedEdge<index_t> > &newVertices,
      std::vector<real_t> &coords, std::vector<float> &metric){
    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }
    newVertices.push_back(DirectedEdge<index_t>(lnn2gnn[n0], lnn2gnn[n1]));

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
      coords.push_back(x);
    }

    // Interpolate new metric and append it to OMP thread's temp storage
    for(size_t i=0;i<msize;i++){
      m = m0[i]+weight*(m1[i] - m0[i]);
      metric.push_back(m);
      if(isnan(m))
        std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                 <<"m0[i] = "<<m0[i]<<std::endl
                 <<"m1[i] = "<<m1[i]<<std::endl
                 <<"weight = "<<weight<<std::endl;
    }
  }

  inline void append_element(const index_t *elem, std::vector<index_t> &ENList){
    for(size_t i=0; i<nloc; ++i)
      ENList.push_back(elem[i]);
  }

  inline void addToNEList(index_t vertex, size_t idx, index_t element, size_t tid){
    NEList[vertex][idx] = element;
    createdByThread[vertex][idx] = tid;
  }

  inline size_t indexInNEList(index_t target, index_t vertex) const{
    size_t pos = 0;
    while(pos < NEList[vertex].size()/2){
      if(NEList[vertex][pos] == target &&
          createdByThread[vertex][pos] == std::numeric_limits<size_t>::max())
        return pos;

      ++pos;
    }

    return std::numeric_limits<size_t>::max();
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;

  index_t *lnn2gnn;
  std::map<index_t, index_t> gnn2lnn;
  std::vector<int> node_owner;

  std::vector< std::vector<index_t> > NNList;
  std::vector< std::vector<index_t> > NEList;
  std::vector< std::vector<index_t> > refined_edges;
  std::vector< std::vector<size_t> > createdByThread;

  static const size_t ndims=2, nloc=3, msize=3;
  int nprocs, rank, nthreads;
};

#endif
