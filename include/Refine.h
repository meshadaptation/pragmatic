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
    
    // Initialise a dynamic vertex list
    std::vector< std::vector<index_t> > refined_edges(NNodes);
    std::vector< std::vector< DirectedEdge<index_t> > > newVertices(nthreads);
    std::vector< std::vector<real_t> > newCoords(nthreads);
    std::vector< std::vector<float> > newMetric(nthreads);
    std::vector< std::vector<index_t> > newElements(nthreads);
    std::vector<size_t> threadIdx(nthreads), splitCnt(nthreads, 0);

#pragma omp parallel
    {
      
      // Establish global node numbering.
#pragma omp master
      {
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
      }

#pragma omp single
      {// Calculate node ownership.
        node_owner.resize(NNodes);
        for(size_t i=0;i<NNodes;i++)
          node_owner[i] = rank;
        
        if(nprocs>1){
          for(int i=0;i<nprocs;i++){
            for(std::vector<int>::const_iterator it=_mesh->recv[i].begin();it!=_mesh->recv[i].end();++it){
              node_owner[*it] = i;
            }
          }
        }
      } // barrier implied
      
      int tid = omp_get_thread_num();

      /* Loop through all edges and select them for refinement if
         its length is greater than L_max in transformed space. */
#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;++i){
        /*
         * Space must be allocated for refined_edges[i] in any case, no matter
         * whether any of the edges adjacent to vertex i will be refined or not.
         * This is because function mark_edge(...) assumes that space has already
         * been allocated. Allocating space for refined_edges[i] on demand, i.e.
         * inside mark_edge(...), is not possible, since mark_edge(...) may be
         * called for the same vertex i by two threads at the same time.
         */
        refined_edges[i].resize(2*_mesh->NNList[i].size(), -1);
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

              refined_edges[i][2*it]   = splitCnt[tid]++;
              refined_edges[i][2*it+1] = tid;

              refine_edge(i, otherVertex, newVertices[tid], newCoords[tid], newMetric[tid]);
            }
          }
      	}
      }
      
      //
      // Insert new vertices into mesh.
      //

      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in mesh._coords and mesh.metric at which new coords and metric should be appended.
      threadIdx[tid] = splitCnt[tid];
      
#pragma omp barrier
      
      unsigned int blockSize = 1, tmp;
      while(blockSize < threadIdx.size()){
        if((tid & blockSize) != 0)
          tmp = threadIdx[tid - ((tid & (blockSize - 1)) + 1)];
        else
          tmp = 0;
        
#pragma omp barrier
        
        threadIdx[tid] += tmp;
        
#pragma omp barrier
        
        blockSize <<= 1;
      }
      
      threadIdx[tid] += NNodes - splitCnt[tid];
      
      // Resize mesh containers. The above must have completed first, thus the barrier.
#pragma omp barrier
#pragma omp single
      {
        const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
        
        _mesh->_coords.resize(ndims*newSize);
        _mesh->metric.resize(msize*newSize);
        node_owner.resize(newSize, -1);
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
       * This approach seems better only for subsequent calls to refine(),where
       * only a few edges are expected to be refined.
       */
      // Fix IDs of new vertices in refined_edges
#pragma omp for schedule(dynamic)
      for(size_t i=0; i<refined_edges.size(); ++i){
        for(typename std::vector<index_t>::iterator it=refined_edges[i].begin(); it!=refined_edges[i].end(); it+=2)
          if(*it != -1)
            *it += threadIdx[*(it+1)];
      }
      
      // Perform element refinement.
      splitCnt[tid] = 0;

#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NElements;i++){
        // Check if this element has been erased - if so continue to next element.
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
        
        // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
        index_t newVertex[3];
        newVertex[0] = _mesh->get_new_vertex(n[1], n[2], refined_edges, lnn2gnn);
        newVertex[1] = _mesh->get_new_vertex(n[2], n[0], refined_edges, lnn2gnn);
        newVertex[2] = _mesh->get_new_vertex(n[0], n[1], refined_edges, lnn2gnn);
        
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
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          splitCnt[tid] += 2;
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
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          splitCnt[tid] += 3;
        }else if(refine_cnt==3){
          const int ele0[] = {n[0], newVertex[2], newVertex[1]};
          const int ele1[] = {n[1], newVertex[0], newVertex[2]};
          const int ele2[] = {n[2], newVertex[1], newVertex[0]};
          const int ele3[] = {newVertex[0], newVertex[1], newVertex[2]};
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          append_element(ele3, newElements[tid]);
          splitCnt[tid] += 4;
        }

        // Remove parent element.
        _mesh->erase_element(i);
      }
      
      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in mesh._ENList at which new elements should be appended.
      threadIdx[tid] = splitCnt[tid];

#pragma omp barrier
      
      blockSize = 1;
      while(blockSize < threadIdx.size())
        {
          if((tid & blockSize) != 0)
            tmp = threadIdx[tid - ((tid & (blockSize - 1)) + 1)];
          else
            tmp = 0;
          
#pragma omp barrier
          
          threadIdx[tid] += tmp;
          
#pragma omp barrier
          
          blockSize *= 2;
        }
      
      threadIdx[tid] += NElements - splitCnt[tid];
      
#pragma omp barrier

      // Resize mesh containers
#pragma omp single
      {
      	const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
        
      	_mesh->_ENList.resize(nloc*newSize);
      }
      
      // Append new elements to the mesh
      memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
          
#pragma omp single
      {
        NNodes = _mesh->get_number_nodes();
        NElements = _mesh->get_number_elements();
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
      
      // Refine surface
#pragma omp single
      {
        _surface->refine(refined_edges, lnn2gnn);
      }

      // Recreate adjancy information.
      _mesh->create_adjancy();
    }

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

  inline void mark_edge(index_t n0, index_t n1, std::vector< std::vector<index_t> > &refined_edges){
    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }

    index_t pos = 0;
    while(_mesh->NNList[n0][pos] != n1) ++pos;
    
    /*
     * WARNING! Code analysis tools may warn about a race condition
     * (write-after-write) for the following line. This is not really
     * a problem, since any thread accessing this place in memory will
     * write the same value (MAX_INT).
     */
    refined_edges[n0][2*pos] = std::numeric_limits<index_t>::max();
  }
  
  inline void append_element(const index_t *elem, std::vector<index_t> &ENList){
    for(size_t i=0; i<nloc; ++i)
      ENList.push_back(elem[i]);
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  
  index_t *lnn2gnn;
  std::map<index_t, index_t> gnn2lnn;
  std::vector<int> node_owner;

  static const size_t ndims=2, nloc=3, msize=3;
  int nprocs, rank, nthreads;
};



/*! \brief Performs 3D mesh refinement.
 *
 */
template<typename real_t, typename index_t> class Refine3D{
 public:
  /// Default constructor.
  Refine3D(Mesh<real_t, index_t> &mesh, Surface3D<real_t, index_t> &surface){
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
                                             _mesh->get_coords(n[2]),
                                             _mesh->get_coords(n[3]));
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
  ~Refine3D(){
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
    
    // Initialise a dynamic vertex list
    std::vector< std::vector<index_t> > refined_edges(NNodes);
    std::vector< std::vector< DirectedEdge<index_t> > > newVertices(nthreads);
    std::vector< std::vector<real_t> > newCoords(nthreads);
    std::vector< std::vector<float> > newMetric(nthreads);
    std::vector< std::vector<index_t> > newElements(nthreads);
    std::vector<size_t> threadIdx(nthreads), splitCnt(nthreads, 0);

    int new_edges_size;

#pragma omp parallel
    {      
      // Establish global node numbering.
#pragma omp master
      {
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
      }

#pragma omp single
      {// Calculate node ownership.
        node_owner.resize(NNodes);
        for(size_t i=0;i<NNodes;i++)
          node_owner[i] = rank;
        
        if(nprocs>1){
          for(int i=0;i<nprocs;i++){
            for(std::vector<int>::const_iterator it=_mesh->recv[i].begin();it!=_mesh->recv[i].end();++it){
              node_owner[*it] = i;
            }
          }
        }
      } // barrier implied
      
      int tid = omp_get_thread_num();
      
      /* Loop through all edges and select them for refinement if
         its length is greater than L_max in transformed space. */
#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;++i){
        /*
         * Space must be allocated for refined_edges[i] in any case, no matter
         * whether any of the edges adjacent to vertex i will be refined or not.
         * This is because function mark_edge(...) assumes that space has already
         * been allocated. Allocating space for refined_edges[i] on demand, i.e.
         * inside mark_edge(...), is not possible, since mark_edge(...) may be
         * called for the same vertex i by two threads at the same time.
         */
        refined_edges[i].resize(2*_mesh->NNList[i].size(), -1);
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

              refined_edges[i][2*it]   = splitCnt[tid]++;
              refined_edges[i][2*it+1] = tid;

              refine_edge(i, otherVertex, newVertices[tid], newCoords[tid], newMetric[tid]);
            }
          }
      	}
      }
      
      /* Given the set of refined edges, apply additional edge-refinement
         to get a regular and conforming element refinement throughout
         the domain. */
      for(;;){
#pragma omp single
        {
          new_edges_size = 0;
        }
#pragma omp for schedule(dynamic) reduction(+:new_edges_size)
        for(size_t i=0;i<NElements;i++){
          // Check if this element has been erased - if so continue to next element.
          const int *n=_mesh->get_element(i);
          if(n[0]<0)
            continue;
          
          // Find what edges have been split in this element.
          typename std::vector< Edge<index_t> > split_set;
          for(size_t j=0;j<nloc;j++){
            for(size_t k=j+1;k<nloc;k++){
              if(_mesh->get_new_vertex(n[j], n[k], refined_edges, lnn2gnn) >= 0)
                split_set.push_back(Edge<index_t>(n[j], n[k]));
            }
          }
          
          int refine_cnt=split_set.size();
          
          switch(refine_cnt){
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
              
              mark_edge(n1, n2, refined_edges);
              new_edges_size++;
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
                    mark_edge(n[j], n[k], refined_edges);
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
                  mark_edge(n[j], n[k], refined_edges);
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
                  mark_edge(n[j], n[k], refined_edges);
                  new_edges_size++;
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
      
        // If there are no new edges then we can jump out of the infinite loop.
#ifdef HAVE_MPI
        if(nprocs>1){
#pragma omp master
          MPI_Allreduce(MPI_IN_PLACE, &new_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#pragma omp barrier
        }
#endif
        
        if(new_edges_size==0)
          break;
        
        // Add additional edges to refined_edges.
        
        // Loop through all edges and refine those which have been marked
#pragma omp for schedule(dynamic)
        for(int i=0;i<(int)NNodes;++i){
          for(index_t it = 0; it < (int)_mesh->NNList[i].size(); ++it){
            if(refined_edges[i][2*it] == std::numeric_limits<index_t>::max()){
              index_t otherVertex = _mesh->NNList[i][it];
              refined_edges[i][2*it]   = splitCnt[tid]++;
              refined_edges[i][2*it+1] = tid;
              refine_edge(i, otherVertex, newVertices[tid], newCoords[tid], newMetric[tid]);
            }
          }
        }
      }
    
      //
      // Insert new vertices into mesh.
      //
      
      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in mesh._coords and mesh.metric at which new coords and metric should be appended.
      threadIdx[tid] = splitCnt[tid];
      
#pragma omp barrier
      
      unsigned int blockSize = 1, tmp;
      while(blockSize < threadIdx.size()){
        if((tid & blockSize) != 0)
          tmp = threadIdx[tid - ((tid & (blockSize - 1)) + 1)];
        else
          tmp = 0;
        
#pragma omp barrier
        
        threadIdx[tid] += tmp;
        
#pragma omp barrier
        
        blockSize <<= 1;
      }
      
      threadIdx[tid] += NNodes - splitCnt[tid];
      
      // Resize mesh containers. The above must have completed first, thus the barrier.
#pragma omp barrier
#pragma omp single
      {
        const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
        
        _mesh->_coords.resize(ndims*newSize);
        _mesh->metric.resize(msize*newSize);
        node_owner.resize(newSize, -1);
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
       * This approach seems better only for subsequent calls to refine(),where
       * only a few edges are expected to be refined.
       */
      // Fix IDs of new vertices in refined_edges
#pragma omp for schedule(dynamic)
      for(size_t i=0; i<refined_edges.size(); ++i){
        for(typename std::vector<index_t>::iterator it=refined_edges[i].begin(); it!=refined_edges[i].end(); it+=2)
          if(*it != -1)
            *it += threadIdx[*(it+1)];
      }
      
      // Perform element refinement.
      splitCnt[tid] = 0;

#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NElements;i++){
        // Check if this element has been erased - if so continue to next element.
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
          
        std::vector<index_t> newVertex;
        std::vector< Edge<index_t> > splitEdges;
        index_t vertexID;
        for(size_t j=0;j<4;j++)
          for(size_t k=j+1;k<4;k++){
            vertexID = _mesh->get_new_vertex(n[j], n[k], refined_edges, lnn2gnn);
            if(vertexID >= 0){
              newVertex.push_back(vertexID);
              splitEdges.push_back(Edge<index_t>(n[j], n[k]));
            }
          }
        int refine_cnt=newVertex.size();
          
        // Apply refinement templates.
        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
          // Find the opposite edge
          int oe[2];
          for(int j=0, pos=0;j<4;j++)
            if(!splitEdges[0].contains(n[j]))
              oe[pos++] = n[j];
            
          // Form and add two new edges.
          const int ele0[] = {splitEdges[0].edge.first, newVertex[0], oe[0], oe[1]};
          const int ele1[] = {splitEdges[0].edge.second, newVertex[0], oe[0], oe[1]};
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          splitCnt[tid] += 2;
        }else if(refine_cnt==2){
          const int ele0[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
          const int ele1[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.second, newVertex[1]};
          const int ele2[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
          const int ele3[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.second, newVertex[1]};
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          append_element(ele3, newElements[tid]);
          splitCnt[tid] += 4;
        }else if(refine_cnt==3){
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
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          append_element(ele3, newElements[tid]);
          splitCnt[tid] += 4;
        }else if(refine_cnt==6){
          const int ele0[] = {n[0], newVertex[0], newVertex[1], newVertex[2]};
          const int ele1[] = {n[1], newVertex[3], newVertex[0], newVertex[4]};
          const int ele2[] = {n[2], newVertex[1], newVertex[3], newVertex[5]};
          const int ele3[] = {newVertex[0], newVertex[3], newVertex[1], newVertex[4]};
          const int ele4[] = {newVertex[0], newVertex[4], newVertex[1], newVertex[2]};
          const int ele5[] = {newVertex[1], newVertex[3], newVertex[5], newVertex[4]};
          const int ele6[] = {newVertex[1], newVertex[4], newVertex[5], newVertex[2]};
          const int ele7[] = {newVertex[2], newVertex[4], newVertex[5], n[3]};
            
          append_element(ele0, newElements[tid]);
          append_element(ele1, newElements[tid]);
          append_element(ele2, newElements[tid]);
          append_element(ele3, newElements[tid]);
          append_element(ele4, newElements[tid]);
          append_element(ele5, newElements[tid]);
          append_element(ele6, newElements[tid]);
          append_element(ele7, newElements[tid]);
          splitCnt[tid] += 8;
        }
          
        // Remove parent element.
        _mesh->erase_element(i);
      }
      
      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in mesh._ENList at which new elements should be appended.
      threadIdx[tid] = splitCnt[tid];
      
#pragma omp barrier
      
      blockSize = 1;
      while(blockSize < threadIdx.size())
        {
          if((tid & blockSize) != 0)
            tmp = threadIdx[tid - ((tid & (blockSize - 1)) + 1)];
          else
            tmp = 0;
          
#pragma omp barrier
          
          threadIdx[tid] += tmp;
          
#pragma omp barrier
          
          blockSize *= 2;
        }
      
      threadIdx[tid] += NElements - splitCnt[tid];
      
      // Resize mesh containers
#pragma omp barrier
#pragma omp single
      {
      	const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
        
      	_mesh->_ENList.resize(nloc*newSize);
      }
      
      // Append new elements to the mesh
      memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
            
      NNodes = _mesh->get_number_nodes();
      NElements = _mesh->get_number_elements();

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
      
      // Refine surface
#pragma omp single
      _surface->refine(refined_edges, lnn2gnn);
      
      // Recreate adjancy information.
      _mesh->create_adjancy();
    }

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

  inline void mark_edge(index_t n0, index_t n1, std::vector< std::vector<index_t> > &refined_edges){
    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }

    index_t pos = 0;
    while(_mesh->NNList[n0][pos] != n1) ++pos;
    
    /*
     * WARNING! Code analysis tools may warn about a race condition
     * (write-after-write) for the following line. This is not really
     * a problem, since any thread accessing this place in memory will
     * write the same value (MAX_INT).
     */
    refined_edges[n0][2*pos] = std::numeric_limits<index_t>::max();
  }
  
  inline void append_element(const index_t *elem, std::vector<index_t> &ENList){
    for(size_t i=0; i<nloc; ++i)
      ENList.push_back(elem[i]);
  }

  Mesh<real_t, index_t> *_mesh;
  Surface3D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  
  index_t *lnn2gnn;
  std::map<index_t, index_t> gnn2lnn;
  std::vector<int> node_owner;

  static const size_t ndims=3, nloc=4, msize=6;
  int nprocs, rank, nthreads;
};

#endif
