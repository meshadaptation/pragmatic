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
#include "Colour.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t> class Swapping2D : public AdaptiveAlgorithm<real_t>{
 public:
  /// Default constructor.
  Swapping2D(Mesh<real_t> &mesh, Surface2D<real_t> &surface){
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

    nthreads = pragmatic_nthreads();
  }

  /// Default destructor.
  ~Swapping2D(){
    if(property!=NULL)
      delete property;
  }

  void swap(real_t quality_tolerance){
    size_t NNodes = _mesh->get_number_nodes();

    size_t NElements = _mesh->get_number_elements();
    int active_colour;

    min_Q = quality_tolerance;

    if(nnodes_reserve<1.5*NNodes){
      nnodes_reserve = 2*NNodes;

      quality.resize(nnodes_reserve * 2, 0.0);
      marked_edges.resize(nnodes_reserve, std::set<index_t>());
    }

    std::vector<char> colouring(NNodes);
    Colour::GebremedhinManne(NNodes, _mesh->NNList, colouring);
    
    int colour_count[64];
#pragma omp parallel firstprivate(NNodes, NElements)
    {
      int tid = pragmatic_thread_id();
      
      // Cache the element quality's. Really need to make this persistent within Mesh.
      // Also, initialize marked_edges.
#pragma omp for schedule(static,8)
      for(size_t i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]>=0){
          const real_t *x0 = _mesh->get_coords(n[0]);
          const real_t *x1 = _mesh->get_coords(n[1]);
          const real_t *x2 = _mesh->get_coords(n[2]);
          
          quality[i] = property->lipnikov(x0, x1, x2,
                                          _mesh->get_metric(n[0]),
                                          _mesh->get_metric(n[1]),
                                          _mesh->get_metric(n[2]));

        }else{
          quality[i] = 0.0;
        }
      }
#pragma omp for schedule(static,8)
      for(size_t i=0;i<NNodes;i++){
        for(std::set<index_t>::const_iterator it=_mesh->NEList[i].begin();it!=_mesh->NEList[i].end();++it){
          if(quality[i]<min_Q){
            for(std::vector<index_t>::const_iterator jt=_mesh->NNList[i].begin();jt!=_mesh->NNList[i].end();++jt){
              if(i<(size_t)*jt)
                marked_edges[i].insert(*jt);
            }
            break;
          }
        }
      }

      for(size_t sweep=0;sweep<100;sweep++){
        // Find the maximal independant set.
        std::vector<int> local_colour_count(64, 0);
#pragma omp single nowait
        for(size_t i=0;i<64;i++)
          colour_count[i] = 0;
#pragma omp for schedule(static,8)
        for(size_t i=0;i<_mesh->NNodes;i++){
          if(!marked_edges[i].empty()){
            ++local_colour_count[colouring[i]];
          }
        }
        for(int i=0;i<64;i++)
	  pragmatic_omp_atomic_update()
	    colour_count[i]+=local_colour_count[i];
#pragma omp barrier
	
        // Find the maximal independant set.
#pragma omp single
        {
          active_colour = -1;
          int cnt=-1;
          size_t i;
          for(i=0;i<64;i++){
            if(colour_count[i]>0){
              cnt=colour_count[i];
              active_colour = i;
              break;
            }
          }
          for(;i<64;i++){
            if(colour_count[i]>cnt){
              cnt = colour_count[i];
              active_colour = i;
            }
          }
        }
        if(active_colour<0)
          break;

#pragma omp for schedule(static,8)
        for(size_t i=0; i<NNodes; ++i){
          // Only process if it is the active colour and has marked edges.
          if(!((colouring[i]==active_colour)&&(!marked_edges[i].empty())))
            continue;

          // Set of elements in this cavity which were modified since the last commit of deferred operations.
          std::set<index_t> modified_elements;
                
          for(typename std::set<index_t>::const_iterator vid=marked_edges[i].begin(); vid!=marked_edges[i].end();){
            std::set<index_t>::const_iterator j_ref = vid; ++vid;
            index_t j = *j_ref;

            // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
            bool skip=false;
            for(std::set<index_t>::const_iterator it=modified_elements.begin();it!=modified_elements.end();++it){
              if(_mesh->NEList[j].find(*it)!=_mesh->NEList[j].end()){
                skip=true;
                break;
              }
            }
            if(skip)
              continue;

            // Mark edge as processed, i.e. remove it from the set of marked edges
            marked_edges[i].erase(j_ref);

            Edge<index_t> edge(i, j);
            swap_kernel(edge, modified_elements, NULL, NULL, pragmatic_thread_id());

            // If edge was swapped
            if(edge.edge.first != (index_t)i){
              index_t k = edge.edge.first;
              index_t l = edge.edge.second;

              Edge<index_t> lateralEdges[] = {
                Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};
              
              // Propagate the operation
              for(size_t ee=0; ee<4; ++ee)
                _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, pragmatic_thread_id());
            }
          }
        }
        _mesh->commit_deferred(tid);
        _mesh->commit_swapping_propagation(marked_edges, tid);
#pragma omp barrier
        Colour::repair(NNodes, _mesh->NNList, marked_edges, colouring);
      }
    }
  }

 private:

  void swap_kernel(Edge<index_t>& edge, std::set<index_t>& modified_elements,
      std::vector<index_t>* ele0, std::vector<index_t>* ele1, size_t tid){
    index_t i = edge.edge.first;
    index_t j = edge.edge.second;

    if(_mesh->is_halo_node(i)&& _mesh->is_halo_node(j))
      return;

    // Find the two elements sharing this edge
    index_t intersection[2];
    {
      size_t loc = 0;
      std::set<index_t>::const_iterator it=_mesh->NEList[i].begin();
      while(loc<2 && it!=_mesh->NEList[i].end()){
        if(_mesh->NEList[j].find(*it)!=_mesh->NEList[j].end()){
          intersection[loc++] = *it;
        }
        ++it;
      }

      // If this is a surface edge, it cannot be swapped.
      if(loc!=2)
        return;
    }

    index_t eid0 = intersection[0];
    index_t eid1 = intersection[1];

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

    if(_mesh->is_halo_node(k)&& _mesh->is_halo_node(l))
      return;

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

  void swap_kernel_single_thr(index_t i, index_t j, index_t k, index_t l){
    // Find the two elements sharing this edge
    std::set<index_t> intersection;
    std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
        _mesh->NEList[j].begin(), _mesh->NEList[j].end(),
        std::inserter(intersection, intersection.begin()));

    index_t eid0 = *intersection.begin();
    index_t eid1 = *intersection.rbegin();

    if(intersection.size() > 2){
      eid0 = -1;
      for(typename std::set<index_t>::const_iterator ee=intersection.begin(); ee!=intersection.end(); ++ee){
        const index_t *ele = _mesh->get_element(*ee);
        for(size_t c=0; c<nloc; ++c)
          if(ele[c]==k){
            eid0 = *ee;
            break;
          }

        if(eid0 >= 0)
          break;
      }

      eid1 = -1;
      for(typename std::set<index_t>::const_iterator ee=intersection.begin(); ee!=intersection.end(); ++ee){
        const index_t *ele = _mesh->get_element(*ee);
        for(size_t c=0; c<nloc; ++c)
          if(ele[c]==l){
            eid1 = *ee;
            break;
          }

        if(eid1 >= 0)
          break;
      }
    }

    const index_t *n = _mesh->get_element(eid0);
    int n_off=-1;
    for(size_t kk=0;kk<3;kk++){
      if((n[kk]!=i) && (n[kk]!=j)){
        n_off = kk;
        break;
      }
    }
    assert(n[n_off]>=0);

    const index_t *m = _mesh->get_element(eid1);
    int m_off=-1;
    for(size_t kk=0;kk<3;kk++){
      if((m[kk]!=i) && (m[kk]!=j)){
        m_off = kk;
        break;
      }
    }
    assert(m[m_off]>=0);

    assert(n[(n_off+2)%3]==m[(m_off+1)%3] && n[(n_off+1)%3]==m[(m_off+2)%3]);

    //index_t k = n[n_off];
    //index_t l = m[m_off];

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

    return;
  }

  inline virtual index_t is_dynamic(index_t vid){
    return (index_t) marked_edges[vid].size();
  }

  Mesh<real_t> *_mesh;
  Surface2D<real_t> *_surface;
  ElementProperty<real_t> *property;

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
                                                   msize*sizeof(double)) / sizeof(int);
  const static size_t idx_owner = sizeof(index_t) / sizeof(int);
  const static size_t idx_coords = idx_owner + 1;
  const static size_t idx_metric = idx_coords + ndims*sizeof(real_t) / sizeof(int);

  int nprocs, rank;
};

#endif
