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

#include "ElementProperty.h"
#include "Mesh.h"
#include "Colour.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, typename index_t> class Swapping2D{
 public:
  /// Default constructor.
  Swapping2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;
    
    dynamic_vertex = NULL;
    tpartition = NULL;

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

#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#else
    nthreads=1;
#endif
  }
  
  /// Default destructor.
  ~Swapping2D(){
    delete property;
  }
  
  void swap(real_t Q_min){
    min_Q = Q_min;
    size_t NNodes = _mesh->get_number_nodes();
    size_t NElements = _mesh->get_number_elements();

    quality.clear();
    quality.resize(NElements);
    
    marked_edges.clear();
    marked_edges.resize(NNodes);
    
    tpartition = new int[NNodes];
    dynamic_vertex = new int[NNodes];

#pragma omp parallel
    {
      int tid = get_tid();

      // Cache the element quality's.
#pragma omp for schedule(static, 32)
      for(size_t i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0){
          quality[i] = 0.0;
          continue;
        }
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        
        quality[i] = property->lipnikov(x0, x1, x2,
                                        _mesh->get_metric(n[0]),
                                        _mesh->get_metric(n[1]),
                                        _mesh->get_metric(n[2]));
      }
      
      // Initialise list of dynamic edges.
#pragma omp for schedule(static, 32)
      for(size_t i=0;i<NNodes;i++){
        for(size_t it=0; it<_mesh->NNList[i].size(); ++it){
          if(i < (size_t) _mesh->NNList[i][it])
            marked_edges[i].push_back(_mesh->NNList[i][it]);
        }
        dynamic_vertex[i] = marked_edges[i].size();
      }

      // Phase 1
      if(nthreads>1){
#pragma omp single
        {
          pragmatic::partition_fast(_mesh->NNList, NNodes, dynamic_vertex, nthreads, tpartition);
        }

        // Each thread creates a list of edges it is responsible for processing
        std::deque< Edge<index_t> > *tdynamic_edge = new std::deque< Edge<index_t> >;

        for(index_t i=0; i<(index_t)NNodes; ++i){
          if(tpartition[i]!=tid)
            continue;

          for(typename std::list<index_t>::const_iterator it=marked_edges[i].begin(); it!=marked_edges[i].end(); ++it){
            index_t opp = *it;
            if(tpartition[opp]!=tid)
              continue;

            tdynamic_edge->push_back(Edge<index_t>(i, opp));
          }
        }

        // Now, each thread starts processing its dynamic list
        process_list(tdynamic_edge, tid);

        delete tdynamic_edge;
      }
    }

    // Phase 2
    std::deque< Edge<index_t> > *tdynamic_edge = new std::deque< Edge<index_t> >;

    for(index_t i=0; i<(index_t)NNodes; ++i){
      tpartition[i] = 0;
      for(typename std::list<index_t>::const_iterator it=marked_edges[i].begin(); it!=marked_edges[i].end(); ++it){
        tdynamic_edge->push_back(Edge<index_t>(i,*it));
      }
    }

    process_list(tdynamic_edge, 0);

    delete tdynamic_edge;

    // Phase 3
    // Future work
    // At this point, all edges in marked_edges are halo edges.

    delete[] tpartition;
    delete[] dynamic_vertex;

    return;
  }

 private:

  struct Rhombus{
    Rhombus(Edge<index_t> edge) : middleEdge(edge) {};
    Edge<index_t> middleEdge;
    index_t eid0;
    index_t eid1;
    size_t n_off;
    size_t m_off;
  };

#pragma GCC diagnostic ignored "-Wuninitialized"
  void swap_kernel(Rhombus &rhombus){
    int eid0 = rhombus.eid0;
    int eid1 = rhombus.eid1;

    real_t worst_q = std::min(quality[eid0], quality[eid1]);
    /*
    if(worst_q>min_Q)
      return;
    */

    index_t i = rhombus.middleEdge.edge.first;
    index_t j = rhombus.middleEdge.edge.second;

    int n_off = rhombus.n_off;
    int m_off = rhombus.m_off;

    const index_t *n = _mesh->get_element(eid0);
    const index_t *m = _mesh->get_element(eid1);

    index_t k = n[n_off];
    index_t l = m[m_off];

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

      rhombus.middleEdge = Edge<index_t>(k, l);
    }
  }

  bool is_eligible(Rhombus &rhombus, int tid){
    index_t i = rhombus.middleEdge.edge.first;
    index_t j = rhombus.middleEdge.edge.second;

    /*
     * For safe MPI execution, it is not allowed at phases 1 and 2 to
     * delete edges or create new edges on the halo of an MPI partition.
     */
    if(_mesh->is_halo_node(i) && _mesh->is_halo_node(j))
      return false;

    // Find the two elements sharing this edge
    std::set<index_t> intersection;
    std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
        _mesh->NEList[j].begin(), _mesh->NEList[j].end(),
        std::inserter(intersection, intersection.begin()));

    // If this is a surface edge, it cannot be swapped, so un-mark it.
    if(intersection.size()!=2){
      typename std::list<index_t>::iterator it;
      it = std::find(marked_edges[i].begin(), marked_edges[i].end(), j);
      marked_edges[i].erase(it);
      return false;
    }

    index_t eid0 = *intersection.begin();
    index_t eid1 = *intersection.rbegin();

    const index_t *n = _mesh->get_element(eid0);
    int n_off=-1;
    for(size_t k=0;k<3;k++){
      if((n[k]!=i) && (n[k]!=j)){
        n_off = k;
        break;
      }
    }
    assert(n[n_off]>=0);
    if(tpartition[n[n_off]]!=tid)
      return false;

    const index_t *m = _mesh->get_element(eid1);
    int m_off=-1;
    for(size_t k=0;k<3;k++){
      if((m[k]!=i) && (m[k]!=j)){
        m_off = k;
        break;
      }
    }
    assert(m[m_off]>=0);
    if(tpartition[m[m_off]]!=tid)
      return false;

    assert(n[(n_off+2)%3]==m[(m_off+1)%3] && n[(n_off+1)%3]==m[(m_off+2)%3]);

    if(_mesh->is_halo_node(n[n_off]) && _mesh->is_halo_node(m[m_off]))
      return false;

    rhombus.eid0 = eid0;
    rhombus.eid1 = eid1;
    rhombus.n_off = n_off;
    rhombus.m_off = m_off;

    return true;
  }

  void process_list(std::deque< Edge<index_t> > *tdynamic_edge, int tid){
    while(!tdynamic_edge->empty()){
      Edge<index_t> oldEdge = *tdynamic_edge->begin();
      tdynamic_edge->pop_front();

      Rhombus rhombus(oldEdge);
      if(!is_eligible(rhombus, tid))
        continue;

      // Process the edge and update the rhombus with
      // the new middle edge if swapping occurs.
      swap_kernel(rhombus);

      // Mark the edge as processed
      typename std::list<index_t>::iterator it;
      it = std::find(marked_edges[oldEdge.edge.first].begin(),
          marked_edges[oldEdge.edge.first].end(), oldEdge.edge.second);
      marked_edges[oldEdge.edge.first].erase(it);

      // If the edge was swapped, propagate the operation
      Edge<index_t> newEdge = rhombus.middleEdge;
      if(newEdge != oldEdge){
        Edge<index_t> lateralEdges[] = {
            Edge<index_t>(oldEdge.edge.first, newEdge.edge.first),
            Edge<index_t>(oldEdge.edge.first, newEdge.edge.second),
            Edge<index_t>(oldEdge.edge.second, newEdge.edge.first),
            Edge<index_t>(oldEdge.edge.second, newEdge.edge.second)};

        for(size_t i=0; i<4; ++i){
          it = std::find(marked_edges[lateralEdges[i].edge.first].begin(),
              marked_edges[lateralEdges[i].edge.first].end(), lateralEdges[i].edge.second);

          // If the edge is already marked, then continue
          if(it != marked_edges[lateralEdges[i].edge.first].end())
            continue;

          marked_edges[lateralEdges[i].edge.first].push_back(lateralEdges[i].edge.second);
          tdynamic_edge->push_back(lateralEdges[i]);
        }
      }
    }
  }

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  static const size_t ndims=2;
  static const size_t nloc=3;
  int nthreads;
  std::vector< std::list<index_t> > marked_edges;
  std::vector<real_t> quality;
  int *tpartition, *dynamic_vertex;
  real_t min_Q;
};

#endif
