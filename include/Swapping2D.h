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
#include <limits>
#include <set>
#include <vector>

#include "Colour.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t> class Swapping2D{
 public:
  /// Default constructor.
  Swapping2D(Mesh<real_t> &mesh, Surface2D<real_t> &surface){
    _mesh = &mesh;
    _surface = &surface;

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

    for(int i=0; i<3; ++i){
      worklist[i] = NULL;
    }

    // We pre-allocate the maximum capacity that may be needed.
    node_colour = NULL;
    GlobalActiveSet_size = 0;
    ind_set_size.resize(max_colour, 0);
    ind_sets.resize(nthreads, std::vector< std::vector<index_t> >(max_colour));
    range_indexer.resize(nthreads, std::vector< std::pair<size_t,size_t> >(max_colour, std::pair<size_t,size_t>(0,0)));
  }

  /// Default destructor.
  ~Swapping2D(){
    if(property!=NULL)
      delete property;

    for(size_t i=0; i<subNNList.size(); ++i)
      if(subNNList[i]!=NULL)
        delete subNNList[i];

    if(node_colour!=NULL)
      delete[] node_colour;

    for(int i=0; i<3; ++i){
      if(worklist[i] != NULL)
        delete[] worklist[i];
    }
  }

  void swap(real_t quality_tolerance){
    size_t NNodes = _mesh->get_number_nodes();
    size_t NElements = _mesh->get_number_elements();

    min_Q = quality_tolerance;

    if(nnodes_reserve<NNodes){
      nnodes_reserve = NNodes;

      subNNList.resize(NNodes, NULL);

      for(int i=0; i<3; ++i){
        if(worklist[i] != NULL)
          delete[] worklist[i];
        worklist[i] = new size_t[NNodes];
      }

      if(node_colour!=NULL)
        delete[] node_colour;

      node_colour = new int[NNodes];

      quality.resize(NElements, 0.0);
      marked_edges.resize(NNodes);
    }
    
#pragma omp parallel
    {
      const int tid = pragmatic_thread_id();

      // Cache the element quality's. Really need to make this
      // persistent within Mesh. Also, initialise marked_edges.
#pragma omp for schedule(guided)
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

          if(quality[i]<min_Q){
            Edge<index_t> edge0(n[1], n[2]);
            Edge<index_t> edge1(n[0], n[2]);
            Edge<index_t> edge2(n[0], n[1]);
            _mesh->deferred_propagate_swapping(edge0.edge.first, edge0.edge.second, tid);
            _mesh->deferred_propagate_swapping(edge1.edge.first, edge1.edge.second, tid);
            _mesh->deferred_propagate_swapping(edge2.edge.first, edge2.edge.second, tid);
          }
        }else{
          quality[i] = 0.0;
        }
      }

#pragma omp for schedule(guided) nowait
      for(int vtid=0; vtid<_mesh->defOp_scaling_factor*nthreads; ++vtid){
        _mesh->commit_swapping_propagation(marked_edges, vtid);
      }

      do{
#pragma omp barrier
#pragma omp single
        {
          for(int i=0; i<max_colour; ++i)
            ind_set_size[i] = 0;
          GlobalActiveSet_size = 0;
        }

        for(int set_no=0; set_no<max_colour; ++set_no){
          ind_sets[tid][set_no].clear();
          range_indexer[tid][set_no].first = std::numeric_limits<size_t>::infinity();
          range_indexer[tid][set_no].second = std::numeric_limits<size_t>::infinity();
        }

        // Construct active sub-mesh
        std::vector<index_t> subSet;
#pragma omp for schedule(guided) nowait
        for(size_t i=0; i<NNodes; ++i){
          if(marked_edges[i].size()>0){
            subSet.push_back(i);
            // Reset the colour of all dynamic vertices
            // It doesn't make sense to reset the colour of any other vertex.
            node_colour[i] = -1;
          }
        }

        if(subSet.size()>0){
          size_t pos;
          pos = pragmatic_omp_atomic_capture(&GlobalActiveSet_size, subSet.size());

          for(typename std::vector<index_t>::const_iterator it=subSet.begin(); it!=subSet.end(); ++it, ++pos){
            if(subNNList[pos]==NULL)
              subNNList[pos] = new std::vector<index_t>;
            else
              subNNList[pos]->clear();

            subNNList[pos]->push_back(*it);
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[*it].begin(); jt!=_mesh->NNList[*it].end(); ++jt)
              if(marked_edges[*jt].size()>0)
                subNNList[pos]->push_back(*jt);
          }
        }

#pragma omp barrier
        if(GlobalActiveSet_size>0){
          Colour::RokosGorman(subNNList, GlobalActiveSet_size,
              node_colour, ind_sets, max_colour, worklist, worklist_size, tid);

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_sets[tid][set_no].size()>0){
              range_indexer[tid][set_no].first = pragmatic_omp_atomic_capture(&ind_set_size[set_no], ind_sets[tid][set_no].size());
              range_indexer[tid][set_no].second = range_indexer[tid][set_no].first + ind_sets[tid][set_no].size();
            }
          }

#pragma omp barrier

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_set_size[set_no] == 0)
              continue;

#pragma omp for schedule(guided)
            for(size_t idx=0; idx<ind_set_size[set_no]; ++idx){
              // Find which vertex corresponds to idx.
              index_t i = -1;
              for(int t=0; t<nthreads; ++t){
                if(idx >= range_indexer[t][set_no].first && idx < range_indexer[t][set_no].second){
                  i = ind_sets[t][set_no][idx - range_indexer[t][set_no].first];
                  break;
                }
              }
              assert(i>=0);

              // If the node has been un-coloured, skip it.
              if(node_colour[i] < 0)
                continue;

              assert(node_colour[i] == set_no);

              // Set of elements in this cavity which were modified since the last commit of deferred operations.
              std::set<index_t> modified_elements;
              std::set<index_t> marked_edges_copy = marked_edges[i];

              for(typename std::set<index_t>::const_iterator vid=marked_edges_copy.begin(); vid!=marked_edges_copy.end(); ++vid){
                index_t j = *vid;

                // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
                bool skip = false;
                for(typename std::set<index_t>::const_iterator it=modified_elements.begin(); it!=modified_elements.end(); ++it){
                  if(_mesh->NEList[j].find(*it) != _mesh->NEList[j].end()){
                    skip = true;
                    break;
                  }
                }
                if(skip)
                  continue;

                // Mark edge as processed, i.e. remove it from the set of marked edges
                marked_edges[i].erase(j);

                Edge<index_t> edge(i, j);
                swap_kernel(edge, modified_elements, NULL, NULL, tid);

                // If edge was swapped
                if(edge.edge.first != i){
                  index_t k = edge.edge.first;
                  index_t l = edge.edge.second;
                  // Uncolour one of the lateral vertices if their colours clash.
                  if((node_colour[k] == node_colour[l]) && (node_colour[k] >= 0))
                    _mesh->deferred_reset_colour(l, tid);

                  Edge<index_t> lateralEdges[] = {
                      Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                  // Propagate the operation
                  for(size_t ee=0; ee<4; ++ee)
                    _mesh->deferred_propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                }
              }
            }

#pragma omp for schedule(guided)
            for(int vtid=0; vtid<_mesh->defOp_scaling_factor*nthreads; ++vtid){
              _mesh->commit_deferred(vtid);
              _mesh->commit_swapping_propagation(marked_edges, vtid);
              _mesh->commit_colour_reset(node_colour, vtid);
            }
          }
#pragma omp barrier
        }
      }while(GlobalActiveSet_size>0);
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
    real_t worst_q = std::min(quality[eid0], quality[eid1]);
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

  Mesh<real_t> *_mesh;
  Surface2D<real_t> *_surface;
  ElementProperty<real_t> *property;

  size_t nnodes_reserve;

  int *node_colour;
  size_t GlobalActiveSet_size;
  std::vector< std::vector<index_t>* > subNNList;
  static const int max_colour = 16;
  std::vector<size_t> ind_set_size;
  std::vector< std::vector< std::vector<index_t> > > ind_sets;
  std::vector< std::vector< std::pair<size_t,size_t> > > range_indexer;
  size_t* worklist[3];
  size_t worklist_size[3];

  static const size_t ndims=2;
  static const size_t nloc=3;
  const static size_t snloc=2;
  const static size_t msize=3;

  std::vector< std::set<index_t> > marked_edges;
  std::vector<real_t> quality;
  real_t min_Q;

  int nthreads;
};

#endif
