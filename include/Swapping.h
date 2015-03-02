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

#ifndef SWAPPING_H
#define SWAPPING_H

#include <algorithm>
#include <atomic>
#include <limits>
#include <list>
#include <set>
#include <vector>

#include "Colour.h"
#include "DeferredOperations.h"
#include "Edge.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, int dim> class Swapping{
 public:
  /// Default constructor.
  Swapping(Mesh<real_t> &mesh){
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
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]));
      else
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]),
                                               _mesh->get_coords(n[3]));
      break;
    }

    nnodes_reserve = 0;
    nthreads = pragmatic_nthreads();

    if(dim==3){
      newElements.resize(nthreads);
      newBoundaries.resize(nthreads);
    }
  }

  /// Default destructor.
  ~Swapping(){
    if(property!=NULL)
      delete property;
  }

  void swap(real_t quality_tolerance){
    if(dim==2)
      swap2d(quality_tolerance);
    else
      swap3d(quality_tolerance);
  }


 private:

  void swap2d(real_t quality_tolerance){
    size_t NNodes = _mesh->get_number_nodes();
    size_t NElements = _mesh->get_number_elements();

    min_Q = quality_tolerance;

    if(nnodes_reserve<NNodes){
      nnodes_reserve = NNodes;

      quality.resize(NElements);
      marked_edges.resize(NNodes);
    }

    std::vector< std::atomic<unsigned int> > vLocks(NNodes);

#pragma omp parallel
    {
#pragma omp for nowait
      for(unsigned int i=0; i<NNodes; ++i){
        vLocks[i].store(0, std::memory_order_relaxed);
        marked_edges[i].clear();
      }

      // Cache the element qualities. Really need to make this
      // persistent within Mesh. Also, initialise marked_edges.
#pragma omp for schedule(guided)
      for(size_t i=0; i<NElements; ++i){
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

      // Vector "retry" is used to store both aborted vertices and propagated vertices.
      std::vector<index_t> retry, new_retry;
#pragma omp for schedule(guided) nowait
      for(index_t node=0; node<NNodes; ++node){
        bool abort = false;
        std::vector<index_t> locks_held;

        int oldval = vLocks[node].fetch_or(1, std::memory_order_acq_rel);
        if((oldval & 1) != 0){
          retry.push_back(node);
          continue;
        }
        locks_held.push_back(node);

        for(auto& it : _mesh->NNList[node]){
          int oldval = vLocks[it].fetch_or(1, std::memory_order_acq_rel);
          if((oldval & 1) != 0){
            abort = true;
            break;
          }
          locks_held.push_back(it);
        }

        if(!abort){
          std::set< Edge<index_t> > active_edges;
          for(auto& ele : _mesh->NEList[node]){
            if(quality[ele] < min_Q){
              const index_t* n = _mesh->get_element(ele);
              for(int i=0; i<nloc; ++i){
                if(n[i] != node)
                  active_edges.insert(Edge<index_t>(node, n[i]));
              }
            }
          }

          for(auto& edge : active_edges){
            index_t i = edge.edge.first;
            index_t j = edge.edge.second;
            bool swapped = swap_kernel2d(const_cast<Edge<index_t>&>(edge));

            if(swapped){
              index_t k = edge.edge.first;
              index_t l = edge.edge.second;

              Edge<index_t> lateralEdges[] = {
                  Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

              // Propagate the operation
              for(size_t ee=0; ee<4; ++ee){
                marked_edges[lateralEdges[ee].edge.first].insert(lateralEdges[ee].edge.second);
                retry.push_back(lateralEdges[ee].edge.first);
              }
            }
          }
        }
        else
          retry.push_back(node);

        for(auto& it : locks_held){
          vLocks[it].store(0, std::memory_order_release);
        }
      }

      while(retry.size()>0){
        new_retry.clear();

        for(auto& node : retry){
          bool abort = false;
          std::vector<index_t> locks_held;

          int oldval = vLocks[node].fetch_or(1, std::memory_order_acq_rel);
          if((oldval & 1) != 0){
            new_retry.push_back(node);
            continue;
          }
          locks_held.push_back(node);

          for(auto& it : _mesh->NNList[node]){
            int oldval = vLocks[it].fetch_or(1, std::memory_order_acq_rel);
            if((oldval & 1) != 0){
              abort = true;
              break;
            }
            locks_held.push_back(it);
          }

          if(!abort){
            if(!marked_edges[node].empty()){
              std::set<index_t> marked_edges_copy = marked_edges[node];
              marked_edges[node].clear();

              for(auto& target : marked_edges_copy){
                Edge<index_t> edge(node, target);
                index_t i = edge.edge.first;
                index_t j = edge.edge.second;
                bool swapped = swap_kernel2d(edge);

                if(swapped){
                  index_t k = edge.edge.first;
                  index_t l = edge.edge.second;

                  Edge<index_t> lateralEdges[] = {
                      Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                  // Propagate the operation
                  for(size_t ee=0; ee<4; ++ee){
                    marked_edges[lateralEdges[ee].edge.first].insert(lateralEdges[ee].edge.second);
                    new_retry.push_back(lateralEdges[ee].edge.first);
                  }
                }
              }
            }
          }
          else
            new_retry.push_back(node);

          for(auto& it : locks_held){
            vLocks[it].store(0, std::memory_order_release);
          }
        }

        retry.swap(new_retry);
      }
    }
  }

  void swap3d(real_t Q_min){
    // Cache the element quality's.
    size_t NElements = _mesh->get_number_elements();
    std::vector<real_t> quality(NElements, -1);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0){
          quality[i] = 0.0;
          continue;
        }

        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        const real_t *x3 = _mesh->get_coords(n[3]);

        quality[i] = property->lipnikov(x0, x1, x2, x3,
                                        _mesh->get_metric(n[0]),
                                        _mesh->get_metric(n[1]),
                                        _mesh->get_metric(n[2]),
                                        _mesh->get_metric(n[3]));
      }
    }

    std::map<int, std::deque<int> > partialEEList;
    for(size_t i=0;i<NElements;i++){
      // Check this is not deleted.
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      // Only start storing information for poor elements.
      if(quality[i]<Q_min){
        bool is_halo = false;
        for(int j=0; j<nloc; ++j){
          if(_mesh->is_halo_node(n[j])){
            is_halo = true;
            break;
          }
        }
        if(is_halo)
          continue;

        partialEEList[i].resize(4);
        std::fill(partialEEList[i].begin(), partialEEList[i].end(), -1);

        for(size_t j=0;j<4;j++){
          std::set<index_t> intersection12;
          set_intersection(_mesh->NEList[n[(j+1)%4]].begin(), _mesh->NEList[n[(j+1)%4]].end(),
                           _mesh->NEList[n[(j+2)%4]].begin(), _mesh->NEList[n[(j+2)%4]].end(),
                           inserter(intersection12, intersection12.begin()));

          std::set<index_t> EE;
          set_intersection(intersection12.begin(),intersection12.end(),
                           _mesh->NEList[n[(j+3)%4]].begin(), _mesh->NEList[n[(j+3)%4]].end(),
                           inserter(EE, EE.begin()));

          for(typename std::set<index_t>::const_iterator it=EE.begin();it!=EE.end();++it){
            if(*it != (index_t)i){
              partialEEList[i][j] = *it;
              break;
            }
          }
        }
      }
    }

    if(partialEEList.empty())
      return;

    // Colour the graph and choose the maximal independent set.
    std::map<int , std::set<int> > graph;
    for(std::map<int, std::deque<int> >::const_iterator it=partialEEList.begin();it!=partialEEList.end();++it){
      for(std::deque<int>::const_iterator jt=it->second.begin();jt!=it->second.end();++jt){
        graph[*jt].insert(it->first);
        graph[it->first].insert(*jt);
      }
    }

    std::deque<int> renumber(graph.size());
    std::map<int, int> irenumber;
    // std::vector<size_t> nedges(graph.size());
    size_t loc=0;
    for(std::map<int , std::set<int> >::const_iterator it=graph.begin();it!=graph.end();++it){
      // nedges[loc] = it->second.size();
      renumber[loc] = it->first;
      irenumber[it->first] = loc;
      loc++;
    }

    std::vector< std::vector<index_t> > NNList(graph.size());
    for(std::map<int , std::set<int> >::const_iterator it=graph.begin();it!=graph.end();++it){
      for(std::set<int>::const_iterator jt=it->second.begin();jt!=it->second.end();++jt){
        NNList[irenumber[it->first]].push_back(irenumber[*jt]);
      }
    }
    std::vector<char> colour(graph.size());
    Colour::greedy(graph.size(), NNList, colour);

    // Assume colour 0 will be the maximal independent set.

    int max_colour=colour[0];
    for(size_t i=1;i<graph.size();i++)
      max_colour = std::max(max_colour, (int)colour[i]);

    // Process face-to-edge swap.
    for(int c=0;c<max_colour;c++){
      for(size_t i=0;i<graph.size();i++){
        int eid0 = renumber[i];

        if(colour[i]==c && (partialEEList.count(eid0)>0)){

          // Check this is not deleted.
          const int *n=_mesh->get_element(eid0);
          if(n[0]<0)
            continue;

          assert(partialEEList[eid0].size()==4);

          // Check adjacency is not toxic.
          bool toxic = false;
          for(int j=0;j<4;j++){
            int eid1 = partialEEList[eid0][j];
            if(eid1==-1)
              continue;

            const int *m=_mesh->get_element(eid1);
            if(m[0]<0){
              toxic = true;
              break;
            }
          }
          if(toxic)
            continue;

          // Create set of nodes for quick lookup.
          std::set<int> ele0_set;
          for(int j=0;j<4;j++)
            ele0_set.insert(n[j]);

          for(int j=0;j<4;j++){
            int eid1 = partialEEList[eid0][j];
            if(eid1==-1)
              continue;

            std::vector<int> hull(5, -1);
            if(j==0){
              hull[0] = n[1];
              hull[1] = n[3];
              hull[2] = n[2];
              hull[3] = n[0];
            }else if(j==1){
              hull[0] = n[2];
              hull[1] = n[3];
              hull[2] = n[0];
              hull[3] = n[1];
            }else if(j==2){
              hull[0] = n[0];
              hull[1] = n[3];
              hull[2] = n[1];
              hull[3] = n[2];
            }else if(j==3){
              hull[0] = n[0];
              hull[1] = n[1];
              hull[2] = n[2];
              hull[3] = n[3];
            }

            const int *m=_mesh->get_element(eid1);
            assert(m[0]>=0);

            for(int k=0;k<4;k++)
              if(ele0_set.count(m[k])==0){
                hull[4] = m[k];
                break;
              }
            assert(hull[4]!=-1);

            // New element: 0143
            real_t q0 = property->lipnikov(_mesh->get_coords(hull[0]),
                                           _mesh->get_coords(hull[1]),
                                           _mesh->get_coords(hull[4]),
                                           _mesh->get_coords(hull[3]),
                                           _mesh->get_metric(hull[0]),
                                           _mesh->get_metric(hull[1]),
                                           _mesh->get_metric(hull[4]),
                                           _mesh->get_metric(hull[3]));

            // New element: 1243
            real_t q1 = property->lipnikov(_mesh->get_coords(hull[1]),
                                           _mesh->get_coords(hull[2]),
                                           _mesh->get_coords(hull[4]),
                                           _mesh->get_coords(hull[3]),
                                           _mesh->get_metric(hull[1]),
                                           _mesh->get_metric(hull[2]),
                                           _mesh->get_metric(hull[4]),
                                           _mesh->get_metric(hull[3]));

            // New element:2043
            real_t q2 = property->lipnikov(_mesh->get_coords(hull[2]),
                                           _mesh->get_coords(hull[0]),
                                           _mesh->get_coords(hull[4]),
                                           _mesh->get_coords(hull[3]),
                                           _mesh->get_metric(hull[2]),
                                           _mesh->get_metric(hull[0]),
                                           _mesh->get_metric(hull[4]),
                                           _mesh->get_metric(hull[3]));

            if(std::min(quality[eid0],quality[eid1]) < std::min(q0, std::min(q1, q2))){
              // Cache boundary values
              int eid0_b0, eid0_b1, eid0_b2, eid1_b0, eid1_b1, eid1_b2;
              for(int face=0; face<nloc; ++face){
                if(n[face] == hull[0])
                  eid0_b0 = _mesh->boundary[eid0*nloc+face];
                else if(n[face] == hull[1])
                  eid0_b1 = _mesh->boundary[eid0*nloc+face];
                else if(n[face] == hull[2])
                  eid0_b2 = _mesh->boundary[eid0*nloc+face];

                if(m[face] == hull[0])
                  eid1_b0 = _mesh->boundary[eid1*nloc+face];
                else if(m[face] == hull[1])
                  eid1_b1 = _mesh->boundary[eid1*nloc+face];
                else if(m[face] == hull[2])
                  eid1_b2 = _mesh->boundary[eid1*nloc+face];
              }

              _mesh->erase_element(eid0);
              _mesh->erase_element(eid1);

              int e0[] = {hull[0], hull[1], hull[4], hull[3]};
              int b0[] = {0, 0, eid0_b2, eid1_b2};
              int eid0 = _mesh->append_element(e0, b0);
              quality.push_back(q0);

              int e1[] = {hull[1], hull[2], hull[4], hull[3]};
              int b1[] = {0, 0, eid0_b0, eid1_b0};
              int eid1 = _mesh->append_element(e1, b1);
              quality.push_back(q1);

              int e2[] = {hull[2], hull[0], hull[4], hull[3]};
              int b2[] = {0, 0, eid0_b1, eid1_b1};
              int eid2 = _mesh->append_element(e2, b2);
              quality.push_back(q2);

              _mesh->NNList[hull[3]].push_back(hull[4]);
              _mesh->NNList[hull[4]].push_back(hull[3]);
              _mesh->NEList[hull[0]].insert(eid0);
              _mesh->NEList[hull[0]].insert(eid2);
              _mesh->NEList[hull[1]].insert(eid0);
              _mesh->NEList[hull[1]].insert(eid1);
              _mesh->NEList[hull[2]].insert(eid1);
              _mesh->NEList[hull[2]].insert(eid2);
              _mesh->NEList[hull[3]].insert(eid0);
              _mesh->NEList[hull[3]].insert(eid1);
              _mesh->NEList[hull[3]].insert(eid2);
              _mesh->NEList[hull[4]].insert(eid0);
              _mesh->NEList[hull[4]].insert(eid1);
              _mesh->NEList[hull[4]].insert(eid2);

              break;
            }
          }
        }
      }
    }

    // Process edge-face swaps.
    for(int c=0;c<max_colour;c++){
      for(size_t i=0;i<graph.size();i++){
        int eid0 = renumber[i];

        if(colour[i]==c && (partialEEList.count(eid0)>0)){

          // Check this is not deleted.
          const int *n=_mesh->get_element(eid0);
          if(n[0]<0)
            continue;

          bool toxic=false, swapped=false;
          for(int k=0;(k<3)&&(!toxic)&&(!swapped);k++){
            for(int l=k+1;l<4;l++){
              Edge<index_t> edge = Edge<index_t>(n[k], n[l]);

              std::set<index_t> neigh_elements;
              set_intersection(_mesh->NEList[n[k]].begin(), _mesh->NEList[n[k]].end(),
                               _mesh->NEList[n[l]].begin(), _mesh->NEList[n[l]].end(),
                               inserter(neigh_elements, neigh_elements.begin()));

              double min_quality = quality[eid0];
              std::vector<index_t> constrained_edges_unsorted;
              std::map<int, std::map<index_t, int> > b;
              std::vector<int> element_order, e_to_eid;

              for(typename std::set<index_t>::const_iterator it=neigh_elements.begin();it!=neigh_elements.end();++it){
                min_quality = std::min(min_quality, quality[*it]);

                const int *m=_mesh->get_element(*it);
                if(m[0]<0){
                  toxic=true;
                  break;
                }

                e_to_eid.push_back(*it);

                for(int j=0;j<4;j++){
                  if((m[j]!=n[k])&&(m[j]!=n[l])){
                    constrained_edges_unsorted.push_back(m[j]);
                  }else if(m[j] == n[k]){
                    b[*it][n[k]] = _mesh->boundary[nloc*(*it)+j];
                  }else{ // if(m[j] == n[l])
                    b[*it][n[l]] = _mesh->boundary[nloc*(*it)+j];
                  }
                }
              }

              if(toxic)
                break;

              size_t nelements = neigh_elements.size();
              assert(nelements*2==constrained_edges_unsorted.size());
              assert(b.size() == nelements);

              // Sort edges.
              std::vector<index_t> constrained_edges;
              std::vector<bool> sorted(nelements, false);
              constrained_edges.push_back(constrained_edges_unsorted[0]);
              constrained_edges.push_back(constrained_edges_unsorted[1]);
              element_order.push_back(e_to_eid[0]);
              for(size_t j=1;j<nelements;j++){
                for(size_t e=1;e<nelements;e++){
                  if(sorted[e])
                    continue;
                  if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2]){
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    element_order.push_back(e_to_eid[e]);
                    sorted[e]=true;
                    break;
                  }else if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2+1]){
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    element_order.push_back(e_to_eid[e]);
                    sorted[e]=true;
                    break;
                  }
                }
              }

              if(*constrained_edges.begin() != *constrained_edges.rbegin()){
                toxic = true;
                break;
              }
              assert(element_order.size() == nelements);

              std::vector< std::vector<index_t> > new_elements;
              std::vector< std::vector<int> > new_boundaries;
              if(nelements==3){
                // This is the 3-element to 2-element swap.
                new_elements.resize(1);
                new_boundaries.resize(1);

                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(b[element_order[1]][n[k]]);
                new_boundaries[0].push_back(b[element_order[2]][n[k]]);
                new_boundaries[0].push_back(b[element_order[0]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(b[element_order[2]][n[l]]);
                new_boundaries[0].push_back(b[element_order[1]][n[l]]);
                new_boundaries[0].push_back(b[element_order[0]][n[l]]);
                new_boundaries[0].push_back(0);
              }else if(nelements==4){
                // This is the 4-element to 4-element swap.
                new_elements.resize(2);
                new_boundaries.resize(2);

                // Option 1.
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[3]][n[k]]);
                new_boundaries[0].push_back(b[element_order[0]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(b[element_order[2]][n[k]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[1]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(b[element_order[3]][n[l]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[0]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[2]][n[l]]);
                new_boundaries[0].push_back(b[element_order[1]][n[l]]);
                new_boundaries[0].push_back(0);

                // Option 2
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[l]);
                new_boundaries[1].push_back(b[element_order[1]][n[k]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(b[element_order[0]][n[k]]);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[l]);
                new_boundaries[1].push_back(b[element_order[2]][n[k]]);
                new_boundaries[1].push_back(b[element_order[3]][n[k]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(n[k]);
                new_boundaries[1].push_back(b[element_order[1]][n[l]]);
                new_boundaries[1].push_back(b[element_order[0]][n[l]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[k]);
                new_boundaries[1].push_back(b[element_order[2]][n[l]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(b[element_order[3]][n[l]]);
                new_boundaries[1].push_back(0);
              }else if(nelements==5){
                // This is the 5-element to 6-element swap.
                new_elements.resize(5);
                new_boundaries.resize(5);

                // Option 1
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(b[element_order[1]][n[k]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[0]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[2]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(b[element_order[4]][n[k]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[3]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[1]][n[l]]);
                new_boundaries[0].push_back(b[element_order[0]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[2]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[4]][n[l]]);
                new_boundaries[0].push_back(b[element_order[3]][n[l]]);
                new_boundaries[0].push_back(0);

                // Option 2
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(n[l]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(b[element_order[4]][n[k]]);
                new_boundaries[1].push_back(b[element_order[0]][n[k]]);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(n[l]);
                new_boundaries[1].push_back(b[element_order[3]][n[k]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[l]);
                new_boundaries[1].push_back(b[element_order[2]][n[k]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(b[element_order[1]][n[k]]);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(n[k]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(b[element_order[0]][n[l]]);
                new_boundaries[1].push_back(b[element_order[4]][n[l]]);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[k]);
                new_boundaries[1].push_back(b[element_order[3]][n[l]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);

                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[k]);
                new_boundaries[1].push_back(b[element_order[2]][n[l]]);
                new_boundaries[1].push_back(b[element_order[1]][n[l]]);
                new_boundaries[1].push_back(0);
                new_boundaries[1].push_back(0);

                // Option 3
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(constrained_edges[2]);
                new_elements[2].push_back(n[l]);
                new_boundaries[2].push_back(b[element_order[0]][n[k]]);
                new_boundaries[2].push_back(b[element_order[1]][n[k]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);

                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(n[l]);
                new_boundaries[2].push_back(b[element_order[4]][n[k]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);

                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[6]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(n[l]);
                new_boundaries[2].push_back(b[element_order[3]][n[k]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(b[element_order[2]][n[k]]);
                new_boundaries[2].push_back(0);

                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[2]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(n[k]);
                new_boundaries[2].push_back(b[element_order[0]][n[l]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(b[element_order[1]][n[l]]);
                new_boundaries[2].push_back(0);

                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(n[k]);
                new_boundaries[2].push_back(b[element_order[4]][n[l]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);

                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(constrained_edges[6]);
                new_elements[2].push_back(n[k]);
                new_boundaries[2].push_back(b[element_order[3]][n[l]]);
                new_boundaries[2].push_back(b[element_order[2]][n[l]]);
                new_boundaries[2].push_back(0);
                new_boundaries[2].push_back(0);

                // Option 4
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(constrained_edges[4]);
                new_elements[3].push_back(n[l]);
                new_boundaries[3].push_back(b[element_order[1]][n[k]]);
                new_boundaries[3].push_back(b[element_order[2]][n[k]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);

                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(n[l]);
                new_boundaries[3].push_back(b[element_order[0]][n[k]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);

                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[8]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(n[l]);
                new_boundaries[3].push_back(b[element_order[4]][n[k]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(b[element_order[3]][n[k]]);
                new_boundaries[3].push_back(0);

                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[4]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(n[k]);
                new_boundaries[3].push_back(b[element_order[1]][n[l]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(b[element_order[2]][n[l]]);
                new_boundaries[3].push_back(0);

                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(n[k]);
                new_boundaries[3].push_back(b[element_order[0]][n[l]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);

                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(constrained_edges[8]);
                new_elements[3].push_back(n[k]);
                new_boundaries[3].push_back(b[element_order[4]][n[l]]);
                new_boundaries[3].push_back(b[element_order[3]][n[l]]);
                new_boundaries[3].push_back(0);
                new_boundaries[3].push_back(0);

                // Option 5
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[0]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(n[l]);
                new_boundaries[4].push_back(b[element_order[0]][n[k]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(b[element_order[4]][n[k]]);
                new_boundaries[4].push_back(0);

                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(n[l]);
                new_boundaries[4].push_back(b[element_order[1]][n[k]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);

                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(constrained_edges[6]);
                new_elements[4].push_back(n[l]);
                new_boundaries[4].push_back(b[element_order[2]][n[k]]);
                new_boundaries[4].push_back(b[element_order[3]][n[k]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);

                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(constrained_edges[0]);
                new_elements[4].push_back(n[k]);
                new_boundaries[4].push_back(b[element_order[0]][n[l]]);
                new_boundaries[4].push_back(b[element_order[4]][n[l]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);

                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(n[k]);
                new_boundaries[4].push_back(b[element_order[1]][n[l]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(0);

                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[6]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(n[k]);
                new_boundaries[4].push_back(b[element_order[2]][n[l]]);
                new_boundaries[4].push_back(0);
                new_boundaries[4].push_back(b[element_order[3]][n[l]]);
                new_boundaries[4].push_back(0);
              }else if(nelements==6){
                // This is the 6-element to 8-element swap.
                new_elements.resize(1);
                new_boundaries.resize(1);

                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[5]][n[k]]);
                new_boundaries[0].push_back(b[element_order[0]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(b[element_order[3]][n[k]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[2]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[1]][n[k]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[l]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[4]][n[k]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(b[element_order[5]][n[l]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[0]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[3]][n[l]]);
                new_boundaries[0].push_back(b[element_order[2]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(b[element_order[1]][n[l]]);
                new_boundaries[0].push_back(0);

                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[k]);
                new_boundaries[0].push_back(b[element_order[4]][n[l]]);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
                new_boundaries[0].push_back(0);
              }else{
                continue;
              }

              nelements = new_elements[0].size()/4;

              // Check new minimum quality.
              std::vector<double> new_min_quality(new_elements.size());
              std::vector< std::vector<double> > newq(new_elements.size());
              int best_option;
              for(int invert=0;invert<2;invert++){
                best_option=0;
                for(size_t option=0;option<new_elements.size();option++){
                  newq[option].resize(nelements);
                  for(size_t j=0;j<nelements;j++){
                    newq[option][j] = property->lipnikov(_mesh->get_coords(new_elements[option][j*4+0]),
                                                         _mesh->get_coords(new_elements[option][j*4+1]),
                                                         _mesh->get_coords(new_elements[option][j*4+2]),
                                                         _mesh->get_coords(new_elements[option][j*4+3]),
                                                         _mesh->get_metric(new_elements[option][j*4+0]),
                                                         _mesh->get_metric(new_elements[option][j*4+1]),
                                                         _mesh->get_metric(new_elements[option][j*4+2]),
                                                         _mesh->get_metric(new_elements[option][j*4+3]));
                  }

                  new_min_quality[option] = newq[option][0];
                  for(size_t j=0;j<nelements;j++)
                    new_min_quality[option] = std::min(newq[option][j], new_min_quality[option]);
                }


                for(size_t option=1;option<new_elements.size();option++){
                  if(new_min_quality[option]>new_min_quality[best_option]){
                    best_option = option;
                  }
                }

                if(new_min_quality[best_option] < 0.0){
                  // Invert elements.
                  std::vector< std::vector<index_t> >::iterator it, bit;
                  for(it=new_elements.begin(), bit=new_boundaries.begin(); it!=new_elements.end(); ++it, ++bit){
                    for(size_t j=0;j<nelements;j++){
                      index_t stash_id = (*it)[j*4];
                      (*it)[j*4] = (*it)[j*4+1];
                      (*it)[j*4+1] = stash_id;
                      int stash_b = (*bit)[j*4];
                      (*bit)[j*4] = (*bit)[j*4+1];
                      (*bit)[j*4+1] = stash_b;
                    }
                  }

                  continue;
                }
                break;
              }

              if(new_min_quality[best_option] <= min_quality)
                continue;

              // Update NNList
              std::vector<index_t>::iterator vit = std::find(_mesh->NNList[n[k]].begin(), _mesh->NNList[n[k]].end(), n[l]);
              assert(vit != _mesh->NNList[n[k]].end());
              _mesh->NNList[n[k]].erase(vit);
              vit = std::find(_mesh->NNList[n[l]].begin(), _mesh->NNList[n[l]].end(), n[k]);
              assert(vit != _mesh->NNList[n[l]].end());
              _mesh->NNList[n[l]].erase(vit);

              // Remove old elements.
              for(typename std::set<index_t>::const_iterator it=neigh_elements.begin();it!=neigh_elements.end();++it)
                _mesh->erase_element(*it);

              // Add new elements.
              for(size_t j=0;j<nelements;j++){
                int eid = _mesh->append_element(&(new_elements[best_option][j*4]), &(new_boundaries[best_option][j*4]));
                quality.push_back(newq[best_option][j]);

                for(int p=0; p<nloc; ++p){
                  index_t v1 = new_elements[best_option][j*4+p];
                  _mesh->NEList[v1].insert(eid);

                  for(int q=p+1; q<nloc; ++q){
                    index_t v2 = new_elements[best_option][j*4+q];
                    std::vector<index_t>::iterator vit = std::find(_mesh->NNList[v1].begin(), _mesh->NNList[v1].end(), v2);
                    if(vit == _mesh->NNList[v1].end()){
                      _mesh->NNList[v1].push_back(v2);
                      _mesh->NNList[v2].push_back(v1);
                    }
                  }
                }
              }

              swapped = true;
              break;
            }
          }
        }
      }
    }

    return;
  }

  bool swap_kernel2d(Edge<index_t>& edge){
    index_t i = edge.edge.first;
    index_t j = edge.edge.second;

    if(_mesh->is_halo_node(i)&& _mesh->is_halo_node(j))
      return false;

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
        return false;
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
      return false;

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
      assert(it != _mesh->NNList[i].end());
      _mesh->NNList[i].erase(it);
      it = std::find(_mesh->NNList[j].begin(), _mesh->NNList[j].end(), i);
      assert(it != _mesh->NNList[j].end());
      _mesh->NNList[j].erase(it);
      _mesh->NNList[k].push_back(l);
      _mesh->NNList[l].push_back(k);

      // Update node-element list.
      _mesh->NEList[n_swap[2]].erase(eid1);
      _mesh->NEList[m_swap[1]].erase(eid0);
      _mesh->NEList[n_swap[0]].insert(eid1);
      _mesh->NEList[n_swap[1]].insert(eid0);

      // Update element-node and boundary list for this element.
      const int *bn = &_mesh->boundary[eid0*nloc];
      const int *bm = &_mesh->boundary[eid1*nloc];
      const int bn_swap[] = {bm[(m_off+2)%3], bn[(n_off+1)%3], 0}; // boundary for n_swap
      const int bm_swap[] = {bm[(m_off+1)%3], 0, bn[(n_off+2)%3]}; // boundary for m_swap

      for(size_t cnt=0;cnt<nloc;cnt++){
        _mesh->_ENList[eid0*nloc+cnt] = n_swap[cnt];
        _mesh->_ENList[eid1*nloc+cnt] = m_swap[cnt];
        _mesh->boundary[eid0*nloc+cnt] = bn_swap[cnt];
        _mesh->boundary[eid1*nloc+cnt] = bm_swap[cnt];
      }

      edge.edge.first = std::min(k, l);
      edge.edge.second = std::max(k, l);

      return true;
    }

    return false;
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

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  size_t nnodes_reserve;

  static const size_t ndims=dim;
  static const size_t nloc=dim+1;
  static const size_t msize=(dim==2?3:6);

  std::vector< std::vector<index_t> > newElements;
  std::vector< std::vector<int> > newBoundaries;

  std::vector< std::set<index_t> > marked_edges;
  std::vector<real_t> quality;
  real_t min_Q;

  int nthreads;
};

#endif
