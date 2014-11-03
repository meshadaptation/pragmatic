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
      threadIdx.resize(nthreads);
      splitCnt.resize(nthreads);
    }

    for(int i=0; i<3; ++i){
      worklist[i] = NULL;
    }

    // We pre-allocate the maximum capacity that may be needed.
    node_colour = NULL;

    for(int i=0; i<3; ++i)
      ind_set_size[i].resize(max_colour, 0);
    ind_sets.resize(nthreads, std::vector< std::vector<index_t> >(max_colour));
    range_indexer.resize(nthreads, std::vector< std::pair<size_t,size_t> >(max_colour, std::pair<size_t,size_t>(0,0)));

    def_ops = new DeferredOperations<real_t>(_mesh, nthreads, defOp_scaling_factor);
  }

  /// Default destructor.
  ~Swapping(){
    if(property!=NULL)
      delete property;

    if(node_colour!=NULL)
      delete[] node_colour;

    for(int i=0; i<3; ++i){
      if(worklist[i] != NULL)
        delete[] worklist[i];
    }

    delete def_ops;
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

      GlobalActiveSet.resize(NNodes);

      for(int i=0; i<3; ++i){
        if(worklist[i] != NULL)
          delete[] worklist[i];
        worklist[i] = new index_t[NNodes];
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

      // Thread-private array of forbidden colours
      std::vector<index_t> forbiddenColours(max_colour, std::numeric_limits<index_t>::max());

#pragma omp single nowait
      memset(node_colour, 0, NNodes*sizeof(int));

#pragma omp single nowait
      {
        for(int i=0; i<max_colour; ++i)
          ind_set_size[0][i] = 0;
        GlobalActiveSet_size[0] = 0;
      }

      // Cache the element quality's. Really need to make this
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

          if(quality[i]<min_Q){
            Edge<index_t> edge0(n[1], n[2]);
            Edge<index_t> edge1(n[0], n[2]);
            Edge<index_t> edge2(n[0], n[1]);
            def_ops->propagate_swapping(edge0.edge.first, edge0.edge.second, tid);
            def_ops->propagate_swapping(edge1.edge.first, edge1.edge.second, tid);
            def_ops->propagate_swapping(edge2.edge.first, edge2.edge.second, tid);
          }
        }else{
          quality[i] = 0.0;
        }
      }

#pragma omp for schedule(guided)
      for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
        for(int i=0; i<nthreads; ++i){
          def_ops->commit_swapping_propagation(marked_edges, i, vtid);
        }
      }

      // Variable for accessing GlobalActiveSet_size[rnd] and ind_set_size[rnd]
      int rnd = 2;

      do{
        // Switch to the next round
        rnd = (rnd+1)%3;

        // Prepare worklists for conflict resolution.
        // Reset GlobalActiveSet_size and ind_set_size for next round.
#pragma omp single nowait
        {
          for(int i=0; i<3; ++i)
            worklist_size[i] = 0;

          int next_rnd = (rnd+1)%3;
          for(int i=0; i<max_colour; ++i)
            ind_set_size[next_rnd][i] = 0;
          GlobalActiveSet_size[next_rnd] = 0;
        }

        // Colour the active sub-mesh
        std::vector<index_t> local_coloured;
#pragma omp for schedule(guided) nowait
        for(size_t i=0; i<NNodes; ++i){
          if(marked_edges[i].size()>0){
            /*
             * Create subNNList for vertex i and also execute the first parallel
             * loop of RokosGorman colouring. This way, two time-consuming barriers,
             * the one at the end of the aforementioned loop and the one a few lines
             * below this comment, are merged into one.
             */
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[i].begin(); jt!=_mesh->NNList[i].end(); ++jt){
              if(marked_edges[*jt].size()>0){
                forbiddenColours[node_colour[*jt]] = (index_t) i;
              }

              for(size_t j=0; j<forbiddenColours.size(); ++j){
                if(forbiddenColours[j] != (index_t) i){
                  node_colour[i] = (int) j;
                  break;
                }
              }
            }

            local_coloured.push_back(i);
          }
        }

        if(local_coloured.size()>0){
          size_t pos;
          pos = pragmatic_omp_atomic_capture(&GlobalActiveSet_size[rnd], local_coloured.size());
          memcpy(&GlobalActiveSet[pos], &local_coloured[0], local_coloured.size() * sizeof(index_t));
        }

#pragma omp barrier
        if(GlobalActiveSet_size[rnd]>0){
          // Continue colouring and swapping
          for(int set_no=0; set_no<max_colour; ++set_no){
            ind_sets[tid][set_no].clear();
            range_indexer[tid][set_no].first = std::numeric_limits<size_t>::infinity();
            range_indexer[tid][set_no].second = std::numeric_limits<size_t>::infinity();
          }

          std::vector<index_t> conflicts;

#pragma omp for schedule(guided) nowait
          for(size_t i=0; i<GlobalActiveSet_size[rnd]; ++i){
            bool defective = false;
            index_t n = GlobalActiveSet[i];
            for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
              if(marked_edges[*jt].size()>0){
                if(node_colour[n] == node_colour[*jt]){
                  // No need to mark both vertices as defectively coloured.
                  // Just mark the one with the lesser ID.
                  if(n < *jt){
                    defective = true;
                    break;
                  }
                }
              }
            }

            if(defective){
              conflicts.push_back(n);

              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                if(marked_edges[*jt].size()>0){
                  int c = node_colour[*jt];
                  forbiddenColours[c] = n;
                }
              }

              for(size_t j=0; j<forbiddenColours.size(); j++){
                if(forbiddenColours[j] != n){
                  node_colour[n] = (int) j;
                  break;
                }
              }
            }else{
              ind_sets[tid][node_colour[n]].push_back(n);
            }
          }

          size_t pos;
          pos = pragmatic_omp_atomic_capture(&worklist_size[0], conflicts.size());

          memcpy(&worklist[0][pos], &conflicts[0], conflicts.size() * sizeof(index_t));

          conflicts.clear();
#pragma omp barrier

          int wl = 0;

          while(worklist_size[wl]){
#pragma omp for schedule(guided) nowait
            for(size_t item=0; item<worklist_size[wl]; ++item){
              index_t n = worklist[wl][item];
              bool defective = false;
              for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                if(marked_edges[*jt].size()>0){
                  if(node_colour[n] == node_colour[*jt]){
                    // No need to mark both vertices as defectively coloured.
                    // Just mark the one with the lesser ID.
                    if(n < *jt){
                      defective = true;
                      break;
                    }
                  }
                }
              }

              if(defective){
                conflicts.push_back(n);

                for(typename std::vector<index_t>::const_iterator jt=_mesh->NNList[n].begin(); jt!=_mesh->NNList[n].end(); ++jt){
                  if(marked_edges[*jt].size()>0){
                    int c = node_colour[*jt];
                    forbiddenColours[c] = n;
                  }
                }

                for(size_t j=0; j<forbiddenColours.size(); j++){
                  if(forbiddenColours[j] != n){
                    node_colour[n] = j;
                    break;
                  }
                }
              }else{
                ind_sets[tid][node_colour[n]].push_back(n);
              }
            }

            // Switch worklist
            wl = (wl+1)%3;

            size_t pos = pragmatic_omp_atomic_capture(&worklist_size[wl], conflicts.size());

            memcpy(&worklist[wl][pos], &conflicts[0], conflicts.size() * sizeof(index_t));

            conflicts.clear();

            // Clear the next worklist
#pragma omp single
            {
              worklist_size[(wl+1)%3] = 0;
            }
          }

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_sets[tid][set_no].size()>0){
              range_indexer[tid][set_no].first = pragmatic_omp_atomic_capture(&ind_set_size[rnd][set_no], ind_sets[tid][set_no].size());
              range_indexer[tid][set_no].second = range_indexer[tid][set_no].first + ind_sets[tid][set_no].size();
            }
          }

#pragma omp barrier

          for(int set_no=0; set_no<max_colour; ++set_no){
            if(ind_set_size[rnd][set_no] == 0)
              continue;

            // Sort range indexer
            std::vector<range_element> range;
            for(int t=0; t<nthreads; ++t){
              if(range_indexer[t][set_no].first != range_indexer[t][set_no].second)
                range.push_back(range_element(range_indexer[t][set_no], t));
            }
            std::sort(range.begin(), range.end(), pragmatic_range_element_comparator);

#pragma omp for schedule(guided)
            for(size_t idx=0; idx<ind_set_size[rnd][set_no]; ++idx){
              // Find which vertex corresponds to idx.
              index_t i = -1;
              std::vector<range_element>::iterator ele = std::lower_bound(range.begin(), range.end(),
                  range_element(std::pair<size_t,size_t> (idx,idx), 0), pragmatic_range_element_finder);
              assert(ele != range.end());
              assert(idx >= range_indexer[ele->second][set_no].first && idx < range_indexer[ele->second][set_no].second);
              i = ind_sets[ele->second][set_no][idx - range_indexer[ele->second][set_no].first];
              assert(i>=0);

              // If the node has been un-coloured, skip it.
              if(node_colour[i] != set_no)
                continue;

              // Set of elements in this cavity which were modified since the last commit of deferred operations.
              std::set<index_t> modified_elements;
              std::set<index_t> marked_edges_new;

              for(typename std::set<index_t>::const_iterator vid=marked_edges[i].begin(); vid!=marked_edges[i].end(); ++vid){
                index_t j = *vid;

                // If vertex j is adjacent to one of the modified elements, then its adjacency list is invalid.
                bool skip = false;
                for(typename std::set<index_t>::const_iterator it=modified_elements.begin(); it!=modified_elements.end(); ++it){
                  if(_mesh->NEList[j].find(*it) != _mesh->NEList[j].end()){
                    skip = true;
                    break;
                  }
                }
                if(skip){
                  marked_edges_new.insert(j);
                  continue;
                }

                Edge<index_t> edge(i, j);
                swap_kernel2d(edge, modified_elements, tid);

                // If edge was swapped
                if(edge.edge.first != i){
                  index_t k = edge.edge.first;
                  index_t l = edge.edge.second;
                  // Uncolour one of the lateral vertices if their colours clash.
                  if((node_colour[k] == node_colour[l]) && (node_colour[k] >= 0))
                    def_ops->reset_colour(l, tid);

                  Edge<index_t> lateralEdges[] = {
                      Edge<index_t>(i, k), Edge<index_t>(i, l), Edge<index_t>(j, k), Edge<index_t>(j, l)};

                  // Propagate the operation
                  for(size_t ee=0; ee<4; ++ee)
                    def_ops->propagate_swapping(lateralEdges[ee].edge.first, lateralEdges[ee].edge.second, tid);
                }
              }

              marked_edges[i].swap(marked_edges_new);
            }

            // Commit deferred operations
#pragma omp for schedule(guided)
            for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid){
              for(int i=0; i<nthreads; ++i){
                def_ops->commit_remNN(i, vtid);
                def_ops->commit_addNN(i, vtid);
                def_ops->commit_remNE(i, vtid);
                def_ops->commit_addNE(i, vtid);
                def_ops->commit_swapping_propagation(marked_edges, i, vtid);
                def_ops->commit_colour_reset(node_colour, i, vtid);
              }
            }
          }
        }
      }while(GlobalActiveSet_size[rnd]>0);
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

  void swap_kernel2d(Edge<index_t>& edge, std::set<index_t>& modified_elements, size_t tid){
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
      assert(it != _mesh->NNList[i].end());
      _mesh->NNList[i].erase(it);
      def_ops->remNN(j, i, tid);
      def_ops->addNN(k, l, tid);
      def_ops->addNN(l, k, tid);

      // Update node-element list.
      def_ops->remNE(n_swap[2], eid1, tid);
      def_ops->remNE(m_swap[1], eid0, tid);
      def_ops->addNE(n_swap[0], eid1, tid);
      def_ops->addNE(n_swap[1], eid0, tid);

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
      modified_elements.insert(eid0);
      modified_elements.insert(eid1);
    }

    return;
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

  // Colouring
  int *node_colour;
  static const int max_colour = (dim==2?16:64);
  std::vector<index_t> GlobalActiveSet;
  size_t GlobalActiveSet_size[3];
  std::vector<size_t> ind_set_size[3];
  std::vector< std::vector< std::vector<index_t> > > ind_sets;
  std::vector< std::vector< std::pair<size_t,size_t> > > range_indexer;

  // Used for iterative colouring
  index_t* worklist[3];
  size_t worklist_size[3];

  static const size_t ndims=dim;
  static const size_t nloc=dim+1;
  static const size_t msize=(dim==2?3:6);

  DeferredOperations<real_t>* def_ops;
  static const int defOp_scaling_factor = 32;

  std::vector< std::vector<index_t> > newElements;
  std::vector< std::vector<int> > newBoundaries;
  std::vector<size_t> threadIdx, splitCnt;

  std::vector< std::set<index_t> > marked_edges;
  std::vector<real_t> quality;
  real_t min_Q;

  int nthreads;
};

#endif
