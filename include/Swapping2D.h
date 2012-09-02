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

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, typename index_t> class Swapping2D{
 public:
  /// Default constructor.
  Swapping2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
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
  }
  
  /// Default destructor.
  ~Swapping2D(){
    delete property;
  }
  
  void swap(real_t Q_min){
    size_t NElements = _mesh->get_number_elements();
    std::vector<real_t> quality(NElements);
    
    typename std::vector< std::vector<char> > marked_edges;
    typename std::vector< std::vector<index_t> > NEList;
    int n_marked_edges = 0;
    
#pragma omp parallel
    {
      size_t NNodes = _mesh->get_number_nodes();

      // Cache the element quality's.
#pragma omp for schedule(dynamic)
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
#pragma omp single nowait
      {
        originalVertexDegree.clear();
        originalVertexDegree.resize(NNodes, 0);
      }
      
#pragma omp single nowait
      {
        marked_edges.resize(NNodes);
      }
      
#pragma omp single 
      {
        NEList.resize(NNodes);
      }
      
#pragma omp for schedule(dynamic) nowait
      for(size_t i=0;i<NNodes;i++){
        if(_mesh->NNList[i].empty())
          continue;
        
        size_t size = _mesh->NNList[i].size();
        
        NEList[i].resize(2 * size, (index_t) -1);
        std::copy(_mesh->NEList[i].begin(), _mesh->NEList[i].end(), NEList[i].begin());
        
        _mesh->NEList[i].clear();
      }

#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;i++){
        if(_mesh->NNList[i].empty())
          continue;
        
        size_t size = _mesh->NNList[i].size();
        
        originalVertexDegree[i] = size;
        _mesh->NNList[i].resize(3 * size, (index_t) -1);
        marked_edges[i].resize(size, (char) 0);
        
        for(size_t it=0; it<size; ++it){
          if(i < (size_t) _mesh->NNList[i][it]){
            marked_edges[i][it] = 1;
          }
        }
      }
      
#pragma omp single
      {
        n_marked_edges = 0;
      }
#pragma omp for schedule(dynamic) reduction(+:n_marked_edges)
      for(size_t i=0;i<NNodes;i++){
        n_marked_edges += std::count(marked_edges[i].begin(), marked_edges[i].end(), (char) 1);
      }

      // -
      while(n_marked_edges > 0){        
#pragma omp for schedule(dynamic)
        for(size_t i=0;i<NNodes;i++){
          if(_mesh->is_halo_node(i)){
            std::fill(marked_edges[i].begin(), marked_edges[i].end(), (char) 0);
            continue;
          }
          
          for(int it=0; it<(int)originalVertexDegree[i]; ++it){
            if(marked_edges[i][it] != 1)
              continue;
            
            index_t opposite = _mesh->NNList[i][it];
            
            if(_mesh->is_halo_node(opposite)){
              marked_edges[i][it] = 0;
              continue;
            }
            
            // Find the two elements sharing this edge
            index_t neigh_elements[2], neigh_elements_cnt=0;
            for(size_t k=0; k<NEList[i].size()/2; ++k){
              if(NEList[i][k] == -1)
                continue;
              
              for(size_t l=0; l<NEList[opposite].size()/2; ++l){
                if(NEList[i][k] == NEList[opposite][l]){
                  neigh_elements[neigh_elements_cnt++] = NEList[i][k];
                  break;
                }
              }

              if(neigh_elements_cnt==2)
                break;
            }
            
            if(neigh_elements_cnt!=2){
              marked_edges[i][it] = 0;
              continue;
            }
            
            int eid0 = neigh_elements[0];
            int eid1 = neigh_elements[1];
            
            /*
              if(std::min(quality[eid0], quality[eid1])>Q_min)
              continue;
            */
            
            const index_t *n = _mesh->get_element(eid0);
            const index_t *m = _mesh->get_element(eid1);
            
            int n_off=-1;
            for(size_t k=0;k<3;k++){
              if((n[k]!=(index_t)i) && (n[k]!=(index_t)opposite)){
                n_off = k;
                break;
              }
            }
            
            int m_off=-1;
            for(size_t k=0;k<3;k++){
              if((m[k]!=(index_t)i) && (m[k]!=(index_t)opposite)){
                m_off = k;
                break;
              }
            }
            
            //
            // Decision algorithm
            //
            
            /*
             * If the following condition is true, it means that this thread had
             * a stale view of NEList and ENList, which in turn means that another
             * thread performed swapping on one of the lateral edges, so anyway
             * this edge would not be a candidate for swapping during this round.
             */
            if(n_off<0 || m_off<0 || n[(n_off+2)%3]!=m[(m_off+1)%3] || n[(n_off+1)%3]!=m[(m_off+2)%3])
              continue;
            
            index_t lateral_n = n[n_off];
            index_t lateral_m = m[m_off];
            
            // i's index in lateral_n's and lateral_m's list
            int idx_in_n = -1, idx_in_m = -1;
            // lateral_n's and lateral_m's index in i's list
            int idx_of_n = -1, idx_of_m = -1;
            // Min and max ID between opposite and lateral_n, max's index in min's list
            int min_opp_n = -1, max_opp_n = -1, idx_opp_n = -1;
            // Min and max ID between opposite and lateral_m, max's index in min's list
            int min_opp_m = -1, max_opp_m = -1, idx_opp_m = -1;
            
            /*
             * Are lateral edges marked for processing?
             * (This also checks whether the four participating
             * vertices are original neighbours of one another)
             */
            if(i > (size_t)lateral_n){
              idx_in_n = originalNeighborIndex(lateral_n, i);
              if(idx_in_n >= (int) originalVertexDegree[lateral_n])
                continue;
              if(marked_edges[lateral_n][idx_in_n] == 1)
                continue;
              
              if(opposite < lateral_n){
                min_opp_n = opposite;
                max_opp_n = lateral_n;
              }else{
                min_opp_n = lateral_n;
                max_opp_n = opposite;
              }
              
              idx_opp_n = originalNeighborIndex(min_opp_n, max_opp_n);
              if(idx_opp_n >= (int) originalVertexDegree[min_opp_n])
                continue;
              if(marked_edges[min_opp_n][idx_opp_n] == 1)
                continue;
            }
            
            if(i > (size_t)lateral_m){
              idx_in_m = originalNeighborIndex(lateral_m, i);
              if(idx_in_m >= (int) originalVertexDegree[lateral_m])
                continue;
              if(marked_edges[lateral_m][idx_in_m] == 1)
                continue;
              
              if(opposite < lateral_m){
                min_opp_m = opposite;
                max_opp_m = lateral_m;
              }else{
                min_opp_m = lateral_m;
                max_opp_m = opposite;
              }
              
              idx_opp_m = originalNeighborIndex(min_opp_m, max_opp_m);
              if(idx_opp_m >= (int) originalVertexDegree[min_opp_m])
                continue;
              if(marked_edges[min_opp_m][idx_opp_m] == 1)
                continue;
            }
            
            /*
             * Are lateral neighbours original ones?
             * (only perform this check if it wasn't
             * performed during the previous decision block)
             */
            if(idx_in_n == -1){
              idx_of_n = originalNeighborIndex(i, lateral_n);
              if(idx_of_n >= (int) originalVertexDegree[i])
                continue;
            }
            
            if(idx_in_m == -1){
              idx_of_m = originalNeighborIndex(i, lateral_m);
              if(idx_of_m >= (int) originalVertexDegree[i])
                continue;
            }
            
            if(idx_opp_n == -1){
              if(opposite < lateral_n){
                min_opp_n = opposite;
                max_opp_n = lateral_n;
              }else{
                min_opp_n = lateral_n;
                max_opp_n = opposite;
              }
              
              idx_opp_n = originalNeighborIndex(min_opp_n, max_opp_n);
              if(idx_opp_n >= (int) originalVertexDegree[min_opp_n])
                continue;
            }
            
            if(idx_opp_m == -1){
              if(opposite < lateral_m){
                min_opp_m = opposite;
                max_opp_m = lateral_m;
              }else{
                min_opp_m = lateral_m;
                max_opp_m = opposite;
              }
              
              idx_opp_m = originalNeighborIndex(min_opp_m, max_opp_m);
              if(idx_opp_m >= (int) originalVertexDegree[min_opp_m])
                continue;
            }
            
            // If execution reaches this point, it means that the edge can be processed
            
            int n_swap[] = {n[n_off], m[m_off],       n[(n_off+2)%3]}; // new eid0
            int m_swap[] = {n[n_off], n[(n_off+1)%3], m[m_off]};       // new eid1
            
            real_t worst_q = std::min(quality[eid0], quality[eid1]);
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
              
              //
              // Update NNList[i], NNList[opposite], NNList[lateral_n] and NNList[lateral_m]
              //
              
              // Remove opposite from i's list
              _mesh->NNList[i][it] = -1;
              
              // Remove i from opposite's list
              _mesh->NNList[opposite][originalNeighborIndex(opposite, i)] = -1;
              
              // Add lateral_m in lateral_n's list
              if(idx_in_n == -1)
                idx_in_n = originalNeighborIndex(lateral_n, i);
              int pos = originalVertexDegree[lateral_n] + idx_in_n;
              if(_mesh->NNList[lateral_n][pos] != -1)
                pos += originalVertexDegree[lateral_n];
              assert(_mesh->NNList[lateral_n][pos] == -1);
              _mesh->NNList[lateral_n][pos] = lateral_m;
              
              // Add lateral_n in lateral_m's list
              if(idx_in_m == -1)
                idx_in_m = originalNeighborIndex(lateral_m, i);
              pos = originalVertexDegree[lateral_m] + idx_in_m;
              if(_mesh->NNList[lateral_m][pos] != -1)
                pos += originalVertexDegree[lateral_m];
              assert(_mesh->NNList[lateral_m][pos] == -1);
              _mesh->NNList[lateral_m][pos] = lateral_n;
              
              //
              // Update node-element list.
              //
              
              // Erase old node-element adjacency.
              index_t vertex;
              size_t halfSize;
              typename std::vector<index_t>::iterator it;
              
              // lateral_n - add eid1
              vertex = n_swap[0];
              halfSize = NEList[vertex].size()/2;
              it = std::find(NEList[vertex].begin(), NEList[vertex].begin() + halfSize, eid0);
              assert(it != NEList[vertex].begin() + halfSize);
              it += halfSize;
              assert(*it == -1);
              *it = eid1;
              
              // lateral_m - add eid0
              vertex = n_swap[1];
              halfSize = NEList[vertex].size()/2;
              it = std::find(NEList[vertex].begin(), NEList[vertex].begin() + halfSize, eid1);
              assert(it != NEList[vertex].begin() + halfSize);
              it += halfSize;
              assert(*it == -1);
              *it = eid0;
              
              // i (or opposite) - remove eid1
              vertex = n_swap[2];
              halfSize = NEList[vertex].size()/2;
              it = std::find(NEList[vertex].begin(), NEList[vertex].begin() + halfSize, eid1);
              assert(it != NEList[vertex].begin() + halfSize);
              assert(*it == eid1);
              *it = -1;
              
              // opposite (or i) - remove eid0
              vertex = m_swap[1];
              halfSize = NEList[vertex].size()/2;
              it = std::find(NEList[vertex].begin(), NEList[vertex].begin() + halfSize, eid0);
              assert(it != NEList[vertex].begin() + halfSize);
              assert(*it == eid0);
              *it = -1;
              
              // Update element-node list for this element.
              for(size_t k=0;k<nloc;k++){
                _mesh->_ENList[eid0*nloc+k] = n_swap[k];
                _mesh->_ENList[eid1*nloc+k] = m_swap[k];
              }
              
              // Also update the edges that have to be rechecked.
              if(i < (size_t)lateral_n)
                marked_edges[i][idx_of_n] = 1;
              else
                marked_edges[lateral_n][idx_in_n] = 1;
              
              if(i < (size_t)lateral_m)
                marked_edges[i][idx_of_m] = 1;
              else
                marked_edges[lateral_m][idx_in_m] = 1;
              
              assert(idx_opp_n!=-1);
              assert(idx_opp_m!=-1);
              marked_edges[min_opp_n][idx_opp_n] = 1;
              marked_edges[min_opp_m][idx_opp_m] = 1;
            }
            
            // Mark the swapped edge as processed
            marked_edges[i][it] = 0;
          }
        }
        
        n_marked_edges = 0;
#pragma omp for schedule(dynamic) reduction(+:n_marked_edges)
        for(size_t i=0;i<NNodes;i++){
          n_marked_edges += std::count(marked_edges[i].begin(), marked_edges[i].end(), (char) 1);
        }

        /*
         * This is used to determine whether swapping is finished.
         * If this is the case, NNList[i] needs not be resized x3.
         * Same for NEList.
         */
        int NNextend = (n_marked_edges > 0 ? 3 : 1);
        int NEextend = (n_marked_edges > 0 ? 2 : 1);
        
        // Compact NNList, NEList
#pragma omp for schedule(dynamic)
        for(size_t i=0;i<NNodes;i++){
          if(_mesh->NNList[i].size() == 0)
            continue;
          
          size_t forward = 0, backward = _mesh->NNList[i].size() - 1;
          
          while(forward < backward){
            while(_mesh->NNList[i][forward] != -1) ++forward;
            while(_mesh->NNList[i][backward] == -1) --backward;
            
            if(forward < backward){
              _mesh->NNList[i][forward] = _mesh->NNList[i][backward];
              _mesh->NNList[i][backward] = -1;
              if(backward < originalVertexDegree[i])
                marked_edges[i][forward] = marked_edges[i][backward];
            }
            else
              break;
            
            ++forward;
            --backward;
          }
          if(_mesh->NNList[i][forward] != -1)
            ++forward;
          
          originalVertexDegree[i] = forward;
          marked_edges[i].resize(forward, (char) 0);
          _mesh->NNList[i].resize(NNextend*forward, (index_t) -1);
          
          forward = 0, backward = NEList[i].size() - 1;
          
          while(forward < backward){
            while(NEList[i][forward] != -1){
              ++forward;
              if(forward>backward)
                break;
            }
            while(NEList[i][backward] == -1){
              --backward;
              if(forward>backward)
                break;
            }
            
            if(forward < backward){
              NEList[i][forward] = NEList[i][backward];
              NEList[i][backward] = -1;
            }
            else
              break;
            
            ++forward;
            --backward;
          }
          if(NEList[i][forward] != -1)
            ++forward;
          
          assert((size_t)i<NEList.size());
          NEList[i].resize(NEextend*forward, (index_t) -1);
        }
      }
      
#pragma omp for schedule(dynamic)
      for(size_t i=0;i<NNodes;i++){
        if(_mesh->NNList[i].empty())
          continue;

        std::copy(NEList[i].begin(), NEList[i].end(), std::inserter(_mesh->NEList[i], _mesh->NEList[i].begin()));
      }
    }

    return;
  }

 private:
  inline size_t originalNeighborIndex(index_t source, index_t target) const{
    size_t pos = 0;
    while(pos < originalVertexDegree[source]){
      if(_mesh->NNList[source][pos] == target)
        return pos;
      ++pos;
    }
    return std::numeric_limits<index_t>::max();
  }

  std::vector<size_t> originalVertexDegree;

  Mesh<real_t, index_t> *_mesh;
  Surface2D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  static const size_t ndims=2;
  static const size_t nloc=3;
  int nthreads;
};

#endif
