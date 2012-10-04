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

#ifndef SWAPPING3D_H
#define SWAPPING3D_H

#include <algorithm>
#include <limits>
#include <list>
#include <set>
#include <vector>

#include "ElementProperty.h"
#include "Mesh.h"
#include "Colour.h"

template<typename real_t, typename index_t> class Swapping3D{
 public:
  /// Default constructor.
  Swapping3D(Mesh<real_t, index_t> &mesh, Surface3D<real_t, index_t> &surface){
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
                                             _mesh->get_coords(n[2]),
                                             _mesh->get_coords(n[3]));
      break;
    }
  }
  
  /// Default destructor.
  ~Swapping3D(){
    delete property;
  }
  
  void swap(real_t Q_min){
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
    std::vector<index_t> colour(graph.size());
    Colour<index_t>::greedy(NNList, &(colour[0]));
    
    // Assume colour 0 will be the maximal independent set.
    
    int max_colour=colour[0];
    for(size_t i=1;i<graph.size();i++)
      max_colour = std::max(max_colour, colour[i]);
    
    // Process face-to-edge swap.
    for(int c=0;c<max_colour;c++)
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
              _mesh->erase_element(eid0);
              _mesh->erase_element(eid1);
              
              int e0[] = {hull[0], hull[1], hull[4], hull[3]};
              _mesh->append_element(e0);
              quality.push_back(q0);
              
              int e1[] = {hull[1], hull[2], hull[4], hull[3]};
              _mesh->append_element(e1);
              quality.push_back(q1);
              
              int e2[] = {hull[2], hull[0], hull[4], hull[3]};
              _mesh->append_element(e2);
              quality.push_back(q2);
              
              break;
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
              for(typename std::set<index_t>::const_iterator it=neigh_elements.begin();it!=neigh_elements.end();++it){
                min_quality = std::min(min_quality, quality[*it]);
                
                const int *m=_mesh->get_element(*it);
                if(m[0]<0){
                  toxic=true;
                  break;
                }
                
                for(int j=0;j<4;j++){
                  if((m[j]!=n[k])&&(m[j]!=n[l])){
                    constrained_edges_unsorted.push_back(m[j]);
                  }
                }
              }
              
              if(toxic)
                break;
              
              size_t nelements = neigh_elements.size();
              assert(nelements*2==constrained_edges_unsorted.size());
              
              // Sort edges.
              std::vector<index_t> constrained_edges;
              std::vector<bool> sorted(nelements, false);
              constrained_edges.push_back(constrained_edges_unsorted[0]);
              constrained_edges.push_back(constrained_edges_unsorted[1]);
              for(size_t j=1;j<nelements;j++){
                for(size_t e=1;e<nelements;e++){
                  if(sorted[e])
                    continue;
                  if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2]){
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    sorted[e]=true;
                    break;
                  }else if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2+1]){
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    sorted[e]=true;
                    break;
                  }
                }
              }
              
              if(*constrained_edges.begin() != *constrained_edges.rbegin()){
                assert(_surface->contains_node(n[k]));
                assert(_surface->contains_node(n[l]));
                
                toxic = true;
                break;
              }
              
              std::vector< std::vector<index_t> > new_elements;
              if(nelements==3){
                // This is the 3-element to 2-element swap.
                new_elements.resize(1);
                
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[k]);
              }else if(nelements==4){
                // This is the 4-element to 4-element swap.
                new_elements.resize(2);
                
                // Option 1.
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(n[k]);
                
                // Option 2
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[l]);
                
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[l]);
                
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(n[k]);
                
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[k]);
              }else if(nelements==5){
                // This is the 5-element to 6-element swap.
                new_elements.resize(5);
                
                // Option 1
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(n[k]);
                
                // Option 2
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(n[l]);
                
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(n[l]);
                
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[l]);
                
                new_elements[1].push_back(constrained_edges[0]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(n[k]);
                
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[8]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(n[k]);
                
                new_elements[1].push_back(constrained_edges[2]);
                new_elements[1].push_back(constrained_edges[6]);
                new_elements[1].push_back(constrained_edges[4]);
                new_elements[1].push_back(n[k]);
                
                // Option 3
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(constrained_edges[2]);
                new_elements[2].push_back(n[l]);
                
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(n[l]);
                
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[6]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(n[l]);
                
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[2]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(n[k]);
                
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[0]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(n[k]);
                
                new_elements[2].push_back(constrained_edges[4]);
                new_elements[2].push_back(constrained_edges[8]);
                new_elements[2].push_back(constrained_edges[6]);
                new_elements[2].push_back(n[k]);
                
                // Option 4
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(constrained_edges[4]);
                new_elements[3].push_back(n[l]);
                
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(n[l]);
                
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[8]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(n[l]);
                
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[4]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(n[k]);
                
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[2]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(n[k]);
                
                new_elements[3].push_back(constrained_edges[6]);
                new_elements[3].push_back(constrained_edges[0]);
                new_elements[3].push_back(constrained_edges[8]);
                new_elements[3].push_back(n[k]);
                
                // Option 5
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[0]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(n[l]);
                
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(n[l]);
                
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(constrained_edges[6]);
                new_elements[4].push_back(n[l]);
                
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(constrained_edges[0]);
                new_elements[4].push_back(n[k]);
                
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(constrained_edges[2]);
                new_elements[4].push_back(n[k]);
                
                new_elements[4].push_back(constrained_edges[8]);
                new_elements[4].push_back(constrained_edges[6]);
                new_elements[4].push_back(constrained_edges[4]);
                new_elements[4].push_back(n[k]);
              }else if(nelements==6){
                // This is the 6-element to 8-element swap.
                new_elements.resize(1);
                
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[l]);
                
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[0]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[6]);
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[2]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(n[k]);
                
                new_elements[0].push_back(constrained_edges[4]);
                new_elements[0].push_back(constrained_edges[10]);
                new_elements[0].push_back(constrained_edges[8]);
                new_elements[0].push_back(n[k]);
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
                  for(typename std::vector< std::vector<index_t> >::iterator it=new_elements.begin();it!=new_elements.end();++it){
                    for(size_t j=0;j<nelements;j++){
                      index_t stash_id = (*it)[j*4];
                      (*it)[j*4] = (*it)[j*4+1];
                      (*it)[j*4+1] = stash_id;           
                    }
                  }
                  
                  continue;
                }
                break;
              }
              
              if(new_min_quality[best_option] <= min_quality)
                continue;
              
              // Remove old elements.
              for(typename std::set<index_t>::const_iterator it=neigh_elements.begin();it!=neigh_elements.end();++it)
                _mesh->erase_element(*it);
              
              // Add new elements.
              for(size_t j=0;j<nelements;j++){
                _mesh->append_element(&(new_elements[best_option][j*4]));
                quality.push_back(newq[best_option][j]);
              }
              
              swapped = true;
              break;
            }
          }
        }
      }
    }

    // recalculate adjacency
#pragma omp parallel
    _mesh->create_adjancy();
    
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
  Surface3D<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  static const size_t ndims=3;
  static const size_t nloc=4;
  int nthreads;
};

#endif
