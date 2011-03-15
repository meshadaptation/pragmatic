/*
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a full list
 *    of copyright holders.
 *
 *    Gerard Gorman
 *    Applied Modelling and Computation Group
 *    Department of Earth Science and Engineering
 *    Imperial College London
 *
 *    amcgsoftware@imperial.ac.uk
 *    
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation,
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
 *    USA
 */

#ifndef COARSEN_H
#define COARSEN_H

#include <algorithm>
#include <set>
#include <vector>

#include "Mesh.h"

/*! \brief Performs mesh coarsening.
 *
 */

template<typename real_t, typename index_t> class Coarsen{
 public:
  /// Default constructor.
  Coarsen(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;
  }
  
  /// Default destructor.
  ~Coarsen(){
  }

  /*! Perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen(real_t L_min){
    // Initialise a dynamic vertex list
    std::deque<index_t> dynamic_vertex;
    {
      std::set<index_t> dynamic_vertex_set;
      for(typename std::set< Edge<real_t, index_t> >::const_iterator it=_mesh->Edges.begin();it!=_mesh->Edges.end();++it){
        if(it->length<L_min){
          if(!_surface->contains_node(it->edge.first))
            dynamic_vertex_set.insert(it->edge.first);
          if(!_surface->contains_node(it->edge.second))
            dynamic_vertex_set.insert(it->edge.second);
        }
      }
      
      for(typename std::set<index_t>::const_iterator it=dynamic_vertex_set.begin();it!=dynamic_vertex_set.end();++it)
        dynamic_vertex.push_back(*it);
    }
    
    while(dynamic_vertex.size()){
      // Vertex under consideration for removal: rm_vertex
      index_t rm_vertex = dynamic_vertex.front();
      dynamic_vertex.pop_front();
      
      // Identify edge to be removed. We choose the shortest edge.
      const Edge<real_t, index_t> *target_edge=NULL;
      {
        // Find shortest edge connected to vertex.
        std::map<real_t, const Edge<real_t, index_t>* > short_edges;
        for(typename std::deque<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
          typename std::set< Edge<real_t, index_t> >::const_iterator edge = _mesh->Edges.find(Edge<real_t, index_t>(rm_vertex, *nn));
          assert(edge!=_mesh->Edges.end());
          short_edges[edge->length] = &(*edge);
        }
        assert(short_edges.size());

        if(short_edges.begin()->first > L_min)
          continue;
        
        target_edge = short_edges.begin()->second;
      }
      
      // Need additional lines of code to check for degenerate elements and swapping
      // ...
      
      // Identify vertex that will be collapsed onto.
      index_t target_vertex = (rm_vertex==target_edge->edge.first)?target_edge->edge.second:target_edge->edge.first;

      std::set<index_t> deleted_elements = target_edge->adjacent_elements;

      // Renumber nodes in elements adjacent to rm_vertex, deleted
      // elements being collapsed, and make these elements adjacent to
      // target_vertex.
      for(typename std::set<index_t>::iterator ee=_mesh->NEList[rm_vertex].begin();ee!=_mesh->NEList[rm_vertex].end();++ee){
        // Delete if element is to be collapsed.
        if(target_edge->adjacent_elements.count(*ee)){
          for(int i=0;i<_mesh->_nloc;i++){
            _mesh->_ENList[_mesh->_nloc*(*ee)+i]=-1;
          }
          continue;
        }
        
        // Renumber
        for(int i=0;i<_mesh->_nloc;i++){
          if(_mesh->_ENList[_mesh->_nloc*(*ee)+i]==rm_vertex){
            _mesh->_ENList[_mesh->_nloc*(*ee)+i]=target_vertex;
            break;
          }
        }
        
        // Add element to target node-elemement adjancy list.
        _mesh->NEList[target_vertex].insert(*ee);

        // Remove elements from node-elemement adjancy list.
        for(typename std::set<index_t>::const_iterator de=deleted_elements.begin(); de!=deleted_elements.end();++de)
          _mesh->NEList[target_vertex].erase(*de);
      }
      
      // Update Edges.
      _mesh->Edges.erase(*target_edge);
      std::set<index_t> adj_nodes_target = _mesh->get_node_patch(target_vertex);
      for(typename std::deque<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){
        // This edge already deleted.
        if(target_vertex==*nn)
          continue;
        
        // We have to extract a copy of the edge being edited.
        Edge<real_t, index_t> edge_modify = *_mesh->Edges.find(Edge<real_t, index_t>(rm_vertex, *nn));
        _mesh->Edges.erase(edge_modify);
        
        // Update vertex id's for this edge.
        edge_modify.edge.first = std::min(target_vertex, *nn);
        edge_modify.edge.second = std::max(target_vertex, *nn);
        
        // Check if this edge is being collapsed onto an existing edge connected to target vertex.
        if(adj_nodes_target.count(*nn)){
          Edge<real_t, index_t> edge_duplicate = *_mesh->Edges.find(Edge<real_t, index_t>(target_vertex, *nn));
          _mesh->Edges.erase(edge_duplicate);
            
          // Add in additional elements from edge being merged onto.
          edge_modify.adjacent_elements.insert(edge_duplicate.adjacent_elements.begin(),
                                               edge_duplicate.adjacent_elements.end());
          
          // Remove deleted elements from the adjancy
          for(typename std::set<index_t>::const_iterator ee=deleted_elements.begin();ee!=deleted_elements.end();++ee){
            typename std::set<index_t>::const_iterator ele = edge_modify.adjacent_elements.find(*ee);
            if(ele!=edge_modify.adjacent_elements.end())
              edge_modify.adjacent_elements.erase(ele);
          }
        }else{
          // Update the length of the edge in metric space.
          edge_modify.length = _mesh->calc_edge_length(target_vertex, *nn);
        }
        // Add in modified edge back in.
        _mesh->Edges.insert(edge_modify);
      }
      
      // Update surrounding NNList and add elements to ENList.
      for(typename std::deque<index_t>::const_iterator nn=_mesh->NNList[rm_vertex].begin();nn!=_mesh->NNList[rm_vertex].end();++nn){          
        if(*nn == target_vertex){
          std::set<index_t> new_patch = adj_nodes_target;
          for(typename std::deque<index_t>::const_iterator inn=_mesh->NNList[rm_vertex].begin();inn!=_mesh->NNList[rm_vertex].end();++inn)
            new_patch.insert(*inn);
          new_patch.erase(target_vertex);
          new_patch.erase(rm_vertex);
          _mesh->NNList[*nn].clear();
          for(typename std::set<index_t>::const_iterator inn=new_patch.begin();inn!=new_patch.end();++inn)
            _mesh->NNList[*nn].push_back(*inn);
        }else if(adj_nodes_target.count(*nn)){
          // Delete element adjancies from NEList.
          for(typename std::set<index_t>::const_iterator de=deleted_elements.begin();de!=deleted_elements.end();++de){
            typename std::set<index_t>::const_iterator ele = _mesh->NEList[*nn].find(*de);
            if(ele!=_mesh->NEList[*nn].end())
              _mesh->NEList[*nn].erase(ele);
          }
          
          // Deletes edges from NNList
          typename std::deque<index_t>::iterator back_reference = find(_mesh->NNList[*nn].begin(),
                                                                       _mesh->NNList[*nn].end(), rm_vertex);
          assert(back_reference!=_mesh->NNList[*nn].end());
          _mesh->NNList[*nn].erase(back_reference);
        }else{
          typename std::deque<index_t>::iterator back_reference = find(_mesh->NNList[*nn].begin(),
                                                                       _mesh->NNList[*nn].end(), rm_vertex);
          assert(back_reference!=_mesh->NNList[*nn].end());
          *back_reference = target_vertex;
        }
      }
      
      // Delete enteries from adjancy lists.
      _mesh->NNList[rm_vertex].clear();
      _mesh->NEList[rm_vertex].clear();
    }
  }

 private:
  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
};

#endif
