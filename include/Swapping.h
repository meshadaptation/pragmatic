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

#ifndef SWAPPING_H
#define SWAPPING_H

#include <algorithm>
#include <set>
#include <vector>

#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, typename index_t> class Swapping{
 public:
  /// Default constructor.
  Swapping(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;
    
    size_t NElements = _mesh->get_number_elements();
    ndims = _mesh->get_number_dimensions();
    nloc = (ndims==2)?3:4;

    // Set the orientation of elements.
    property = NULL;
    for(size_t i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      if(ndims==2)
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
  }
  
  /// Default destructor.
  ~Swapping(){
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
        if(ndims==2){
          const real_t *x0 = _mesh->get_coords(n[0]);
          const real_t *x1 = _mesh->get_coords(n[1]);
          const real_t *x2 = _mesh->get_coords(n[2]);
          
          quality[i] = property->lipnikov(x0, x1, x2,
                                          _mesh->get_metric(n[0]),
                                          _mesh->get_metric(n[1]),
                                          _mesh->get_metric(n[2]));
        }else{
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
    }

    // Initialise list of dynamic edges.
    typename std::set<Edge<real_t, index_t> > dynamic_edges;
    for(typename std::set< Edge<real_t, index_t> >::iterator it=_mesh->Edges.begin();it!=_mesh->Edges.end();++it){
      if(it->adjacent_elements.size()!=2)
        continue;
      
      for(std::set<int>::const_iterator jt=it->adjacent_elements.begin();jt!=it->adjacent_elements.end();++jt){
        if(quality[*jt]<Q_min){
          dynamic_edges.insert(*it);
          break;
        }
      }
    }

    // -
    while(!dynamic_edges.empty()){
      Edge<real_t, index_t> target_edge = *_mesh->Edges.find(*dynamic_edges.begin());
      dynamic_edges.erase(dynamic_edges.begin());

      if(target_edge.adjacent_elements.size()!=2)
        continue;
      
      if(_mesh->is_halo_node(target_edge.edge.first) || _mesh->is_halo_node(target_edge.edge.second))
        continue;

      int eid0 = *target_edge.adjacent_elements.begin();
      int eid1 = *target_edge.adjacent_elements.rbegin();
      
      if(std::min(quality[eid0], quality[eid1])>Q_min)
        continue;
      
      const int *n = _mesh->get_element(eid0);
      const int *m = _mesh->get_element(eid1);
      
      int n_off=-1;
      for(size_t i=0;i<3;i++){
        if((n[i]!=target_edge.edge.first) && (n[i]!=target_edge.edge.second)){
          n_off = i;
          break;
        }
      }
      assert(n_off>=0);

      int m_off=-1;
      for(size_t i=0;i<3;i++){
        if((m[i]!=target_edge.edge.first) && (m[i]!=target_edge.edge.second)){
          m_off = i;
          break;
        }
      }
      assert(m_off>=0);

      assert(n[(n_off+2)%3]==m[(m_off+1)%3]);
      assert(n[(n_off+1)%3]==m[(m_off+2)%3]);

      int n_swap[] = {n[n_off], m[m_off],       n[(n_off+2)%3]}; // new eid0
      int m_swap[] = {n[n_off], n[(n_off+1)%3], m[(m_off)%3]};   // new eid1

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
        // Update Edges.
        //
        
        // Delete old element from edge-element adjancy.
        for(size_t i=0;i<nloc;i++){
          typename std::set< Edge<real_t, index_t> >::iterator edge =  _mesh->Edges.find(Edge<real_t, index_t>(n[i], n[(i+1)%3]));
          assert(edge!=_mesh->Edges.end());
          
          Edge<real_t, index_t> modify_edge = *edge;
          _mesh->Edges.erase(edge);
          
          assert(modify_edge.adjacent_elements.count(eid0));
          modify_edge.adjacent_elements.erase(eid0);
          _mesh->Edges.insert(modify_edge);
        }
        for(size_t i=0;i<nloc;i++){
          typename std::set< Edge<real_t, index_t> >::iterator edge =  _mesh->Edges.find(Edge<real_t, index_t>(m[i], m[(i+1)%3]));
          assert(edge!=_mesh->Edges.end());

          Edge<real_t, index_t> modify_edge = *edge;
          _mesh->Edges.erase(edge);
          
          assert(modify_edge.adjacent_elements.count(eid1));
          modify_edge.adjacent_elements.erase(eid1);
          _mesh->Edges.insert(modify_edge);
        }

        // Add new edge
        _mesh->Edges.insert(Edge<real_t, index_t>(n[n_off], m[m_off]));
        
        // Add new element to the edge-element adjancy.
        for(size_t i=0;i<nloc;i++){
          // eid0
          typename std::set< Edge<real_t, index_t> >::iterator edge0 =  _mesh->Edges.find(Edge<real_t, index_t>(n_swap[i], n_swap[(i+1)%3]));
          Edge<real_t, index_t> modify_edge0 = *edge0;
          _mesh->Edges.erase(edge0);
          
          modify_edge0.adjacent_elements.insert(eid0);
          _mesh->Edges.insert(modify_edge0);
          
          // eid1
          typename std::set< Edge<real_t, index_t> >::iterator edge1 =  _mesh->Edges.find(Edge<real_t, index_t>(m_swap[i], m_swap[(i+1)%3]));
          Edge<real_t, index_t> modify_edge1 = *edge1;
          _mesh->Edges.erase(edge1);
          
          modify_edge1.adjacent_elements.insert(eid1);
          _mesh->Edges.insert(modify_edge1);
        }
        
        // Delete edge being swapped out
        _mesh->Edges.erase(Edge<real_t, index_t>(n[(n_off+1)%3], n[(n_off+2)%3]));

        //
        // Update node-node list.
        //
        
        // Make local partial copy of nnlist
        std::map<int, std::set<int> > nnlist;
        for(size_t i=0;i<nloc;i++){
          for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[n[i]].begin();it!=_mesh->NNList[n[i]].end();++it){
            if(*it>=0)
              nnlist[n[i]].insert(*it);
          }
        }
        for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[m[m_off]].begin();it!=_mesh->NNList[m[m_off]].end();++it){
          if(*it>=0)
            nnlist[m[m_off]].insert(*it);
        }
        nnlist[n[(n_off+1)%3]].erase(n[(n_off+2)%3]);
        nnlist[n[(n_off+2)%3]].erase(n[(n_off+1)%3]);
        
        nnlist[n[n_off]].insert(m[m_off]);
        nnlist[m[m_off]].insert(n[n_off]);

        // Put back in new adjancy info
        for(std::map<int, std::set<int> >::const_iterator it=nnlist.begin();it!=nnlist.end();++it){
          _mesh->NNList[it->first].clear();
          for(typename std::set<index_t>::const_iterator jt=it->second.begin();jt!=it->second.end();++jt)
            _mesh->NNList[it->first].push_back(*jt);
        }

        //
        // Update node-element list.
        //
        
        // Erase old node-element adjancies.
        for(size_t i=0;i<nloc;i++){
          _mesh->NEList[n[i]].erase(eid0);
          _mesh->NEList[m[i]].erase(eid1);
        }
        for(size_t i=0;i<nloc;i++){
          _mesh->NEList[n_swap[i]].insert(eid0);
          _mesh->NEList[m_swap[i]].insert(eid1);
        }

        // Update element-node list for this element.
        for(size_t i=0;i<nloc;i++){
          _mesh->_ENList[eid0*nloc+i] = n_swap[i];
          _mesh->_ENList[eid1*nloc+i] = m_swap[i];

          // Also update the edges that have to be rechecked.
          dynamic_edges.insert(Edge<real_t, index_t>(n_swap[i], n_swap[(i+1)%3]));
          dynamic_edges.insert(Edge<real_t, index_t>(m_swap[i], m_swap[(i+1)%3]));
        }
      }
    }

    return;
  }

 private:
  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  size_t ndims, nloc;
};

#endif
