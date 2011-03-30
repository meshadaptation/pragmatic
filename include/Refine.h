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

#ifndef REFINE_H
#define REFINE_H

#include <algorithm>
#include <set>
#include <vector>

#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs mesh refinement.
 *
 */
template<typename real_t, typename index_t> class Refine{
 public:
  /// Default constructor.
  Refine(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
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
  ~Refine(){
    delete property;
  }

  void refine(real_t L_max){
    for(;;){
      int refine_cnt = refine_level(L_max);
      _mesh->create_adjancy();
      _mesh->calc_edge_lengths();

      if(refine_cnt==0)
        break;
    }
  }

  /*! Perform one level of refinement
   * See Figure 25; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  int refine_level(real_t L_max){
    // Initialise a dynamic vertex list
    std::map< Edge<real_t, index_t>, index_t> refined_edges;
    for(typename std::set< Edge<real_t, index_t> >::const_iterator it=_mesh->Edges.begin();it!=_mesh->Edges.end();++it){
      // Split edge if it's length is greater than L_max in transformed space.
      if(it->length>L_max){
        // Calculate the position of the new point. From equation 16 in
        // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
        real_t x[3], m[9];
        index_t n0 = it->edge.first;
        const real_t *x0 = _mesh->get_coords(n0);
        const real_t *m0 = _mesh->get_metric(n0);

        index_t n1 = it->edge.second;
        const real_t *x1 = _mesh->get_coords(n1);
        const real_t *m1 = _mesh->get_metric(n1);

        real_t weight = 1.0/(1.0 + sqrt(property->length(x0, x1, m0)/
                                        property->length(x0, x1, m1)));
        
        // Calculate position of new vertex
        for(size_t i=0;i<ndims;i++)
          x[i] = x0[i]+weight*(x1[i] - x0[i]);
        
        // Interpolate new metric
        for(size_t i=0;i<ndims*ndims;i++)
          m[i] = m0[i]+weight*(m1[i] - m0[i]);
        
        // Append this new vertex and metric into mesh data structures.
        index_t nid = _mesh->append_vertex(x, m);
        
        // Insert split edge into refined_edges
        refined_edges[*it] = nid;
      }
    }

    int refine_cnt = refined_edges.size();
    if(refine_cnt==0){
      return refine_cnt;
    }

    // Given these refined edges, refine elements. Because we are
    // using templates to refine elements, 2D and 3D are dealt with
    // seperately in software.
    int NElements = _mesh->get_number_elements();
    if(ndims==2){
      for(int i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
        
        // Note the order of the edges - the i'th edge is opposit the i'th node in the element. 
        typename std::map< Edge<real_t, index_t>, index_t>::const_iterator edge[3];
        edge[0] = refined_edges.find(Edge<real_t, index_t>(n[1], n[2]));
        edge[1] = refined_edges.find(Edge<real_t, index_t>(n[2], n[0]));
        edge[2] = refined_edges.find(Edge<real_t, index_t>(n[0], n[1]));

        int refine_cnt=0;
        for(int j=0;j<3;j++)
          if(edge[j]!=refined_edges.end())
            refine_cnt++;
        
        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
          // Single edge split.
          typename std::map< Edge<real_t, index_t>, index_t>::const_iterator split;
          int rotated_ele[3];
          for(int j=0;j<3;j++)
            if(edge[j]!=refined_edges.end()){
              split = edge[j];
              for(int k=0;k<3;k++)
                rotated_ele[k] = n[(j+k)%3];
              break;
            }
          
          const int ele0[] = {rotated_ele[0], rotated_ele[1], split->second};
          const int ele1[] = {rotated_ele[0], split->second, rotated_ele[2]};
          
          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
        }else if(refine_cnt==2){
          // Two edges split - have to mesh resulting quadrilateral.
          typename std::map< Edge<real_t, index_t>, index_t>::const_iterator split[2];
          int rotated_ele[3];
          for(int j=0;j<3;j++)
            if(edge[j]==refined_edges.end()){
              split[0] = edge[(j+1)%3];
              split[1] = edge[(j+2)%3];
              for(int k=0;k<3;k++)
                rotated_ele[k] = n[(j+k)%3];
              break;
            }
          
          // Calculate lengths of diagonals in quadrilateral part.
          real_t d1 = _mesh->calc_edge_length(rotated_ele[2], split[1]->second);
          real_t d2 = _mesh->calc_edge_length(rotated_ele[1], split[0]->second);
          
          const int ele0[] = {rotated_ele[0], split[1]->second, split[0]->second};
          _mesh->append_element(ele0);
          if(d1<d2){
            const int ele1[] = {rotated_ele[1], rotated_ele[2], split[1]->second};
            const int ele2[] = {rotated_ele[2], split[0]->second, split[1]->second};

            _mesh->append_element(ele1);
            _mesh->append_element(ele2);
          }else{
            const int ele1[] = {rotated_ele[1], rotated_ele[2], split[0]->second};
            const int ele2[] = {split[0]->second, split[1]->second, rotated_ele[1]};
            
            _mesh->append_element(ele1);
            _mesh->append_element(ele2);
          }
          
          // Use the shortest edge - insert into refined_edges so that it can be picked up by surface refinement.
          // ... do nothing here for 2D - delete once 3D is implemented.
        }else if(refine_cnt==3){
          const int ele0[] = {n[0], edge[2]->second, edge[1]->second};
          const int ele1[] = {n[1], edge[0]->second, edge[2]->second};
          const int ele2[] = {n[2], edge[1]->second, edge[0]->second};
          const int ele3[] = {edge[0]->second, edge[1]->second, edge[2]->second};

          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
          _mesh->append_element(ele2);
          _mesh->append_element(ele3);
        }
        // Remove parent element.
        _mesh->erase_element(i);
      }
    }else{
      for(int i=0;i<NElements;i++){
      }
    }
    return refine_cnt;
  }

 private:
  Mesh<real_t, index_t> *_mesh;
  ElementProperty<real_t> *property;

  Surface<real_t, index_t> *_surface;

  size_t ndims, nloc;
};

#endif
