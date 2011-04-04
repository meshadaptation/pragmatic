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

  /*! Perform one level of refinement See Figure 25; X Li et al, Comp
   * Methods Appl Mech Engrg 194 (2005) 4915-4950. The actual
   * templates used for 3D refinement follows Rupak Biswas, Roger
   * C. Strawn, "A new procedure for dynamic adaption of
   * three-dimensional unstructured grids", Applied Numerical
   * Mathematics, Volume 13, Issue 6, February 1994, Pages 437-452.
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

    if(refined_edges.size()==0){
      return 0;
    }

    // Given these refined edges, refine elements.
    int NElements = _mesh->get_number_elements();
    if(ndims==2){
      for(int i=0;i<NElements;i++){
        // Check if this element has been erased - if so continue to next element.
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
          int rotated_ele[] = {-1, -1, -1};
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
          int rotated_ele[] = {-1, -1, -1};
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
      for(;;){
        typename std::set< Edge<real_t, index_t> > new_edges;
        
        for(int i=0;i<NElements;i++){
          // Check if this element has been erased - if so continue to next element.
          const int *n=_mesh->get_element(i);
          if(n[0]<0)
            continue;
          
          std::vector<typename std::map< Edge<real_t, index_t>, index_t>::const_iterator> split;
          typename std::set< Edge<real_t, index_t> > split_set;
          for(size_t j=0;j<4;j++){
            for(size_t k=j+1;k<4;k++){
              typename std::map< Edge<real_t, index_t>, index_t>::const_iterator it =
                refined_edges.find(Edge<real_t, index_t>(n[j], n[k]));
              if(it!=refined_edges.end()){
                split.push_back(it);
                split_set.insert(it->first);
              }
            }
          }
          int refine_cnt=split.size();

          if(refine_cnt==0){
            // No refinement - continue to next element.
          }else if(refine_cnt==1){
            // 1:2 refinement is ok.
          }else if(refine_cnt==2){
            // Here there are two possibilities. Either the two split
            // edges share a vertex (case 1) or there are opposit (case
            // 2). Case 1 results in a 1:3 subdivision and a possible mismatch on the
            // surface. So we have to spit an additional edge. Case 2 results in a
            // 1:4 with no issues so is left as is.
            
            int n0=split[0]->first.connected(split[1]->first);
            if(n0>=0){
              // Case 1.
              int n1 = (n0==split[0]->first.edge.first)?split[0]->first.edge.second:split[0]->first.edge.first;
              int n2 = (n0==split[1]->first.edge.first)?split[1]->first.edge.second:split[1]->first.edge.first;
              new_edges.insert(Edge<real_t, index_t>(n1, n2));
            }else{
              // Case 2
            }
          }else if(refine_cnt==3){
            // There are 3 cases that need to be considered. They can be
            // distinguished by the total number of nodes that are common between any
            // pair of edges.
            std::set<index_t> shared;
            for(int j=0;j<refine_cnt;j++){
              for(int k=j+1;k<refine_cnt;k++){
                index_t nid = split[j]->first.connected(split[k]->first);
                if(nid>=0)
                  shared.insert(nid);
              }
            }
            size_t nshared = shared.size();
            
            if(nshared==3){
              // 1:4
            }else{
              // Refine unsplit edges.
              for(int j=0;j<4;j++)
                for(int k=j+1;k<4;k++){
                  Edge<real_t, index_t> test_edge(n[j], n[k]);
                  if(split_set.count(test_edge)==0)
                    new_edges.insert(test_edge);
                }
            }
          }else if(refine_cnt==4){
            // Refine unsplit edges.
            for(int j=0;j<4;j++)
              for(int k=j+1;k<4;k++){
                Edge<real_t, index_t> test_edge(n[j], n[k]);
                if(split_set.count(test_edge)==0)
                  new_edges.insert(test_edge);
              }
          }else if(refine_cnt==5){
            // Refine unsplit edges.
            for(int j=0;j<4;j++)
              for(int k=j+1;k<4;k++){
                Edge<real_t, index_t> test_edge(n[j], n[k]);
                if(split_set.count(test_edge)==0)
                  new_edges.insert(test_edge);
              }
          }else if(refine_cnt==6){
            // All edges spit. Nothing to do.
          }
        }
        
        for(typename std::set< Edge<real_t, index_t> >::const_iterator it=new_edges.begin();it!=new_edges.end();++it){
          assert(refined_edges.find(*it)==refined_edges.end());
          
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
        
        if(new_edges.size()==0)
          break;
      }
      
      for(int i=0;i<NElements;i++){
        // Check if this element has been erased - if so continue to next element.
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
        
        std::vector<typename std::map< Edge<real_t, index_t>, index_t>::const_iterator> split;
        for(size_t j=0;j<4;j++)
          for(size_t k=j+1;k<4;k++){
            typename std::map< Edge<real_t, index_t>, index_t>::const_iterator it =
              refined_edges.find(Edge<real_t, index_t>(n[j], n[k]));
            if(it!=refined_edges.end())
              split.push_back(it);
          }
        int refine_cnt=split.size();
        
        // Apply refinement templates.
        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
          // Find the opposit edge
          int oe[2];
          for(int j=0, pos=0;j<4;j++)
            if(!split[0]->first.contains(n[j]))
              oe[pos++] = n[j];
          
          // Form and add two new edges.
          const int ele0[] = {split[0]->first.edge.first, split[0]->second, oe[0], oe[1]};
          const int ele1[] = {split[0]->first.edge.second, split[0]->second, oe[0], oe[1]};
          
          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
        }else if(refine_cnt==2){
          const int ele0[] = {split[0]->first.edge.first, split[0]->second, split[1]->first.edge.first, split[1]->second};
          const int ele1[] = {split[0]->first.edge.first, split[0]->second, split[1]->first.edge.second, split[1]->second};
          const int ele2[] = {split[0]->first.edge.second, split[0]->second, split[1]->first.edge.first, split[1]->second};
          const int ele3[] = {split[0]->first.edge.second, split[0]->second, split[1]->first.edge.second, split[1]->second};
          
          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
          _mesh->append_element(ele2);
          _mesh->append_element(ele3);
        }else if(refine_cnt==3){
          index_t m[] = {-1, -1, -1, -1, -1, -1, -1};
          m[0] = split[0]->first.edge.first;
          m[1] = split[0]->second;
          m[2] = split[0]->first.edge.second;
          if(split[1]->first.contains(m[2])){
            m[3] = split[1]->second;
            if(split[1]->first.edge.first!=m[2])
              m[4] = split[1]->first.edge.first;
            else
              m[4] = split[1]->first.edge.second;
            m[5] = split[2]->second;
          }else{
            m[3] = split[2]->second;
            if(split[2]->first.edge.first!=m[2])
              m[4] = split[2]->first.edge.first;
            else
              m[4] = split[2]->first.edge.second;
            m[5] = split[1]->second;
          }
          for(int j=0;j<4;j++)
            if((n[j]!=m[0])&&(n[j]!=m[2])&&(n[j]!=m[4])){
              m[6] = n[j];
              break;
            }
          
          const int ele0[] = {m[0], m[1], m[5], m[6]};
          const int ele1[] = {m[1], m[2], m[3], m[6]};
          const int ele2[] = {m[5], m[3], m[4], m[6]};
          const int ele3[] = {m[1], m[3], m[5], m[6]};
          
          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
          _mesh->append_element(ele2);
          _mesh->append_element(ele3);
        }else if(refine_cnt==6){
          const int ele0[] = {n[0], split[0]->second, split[1]->second, split[2]->second};
          const int ele1[] = {n[1], split[3]->second, split[0]->second, split[4]->second};
          const int ele2[] = {n[2], split[1]->second, split[3]->second, split[5]->second};
          const int ele3[] = {split[0]->second, split[3]->second, split[1]->second, split[4]->second};
          const int ele4[] = {split[0]->second, split[4]->second, split[1]->second, split[2]->second};
          const int ele5[] = {split[1]->second, split[3]->second, split[5]->second, split[4]->second};
          const int ele6[] = {split[1]->second, split[4]->second, split[5]->second, split[2]->second};
          const int ele7[] = {split[2]->second, split[4]->second, split[5]->second, n[3]};

          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
          _mesh->append_element(ele2);
          _mesh->append_element(ele3);
          _mesh->append_element(ele4);
          _mesh->append_element(ele5);
          _mesh->append_element(ele6);
          _mesh->append_element(ele7);
        }
        // Remove parent element.
        _mesh->erase_element(i);
      }
    }
    
    // Fix orientations of new elements.
    for(int i=NElements;i<_mesh->get_number_elements();i++){
      int *n=&(_mesh->_ENList[i*nloc]);
      
      real_t av;
      if(ndims==2)
        av = property->area(_mesh->get_coords(n[0]),
                            _mesh->get_coords(n[1]),
                            _mesh->get_coords(n[2]));
      else
        av = property->volume(_mesh->get_coords(n[0]),
                              _mesh->get_coords(n[1]),
                              _mesh->get_coords(n[2]),
                              _mesh->get_coords(n[3]));
      if(av<0){
        // Flip element
        int ntmp = n[0];
        n[0] = n[1];
        n[1] = ntmp;
      }
    }

    real_t total_volume=0;
    for(int i=0;i<_mesh->get_number_elements();i++){
      int *n=&(_mesh->_ENList[i*nloc]);
      if(n[0]<0)
        continue;

      real_t av;
      if(ndims==2)
        av = property->area(_mesh->get_coords(n[0]),
                            _mesh->get_coords(n[1]),
                            _mesh->get_coords(n[2]));
      else
        av = property->volume(_mesh->get_coords(n[0]),
                              _mesh->get_coords(n[1]),
                              _mesh->get_coords(n[2]),
                              _mesh->get_coords(n[3]));
      total_volume+=av;
    }

    // Finally, refine surface
    _surface->refine(refined_edges);

    return refined_edges.size();
  }

 private:
  Mesh<real_t, index_t> *_mesh;
  ElementProperty<real_t> *property;

  Surface<real_t, index_t> *_surface;

  size_t ndims, nloc;
};

#endif
