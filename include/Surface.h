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

#ifndef SURFACE_H
#define SURFACE_H

#include <vector>
#include <set>
#include <map>

#include "Mesh.h"

/*! \brief Manages surface information and classification.
 *
 * This class is used to: identify the boundary of the domain;
 * uniquely label connected co-linear patches of surface elements
 * (these can be used to prevent adaptivity coarsening these patches
 * and smoothening out features); evaluate a characteristic length
 * scale for these patches (these "constraints" can be added to the
 * metric tensor field before gradation is applied in order to get
 * good quality elements near the geometry).
 */

template<typename real_t, typename index_t>
  class Surface{
 public:
  
  /// Default constructor.
  Surface(Mesh<real_t, index_t> &mesh){
    _mesh = &mesh;

    ndims = mesh.get_number_dimensions();
    nloc = (ndims==2)?3:4;
    snloc = (ndims==2)?2:3;

    set_coplanar_tolerance(0.9999999);

    find_surface();
  }
  
  /// Default destructor.
  ~Surface(){
  }

  /// True if surface contains vertex nid.
  bool contains_node(index_t nid){
    return surface_nodes[nid];
  }

  /// True if node nid is a corner vertex.
  bool is_corner_vertex(index_t nid){
    std::set<int> incident_plane;
    for(typename std::set<index_t>::const_iterator it=SNEList[nid].begin();it!=SNEList[nid].end();++it)
      incident_plane.insert(coplanar_ids[*it]);
    
    return (incident_plane.size()>=ndims);
  }

  bool is_collapsible(index_t nid_free, index_t nid_target){
    // If nid_free is not on the surface then it's unconstrained.
    if(!surface_nodes[nid_free])
      return true;

    std::set<int> incident_plane_free;
    for(typename std::set<index_t>::const_iterator it=SNEList[nid_free].begin();it!=SNEList[nid_free].end();++it)
      incident_plane_free.insert(coplanar_ids[*it]);
    
    // Non-collapsible if nid_free is a corner node.
    if(incident_plane_free.size()>=ndims)
      return false;

    std::set<int> incident_plane_target;
    for(typename std::set<index_t>::const_iterator it=SNEList[nid_target].begin();it!=SNEList[nid_target].end();++it)
      incident_plane_target.insert(coplanar_ids[*it]);
    
    if(ndims==3){
      // Logic if nid_free is on a geometric edge. This only applies for 3D.
      if(incident_plane_free.size()==2){
        return
          (incident_plane_target.count(*incident_plane_free.begin()))
          &&
          (incident_plane_target.count(*incident_plane_free.rbegin()));
      }
    }
    
    // The final case is that the vertex is on a plane and can be
    // collapsed to any other vertex on that plane.
    assert(incident_plane_free.size()==1);
    
    return incident_plane_target.count(*incident_plane_free.begin())>0;
  }

  void collapse(index_t nid_free, index_t nid_target){
    assert(is_collapsible(nid_free, nid_target));
    
    surface_nodes[nid_free] = false; 
    
    // Find deleted facets.
    std::set<index_t> deleted_facets;
    for(typename std::set<index_t>::const_iterator it=SNEList[nid_free].begin();it!=SNEList[nid_free].end();++it){
      if(SNEList[nid_target].count(*it))
        deleted_facets.insert(*it);
    }

    // Renumber nodes in elements adjacent to rm_vertex, deleted
    // elements being collapsed, and make these elements adjacent to
    // target_vertex.
    for(typename std::set<index_t>::const_iterator ee=SNEList[nid_free].begin();ee!=SNEList[nid_free].end();++ee){
      // Delete if element is to be collapsed.
      if(deleted_facets.count(*ee)){
        for(size_t i=0;i<snloc;i++){
          SENList[snloc*(*ee)+i]=-1;
        }
        continue;
      }
      
      // Renumber
      for(size_t i=0;i<snloc;i++){
        if(SENList[snloc*(*ee)+i]==nid_free){
          SENList[snloc*(*ee)+i]=nid_target;
          break;
        }
      }
      
      // Add element to target node-elemement adjancy list.
      SNEList[nid_target].insert(*ee);
    }
    
    // Remove deleted facets node-elemement adjancy list.
    for(typename std::set<index_t>::const_iterator de=deleted_facets.begin(); de!=deleted_facets.end();++de)
      SNEList[nid_target].erase(*de);
  }

  /*! Defragment surface mesh. This compresses the storage of internal data
    structures after the mesh has been defraged.*/
  void defragment(std::map<index_t, index_t> *active_vertex_map){
    size_t NNodes = _mesh->get_number_nodes();
    surface_nodes.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      surface_nodes[i] = false;
    
    std::vector<index_t> defrag_SENList, defrag_coplanar_ids;
    std::vector<real_t> defrag_normals;
    size_t NSElements = get_number_facets();
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);
      if(n[0]<0)
        continue;
      
      for(size_t j=0;j<snloc;j++){
        int nid = (*active_vertex_map)[n[j]];
        defrag_SENList.push_back(nid);
        surface_nodes[nid] = true;
      }
      defrag_coplanar_ids.push_back(coplanar_ids[i]);
      for(size_t j=0;j<ndims;j++)
        defrag_normals.push_back(normals[i*ndims+j]);
    }
    defrag_SENList.swap(SENList);
    defrag_coplanar_ids.swap(coplanar_ids);
    defrag_normals.swap(normals);
    
    // Finally, fix the Node-Element adjancy list.
    SNEList.clear();
    NSElements = get_number_facets();
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);
      
      for(size_t j=0;j<snloc;j++)
        SNEList[n[j]].insert(i);
    }
  }

  void refine(std::map< Edge<real_t, index_t>, index_t> &refined_edges){
    if(refined_edges.size()==0){
      return;
    }
    
    // Given the refined edges, refine facets.
    int lNSElements = get_number_facets();
    if(ndims==2){
      for(int i=0;i<lNSElements;i++){
        // Check if this element has been erased - if so continue to next element.
        int *n=&(SENList[i*snloc]);
        if(n[0]<0)
          continue;
        
        // Check if this edge has been refined.
        typename std::map< Edge<real_t, index_t>, index_t>::const_iterator edge = 
          refined_edges.find(Edge<real_t, index_t>(n[0], n[1]));
        
        // If it's refined then just jump onto the next one.
        if(edge==refined_edges.end())
          continue;

        // Renumber existing facet and add the new one.
        index_t cache_n1 = n[1];
        n[1] = edge->second;

        SENList.push_back(edge->second);
        SENList.push_back(cache_n1);

        coplanar_ids.push_back(coplanar_ids[i]);
        for(size_t j=0;j<ndims;j++)
          normals.push_back(normals[ndims*i+j]);
      }
    }else{      
      for(int i=0;i<lNSElements;i++){
        /* Check if this element has been erased - if so continue to
           next element.*/
        int *n=&(SENList[i*snloc]);
        if(n[0]<0)
          continue;
    
        /* Delete this facet if it's parent element has been deleted.*/
        bool erase_facet=true;
        for(size_t j=0;j<3;j++)
          if(!_mesh->is_halo_node(n[j])){
            erase_facet = false;
            break;
          }
        if(erase_facet){
          for(size_t j=0;j<3;j++)
            n[j] = -1;
          continue;
        }

        std::vector<typename std::map< Edge<real_t, index_t>, index_t>::const_iterator> split;
        for(size_t j=0;j<3;j++)
          for(size_t k=j+1;k<3;k++){
            typename std::map< Edge<real_t, index_t>, index_t>::const_iterator it =
              refined_edges.find(Edge<real_t, index_t>(n[j], n[k]));
            if(it!=refined_edges.end())
              split.push_back(it);
          }
        int refine_cnt=split.size();
        
        if(refine_cnt==0)
          continue;
        
        // Apply refinement templates.
        if(refine_cnt==1){
          // Find the opposit vertex
          int n0;
          for(size_t j=0;j<snloc;j++){
            if((split[0]->first.edge.first!=n[j])&&(split[0]->first.edge.second!=n[j])){
              n0 = n[j];
              break;
            }
          }
            
          // Renumber existing facet and add the new one.
          n[0] = split[0]->first.edge.first;
          n[1] = split[0]->second;
          n[2] = n0;

          SENList.push_back(split[0]->second);
          SENList.push_back(split[0]->first.edge.second);
          SENList.push_back(n0);
          
          coplanar_ids.push_back(coplanar_ids[i]);
          for(size_t j=0;j<ndims;j++)
            normals.push_back(normals[ndims*i+j]);
        }else{
          assert(refine_cnt==3);

          index_t m[6];
          m[0] = n[0];
          m[1] = refined_edges.find(Edge<real_t, index_t>(n[0], n[1]))->second;
          m[2] = n[1];
          m[3] = refined_edges.find(Edge<real_t, index_t>(n[1], n[2]))->second;
          m[4] = n[2];
          m[5] = refined_edges.find(Edge<real_t, index_t>(n[2], n[0]))->second;
          
          // Renumber existing facet and add the new one.
          n[0] = m[0];
          n[1] = m[1];
          n[2] = m[5];
          
          SENList.push_back(m[1]);
          SENList.push_back(m[3]);
          SENList.push_back(m[5]);
          
          coplanar_ids.push_back(coplanar_ids[i]);
          for(size_t j=0;j<ndims;j++)
            normals.push_back(normals[ndims*i+j]);

          SENList.push_back(m[1]);
          SENList.push_back(m[2]);
          SENList.push_back(m[3]);
          
          coplanar_ids.push_back(coplanar_ids[i]);
          for(size_t j=0;j<ndims;j++)
            normals.push_back(normals[ndims*i+j]);

          SENList.push_back(m[3]);
          SENList.push_back(m[4]);
          SENList.push_back(m[5]);
          
          coplanar_ids.push_back(coplanar_ids[i]);
          for(size_t j=0;j<ndims;j++)
            normals.push_back(normals[ndims*i+j]);
        }
      }
    }

    size_t NNodes = _mesh->get_number_nodes();
    size_t NSElements = get_number_facets();

    SNEList.clear();
    surface_nodes.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      surface_nodes[i] = false;
    
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);
      if(n[0]<0)
        continue;
      
      for(size_t j=0;j<snloc;j++){
        const int *n=get_facet(i);
        
        SNEList[n[j]].insert(i);
        surface_nodes[n[j]] = true;
      }
    }
  }

  int get_number_facets() const{
    return SENList.size()/snloc;
  }

  const int* get_facet(index_t id) const{
    return &(SENList[id*snloc]);
  }
  
  int get_number_dimensions() const{
    return _mesh->get_number_dimensions();
  }

  int get_number_nodes() const{
    return _mesh->get_number_nodes();
  }

  int get_coplanar_id(int eid) const{
    return coplanar_ids[eid];
  }

  const int* get_coplanar_ids() const{
    return &(coplanar_ids[0]);
  }

  const real_t* get_normal(int eid) const{
    return &(normals[eid*ndims]);
  }

  std::set<index_t> get_surface_patch(int i){
    return SNEList[i];
  }

  /// Set dot product tolerence - used to decide if elements are co-planar
  void set_coplanar_tolerance(real_t tol){
    COPLANAR_MAGIC_NUMBER = tol;
  }
  
 private:
  template<typename _real_t, typename _index_t> friend class VTKTools;

  /// Detects the surface nodes of the domain.
  void find_surface(){
    surface_nodes.resize(_mesh->_NNodes);
    for(size_t i=0;i<_mesh->_NNodes;i++)
      surface_nodes[i] = false;

    std::map< std::set<index_t>, std::vector<int> > facets;
    for(size_t i=0;i<_mesh->_NElements;i++){
      for(size_t j=0;j<nloc;j++){
        std::set<index_t> facet;
        for(size_t k=1;k<nloc;k++){
          facet.insert(_mesh->_ENList[i*nloc+(j+k)%nloc]);
        }
        if(facets.count(facet)){
          facets.erase(facet);
        }else{
          std::vector<int> element;
          if(snloc==3){
            if(j==0){
              element.push_back(_mesh->_ENList[i*nloc+1]);
              element.push_back(_mesh->_ENList[i*nloc+3]);
              element.push_back(_mesh->_ENList[i*nloc+2]);
            }else if(j==1){
              element.push_back(_mesh->_ENList[i*nloc+2]);
              element.push_back(_mesh->_ENList[i*nloc+3]);
              element.push_back(_mesh->_ENList[i*nloc+0]);
            }else if(j==2){
              element.push_back(_mesh->_ENList[i*nloc+0]);
              element.push_back(_mesh->_ENList[i*nloc+3]);
              element.push_back(_mesh->_ENList[i*nloc+1]);
            }else if(j==3){
              element.push_back(_mesh->_ENList[i*nloc+0]);
              element.push_back(_mesh->_ENList[i*nloc+1]);
              element.push_back(_mesh->_ENList[i*nloc+2]);
            }
          }else{
            element.push_back(_mesh->_ENList[i*nloc+(j+1)%nloc]);
            element.push_back(_mesh->_ENList[i*nloc+(j+2)%nloc]);
          }
          facets[facet] = element;
        }
      }
    }
    
    for(typename std::map<std::set<index_t>, std::vector<int> >::const_iterator it=facets.begin(); it!=facets.end(); ++it){
      SENList.insert(SENList.end(), it->second.begin(), it->second.end());
      for(typename std::set<index_t>::const_iterator jt=it->first.begin();jt!=it->first.end();++jt)
        surface_nodes[*jt] = true;
    }

    calculate_coplanar_ids();
  }

  /// Calculate co-planar patches.
  void calculate_coplanar_ids(){
    // Calculate all element normals
    size_t NSElements = get_number_facets();
    normals.resize(NSElements*ndims);
    if(ndims==2){
      for(size_t i=0;i<NSElements;i++){
        normals[i*2] = sqrt(1 - pow((get_x(SENList[2*i+1]) - get_x(SENList[2*i]))
                                    /(get_y(SENList[2*i+1]) - get_y(SENList[2*i])), 2));
        if(isnan(normals[i*2])){
          normals[i*2] = 0;
          normals[i*2+1] = 1;
        }else{
          normals[i*2+1] = sqrt(1 - pow(normals[i*2], 2));
        }
        
        if(get_y(SENList[2*i+1]) - get_y(SENList[2*i])>0)
          normals[i*2] *= -1;

        if(get_x(SENList[2*i]) - get_x(SENList[2*i+1])>0)
          normals[i*2+1] *= -1;
      }
    }else{
      for(size_t i=0;i<NSElements;i++){
        real_t x1 = get_x(SENList[3*i+1]) - get_x(SENList[3*i]);
        real_t y1 = get_y(SENList[3*i+1]) - get_y(SENList[3*i]);
        real_t z1 = get_z(SENList[3*i+1]) - get_z(SENList[3*i]);
        
        real_t x2 = get_x(SENList[3*i+2]) - get_x(SENList[3*i]);
        real_t y2 = get_y(SENList[3*i+2]) - get_y(SENList[3*i]);
        real_t z2 = get_z(SENList[3*i+2]) - get_z(SENList[3*i]);
        
        normals[i*3  ] = y1*z2 - y2*z1;
        normals[i*3+1] =-x1*z2 + x2*z1;
        normals[i*3+2] = x1*y2 - x2*y1;
        
        real_t invmag = 1/sqrt(normals[i*3]*normals[i*3]+normals[i*3+1]*normals[i*3+1]+normals[i*3+2]*normals[i*3+2]);
        normals[i*3  ]*=invmag;
        normals[i*3+1]*=invmag;
        normals[i*3+2]*=invmag;
      }
    }
    
    // Create EEList for surface
    for(size_t i=0;i<NSElements;i++){
      for(size_t j=0;j<snloc;j++){
        SNEList[SENList[snloc*i+j]].insert(i);
      }
    }
    
    std::vector<int> EEList(NSElements*snloc);
    for(size_t i=0;i<NSElements;i++){
      if(snloc==2){
        for(size_t j=0;j<2;j++){
          index_t nid=SENList[i*2+j];
          for(typename std::set<index_t>::iterator it=SNEList[nid].begin();it!=SNEList[nid].end();++it){
            if(*it==(index_t)i){
              continue;
            }else{
              EEList[i*2+j] = *it;
              break;
            }
          }
        }
      }else{
        for(size_t j=0;j<3;j++){
          index_t nid1=SENList[i*3+(j+1)%3];
          index_t nid2=SENList[i*3+(j+2)%3];
          for(typename std::set<index_t>::iterator it=SNEList[nid1].begin();it!=SNEList[nid1].end();++it){
            if(*it==(index_t)i){
              continue;
            }       
            if(SNEList[nid2].find(*it)!=SNEList[nid2].end()){
              EEList[i*3+j] = *it;
              break;
            }
          }
        }
      }
    }
    
    // Form patches
    coplanar_ids.resize(NSElements);
    for(std::vector<int>::iterator it=coplanar_ids.begin(); it!=coplanar_ids.end(); ++it)
      *it = 0;
  
    int current_id = 1;
    for(size_t pos = 0;pos<NSElements;){
      // Create a new starting point
      const real_t *ref_normal=NULL;
      for(size_t i=pos;i<NSElements;i++){
        if(coplanar_ids[i]==0){
          // This is the first element in the new patch
          pos = i;
          coplanar_ids[pos] = current_id;
          ref_normal = &(normals[pos*ndims]);
          break;
        }
      }
      if(ref_normal==NULL)
        break;

      // Initialise the front
      std::set<int> front;
      front.insert(pos);
      
      // Advance this front
      while(!front.empty()){
        int sele = *front.begin();
        front.erase(front.begin());
        
        // Check surrounding surface elements:      
        for(size_t i=0; i<snloc; i++){
          int sele2 = EEList[sele*snloc+i];
          if(coplanar_ids[sele2]>0)
            continue;
          
          double coplanar = 0.0;
          for(size_t d=0;d<ndims;d++)
            coplanar += ref_normal[d]*normals[sele2*ndims+d];
          
          if(coplanar>=COPLANAR_MAGIC_NUMBER){
            front.insert(sele2);
            coplanar_ids[sele2] = current_id;
          }
        }
      }
      current_id++;
      pos++;
    }
  }

  inline real_t get_x(index_t nid) const{
    return _mesh->_coords[nid*ndims];
  }

  inline real_t get_y(index_t nid) const{
    return _mesh->_coords[nid*ndims+1];
  }

  inline real_t get_z(index_t nid) const{
    return _mesh->_coords[nid*ndims+2];
  }

  size_t ndims, nloc, snloc;
  std::map<int, std::set<index_t> > SNEList;
  std::vector<bool> surface_nodes;
  std::vector<index_t> SENList, coplanar_ids;
  std::vector<real_t> normals;
  real_t COPLANAR_MAGIC_NUMBER;
  
  Mesh<real_t, index_t> *_mesh;
};
#endif
