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
    _NNodes = mesh.get_number_nodes();
    _NElements = mesh.get_number_elements();
    _ndims = mesh.get_number_dimensions();
    nloc = (_ndims==2)?3:4;
    snloc = (_ndims==2)?2:3;
    _ENList = mesh.get_enlist();;
    _coords = mesh.get_coords();
#ifdef HAVE_MPI
    _mesh_comm = mesh.get_mpi_comm();
#endif
    set_coplanar_tolerance(0.9999999);

    find_surface();
  }
  
  /// Default destructor.
  ~Surface(){
  }

  bool contains_node(index_t nid){
    return surface_nodes.count(nid)>0;
  }

  int get_number_facets(){
    return NSElements;
  }

  const int* get_facets(){
    return &(SENList[0]);
  }
  
  int get_coplanar_id(int eid){
    return coplanar_ids[eid];
  }

  const int* get_coplanar_ids(){
    return &(coplanar_ids[0]);
  }

  const real_t* get_normal(int eid){
    return &(normals[eid*_ndims]);
  }

  std::set<size_t> get_surface_patch(int i){
    return SNEList[i];
  }

  /// Set dot product tolerence - used to decide if elements are co-planar
  void set_coplanar_tolerance(real_t tol){
    COPLANAR_MAGIC_NUMBER = tol;
  }
  
 private:
  /// Detects the surface nodes of the domain.
  void find_surface(){
    std::map< std::set<index_t>, std::vector<int> > facets;
    for(int i=0;i<_NElements;i++){
      for(int j=0;j<nloc;j++){
        std::set<index_t> facet;
        for(int k=1;k<nloc;k++){
          facet.insert(_ENList[i*nloc+(j+k)%nloc]);
        }
        if(facets.count(facet)){
          facets.erase(facet);
        }else{
          std::vector<int> element;
          if(snloc==3){
            if(j==0){
              element.push_back(_ENList[i*nloc+1]);
              element.push_back(_ENList[i*nloc+3]);
              element.push_back(_ENList[i*nloc+2]);
            }else if(j==1){
              element.push_back(_ENList[i*nloc+2]);
              element.push_back(_ENList[i*nloc+3]);
              element.push_back(_ENList[i*nloc+0]);
            }else if(j==2){
              element.push_back(_ENList[i*nloc+0]);
              element.push_back(_ENList[i*nloc+3]);
              element.push_back(_ENList[i*nloc+1]);
            }else if(j==3){
              element.push_back(_ENList[i*nloc+0]);
              element.push_back(_ENList[i*nloc+1]);
              element.push_back(_ENList[i*nloc+2]);
            }
          }else{
            element.push_back(_ENList[i*nloc+(j+1)%nloc]);
            element.push_back(_ENList[i*nloc+(j+2)%nloc]);
          }
          facets[facet] = element;
        }
      }
    }
    
    NSElements = facets.size();
    for(typename std::map<std::set<index_t>, std::vector<int> >::const_iterator it=facets.begin(); it!=facets.end(); ++it){
      SENList.insert(SENList.end(), it->second.begin(), it->second.end());
      surface_nodes.insert(it->first.begin(), it->first.end());
    }

    calculate_coplanar_ids();
  }

  /// Calculate co-planar patches.
  void calculate_coplanar_ids(){
    // Calculate all element normals
    normals.resize(NSElements*_ndims);
    if(_ndims==2){
      for(int i=0;i<NSElements;i++){
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
      for(int i=0;i<NSElements;i++){
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
    for(int i=0;i<NSElements;i++){
      for(int j=0;j<snloc;j++){
        SNEList[SENList[snloc*i+j]].insert(i);
      }
    }
    
    std::vector<int> EEList(NSElements*snloc);
    for(int i=0;i<NSElements;i++){
      if(snloc==2){
        for(int j=0;j<2;j++){
          int nid=SENList[i*2+j];
          for(std::set<size_t>::iterator it=SNEList[nid].begin();it!=SNEList[nid].end();++it){
            if((int)(*it)==i){
              continue;
            }else{
              EEList[i*2+j] = *it;
              break;
            }
          }
        }
      }else{
        for(int j=0;j<3;j++){
          int nid1=SENList[i*3+(j+1)%3];
          int nid2=SENList[i*3+(j+2)%3];
          for(std::set<size_t>::iterator it=SNEList[nid1].begin();it!=SNEList[nid1].end();++it){
            if((int)(*it)==i){
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
    for(int pos = 0;pos<NSElements;){
      // Create a new starting point
      const real_t *ref_normal=NULL;
      for(int i=pos;i<NSElements;i++){
        if(coplanar_ids[i]==0){
          // This is the first element in the new patch
          pos = i;
          coplanar_ids[pos] = current_id;
          ref_normal = &(normals[pos*_ndims]);
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
        for(int i=0; i<snloc; i++){
          int sele2 = EEList[sele*snloc+i];
          if(coplanar_ids[sele2]>0)
            continue;
          
          double coplanar = 0.0;
          for(int d=0;d<_ndims;d++)
            coplanar += ref_normal[d]*normals[sele2*_ndims+d];
          
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

  inline real_t get_x(index_t nid){
    return _coords[nid*_ndims];
  }

  inline real_t get_y(index_t nid){
    return _coords[nid*_ndims+1];
  }

  inline real_t get_z(index_t nid){
    return _coords[nid*_ndims+2];
  }

  int _NNodes, _NElements, NSElements, _ndims, nloc, snloc;
  const index_t *_ENList, *_node_distribution;
  const real_t *_coords;
  std::vector< std::set<index_t> > NNList;
  std::map<int, std::set<size_t> > SNEList;
  std::vector<int> norder;
  std::set<index_t> surface_nodes;
  std::vector<int> SENList, coplanar_ids;
  std::vector<real_t> normals;
  real_t COPLANAR_MAGIC_NUMBER;

#ifdef HAVE_MPI
  const MPI_Comm *_mesh_comm;
#endif
};
#endif
