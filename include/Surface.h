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

/** This class is used to: identify the boundary of the domain;
    uniquely label connected co-linear patches of surface elements
    (these can be used to prevent adaptivity coarsening these patches
    and smoothening out features); evaluate a characteristic length
    scale for these patches (these "constraints" can be added to the
    metric tensor field before gradation is applied in order to get
    good quality elements near the geometry).
*/

template<typename real_t, typename index_t>
  class Surface{
 public:
  
  /// Default constructor.
  Surface(){
    _NNodes = 0;
    _NElements = 0;
    _ndims = 0;
    nloc = 0;
    snloc = 0;
    _ENList = NULL;
    _node_distribution = NULL;
    _x = NULL;
    _y = NULL;
    _z = NULL;
#ifdef HAVE_MPI
    _mesh_comm = NULL;
#endif
    set_coplanar_tolerance(0.9999999);
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

  const int* get_coplanar_ids(){
    return &(coplanar_ids[0]);
  }

  /*! Set the source mesh (2D triangular meshes).
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param node_distribution gives the number of nodes owned
   *        by each process when this is using MPI, where it is
   *        assumed that each processes ownes n_i consecutive
   *        vertices of the mesh. Its contents are identical for every process.
   */
  void set_mesh(int NNodes, int NElements, const index_t *ENList,
                const real_t *x, const real_t *y, const index_t *node_distribution=NULL){
    nloc = 3;
    snloc = 2;
    _ndims = 2;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _x = x;
    _y = y;
    _z = NULL;
    _node_distribution = node_distribution;

    find_surface();
  }
  
  /*! Set the source mesh (3D tetrahedral meshes).
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   * @param node_distribution gives the number of nodes owned
   *        by each process when this is using MPI, where it is
   *        assumed that each processes ownes n_i consecutive
   *        vertices of the mesh. Its contents are identical for every process.
   */
  void set_mesh(int NNodes, int NElements, const index_t *ENList,
                const real_t *x, const real_t *y, const real_t *z, const index_t *node_distribution=NULL){
    nloc = 4;
    snloc = 3;
    _ndims = 3;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _x = x;
    _y = y;
    _z = z;
    _node_distribution = node_distribution;

    find_surface();
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
      // for(typename std::vector<index_t>::const_iterator jt=it->second.begin(); jt!=it->second.end(); ++jt)
      SENList.insert(SENList.end(), it->second.begin(), it->second.end());
      surface_nodes.insert(it->first.begin(), it->first.end());
    }

    calculate_coplanar_ids();
  }

  /// Calculate co-planar patches.
  void calculate_coplanar_ids(){
    // Calculate all element normals
    size_t NSElements = SENList.size()/snloc;
    std::vector<real_t> normals(NSElements*_ndims);
    if(_ndims==2){
      for(size_t i=0;i<NSElements;i++){
        normals[i*2] = sqrt(1 - pow((_x[SENList[2*i+1]] - _x[SENList[2*i]])
                                    /(_y[SENList[2*i+1]] - _y[SENList[2*i]]), 2));
        if(isnan(normals[i*2])){
          normals[i*2] = 0;
          if(_x[SENList[2*i+1]] - _x[SENList[2*i]]>0)
            normals[i*2+1] = -1;
          else
            normals[i*2+1] = 1;
        }else{
          normals[i*2+1] = sqrt(1 - pow(normals[i*2], 2));
        }
      }
    }else{
      for(size_t i=0;i<NSElements;i++){
        real_t x1 = _x[SENList[3*i+1]] - _x[SENList[3*i]];
        real_t y1 = _y[SENList[3*i+1]] - _y[SENList[3*i]];
        real_t z1 = _z[SENList[3*i+1]] - _z[SENList[3*i]];
        
        real_t x2 = _x[SENList[3*i+2]] - _x[SENList[3*i]];
        real_t y2 = _y[SENList[3*i+2]] - _y[SENList[3*i]];
        real_t z2 = _z[SENList[3*i+2]] - _z[SENList[3*i]];
        
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
    std::map< int, std::set<size_t> > SNEList;
    for(size_t i=0;i<NSElements;i++){
      for(size_t j=0;j<snloc;j++){
        SNEList[SENList[snloc*i+j]].insert(i);
      }
    }
    
    std::vector<int> EEList(NSElements*snloc);
    for(size_t i=0;i<NSElements;i++){
      if(snloc=2){
        for(size_t j=0;j<2;j++){
          size_t nid=SENList[i*2+j];
          for(std::set<size_t>::iterator it=SNEList[nid].begin();it!=SNEList[nid].end();++it){
            if((*it)==i){
              continue;
            }else{
              EEList[i*2+j] = *it;
              break;
            }
          }
        }
      }else{
        for(size_t j=0;j<3;j++){
          size_t nid1=SENList[i*3+(j+1)%3];
          size_t nid2=SENList[i*3+(j+2)%3];
          for(std::set<size_t>::iterator it=SNEList[nid1].begin();it!=SNEList[nid1].end();++it){
            if((*it)==i){
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
  
    size_t current_id = 1;
    for(size_t pos = 0;pos<NSElements;){
      // Create a new starting point
      real_t *ref_normal=NULL;
      for(size_t i=pos;i<NSElements;i++){{
          if(coplanar_ids[i]==0){
            // This is the first element in the new patch
            pos = i;
            coplanar_ids[pos] = current_id;
            ref_normal = &(normals[pos*_ndims]);
            break;
          }
        }
        
        // Jump out of this while loop if we are finished
        if(i==NSElements)
          break;
      }
      
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
          
          double coplanar = 0;
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

  int _NNodes, _NElements, NSElements, _ndims, nloc, snloc;
  const index_t *_ENList, *_node_distribution;
  const real_t *_x, *_y, *_z;
  std::vector< std::set<index_t> > NNList;
  std::vector<int> norder;
  std::set<index_t> surface_nodes;
  std::vector<int> SENList, coplanar_ids;
  real_t COPLANAR_MAGIC_NUMBER;

#ifdef HAVE_MPI
  const MPI_Comm *_mesh_comm;
#endif
};
#endif
