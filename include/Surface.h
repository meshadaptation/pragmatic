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

#ifndef SURFACE_H
#define SURFACE_H

#include <vector>
#include <set>
#include <map>

#include <errno.h>

#include "Mesh.h"
#include "Edge.h"

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

    size_t NNodes = _mesh->get_number_nodes();
    surface_nodes.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      surface_nodes[i] = false;
    
    // Default coplanar tolerance.
    set_coplanar_tolerance(0.9999999);
  }
  
  /// Default destructor.
  ~Surface(){
  }

  /// Append a facet to the surface
  void append_facet(const int *facet, int boundary_id, int coplanar_id, bool check_duplicates=false){
    if(check_duplicates){
      std::set<index_t> Intersection;
      set_intersection(SNEList[facet[0]].begin(), SNEList[facet[0]].end(),
                       SNEList[facet[1]].begin(), SNEList[facet[1]].end(),
                       inserter(Intersection,Intersection.begin()));
      for(size_t i=2;i<snloc;i++){
        std::set<index_t> tmp_intersection;
        set_intersection(Intersection.begin(), Intersection.end(),
                         SNEList[facet[i]].begin(), SNEList[facet[i]].end(),
                         inserter(tmp_intersection,tmp_intersection.begin()));
        Intersection.swap(tmp_intersection);
      }
      if(Intersection.size())
        return;
    }

    index_t eid = coplanar_ids.size();

    boundary_ids.push_back(boundary_id);
    coplanar_ids.push_back(coplanar_id);
    for(size_t i=0;i<snloc;i++){
      index_t nid = facet[i];
      SNEList[nid].insert(eid);

      // This is required because additional vertices may have been added.
      while(surface_nodes.size()<=(size_t)nid)
        surface_nodes.push_back(false);
      surface_nodes[nid] = true;

      SENList.push_back(nid);
    }
    
    double normal[3];
    if(ndims==2)
      calculate_normal_2d(facet, normal);
    else
      calculate_normal_3d(facet, normal);
    
    for(size_t i=0;i<ndims;i++)
      normals.push_back(normal[i]);
  }

  /// True if surface contains vertex nid.
  bool contains_node(index_t nid) const{
    assert(nid>=0);
    // assert(nid<(index_t)surface_nodes.size()); 
    if(nid>=(index_t)surface_nodes.size()){
      return false;
    }

    return surface_nodes[nid];
  }

  /// Detects the surface nodes of the domain.
  void find_surface(bool assume_bounding_box=false){
    use_bbox = assume_bounding_box;

    size_t NElements = _mesh->get_number_elements();
    
    std::map< std::set<index_t>, std::vector<int> > facets;
    for(size_t i=0;i<NElements;i++){
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

          // Only recognise this as a valid boundary node if at least one node is owned.
          bool interpartition_boundary=true;
          for(std::vector<int>::const_iterator it=element.begin();it!=element.end();++it)
            if(_mesh->is_owned_node(*it)){
              interpartition_boundary=false;
              break;
            }
          if(!interpartition_boundary)
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

  /// True if node nid is a corner vertex OR if more than ndims boundary lables are incident on the vertex.
  bool is_corner_vertex(index_t nid) const{
    typename std::map<int, std::set<index_t> >::const_iterator iSNEList = SNEList.find(nid);
    if(iSNEList==SNEList.end())
      return false;

    std::set<int> incident_plane;
    for(typename std::set<index_t>::const_iterator it=iSNEList->second.begin();it!=iSNEList->second.end();++it){
      incident_plane.insert(coplanar_ids[*it]);
    }
    
    return (incident_plane.size()>=ndims);
  }

  bool is_collapsible(index_t nid_free, index_t nid_target) const{
    if((nid_free>=(index_t)surface_nodes.size())||(nid_target>=(index_t)surface_nodes.size())){
      std::cerr<<"WARNING: have yet to migrate surface\n";
      return true;
    }

    // If nid_free is not on the surface then it's unconstrained.
    if(!surface_nodes[nid_free])
      return true;

    // However, is nid_free is on the surface then nid_target must
    // also lie on a surface for it to be considered for collapse.
    if(!surface_nodes[nid_target]){
      return false;
    }

    std::set<int> incident_plane_free;
    typename std::map<int, std::set<index_t> >::const_iterator iSNEList = SNEList.find(nid_free);
    assert(iSNEList!=SNEList.end());
    for(typename std::set<index_t>::const_iterator it=iSNEList->second.begin();it!=iSNEList->second.end();++it){
      incident_plane_free.insert(coplanar_ids[*it]);
    }

    // Non-collapsible if nid_free is a corner node.
    if(incident_plane_free.size()>=ndims){
      return false;
    }
    
    std::set<int> incident_plane_target;
    typename std::map<int, std::set<index_t> >::const_iterator jSNEList = SNEList.find(nid_target);
    assert(jSNEList!=SNEList.end());
    for(typename std::set<index_t>::const_iterator it=jSNEList->second.begin();it!=jSNEList->second.end();++it)
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
    assert(SNEList.count(nid_free));
    assert(SNEList.count(nid_target));

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
    
    std::vector<index_t> defrag_SENList;
    std::vector<int> defrag_boundary_ids, defrag_coplanar_ids;
    std::vector<real_t> defrag_normals;
    size_t NSElements = get_number_facets();
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);

      // Check if deleted.
      if(n[0]<0)
        continue;
      
      // Check all vertices are active.
      bool valid_facet=true;
      for(size_t j=0;j<snloc;j++){
        if(active_vertex_map->count(n[j])==0){
          valid_facet = false;
          break;
        }
      }
      if(!valid_facet)
        continue;
      
      for(size_t j=0;j<snloc;j++){
        int nid = (*active_vertex_map)[n[j]];
        defrag_SENList.push_back(nid);
        surface_nodes[nid] = true;
      }
      defrag_boundary_ids.push_back(boundary_ids[i]);
      defrag_coplanar_ids.push_back(coplanar_ids[i]);
      for(size_t j=0;j<ndims;j++)
        defrag_normals.push_back(normals[i*ndims+j]);
    }
    defrag_SENList.swap(SENList);
    defrag_boundary_ids.swap(boundary_ids);
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

  void find_facets(const int *element, std::vector<int> &facet_ids) const{
    for(size_t i=0;i<nloc;i++){
      std::set<index_t> Intersection;
      
      typename std::map<int, std::set<index_t> >::const_iterator iSNEList = SNEList.find(element[i]);
      if(iSNEList==SNEList.end())
        continue;
      
      typename std::map<int, std::set<index_t> >::const_iterator jSNEList = SNEList.find(element[(i+1)%nloc]);
      if(jSNEList==SNEList.end())
        continue;
      
      set_intersection(iSNEList->second.begin(), iSNEList->second.end(),
                       jSNEList->second.begin(), jSNEList->second.end(),
                       inserter(Intersection,Intersection.begin()));
      
      if(ndims==3){
        typename std::map<int, std::set<index_t> >::const_iterator kSNEList = SNEList.find(element[(i+2)%nloc]);
        if(kSNEList==SNEList.end())
          continue;

        std::set<index_t> tmp_intersection;
        set_intersection(Intersection.begin(), Intersection.end(),
                         kSNEList->second.begin(), kSNEList->second.end(),
                         inserter(tmp_intersection,tmp_intersection.begin()));
        Intersection.swap(tmp_intersection);
      }

      if(Intersection.size()){
#ifndef NDEBUG
        if(Intersection.size()>1){
          std::cerr<<"WARNING: duplicate facets\nIntersection.size() = "<<Intersection.size()<<std::endl<<"Intersection = ";
          for(typename std::set<index_t>::iterator ii=Intersection.begin();ii!=Intersection.end();++ii){
            std::cerr<<*ii<<" (";
            for(size_t jj=0;jj<snloc;jj++)
              std::cerr<<SENList[(*ii)*snloc+jj]<<" ";
            std::cerr<<") ";
          }
          std::cerr<<std::endl;  
        }
#endif
        facet_ids.push_back(*Intersection.begin());
      }
    }
  }

  void refine(std::vector< std::vector<index_t> > &refined_edges, std::vector<index_t> &lnn2gnn){
    // Given the refined edges, refine facets.
    std::vector< std::vector<index_t> > private_SENList;
    std::vector< std::vector<int> > private_boundary_ids;
    std::vector< std::vector<int> > private_coplanar_ids;
    std::vector< std::vector<real_t> > private_normals;
    std::vector<unsigned int> threadIdx, splitCnt;
    unsigned int nthreads;
    
#pragma omp parallel
    {
      nthreads = omp_get_num_threads();
      
#pragma omp master
      {
      	private_SENList.resize(nthreads);
      	private_boundary_ids.resize(nthreads);
      	private_coplanar_ids.resize(nthreads);
      	private_normals.resize(nthreads);
      	threadIdx.resize(nthreads);
      	splitCnt.resize(nthreads);
      }
#pragma omp barrier
    }
    
    int lNSElements = get_number_facets();
    
    if(ndims==2){
#pragma omp parallel
      {
        const unsigned int tid = omp_get_thread_num();
        splitCnt[tid] = 0;
        
#pragma omp for schedule(dynamic)
        for(int i=0;i<lNSElements;i++){
          // Check if this element has been erased - if so continue to next element.
          int *n=&(SENList[i*snloc]);
          if(n[0]<0)
            continue;
          
          // Check if this edge has been refined.
          index_t newVertex = _mesh->get_new_vertex(n[0], n[1], refined_edges, lnn2gnn);
          
          // If it's not refined then just jump onto the next one.
          if(newVertex < 0)
            continue;
          
          // Renumber existing facet and add the new one.
          index_t cache_n1 = n[1];
          n[1] = newVertex;
          
          private_SENList[tid].push_back(newVertex);
          private_SENList[tid].push_back(cache_n1);
          
          private_boundary_ids[tid].push_back(boundary_ids[i]);
          private_coplanar_ids[tid].push_back(coplanar_ids[i]);
          for(size_t j=0;j<ndims;j++)
            private_normals[tid].push_back(normals[ndims*i+j]);
          
          splitCnt[tid]++;
        }
      }
    }else{
#pragma omp parallel
      {
        const unsigned int tid = omp_get_thread_num();
        splitCnt[tid] = 0;
        
#pragma omp for schedule(dynamic)
        for(int i=0;i<lNSElements;i++){
          // Check if this element has been erased - if so continue to next element.
          int *n=&(SENList[i*snloc]);
          if(n[0]<0)
            continue;
          
          // Delete this facet if it's parent element has been deleted.
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
          
          std::vector< Edge<index_t> > splitEdges;
          std::vector<index_t> newVertex;
          index_t vertexID;
          for(size_t j=0;j<3;j++)
            for(size_t k=j+1;k<3;k++){
              vertexID = _mesh->get_new_vertex(n[j], n[k], refined_edges, lnn2gnn);
              if(vertexID >= 0){
              	splitEdges.push_back(Edge<index_t>(n[j], n[k]));
              	newVertex.push_back(vertexID);
              }
            }
          int refine_cnt=splitEdges.size();
          
          if(refine_cnt==0)
            continue;
          
          // Apply refinement templates.
          if(refine_cnt==1){
            // Find the opposite vertex
            int n0;
            for(size_t j=0;j<snloc;j++){
              if((splitEdges[0].edge.first!=n[j])&&(splitEdges[0].edge.second!=n[j])){
                n0 = n[j];
                break;
              }
            }
            
            // Renumber existing facet and add the new one.
            n[0] = splitEdges[0].edge.first;
            n[1] = newVertex[0];
            n[2] = n0;
            
            private_SENList[tid].push_back(newVertex[0]);
            private_SENList[tid].push_back(splitEdges[0].edge.second);
            private_SENList[tid].push_back(n0);
            
            private_boundary_ids[tid].push_back(boundary_ids[i]);
            private_coplanar_ids[tid].push_back(coplanar_ids[i]);
            for(size_t j=0;j<ndims;j++)
              private_normals[tid].push_back(normals[ndims*i+j]);
            
            splitCnt[tid]++;
          }else{
            assert(refine_cnt==3);
            
            index_t m[6];
            m[0] = n[0];
            m[1] = newVertex[0];
            m[2] = n[1];
            m[3] = newVertex[2];
            m[4] = n[2];
            m[5] = newVertex[1];

            // Renumber existing facet and add the new one.
            n[0] = m[0];
            n[1] = m[1];
            n[2] = m[5];

            private_SENList[tid].push_back(m[1]);
            private_SENList[tid].push_back(m[3]);
            private_SENList[tid].push_back(m[5]);

            private_boundary_ids[tid].push_back(boundary_ids[i]);
            private_coplanar_ids[tid].push_back(coplanar_ids[i]);
            for(size_t j=0;j<ndims;j++)
              private_normals[tid].push_back(normals[ndims*i+j]);

            private_SENList[tid].push_back(m[1]);
            private_SENList[tid].push_back(m[2]);
            private_SENList[tid].push_back(m[3]);

            private_boundary_ids[tid].push_back(boundary_ids[i]);
            private_coplanar_ids[tid].push_back(coplanar_ids[i]);
            for(size_t j=0;j<ndims;j++)
            	private_normals[tid].push_back(normals[ndims*i+j]);

            private_SENList[tid].push_back(m[3]);
            private_SENList[tid].push_back(m[4]);
            private_SENList[tid].push_back(m[5]);

            private_boundary_ids[tid].push_back(boundary_ids[i]);
            private_coplanar_ids[tid].push_back(coplanar_ids[i]);
            for(size_t j=0;j<ndims;j++)
            	private_normals[tid].push_back(normals[ndims*i+j]);

            splitCnt[tid] += 3;
          }
        }
      }
    }
    
#pragma omp parallel
    {
      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in SENList at which new elements should be appended.
      const unsigned int tid = omp_get_thread_num();
      threadIdx[tid] = splitCnt[tid];
      
#pragma omp barrier
      
      unsigned int blockSize = 1, tmp;
      while(blockSize < threadIdx.size())
        {
          if((tid & blockSize) != 0)
            tmp = threadIdx[tid - ((tid & (blockSize - 1)) + 1)];
          else
            tmp = 0;
          
#pragma omp barrier
          
          threadIdx[tid] += tmp;
          
#pragma omp barrier
          
          blockSize *= 2;
        }
      
      threadIdx[tid] += get_number_facets() - splitCnt[tid];
      
#pragma omp barrier
      
      // Resize mesh containers
#pragma omp master
      {
      	const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];
        
      	SENList.resize(snloc*newSize);
        boundary_ids.resize(newSize);
        coplanar_ids.resize(newSize);
        normals.resize(ndims*newSize);
      }
#pragma omp barrier
      
      // Append new elements to the surface
      memcpy(&SENList[snloc*threadIdx[tid]], &private_SENList[tid][0], snloc*splitCnt[tid]*sizeof(index_t));
      memcpy(&boundary_ids[threadIdx[tid]], &private_boundary_ids[tid][0], splitCnt[tid]*sizeof(int));
      memcpy(&coplanar_ids[threadIdx[tid]], &private_coplanar_ids[tid][0], splitCnt[tid]*sizeof(int));
      memcpy(&normals[ndims*threadIdx[tid]], &private_normals[tid][0], ndims*splitCnt[tid]*sizeof(real_t));
    }
    
    size_t NNodes = _mesh->get_number_nodes();
    size_t NSElements = get_number_facets();
    
    SNEList.clear();
    surface_nodes.clear();
    surface_nodes.resize(NNodes, false);
    
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);
      if(n[0]<0)
        continue;
      
      for(size_t j=0;j<snloc;j++){
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

  int get_boundary_id(int eid) const{
    return boundary_ids[eid];
  }

  const int* get_boundary_ids() const{
    return &(boundary_ids[0]);
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

  std::set<index_t> get_surface_patch(int nid) const{
    assert(SNEList.find(nid)!=SNEList.end());
    return SNEList.find(nid)->second;
  }

  /// Set dot product tolerence - used to decide if elements are co-planar
  void set_coplanar_tolerance(real_t tol){
    COPLANAR_MAGIC_NUMBER = tol;
  }

  /// Set surface.
  void set_surface(int NSElements, const int *senlist, const int *boundary_ids_, const int *coplanar_ids_){
    boundary_ids.resize(NSElements);
    coplanar_ids.resize(NSElements);
    SENList.resize(NSElements*snloc);
    surface_nodes.resize(_mesh->get_number_nodes());
    fill(surface_nodes.begin(), surface_nodes.end(), false);
    SNEList.clear();
    for(int i=0;i<NSElements;i++){
      boundary_ids[i] = boundary_ids_[i];
      coplanar_ids[i] = coplanar_ids_[i];
      const int *n = senlist+i*snloc;
      for(size_t j=0;j<snloc;j++){
        SNEList[n[j]].insert(i);
        surface_nodes[n[j]] = true;
        
        SENList[i*snloc+j] = n[j];
      }
    }

    calculate_normals();
  }

 private:
  template<typename _real_t, typename _index_t> friend class VTKTools;
  template<typename _real_t, typename _index_t> friend class CUDATools;

  /// Calculate surface normals.
  void calculate_normals(){
    // Calculate all element normals
    size_t NSElements = get_number_facets();
    normals.resize(NSElements*ndims);
    if(ndims==2){
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for(size_t i=0;i<NSElements;i++)
          calculate_normal_2d(&(SENList[2*i]), &(normals[i*2]));
      }
    }else{
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for(size_t i=0;i<NSElements;i++){
          calculate_normal_3d(&(SENList[3*i]), &(normals[i*3]));
        }
      }
    }
  }

  /// Calculate facet normal (2D).
  inline void calculate_normal_2d(const index_t *facet, double *normal){
    normal[0] = sqrt(1 - pow((get_x(facet[1]) - get_x(facet[0]))
                             /(get_y(facet[1]) - get_y(facet[0])), 2));
    if(isnan(normal[0])){
      errno = 0;
      normal[0] = 0;
      normal[1] = 1;
    }else{
      normal[1] = sqrt(1 - pow(normal[0], 2));
    }
    
    if(get_y(facet[1]) - get_y(facet[0])>0)
      normal[0] *= -1;
    
    if(get_x(facet[0]) - get_x(facet[1])>0)
      normal[1] *= -1;
  }


  /// Calculate facet normal (3D).
  inline void calculate_normal_3d(const index_t *facet, double *normal){
    real_t x1 = get_x(facet[1]) - get_x(facet[0]);
    real_t y1 = get_y(facet[1]) - get_y(facet[0]);
    real_t z1 = get_z(facet[1]) - get_z(facet[0]);
    
    real_t x2 = get_x(facet[2]) - get_x(facet[0]);
    real_t y2 = get_y(facet[2]) - get_y(facet[0]);
    real_t z2 = get_z(facet[2]) - get_z(facet[0]);
    
    normal[0] = y1*z2 - y2*z1;
    normal[1] =-x1*z2 + x2*z1;
    normal[2] = x1*y2 - x2*y1;
    
    real_t invmag = 1/sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    normal[0]*=invmag;
    normal[1]*=invmag;
    normal[2]*=invmag;
  }

  /// Calculate co-planar patches.
  void calculate_coplanar_ids(){
    if(normals.empty())
      calculate_normals();

    size_t NSElements = get_number_facets();

    // Create Node-Element list
    for(size_t i=0;i<NSElements;i++){
      for(size_t j=0;j<snloc;j++){
        SNEList[SENList[snloc*i+j]].insert(i);
      }
    }

    boundary_ids.resize(NSElements);
    std::fill(boundary_ids.begin(), boundary_ids.end(), 0);

    coplanar_ids.resize(NSElements);
    std::fill(coplanar_ids.begin(), coplanar_ids.end(), 0);

    if(use_bbox){
      for(size_t i=0;i<NSElements;i++){
        if(normals[i*ndims]<-0.9){
          coplanar_ids[i]=1;
        }else if(normals[i*ndims]>0.9){
          coplanar_ids[i]=2;
        }else if(normals[i*ndims+1]<-0.9){
          coplanar_ids[i]=3;
        }else if(normals[i*ndims+1]>0.9){
          coplanar_ids[i]=4;
        }else if(ndims==3){
          if(normals[i*ndims+2]<-0.9){
            coplanar_ids[i]=5;
          }else if(normals[i*ndims+2]>0.9){
            coplanar_ids[i]=6;
          } 
        }
      }

    }else{
      // Create EEList for surface
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
    
      // Grow patches        
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
  std::vector<index_t> SENList;
  std::vector<int> boundary_ids, coplanar_ids;
  std::vector<real_t> normals;
  real_t COPLANAR_MAGIC_NUMBER;
  bool use_bbox;

  Mesh<real_t, index_t> *_mesh;
};
#endif
