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

#ifndef SURFACE2D_H
#define SURFACE2D_H

#include <vector>
#include <set>
#include <map>

#include <errno.h>

#include "Mesh.h"
#include "Edge.h"
#include "PragmaticTypes.h"

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

template<typename real_t>
  class Surface2D{
 public:

  /// Default constructor.
  Surface2D(Mesh<real_t> &mesh){
    _mesh = &mesh;
    nthreads = pragmatic_nthreads();

    size_t NNodes = _mesh->get_number_nodes();
    surface_nodes.resize(NNodes);
    std::fill(surface_nodes.begin(), surface_nodes.end(), false);

    // Default coplanar tolerance.
    set_coplanar_tolerance(0.9999999);

    deferred_operations.resize(nthreads);
    for(size_t i=0; i<(size_t) nthreads; ++i)
      deferred_operations[i].resize(nthreads);
  }

  /// Default destructor.
  ~Surface2D(){
  }

  /// Append a facet to the surface
  void append_facet(const int *facet, int boundary_id, bool check_duplicates=false){
    if(check_duplicates){
      std::set<index_t> Intersection;
      std::set_intersection(SNEList[facet[0]].begin(), SNEList[facet[0]].end(),
                       SNEList[facet[1]].begin(), SNEList[facet[1]].end(),
                       std::inserter(Intersection,Intersection.begin()));
      if(Intersection.size())
        return;
    }

    // This is required because additional vertices may have been added.
    if(surface_nodes.size() < _mesh->NNodes)
      surface_nodes.resize(_mesh->NNodes, 0);

    index_t eid = boundary_ids.size();

    boundary_ids.push_back(boundary_id);
    for(size_t i=0;i<snloc;i++){
      index_t nid = facet[i];
      SNEList[nid].insert(eid);

      surface_nodes[nid] = (char) 1;

      SENList.push_back(nid);
    }
  }

  /// True if surface contains vertex nid.
  bool contains_node(index_t nid) const{
    assert(nid>=0);
    if(nid>=(index_t)surface_nodes.size()){
      return false;
    }

    return surface_nodes[nid] == (char) 0 ? false : true;
  }

  /// Detects the surface nodes of the domain.
  void find_surface(){

    size_t NElements = _mesh->get_number_elements();

    std::map< std::set<index_t>, std::vector<int> > facets;
    for(size_t i=0;i<NElements;i++){
      for(size_t j=0;j<nloc;j++){
        std::set<index_t> facet;
        for(size_t k=1;k<nloc;k++){
          facet.insert(_mesh->_ENList[i*nloc+(j+k)%nloc]);
        }
        typename std::map< std::set<index_t>, std::vector<int> >::iterator target_facet=facets.find(facet);
        if(target_facet!=facets.end()){
          facets.erase(target_facet);
        }else{
          std::vector<int> element;
          element.reserve(2);

          element.push_back(_mesh->_ENList[i*nloc+(j+1)%nloc]);
          element.push_back(_mesh->_ENList[i*nloc+(j+2)%nloc]);

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
        surface_nodes[*jt] = (char) 1;
    }

    // Calculate the boundary representation.
    calculate_brep();
  }

  /// True if node nid is a corner vertex OR if more than ndims boundary labels are incident on the vertex.
  bool is_corner(index_t nid) const{
    typename SNEList_t::const_iterator iSNEList = SNEList.find(nid);
    if(iSNEList==SNEList.end())
      return false;

    std::set<int> incident_plane;
    for(typename std::set<index_t>::const_iterator it=iSNEList->second.begin();it!=iSNEList->second.end();++it){
      incident_plane.insert(boundary_ids[*it]);
    }

    return (incident_plane.size()>=ndims);
  }

  bool is_collapsible(index_t nid_free, index_t nid_target){
    if((nid_free>=(index_t)surface_nodes.size())||(nid_target>=(index_t)surface_nodes.size())){
      surface_nodes.resize(_mesh->NNodes, 0);
    }

    // If nid_free is not on the surface then it's unconstrained.
    if(surface_nodes[nid_free] == (char) 0){
      return true;
    }

    // However, is nid_free is on the surface then nid_target must
    // also lie on a surface for it to be considered for collapse.
    if(surface_nodes[nid_target] == (char) 0){
      return false;
    }

    std::set<int> incident_plane_free;
    typename SNEList_t::const_iterator iSNEList = SNEList.find(nid_free);
    assert(iSNEList!=SNEList.end());
    for(typename std::set<index_t>::const_iterator it=iSNEList->second.begin();it!=iSNEList->second.end();++it){
      incident_plane_free.insert(boundary_ids[*it]);
    }

    // Non-collapsible if nid_free is a corner node.
    if(incident_plane_free.size()>=ndims){
      return false;
    }

    std::set<index_t> Intersection;
    std::set_intersection(SNEList[nid_target].begin(), SNEList[nid_target].end(),
                          SNEList[nid_free].begin(), SNEList[nid_free].end(),
                          std::inserter(Intersection,Intersection.begin()));
    if(Intersection.size()==0)
      return false;

    return true;
  }

  /*! Defragment surface mesh. This compresses the storage of internal data
    structures after the mesh has been defraged.*/
  void defragment(std::vector<index_t> *active_vertex_map){
    size_t NNodes = _mesh->get_number_nodes();
    surface_nodes.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      surface_nodes[i] = (char) 0;

    std::vector<index_t> defrag_SENList;
    std::vector<int> defrag_boundary_ids;
    size_t NSElements = get_number_facets();
    for(size_t i=0;i<NSElements;i++){
      const int *n=get_facet(i);

      // Check if deleted.
      if(n[0]<0)
        continue;

      // Check all vertices are active.
      bool valid_facet=true;
      for(size_t j=0;j<snloc;j++){
        if((*active_vertex_map)[n[j]]<0){
          valid_facet = false;
          break;
        }
      }
      if(!valid_facet)
        continue;

      for(size_t j=0;j<snloc;j++){
        int nid = (*active_vertex_map)[n[j]];
        defrag_SENList.push_back(nid);
        surface_nodes[nid] = (char) 1;
      }
      defrag_boundary_ids.push_back(boundary_ids[i]);
    }
    defrag_SENList.swap(SENList);
    defrag_boundary_ids.swap(boundary_ids);

    // Finally, fix the Node-Element adjacency list.
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

      typename SNEList_t::const_iterator iSNEList = SNEList.find(element[i]);
      if(iSNEList==SNEList.end())
        continue;

      typename SNEList_t::const_iterator jSNEList = SNEList.find(element[(i+1)%nloc]);
      if(jSNEList==SNEList.end())
        continue;

      std::set_intersection(iSNEList->second.begin(), iSNEList->second.end(),
                       jSNEList->second.begin(), jSNEList->second.end(),
                       std::inserter(Intersection,Intersection.begin()));

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

  struct DeferredOperations{
    std::vector<index_t> addSNE; // addSNE -> [i, e] : Add facet e to SNEList[i].
    std::vector<index_t> remSNE; // remNN -> [i, e] : Remove facet e from SNEList[i].
    std::vector<index_t> addNode; // addNE -> [n] : Add node n to SNEList.
    std::vector<index_t> remNode; // remNE -> [n] : Remove node n from SNEList.
  };

  inline void deferred_addSNE(index_t i, index_t e, size_t tid){
    deferred_operations[tid][i % nthreads].addSNE.push_back(i);
    deferred_operations[tid][i % nthreads].addSNE.push_back(e);
  }

  inline void deferred_remSNE(index_t i, index_t e, size_t tid){
    deferred_operations[tid][i % nthreads].remSNE.push_back(i);
    deferred_operations[tid][i % nthreads].remSNE.push_back(e);
  }

  // SNEList is a map. Adding/removing items is not thread safe,
  // so only one thread will perform these operations. Let's say thread 0.
  inline void deferred_addNode(index_t n, size_t tid){
    deferred_operations[tid][0].addNode.push_back(n);
  }

  inline void deferred_remNode(index_t n, size_t tid){
    deferred_operations[tid][0].remNode.push_back(n);
  }

  void commit_deferred(size_t tid){
    for(int i=0; i<nthreads; ++i){
      DeferredOperations& pending = deferred_operations[i][tid];

      // Commit element removals from SNEList sets.
      for(typename std::vector<index_t>::const_iterator it=pending.remSNE.begin(); it!=pending.remSNE.end(); it+=2){
        assert(*it % nthreads == (int) tid);
        SNEList[*it].erase(*(it+1));
      }
      pending.remSNE.clear();

      // Commit element additions to SNEList sets.
      for(typename std::vector<index_t>::const_iterator it=pending.addSNE.begin(); it!=pending.addSNE.end(); it+=2){
        assert(*it % nthreads == (int) tid);
        SNEList[*it].insert(*(it+1));
      }
      pending.addSNE.clear();

      // Commit vertex removals from SNEList

      // Commit vertex additions to SNEList
    }
  }

  void collapse(index_t nid_free, index_t nid_target, size_t tid){
    assert(SNEList.count(nid_free));
    assert(SNEList.count(nid_target));
    assert(is_collapsible(nid_free, nid_target));

    surface_nodes[nid_free] = (char) 0;

    // Find the facet which is about to be deleted.
    std::set<index_t> deleted_facets;
    std::set_intersection(SNEList[nid_free].begin(), SNEList[nid_free].end(),
                     SNEList[nid_target].begin(), SNEList[nid_target].end(),
                     std::inserter(deleted_facets, deleted_facets.begin()));

    // Delete collapsing facet and remove it from target_vertex's adjacency list.
    assert(deleted_facets.size()==1);
    index_t de = *deleted_facets.begin();
    SENList[snloc*de]=-1;
    deferred_remSNE(nid_target, de, tid);

    // Renumber nodes in the other facet adjacent to rm_vertex
    // and make this facet adjacent to target_vertex.
    SNEList[nid_free].erase(de);
    assert(SNEList[nid_free].size()==1);
    index_t ele = *SNEList[nid_free].begin();

    for(size_t i=0;i<snloc;i++){
      if(SENList[snloc*ele+i]==nid_free){
        SENList[snloc*ele+i]=nid_target;
        break;
      }
    }

    // Add facet to target node-element adjacency list.
    deferred_addSNE(nid_target, ele, tid);

    SNEList[nid_free].clear();
    //SNEList.erase(nid_free);
  }

  void refine(std::vector< std::vector<DirectedEdge<index_t> > > surfaceEdges){
    unsigned int nthreads = pragmatic_nthreads();

    // Given the refined edges, refine facets.
    std::vector< std::vector<index_t> > private_SENList(nthreads);
    std::vector< std::vector<int> > private_boundary_ids(nthreads);
    std::vector<unsigned int> threadIdx(nthreads), splitCnt(nthreads);
    std::vector< DirectedEdge<index_t> > refined_edges;

#pragma omp parallel
    {
      const int tid = pragmatic_thread_id();
      splitCnt[tid] = 0;

      // Serialise surfaceEdges
      threadIdx[tid] = 0;
      for(int id=0; id<tid; ++id)
        threadIdx[tid] += surfaceEdges[id].size();

#pragma omp barrier
#pragma omp single
      {
        refined_edges.resize(threadIdx[nthreads-1] + surfaceEdges[nthreads-1].size());
      }

      memcpy(&refined_edges[threadIdx[tid]], &surfaceEdges[tid][0], surfaceEdges[tid].size()*sizeof(DirectedEdge<index_t>));

#pragma omp barrier
#pragma omp for schedule(static)
      for(index_t i=0;i < (index_t)refined_edges.size();++i){
        index_t v1 = refined_edges[i].edge.first;
        index_t v2 = refined_edges[i].edge.second;

        /* Check whether the split edge is not on the surface, despite the fact
         * that both vertices are on the surface. This is the case of an edge
         * opposite a mesh corner. We can detect such edges from the fact that
         * the intersection of SNEList[v1] and SNEList[v2] is empty. If v1 and
         * v2 are on the same plane, then they have one facet in common.
         */
        std::set<index_t> intersection;
        std::set_intersection(SNEList[v1].begin(), SNEList[v1].end(), SNEList[v2].begin(),
            SNEList[v2].end(), std::inserter(intersection, intersection.begin()));

        if(intersection.size() != 1)
          continue;

        index_t seid = *intersection.begin();
        int *n=&(SENList[seid*snloc]);
        index_t newVertex = refined_edges[i].id;

        // Renumber existing facet and add the new one.
        index_t cache_n1 = n[1];
        n[1] = newVertex;

        private_SENList[tid].push_back(newVertex);
        private_SENList[tid].push_back(cache_n1);

        private_boundary_ids[tid].push_back(boundary_ids[i]);

        splitCnt[tid]++;
      }

      // Perform prefix sum to find (for each OMP thread) the starting position
      // in SENList at which new elements should be appended.
      threadIdx[tid] = 0;
      for(int id=0; id<tid; ++id)
        threadIdx[tid] += splitCnt[id];

      threadIdx[tid] += get_number_facets();

#pragma omp barrier

      // Resize mesh containers
#pragma omp master
      {
      	const int newSize = threadIdx[nthreads - 1] + splitCnt[nthreads - 1];

      	SENList.resize(snloc*newSize);
        boundary_ids.resize(newSize);
      }
#pragma omp barrier

      // Append new elements to the surface
      memcpy(&SENList[snloc*threadIdx[tid]], &private_SENList[tid][0], snloc*splitCnt[tid]*sizeof(index_t));
      memcpy(&boundary_ids[threadIdx[tid]], &private_boundary_ids[tid][0], splitCnt[tid]*sizeof(int));
    }

    size_t NNodes = _mesh->get_number_nodes();
    size_t NSElements = get_number_facets();

    SNEList.clear();
    surface_nodes.resize(NNodes);
    std::fill(surface_nodes.begin(), surface_nodes.end(), (char) 0);

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
    return ndims;
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

  std::set<index_t> get_surface_patch(int nid) const{
    assert(SNEList.find(nid)!=SNEList.end());
    return SNEList.find(nid)->second;
  }

  /// Set dot product tolerence - used to decide if elements are co-planar
  void set_coplanar_tolerance(real_t tol){
    COPLANAR_MAGIC_NUMBER = tol;
  }

  /// Set surface.
  void set_surface(int NSElements, const int *senlist, const int *boundary_ids_){
    boundary_ids.resize(NSElements);
    SENList.resize(NSElements*snloc);
    surface_nodes.resize(_mesh->get_number_nodes());
    std::fill(surface_nodes.begin(), surface_nodes.end(), (char) 0);
    SNEList.clear();
    for(int i=0;i<NSElements;i++){
      boundary_ids[i] = boundary_ids_[i];
      const int *n = senlist+i*snloc;
      for(size_t j=0;j<snloc;j++){
        SNEList[n[j]].insert(i);
        surface_nodes[n[j]] = (char) 1;

        SENList[i*snloc+j] = n[j];
      }
    }
  }

 private:
  template<typename _real_t> friend class VTKTools;
  template<typename _real_t> friend class CUDATools;

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

  /// Calculate BRep.
  void calculate_brep(){
    size_t NSElements = get_number_facets();
    std::vector<real_t> normals(NSElements*2);

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(size_t i=0;i<NSElements;i++)
        calculate_normal_2d(&(SENList[2*i]), &(normals[i*2]));
    }

    // Create Node-Element list
    for(size_t i=0;i<NSElements;i++){
      for(size_t j=0;j<snloc;j++){
        SNEList[SENList[snloc*i+j]].insert(i);
      }
    }

    boundary_ids.resize(NSElements);
    std::fill(boundary_ids.begin(), boundary_ids.end(), 0);

    // Create EEList for surface
    std::vector<int> EEList(NSElements*snloc);
    for(size_t i=0;i<NSElements;i++){
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
    }

    // Grow patches
    int current_id = 1;
    for(size_t pos = 0;pos<NSElements;){
      // Create a new starting point
      const real_t *ref_normal=NULL;
      for(size_t i=pos;i<NSElements;i++){
        if(boundary_ids[i]==0){
          // This is the first element in the new patch
          pos = i;
          boundary_ids[pos] = current_id;
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
          if(boundary_ids[sele2]>0)
            continue;

          double coplanar = 0.0;
          for(size_t d=0;d<ndims;d++)
            coplanar += ref_normal[d]*normals[sele2*ndims+d];

          if(coplanar>=COPLANAR_MAGIC_NUMBER){
            front.insert(sele2);
            boundary_ids[sele2] = current_id;
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

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t snloc=2;

  SNEList_t SNEList;
  std::vector<char> surface_nodes;
  std::vector<index_t> SENList;
  std::vector<int> boundary_ids;
  real_t COPLANAR_MAGIC_NUMBER;
  bool use_bbox;

  std::vector< std::vector<DeferredOperations> > deferred_operations;

  Mesh<real_t> *_mesh;
  int nthreads;
};

#endif
