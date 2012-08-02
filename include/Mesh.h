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

#ifndef MESH_H
#define MESH_H

#include "pragmatic_config.h"

#include <algorithm>
#include <deque>
#include <vector>
#include <set>
#include <stack>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "Metis.h"
#include "ElementProperty.h"
#include "MetricTensor.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t, typename index_t> class Mesh{
 public:

  /*! 2D triangular mesh constructor. This is for use when there is no MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList, const real_t *x, const real_t *y){
#ifdef HAVE_MPI
    _mpi_comm = MPI_COMM_WORLD;      
#endif
    _init(NNodes, NElements, ENList, x, y, NULL, NULL, NULL);
  }
  
#ifdef HAVE_MPI  
  /*! 2D triangular mesh constructor. This is used when parallelised with MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param lnn2gnn mapping of local node numbering to global node numbering.
   * @param owner_range range of node id's owned by each partition.
   * @param mpi_comm the mpi communicator.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const index_t *lnn2gnn,
       const index_t *owner_range, MPI_Comm mpi_comm){
    _mpi_comm = mpi_comm;
    _init(NNodes, NElements, ENList, x, y, NULL, lnn2gnn, owner_range);
  }
#endif

  /*! 3D tetrahedra mesh constructor. This is for use when there is no MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const real_t *z){
#ifdef HAVE_MPI
      _mpi_comm = MPI_COMM_WORLD;      
#endif
    _init(NNodes, NElements, ENList, x, y, z, NULL, NULL);
  }

#ifdef HAVE_MPI
  /*! 3D tetrahedra mesh constructor. This is used when parallelised with MPI.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   * @param lnn2gnn mapping of local node numbering to global node numbering.
   * @param owner_range range of node id's owned by each partition.
   * @param mpi_comm the mpi communicator.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const real_t *z, const index_t *lnn2gnn,
       const index_t *owner_range, MPI_Comm mpi_comm){
    _mpi_comm = mpi_comm;
    _init(NNodes, NElements, ENList, x, y, z, lnn2gnn, owner_range);
  }
#endif

  /*! Defragment mesh. This compresses the storage of internal data
    structures. This is useful if the mesh has been significently
    coarsened. */
  void defragment(std::map<index_t, index_t> *active_vertex_map=NULL){
    // Discover which verticies and elements are active.
    bool local_active_vertex_map=(active_vertex_map==NULL);
    if(local_active_vertex_map)
      active_vertex_map = new std::map<index_t, index_t>;

    // Create look-up tables for halos.
    std::map<index_t, std::set<int> > send_set, recv_set;
    if(num_processes>1){
      for(int k=0;k<num_processes;k++){
        for(std::vector<int>::iterator jt=send[k].begin();jt!=send[k].end();++jt)
          send_set[k].insert(*jt);
        for(std::vector<int>::iterator jt=recv[k].begin();jt!=recv[k].end();++jt)
          recv_set[k].insert(*jt);
      }
    }

    // Identify active vertices and elements.
    std::deque<index_t> active_vertex, active_element;
    std::map<index_t, std::set<int> > new_send_set, new_recv_set;
    size_t NElements = get_number_elements();
    for(size_t e=0;e<NElements;e++){
      index_t nid = _ENList[e*nloc];
      
      // Check if deleted.
      if(nid<0)
        continue;

      // Check if wholly owned by another process or if halo node.
      bool local=false, halo_element=false;
      for(size_t j=0;j<nloc;j++){
        nid = _ENList[e*nloc+j];
        if(recv_halo.count(nid)){
          halo_element = true;
        }else{
          local = true;
        }
      }
      if(!local)
        continue;
      
      // Need these mesh entities.
      active_element.push_back(e);

      std::set<int> neigh;
      for(size_t j=0;j<nloc;j++){
        nid = _ENList[e*nloc+j];
        (*active_vertex_map)[nid]=0;
      
        if(halo_element){
          for(int k=0;k<num_processes;k++){
            if(recv_set[k].count(nid)){
              new_recv_set[k].insert(nid);
              neigh.insert(k);
            }
          }
        }
      }
      for(size_t j=0;j<nloc;j++){
        nid = _ENList[e*nloc+j];
        for(std::set<int>::iterator kt=neigh.begin();kt!=neigh.end();++kt){
          if(send_set[*kt].count(nid))
            new_send_set[*kt].insert(nid);
        }
      }
    }

    // Create a new numbering.
    index_t cnt=0;
    for(typename std::map<index_t, index_t>::iterator it=active_vertex_map->begin();it!=active_vertex_map->end();++it){
      it->second = cnt++;
      active_vertex.push_back(it->first);
    }
    
    int metis_nnodes = active_vertex.size();
    int metis_nelements = active_element.size();
    std::vector< std::set<index_t> > graph(metis_nnodes);
    for(typename std::deque<index_t>::iterator ie=active_element.begin();ie!=active_element.end();++ie){
      for(size_t i=0;i<nloc;i++){
        index_t nid0 = (*active_vertex_map)[_ENList[(*ie)*nloc+i]];
        for(size_t j=i+1;j<nloc;j++){
          index_t nid1 = (*active_vertex_map)[_ENList[(*ie)*nloc+j]];
          graph[nid0].insert(nid1);
          graph[nid1].insert(nid0);
        }
      }
    }
    
    // Compress graph
    std::vector<idxtype> xadj(metis_nnodes+1), adjncy;
    int pos=0;
    xadj[0]=0;
    for(int i=0;i<metis_nnodes;i++){
      for(typename std::set<index_t>::const_iterator jt=graph[i].begin();jt!=graph[i].end();jt++){
        assert((*jt)>=0);
        assert((*jt)<metis_nnodes);
        adjncy.push_back(*jt);
        pos++;
      }
      xadj[i+1] = pos;
    }
    
    std::vector<int> norder(metis_nnodes);
    std::vector<int> inorder(metis_nnodes);
    int numflag=0, options[] = {0};
    
    METIS_NodeND(&metis_nnodes, &(xadj[0]), &(adjncy[0]), &numflag, options, &(norder[0]), &(inorder[0]));
    
    std::vector<index_t> metis_vertex_renumber(metis_nnodes);
    for(int i=0;i<metis_nnodes;i++){
      metis_vertex_renumber[i] = inorder[i];
    }

    // Update active_vertex_map
    for(typename std::map<index_t, index_t>::iterator it=active_vertex_map->begin();it!=active_vertex_map->end();++it){
      it->second = metis_vertex_renumber[it->second];
    }
    
    // Renumber elements
    std::map< std::set<index_t>, index_t > ordered_elements;
    for(int i=0;i<metis_nelements;i++){
      index_t old_eid = active_element[i];
      std::set<index_t> sorted_element;
      for(size_t j=0;j<nloc;j++){
        index_t new_nid = (*active_vertex_map)[_ENList[old_eid*nloc+j]];
        sorted_element.insert(new_nid);
      }
      assert(ordered_elements.find(sorted_element)==ordered_elements.end());
      ordered_elements[sorted_element] = old_eid;
    }
    std::vector<index_t> metis_element_renumber;
    metis_element_renumber.reserve(metis_nelements);
    for(typename std::map< std::set<index_t>, index_t >::const_iterator it=ordered_elements.begin();it!=ordered_elements.end();++it){
      metis_element_renumber.push_back(it->second);
    }
    
    // end of renumbering
    
    // Compress data structures.
    index_t NNodes = active_vertex.size();
    NElements = active_element.size();

    std::vector<index_t> defrag_ENList(NElements*nloc);
    std::vector<real_t> defrag_coords(NNodes*ndims);
    std::vector<float> defrag_metric(NNodes*msize);

    assert(NElements==(size_t)metis_nelements);

    // This first touch is to bind memory locally.
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)NElements;i++){
        for(size_t j=0;j<nloc;j++){
          defrag_ENList[i*nloc+j] = 0;
        }
      }

#pragma omp for schedule(static)
      for(int i=0;i<(int)NNodes;i++){
        for(size_t j=0;j<ndims;j++)
          defrag_coords[i*ndims+j] = 0.0;
        for(size_t j=0;j<msize;j++)
          defrag_metric[i*msize+j] = 0.0;
      }
    }
    
    // Second sweep writes elements with new numbering.
    for(int i=0;i<metis_nelements;i++){
      index_t old_eid = metis_element_renumber[i];
      index_t new_eid = i;
      for(size_t j=0;j<nloc;j++){
        index_t new_nid = (*active_vertex_map)[_ENList[old_eid*nloc+j]];
        assert(new_nid<NNodes);
        defrag_ENList[new_eid*nloc+j] = new_nid;
      }
    }

    // Second sweep writes node wata with new numbering.
    for(typename std::map<index_t, index_t>::iterator it=active_vertex_map->begin();it!=active_vertex_map->end();++it){
      index_t old_nid = it->first;
      index_t new_nid = it->second;
      
      for(size_t j=0;j<ndims;j++)
        defrag_coords[new_nid*ndims+j] = _coords[old_nid*ndims+j];
      for(size_t j=0;j<msize;j++)
        defrag_metric[new_nid*msize+j] = metric[old_nid*msize+j];
    }

    _ENList.swap(defrag_ENList);
    _coords.swap(defrag_coords);
    metric.swap(defrag_metric);
    
    // Renumber halo
    if(num_processes>1){
      for(int k=0;k<num_processes;k++){
        std::vector<int> new_halo;
        for(std::vector<int>::iterator jt=send[k].begin();jt!=send[k].end();++jt){
          if(new_send_set[k].count(*jt))
            new_halo.push_back((*active_vertex_map)[*jt]);
        }
        send[k].swap(new_halo);
      }
      for(int k=0;k<num_processes;k++){
        std::vector<int> new_halo;
        for(std::vector<int>::iterator jt=recv[k].begin();jt!=recv[k].end();++jt){
          if(new_recv_set[k].count(*jt))
            new_halo.push_back((*active_vertex_map)[*jt]);
        }
        recv[k].swap(new_halo);
      }
      
      {
        send_halo.clear();
        for(int k=0;k<num_processes;k++){
          for(std::vector<int>::iterator jt=send[k].begin();jt!=send[k].end();++jt){
            send_halo.insert(*jt);
          }
        }
      }    
      {
        recv_halo.clear();
        for(int k=0;k<num_processes;k++){
          for(std::vector<int>::iterator jt=recv[k].begin();jt!=recv[k].end();++jt){
            recv_halo.insert(*jt);
          }
        }
      }
    }

    create_adjancy();

    if(local_active_vertex_map)
      delete active_vertex_map;
  }

  /// Add a new vertex
  index_t append_vertex(const real_t *x, const float *m){
    for(size_t i=0;i<ndims;i++)
      _coords.push_back(x[i]);
    
    for(size_t i=0;i<msize;i++)
      metric.push_back(m[i]);
    
    NEList.push_back(std::set<index_t>());
    NNList.push_back(std::deque<index_t>());
      
    return get_number_nodes()-1;
  }

  /// Erase a vertex
  void erase_vertex(const index_t nid){
    NNList[nid].clear();
    NEList[nid].clear();
  }

  /// Add a new element
  index_t append_element(const int *n){
    for(size_t i=0;i<nloc;i++)
      _ENList.push_back(n[i]);
    
    return get_number_elements()-1;
  }

  /// Erase an element
  void erase_element(const index_t eid){
    _ENList[eid*nloc] = -1;
    // Something for NEList?
  }

  /// Return the number of nodes in the mesh.
  int get_number_nodes() const{
    return _coords.size()/ndims;
  }

  /// Return the number of elements in the mesh.
  int get_number_elements() const{
    return _ENList.size()/nloc;
  }

  /// Return the number of spatial dimensions.
  int get_number_dimensions() const{
    return ndims;
  }

  /// Get the mean edge length metric space.
  real_t get_lmean(){
    int NNodes = get_number_nodes();
    double total_length=0;
    int nedges=0;
#pragma omp parallel reduction(+:total_length,nedges)
    {
#pragma omp for schedule(static)
      for(int i=0;i<NNodes;i++){
        if(is_owned_node(i) && (NNList[i].size()>0))
          for(typename std::deque<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
            if(i<*it){ // Ensure that every edge length is only calculated once. 
              total_length += calc_edge_length(i, *it);
              nedges++;
            }
          }
      }
    }
    
#ifdef HAVE_MPI
    if(num_processes>1){
      MPI_Allreduce(MPI_IN_PLACE, &total_length, 1, MPI_DOUBLE, MPI_SUM, _mpi_comm);
      MPI_Allreduce(MPI_IN_PLACE, &nedges, 1, MPI_INT, MPI_SUM, _mpi_comm);
    }
#endif
    
    double mean = total_length/nedges;
    
    return mean;
  }

  /// Get the edge length RMS value in metric space.
  real_t get_lrms(){
    double mean = get_lmean();
    int NNodes = get_number_nodes();

    double rms=0;
    int nedges=0;
#pragma omp parallel reduction(+:rms,nedges)
    {
#pragma omp for schedule(static)
      for(int i=0;i<NNodes;i++){
        if(is_owned_node(i) && (NNList[i].size()>0))
          for(typename std::deque<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
            if(i<*it){ // Ensure that every edge length is only calculated once. 
              rms+=pow(calc_edge_length(i, *it) - mean, 2);
              nedges++;
            }
          }
      }
    }

#ifdef HAVE_MPI
    if(num_processes>1){
      MPI_Allreduce(MPI_IN_PLACE, &rms, 1, MPI_DOUBLE, MPI_SUM, _mpi_comm);
      MPI_Allreduce(MPI_IN_PLACE, &nedges, 1, MPI_INT, MPI_SUM, _mpi_comm);
    }
#endif
    
    rms = sqrt(rms/nedges);
    
    return rms;
  }

  /// Get the element mean quality in metric space.
  real_t get_qmean() const{
    int NElements = get_number_elements();
    real_t sum=0;
    int nele=0;
    
#pragma omp parallel reduction(+:sum, nele)
    {
#pragma omp for schedule(static)
      for(int i=0;i<NElements;i++){
        const index_t *n=get_element(i);
        if(n[0]<0)
          continue;
        
        if(ndims==2){
          sum += property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), 
                                    get_metric(n[0]), get_metric(n[1]), get_metric(n[2]));
        }else{
          sum += property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]), 
                                    get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3]));
        }
        nele++;
      }
    }

#ifdef HAVE_MPI
    if(num_processes>1){
      MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, _mpi_comm);
      MPI_Allreduce(MPI_IN_PLACE, &nele, 1, MPI_INT, MPI_SUM, _mpi_comm);
    }
#endif

    double mean = sum/nele;

    return mean;
  }

  /// Get the element minimum quality in metric space.
  real_t get_qmin() const{
    size_t NElements = get_number_elements();
    float qmin=1; // Where 1 is ideal.

    // This is not urgent - but when OpenMP 3.1 is more common we
    // should stick a proper min reduction in here.
    for(size_t i=0;i<NElements;i++){
      const index_t *n=get_element(i);
      if(n[0]<0)
        continue;
      
      if(ndims==2)
        qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), 
                                                 get_metric(n[0]), get_metric(n[1]), get_metric(n[2])));
      else
        qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]), 
                                                 get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3])));
    }
#ifdef HAVE_MPI
    if(num_processes>1){
      MPI_Allreduce(MPI_IN_PLACE, &qmin, 1, MPI_DOUBLE, MPI_MIN, _mpi_comm);
    }
#endif
    return qmin;
  }

  /// Get the element quality RMS value in metric space, where the ideal element has a value of unity.
  real_t get_qrms() const{
    size_t NElements = get_number_elements();
    double mean = get_qmean();

    real_t rms=0;
    int nele=0;
    for(size_t i=0;i<NElements;i++){
      const index_t *n=get_element(i);
      if(n[0]<0)
        continue;

      real_t q;
      if(ndims==2)
        q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), 
                               get_metric(n[0]), get_metric(n[1]), get_metric(n[2]));
      else
        q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]), 
                               get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3]));

      rms += pow(q-mean, 2);
      nele++;
    }
    rms = sqrt(rms/nele);

    return rms;
  }

#ifdef HAVE_MPI
  /// Return the MPI communicator.
  MPI_Comm get_mpi_comm() const{
    return _mpi_comm;
  }
#endif

  /// Return a pointer to the element-node list.
  const int *get_element(size_t eid) const{
    return &(_ENList[eid*nloc]);
  }

  /// Return copy of element-node list.
  void get_element(size_t eid, int *ele) const{
    for(size_t i=0;i<nloc;i++)
      ele[i] = _ENList[eid*nloc+i];
    return;
  }

  /// Return the node id's connected to the specified node_id
  std::set<index_t> get_node_patch(index_t nid) const{
    assert(nid<(index_t)NNList.size());
    std::set<index_t> patch;
    for(typename std::deque<index_t>::const_iterator it=NNList[nid].begin();it!=NNList[nid].end();++it)
      patch.insert(patch.end(), *it);
    return patch;
  }

  /// Grow a node patch around node id's until it reaches a minimum size.
  std::set<index_t> get_node_patch(index_t nid, size_t min_patch_size){
    std::set<index_t> patch = get_node_patch(nid);
    size_t NNodes = get_number_nodes();

    if(patch.size()<min_patch_size){
      std::set<index_t> front = patch, new_front;
      for(;;){
        for(typename std::set<index_t>::const_iterator it=front.begin();it!=front.end();it++){
          for(typename std::deque<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();jt++){
            if(patch.find(*jt)==patch.end()){
              new_front.insert(*jt);
              patch.insert(*jt);
            }
          }
        }
        
        if(patch.size()>=std::min(min_patch_size, NNodes))
          break;
        
        front.swap(new_front);
      }
    }
    
    return patch;
  }

  /// Return positions vector.
  const real_t *get_coords(size_t nid) const{
    return &(_coords[nid*ndims]);
  }

  /// Return copy of the coordinate.
  void get_coords(size_t nid, real_t *x) const{
    for(size_t i=0;i<ndims;i++)
      x[i] = _coords[nid*ndims+i];
    return;
  }

  /// Return metric at that vertex.
  const float *get_metric(size_t nid) const{
    assert(metric.size()>0);
    return &(metric[nid*msize]);
  }

  /// Return copy of metric.
  void get_metric(size_t nid, float *m) const{
    assert(metric.size()>0);
    for(size_t i=0;i<msize;i++)
      m[i] = metric[nid*msize+i];
    return;
  }

  /// Returns true if the node is in any of the partitioned elements.
  bool is_halo_node(int nid) const{
    return (send_halo.count(nid)+recv_halo.count(nid))>0;
  }

  /// Flip orientation of element.
  void invert_element(size_t eid){
    int tmp = _ENList[eid*nloc];
    _ENList[eid*nloc] = _ENList[eid*nloc+1];
    _ENList[eid*nloc+1] = tmp;
  }

  /// Returns true if the node is assigned to the local partition.
  bool is_owned_node(int nid) const{
    return recv_halo.count(nid)==0;
  }

  /// Default destructor.
  ~Mesh(){
    delete property;
  }

  /// Calculates the edge lengths in metric space.
  real_t calc_edge_length(index_t nid0, index_t nid1) const{
    real_t length=-1.0;
    if(ndims==2){
      float m[3];
      m[0] = (metric[nid0*msize]+metric[nid1*msize])*0.5;
      m[1] = (metric[nid0*msize+1]+metric[nid1*msize+1])*0.5;
      m[2] = (metric[nid0*msize+2]+metric[nid1*msize+2])*0.5;

      length = ElementProperty<real_t>::length2d(get_coords(nid0), get_coords(nid1), m);
    }else{
      float m[6];
      m[0] = (metric[nid0*msize  ]+metric[nid1*msize  ])*0.5;
      m[1] = (metric[nid0*msize+1]+metric[nid1*msize+1])*0.5;
      m[2] = (metric[nid0*msize+2]+metric[nid1*msize+2])*0.5;
      
      m[3] = (metric[nid0*msize+3]+metric[nid1*msize+3])*0.5;
      m[4] = (metric[nid0*msize+4]+metric[nid1*msize+4])*0.5;
      
      m[5] = (metric[nid0*msize+5]+metric[nid1*msize+5])*0.5;
      
      length = ElementProperty<real_t>::length3d(get_coords(nid0), get_coords(nid1), m);
    }
    return length;
  }
  
  real_t maximal_edge_length(){
    double L_max = 0;
    index_t NNodes = get_number_nodes();

    for(index_t i=0;i<NNodes;i++){
      if(is_owned_node(i) && (NNList[i].size()>0))
        for(typename std::deque<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
          if(i<*it){ // Ensure that every edge length is only calculated once. 
            L_max = std::max(L_max, calc_edge_length(i, *it));
          }
        }
    }
    
#ifdef HAVE_MPI
    if(num_processes>1)
      MPI_Allreduce(MPI_IN_PLACE, &L_max, 1, MPI_DOUBLE, MPI_MAX, _mpi_comm);
#endif

    return L_max;
  }
  
  /// This is used to verify that the mesh and its metadata is correct.
  bool verify() const{
    bool state = true;

    size_t NElements = get_number_elements();
    size_t NNodes = get_number_nodes();

    std::vector<int> mpi_node_owner(NNodes, rank);
    if(num_processes>1)
      for(int p=0;p<num_processes;p++)
        for(std::vector<int>::const_iterator it=recv[p].begin();it!=recv[p].end();++it){
          mpi_node_owner[*it] = p;
        }
    std::vector<int> mpi_ele_owner(NElements, rank);
    if(num_processes>1)
      for(size_t i=0;i<NElements;i++){
        if(_ENList[i*nloc]<0)
          continue;
        int owner = mpi_node_owner[_ENList[i*nloc]];
        for(size_t j=1;j<nloc;j++)
          owner = std::min(owner, mpi_node_owner[_ENList[i*nloc+j]]);
        mpi_ele_owner[i] = owner;
      }
    
    // Check for the correctness of NNList and NEList.
    std::vector< std::set<index_t> > local_NEList(NNodes);
    std::vector< std::set<index_t> > local_NNList(NNodes);
    for(size_t i=0; i<NElements; i++){
      if(_ENList[i*nloc]<0)
        continue;
      
      for(size_t j=0;j<nloc;j++){
        index_t nid_j = _ENList[i*nloc+j];

        local_NEList[nid_j].insert(i);
        for(size_t k=j+1;k<nloc;k++){
          index_t nid_k = _ENList[i*nloc+k];
          local_NNList[nid_j].insert(nid_k);
          local_NNList[nid_k].insert(nid_j);
        }
      }
    }
    {
      if(rank==0) std::cout<<"VERIFY: NNList..................";
      if(NNList.size()==0){
        if(rank==0) std::cout<<"empty\n";
      }else{
        bool valid_nnlist=true;
        for(size_t i=0;i<NNodes;i++){
          size_t active_cnt=0;
          for(size_t j=0;j<NNList[i].size();j++){
            if(NNList[i][j]>=0)
              active_cnt++;
          }
          if(active_cnt!=local_NNList[i].size()){
            valid_nnlist=false;

            std::cerr<<"active_cnt!=local_NNList[i].size() "<<active_cnt<<", "<<local_NNList[i].size()<<std::endl;
            std::cerr<<"NNList["
                     <<i
                     <<"("
                     <<get_coords(i)[0]<<", "
                     <<get_coords(i)[1]<<", ";
            if(ndims==3) std::cerr<<get_coords(i)[2]<<", ";
            std::cerr<<send_halo.count(i)<<", "<<recv_halo.count(i)<<")] =       ";
            for(size_t j=0;j<NNList[i].size();j++)
              std::cerr<<NNList[i][j]<<"("<<NNList[NNList[i][j]].size()<<", "
                       <<send_halo.count(NNList[i][j])<<", "
                       <<recv_halo.count(NNList[i][j])<<") ";
            std::cerr<<std::endl;
            std::cerr<<"local_NNList["<<i<<"] = ";
            for(typename std::set<index_t>::iterator kt=local_NNList[i].begin();kt!=local_NNList[i].end();++kt)
              std::cerr<<*kt<<" ";
            std::cerr<<std::endl;
            
            state = false;
          }
        }
        if(rank==0){
          if(valid_nnlist){
            std::cout<<"pass\n";
          }else{
            state = false;
            std::cout<<"warn\n";
          }
        }
      }
    }
    {
      if(rank==0) std::cout<<"VERIFY: NEList..................";
      std::string result="pass\n";
      if(NEList.size()==0){
        result = "empty\n";
      }else{
        if(NEList.size()!=local_NEList.size()){
          result = "fail (NEList.size()!=local_NEList.size())\n";
          state = false;
        }else{
          for(size_t i=0;i<NNodes;i++){
            if(local_NEList[i].size()!=NEList[i].size()){
              result = "fail (NEList[i].size()!=local_NEList[i].size())\n";
              state = false;
              break;
            }
            if(local_NEList[i].size()==0)
              continue;
            if(local_NEList[i]!=NEList[i]){
              result = "fail (local_NEList[i]!=NEList[i])\n";
              state = false;
              break;
            }
          }
        }
      }
      if(rank==0) std::cout<<result;
    }
    if(ndims==2){
      real_t area=0, min_ele_area=0, max_ele_area=0;
      size_t i=0;
      for(;i<NElements;i++){
        const index_t *n=get_element(i);
        if((mpi_ele_owner[i]!=rank) || (n[0]<0))
          continue;
        
        area = property->area(get_coords(n[0]),
                              get_coords(n[1]),
                              get_coords(n[2]));
        min_ele_area = area;
        max_ele_area = area;
        i++; break;
      }
      for(;i<NElements;i++){
        const index_t *n=get_element(i);
        if((mpi_ele_owner[i]!=rank) || (n[0]<0))
          continue;
        
        real_t larea = property->area(get_coords(n[0]),
                                      get_coords(n[1]),
                                      get_coords(n[2]));
        if(isnan(larea)){
          std::cerr<<"ERROR: Bad element "<<n[0]<<", "<<n[1]<<", "<<n[2]<<std::endl;
        }

        area += larea;
        min_ele_area = std::min(min_ele_area, larea);
        max_ele_area = std::max(max_ele_area, larea);
      }

      if(MPI::Is_initialized()){
        MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, get_mpi_comm());
        MPI_Allreduce(MPI_IN_PLACE, &min_ele_area, 1, MPI_DOUBLE, MPI_MIN, get_mpi_comm());
        MPI_Allreduce(MPI_IN_PLACE, &max_ele_area, 1, MPI_DOUBLE, MPI_MAX, get_mpi_comm());
      }

      if(rank==0){
        std::cout<<"VERIFY: total area  ............"<<area<<std::endl;
        if(fabs(area-1.0)>1.0e-6)
          state = false;
        std::cout<<"VERIFY: minimum element area...."<<min_ele_area<<std::endl;
        std::cout<<"VERIFY: maximum element area...."<<max_ele_area<<std::endl;
      }
    }else{
      real_t volume=0, min_ele_vol=0, max_ele_vol=0;
      size_t i=0;
      for(;i<NElements;i++){
        const index_t *n=get_element(i);
        if((mpi_ele_owner[i]!=rank) || (n[0]<0))
          continue;
        
        volume = property->volume(get_coords(n[0]),
                                  get_coords(n[1]),
                                  get_coords(n[2]),
                                  get_coords(n[3]));
        min_ele_vol = volume;
        max_ele_vol = volume;
        i++; break;
      }
      for(;i<NElements;i++){
        const index_t *n=get_element(i);
        if((mpi_ele_owner[i]!=rank) || (n[0]<0))
          continue;
        
        real_t lvolume = property->volume(get_coords(n[0]),
                                          get_coords(n[1]),
                                          get_coords(n[2]),
                                          get_coords(n[3]));
        volume += lvolume;
        min_ele_vol = std::min(min_ele_vol, lvolume);
        max_ele_vol = std::max(max_ele_vol, lvolume);
      }
      if(MPI::Is_initialized()){
        MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, get_mpi_comm());
        MPI_Allreduce(MPI_IN_PLACE, &min_ele_vol, 1, MPI_DOUBLE, MPI_MIN, get_mpi_comm());
        MPI_Allreduce(MPI_IN_PLACE, &max_ele_vol, 1, MPI_DOUBLE, MPI_MAX, get_mpi_comm());
      }

      if(rank==0){
        std::cout<<"VERIFY: total volume.............."<<volume<<std::endl;
        std::cout<<"VERIFY: minimum element volume...."<<min_ele_vol<<std::endl;
        std::cout<<"VERIFY: maximum element volume...."<<max_ele_vol<<std::endl;
      }
    }
    double qmean = get_qmean();
    double qmin = get_qmin();
    double qrms = get_qrms();
    if(rank==0){
      std::cout<<"VERIFY: mean quality...."<<qmean<<std::endl;
      std::cout<<"VERIFY: min quality...."<<qmin<<std::endl;
      std::cout<<"VERIFY: rms quality...."<<qrms<<std::endl;
    }

#ifdef HAVE_MPI
    int false_cnt = state?0:1;
    if(num_processes>1){
      MPI_Allreduce(MPI_IN_PLACE, &false_cnt, 1, MPI_INT, MPI_SUM, _mpi_comm);
    }
    state = (false_cnt == 0);
#endif

    return state;
  }

  void get_global_node_numbering(std::vector<int>& NPNodes, std::vector<int> &lnn2gnn){
    int NNodes = get_number_nodes();
    
    NPNodes.resize(num_processes);    
    if(num_processes>1){
      NPNodes[rank] = NNodes - recv_halo.size();
      
      // Allgather NPNodes
      MPI_Allgather(&(NPNodes[rank]), 1, MPI_INT, &(NPNodes[0]), 1, MPI_INT, get_mpi_comm());
      
      // Calculate the global numbering offset for this partition.
      int gnn_offset=0;
      for(int i=0;i<rank;i++)
        gnn_offset+=NPNodes[i];

      // Write global node numbering.
      lnn2gnn.resize(NNodes);
      for(int i=0;i<NNodes;i++){
        if(recv_halo.count(i)){
          lnn2gnn[i] = 0;
        }else{
          lnn2gnn[i] = gnn_offset++;
        }
      }
      
      // Update GNN's for the halo nodes.
      halo_update(&(lnn2gnn[0]), 1);
    }else{
      NPNodes[0] = NNodes;
      lnn2gnn.resize(NNodes);
      for(int i=0;i<NNodes;i++){
        lnn2gnn[i] = i;
      }
    }
  }

 private:
  template<typename _real_t, typename _index_t> friend class MetricField2D;
  template<typename _real_t, typename _index_t> friend class MetricField3D;
  template<typename _real_t, typename _index_t> friend class Smooth;
  template<typename _real_t, typename _index_t> friend class Swapping;
  template<typename _real_t, typename _index_t> friend class Coarsen;
  template<typename _real_t, typename _index_t> friend class Refine;
  template<typename _real_t, typename _index_t> friend class Surface;
  template<typename _real_t, typename _index_t> friend class VTKTools;
  template<typename _real_t, typename _index_t> friend class CUDATools;

  void _init(int NNodes, int NElements, const index_t *globalENList,
             const real_t *x, const real_t *y, const real_t *z,
             const index_t *lnn2gnn, const index_t *owner_range){
    num_processes = 1;
    rank=0;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_size(_mpi_comm, &num_processes);
      MPI_Comm_rank(_mpi_comm, &rank);
    }
#endif

    num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    num_uma = num_threads; // Assume the worst
#ifdef HAVE_LIBNUMA
    num_uma = numa_max_node()+1;
#endif
    
    if(z==NULL){
      nloc = 3;
      ndims = 2;
      msize = 3;
    }else{
      nloc = 4;
      ndims = 3;
      msize = 6;
    }

    // From the globalENList, create the halo and a local ENList if num_processes>1.
    const index_t *ENList;
    std::map<index_t, index_t> gnn2lnn;
    if(num_processes==1){
      ENList = globalENList;
    }else{
#ifdef HAVE_MPI
      assert(lnn2gnn!=NULL);
      for(size_t i=0;i<(size_t)NNodes;i++){
        gnn2lnn[lnn2gnn[i]] = i;
      }

      std::vector< std::set<int> > recv_set(num_processes);
      index_t *localENList = new index_t[NElements*nloc];
      for(size_t i=0;i<(size_t)NElements*nloc;i++){
        index_t gnn = globalENList[i];
        for(int j=0;j<num_processes;j++){
          if(gnn<owner_range[j+1]){
            if(j!=rank)
              recv_set[j].insert(gnn);
            break;
          }
        }
        localENList[i] = gnn2lnn[gnn];
      }
      std::vector<int> recv_size(num_processes);
      recv.resize(num_processes);
      for(int j=0;j<num_processes;j++){
        for(typename std::set<int>::const_iterator it=recv_set[j].begin();it!=recv_set[j].end();++it){
          recv[j].push_back(*it);
        }
        recv_size[j] = recv[j].size();
      }
      std::vector<int> send_size(num_processes);
      MPI_Alltoall(&(recv_size[0]), 1, MPI_INT,
                   &(send_size[0]), 1, MPI_INT, _mpi_comm);
      
      // Setup non-blocking receives
      send.resize(num_processes);      
      std::vector<MPI_Request> request(num_processes*2);
      for(int i=0;i<num_processes;i++){
        if((i==rank)||(send_size[i]==0)){
          request[i] =  MPI_REQUEST_NULL;
        }else{
          send[i].resize(send_size[i]);
          MPI_Irecv(&(send[i][0]), send_size[i], MPI_INT, i, 0, _mpi_comm, &(request[i]));
        }
      }
      
      // Non-blocking sends.
      for(int i=0;i<num_processes;i++){
        if((i==rank)||(recv_size[i]==0)){
          request[num_processes+i] =  MPI_REQUEST_NULL;
        }else{
          MPI_Isend(&(recv[i][0]), recv_size[i], MPI_INT, i, 0, _mpi_comm, &(request[num_processes+i]));
        }
      }
      
      std::vector<MPI_Status> status(num_processes*2);
      MPI_Waitall(num_processes, &(request[0]), &(status[0]));
      MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

      for(int j=0;j<num_processes;j++){
        for(int k=0;k<recv_size[j];k++)
          recv[j][k] = gnn2lnn[recv[j][k]];
        
        for(int k=0;k<send_size[j];k++)
          send[j][k] = gnn2lnn[send[j][k]];
      }

      ENList = localENList;
#endif
    }

    _ENList.resize(NElements*nloc);
    _coords.resize(NNodes*ndims);
    
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)NElements;i++){
        for(size_t j=0;j<nloc;j++){
          _ENList[i*nloc+j] = ENList[i*nloc+j];
        }
      }
      if(ndims==2){
#pragma omp for schedule(static)
        for(int i=0;i<(int)NNodes;i++){
          _coords[i*ndims  ] = x[i];
          _coords[i*ndims+1] = y[i];
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0;i<(int)NNodes;i++){
          _coords[i*ndims  ] = x[i];
          _coords[i*ndims+1] = y[i];
          _coords[i*ndims+2] = z[i];
        }
      }
    }

    if(num_processes>1){
      // Take into account renumbering for halo.
      for(int j=0;j<num_processes;j++){
        for(size_t k=0;k<recv[j].size();k++){
          recv_halo.insert(recv[j][k]);
        }
        for(size_t k=0;k<send[j].size();k++){
          send_halo.insert(send[j][k]);
        }
      }
    }
    
    if(num_processes>1){
      delete [] ENList;
    }
    
    create_adjancy();
    
    // Set the orientation of elements.
    {
      const int *n=get_element(0);
      assert(n[0]>=0);
      
      if(ndims==2)
        property = new ElementProperty<real_t>(get_coords(n[0]),
                                               get_coords(n[1]),
                                               get_coords(n[2]));
      else
        property = new ElementProperty<real_t>(get_coords(n[0]),
                                               get_coords(n[1]),
                                               get_coords(n[2]),
                                               get_coords(n[3]));
    }

    for(size_t i=1;i<(size_t)NElements;i++){
      const int *n=get_element(i);
      assert(n[0]>=0);
      
      double volarea;
      if(ndims==2)
        volarea = property->area(get_coords(n[0]),
                                 get_coords(n[1]),
                                 get_coords(n[2]));
      else
        volarea = property->volume(get_coords(n[0]),
                                   get_coords(n[1]),
                                   get_coords(n[2]),
                                   get_coords(n[2]));
      
      if(volarea<0)
        invert_element(i);
    }
  }

  void halo_update(float *vec, int block){
#ifdef HAVE_MPI
    if(num_processes<2)
      return;

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);
    
    // Setup non-blocking receives.
    std::vector< std::vector<float> > recv_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(recv[i].size()==0)){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv[i].size()*block);  
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_FLOAT, i, 0, _mpi_comm, &(request[i]));
      }
    }
    
    // Non-blocking sends.
    std::vector< std::vector<float> > send_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(send[i].size()==0)){
        request[num_processes+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].begin();it!=send[i].end();++it)
          for(int j=0;j<block;j++){
            send_buff[i].push_back(vec[(*it)*block+j]);
          }
        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_FLOAT, i, 0, _mpi_comm, &(request[num_processes+i]));
      }
    }
    
    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));
    
    for(int i=0;i<num_processes;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].begin();it!=recv[i].end();++it, ++k)
        for(int j=0;j<block;j++)
          vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }
#endif
  }

  void halo_update(double *vec, int block){
#ifdef HAVE_MPI
    if(num_processes<2)
      return;
    
    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);
    
    // Setup non-blocking receives.
    std::vector< std::vector<double> > recv_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(recv[i].size()==0)){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv[i].size()*block);  
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_DOUBLE, i, 0, _mpi_comm, &(request[i]));
      }
    }
    
    // Non-blocking sends.
    std::vector< std::vector<double> > send_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(send[i].size()==0)){
        request[num_processes+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].begin();it!=send[i].end();++it)
          for(int j=0;j<block;j++){
            send_buff[i].push_back(vec[(*it)*block+j]);
          }
        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_DOUBLE, i, 0, _mpi_comm, &(request[num_processes+i]));
      }
    }
    
    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));
    
    for(int i=0;i<num_processes;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].begin();it!=recv[i].end();++it, ++k)
        for(int j=0;j<block;j++)
          vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }
#endif
  }


  void halo_update(index_t *vec, int block){
#ifdef HAVE_MPI
    if(num_processes<2)
      return;

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);
    
    // Setup non-blocking receives.
    std::vector< std::vector<index_t> > recv_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(recv[i].size()==0)){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv[i].size()*block);  
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_INT, i, 0, _mpi_comm, &(request[i]));
      }
    }
    
    // Non-blocking sends.
    std::vector< std::vector<index_t> > send_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if((i==rank)||(send[i].size()==0)){
        request[num_processes+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].begin();it!=send[i].end();++it)
          for(int j=0;j<block;j++){
            send_buff[i].push_back(vec[(*it)*block+j]);
          }
        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_INT, i, 0, _mpi_comm, &(request[num_processes+i]));
      }
    }
    
    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));
    
    for(int i=0;i<num_processes;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].begin();it!=recv[i].end();++it, ++k)
        for(int j=0;j<block;j++)
          vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }
#endif
  }

  /// Create required adjacency lists.
  void create_adjancy(){
    size_t NNodes = get_number_nodes();
    size_t NElements = get_number_elements();

    // Create new NNList and NEList.
    std::vector< std::set<index_t> > NNList_set(NNodes);
    NEList.clear();
    NEList.resize(NNodes);

    for(size_t i=0; i<NElements; i++){
      for(size_t j=0;j<nloc;j++){
        index_t nid_j = _ENList[i*nloc+j];
        if(nid_j<0)
          break;
        NEList[nid_j].insert(i);
        for(size_t k=j+1;k<nloc;k++){
          index_t nid_k = _ENList[i*nloc+k];
          NNList_set[nid_j].insert(nid_k);
          NNList_set[nid_k].insert(nid_j);
        }
      }
    }

    // Compress NNList
    NNList.clear();
    NNList.resize(NNodes);
    for(size_t i=0;i<NNodes;i++)
      NNList[i].insert(NNList[i].end(), NNList_set[i].begin(), NNList_set[i].end());
  }

  void create_global_node_numbering(int &NPNodes, std::vector<int> &lnn2gnn, std::vector<size_t> &owner){
    int NNodes = get_number_nodes();

    if(num_processes>1){
      NPNodes = NNodes - recv_halo.size();
      
      // Calculate the global numbering offset for this partition.
      int gnn_offset;
      MPI_Scan(&NPNodes, &gnn_offset, 1, MPI_INT, MPI_SUM, get_mpi_comm());
      gnn_offset-=NPNodes;

      // Write global node numbering and ownership for nodes assigned to local process.
      lnn2gnn.resize(NNodes);
      owner.resize(NNodes);
      for(int i=0;i<NNodes;i++){
        if(recv_halo.count(i)){
          lnn2gnn[i] = 0;
        }else{
          lnn2gnn[i] = gnn_offset++;
          owner[i] = rank;
        }
      }
      
      // Update GNN's for the halo nodes.
      halo_update(&(lnn2gnn[0]), 1);
      
      
      // Finish writing node ownerships.
      for(int i=0;i<num_processes;i++){
        for(std::vector<int>::const_iterator it=recv[i].begin();it!=recv[i].end();++it){
          owner[*it] = i;
        }
      }
    }else{
      NPNodes = NNodes;
      lnn2gnn.resize(NNodes);
      owner.resize(NNodes);
      for(int i=0;i<NNodes;i++){
        lnn2gnn[i] = i;
        owner[i] = 0;
      }
    }
  }
  
  inline index_t get_new_vertex(index_t n0, index_t n1,
      std::vector< std::vector<index_t> > &refined_edges, std::vector<index_t> &lnn2gnn) const{
    
    assert((size_t)n0<lnn2gnn.size());
    assert((size_t)n1<lnn2gnn.size());
    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }

    for(size_t i=0;i<NNList[n0].size();i++){
      if(NNList[n0][i]==n1){
        return refined_edges[n0][2*i];
      }
    }
    
    return -1;
  }

  size_t ndims, nloc, msize;
  std::vector<index_t> _ENList;
  std::vector<real_t> _coords;

  // Adjacency lists
  std::vector< std::set<index_t> > NEList;
  std::vector< std::deque<index_t> > NNList;

  ElementProperty<real_t> *property;

  // Metric tensor field.
  std::vector<float> metric;

  // Parallel support.
  int rank, num_processes, num_uma, num_threads;
  std::vector< std::vector<int> > send, recv;
  std::set<int> send_halo, recv_halo;

#ifdef HAVE_MPI
  MPI_Comm _mpi_comm;
#endif
};

inline int get_tid(){
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

#endif
