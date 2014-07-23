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
#include <vector>
#include <set>
#include <stack>
#include <cmath>
#include <stdint.h>

#include <petsc-private/dmpleximpl.h>
#include <petsc-private/isimpl.h>
#include <petscdmda.h>
#include <petscsf.h>
#include <petsc.h>
#include <petscsys.h>
#include <petscerror.h>
#include <petscdmplex.h>
#include <petscviewertypes.h>
#include <petscsf.h>
#include <petscdm.h>
#include "petscdmplex.h"   

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mpi_tools.h"

#include "PragmaticTypes.h"
#include "PragmaticMinis.h"

#include "Metis.h"
#include "ElementProperty.h"
#include "MetricTensor.h"
#include "HaloExchange.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t> class Mesh{
 public:

  /*! Constructor from PETSc/DMPlex
   *
   * @param plex DMPlex object that we import from.
   */
  Mesh(DM plex, MPI_Comm comm){
    _mpi_comm = comm;

    MPI_Comm_size(_mpi_comm, &num_processes);
    MPI_Comm_rank(_mpi_comm, &rank);

    // Assign the correct MPI data type to MPI_INDEX_T
    mpi_type_wrapper<index_t> mpi_index_t_wrapper;
    MPI_INDEX_T = mpi_index_t_wrapper.mpi_type;

    nthreads = pragmatic_nthreads();

    /* Establish sizes */
    PetscErrorCode ierr;
    PetscInt dim, cStart, cEnd, vStart, vEnd;
    ierr = DMPlexGetDimension(plex, &dim);assert(ierr==0);
    ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);assert(ierr==0);
    NNodes = vEnd - vStart;
    ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);assert(ierr==0);
    NElements = cEnd - cStart;

    if(dim == 2){
      nloc = 3;
      ndims = 2;
      msize = 3;
    }else{
      nloc = 4;
      ndims = 3;
      msize = 6;
    }

    /* Pragmatic assumes that local facet numbers correspond to the local vertex number
       of the opposite vertex. Since this is not given by the Plex we have to perform a
       re-ordering of Plex closures in order to enforce local entity numbering according
       to Fenics rules. The universal vertex numbering is provided by the Plex's
       coordinate section.
    */
    PetscSection coord_sec;
    ierr = DMGetCoordinateSection(plex, &coord_sec);
    std::vector<std::pair<PetscInt, PetscInt>> fenics_local_numbering;
    derive_fenics_local_numbering(plex, coord_sec, nloc, &fenics_local_numbering);

    /* Make sure label "boundary_ids" exists */
    PetscBool has_label;
    ierr = DMPlexHasLabel(plex, "boundary_ids", &has_label);
    if (!has_label) ierr = DMPlexCreateLabel(plex, "boundary_ids");

    /* Import local element node list and set boundary IDs according to Plex label "boundary_ids" */
    _ENList.resize(NElements*nloc);
    boundary.resize(NElements*nloc);
    int bid, ind = 0;
    for(std::vector<std::pair<PetscInt, PetscInt>>::iterator it = fenics_local_numbering.begin();
        it != fenics_local_numbering.end(); it++) {
      _ENList[ind] = it->first - vStart;

      ierr = DMPlexGetLabelValue(plex, "boundary_ids", it->second, &bid);
      if (bid > 0) boundary[ind] = bid;
      else boundary[ind] = 0;
      ind++;
    }

    // Import local coordinates
    _coords.resize(NNodes*ndims);
    Vec coords_vec;
    PetscReal *plex_coords;
    ierr = DMGetCoordinatesLocal(plex, &coords_vec);
    ierr = VecGetArray(coords_vec, &plex_coords);
    /* Not sure we really need to deep copy here */
    for (int i=0; i<NNodes*ndims; i++) _coords[i] = plex_coords[i];
    ierr = VecRestoreArray(coords_vec, &plex_coords);

    NNList.resize(NNodes);
    NEList.resize(NNodes);

    /* This std::vector< std::vector<index_t> > send, recv defines the halo
     * where node index send[i][k] is sent to rank i, and is received by index recv[j][k], 
     * where j is the sending rank. Note that k on the send side is the same as k on the recv size.
     */
    send.resize(num_processes);
    recv.resize(num_processes);
    send_map.resize(num_processes);
    recv_map.resize(num_processes);

    // Stores the rank that owns each vertex.
    node_owner.resize(NNodes);

    PetscSF sf = NULL;
    PetscInt nroots, nleaves;
    const PetscInt *local;
    const PetscSFNode *remote;

    /* Hack alert: Build a scalar CG1 section to enforce a meaningful SF */
    PetscSection section;
    PetscInt  comp[1] = {1};
    PetscInt  dof[4] = {1, 0, 0, 0};
    ierr = DMPlexCreateSection(plex, ndims, 1, comp, dof, 0, NULL, NULL, NULL, &section); assert(ierr==0);
    ierr = DMSetDefaultSection(plex, section); assert(ierr==0);
    ierr = DMGetDefaultSF(plex, &sf); assert(ierr==0);

    /* Derive local receives, the according remote sends and a local
       root->leaf mapping */
    std::vector< std::vector<index_t> > remote_send(num_processes);
    std::map<index_t, index_t> root_leaf_map;
    ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &local, &remote); assert(ierr==0);
    assert(nleaves == NNodes);
    for (int i=0; i<nleaves; i++) {
      if (remote[i].rank != rank) {
        recv[remote[i].rank].push_back( local[i] );
        remote_send[remote[i].rank].push_back( remote[i].index );
      } else {
        root_leaf_map.insert( std::pair<index_t, index_t>(remote[i].index, local[i]) );
      }
      node_owner[i] = remote[i].rank;
    }

    /* Propagate remote send lists to the actual sender */
    int tag = 123456;
    std::vector<MPI_Request> send_req(num_processes);
    for (int proc=0; proc<num_processes; proc++) {
      if (proc != rank) {
        ierr = MPI_Isend(remote_send[proc].data(), remote_send[proc].size(), MPI_INT,
                         proc, tag, comm, &send_req[proc]); assert(ierr==0);
      }
    }

    /* Receive local send list from receiving proc */
    MPI_Status status;
    std::vector<int> recv_size(num_processes);
    std::vector<MPI_Request> recv_req(num_processes);
    std::vector<index_t*> recv_buffer(num_processes);
    for (int proc=0; proc<num_processes; proc++) {
      if (proc != rank) {
        ierr = MPI_Probe(proc, tag, comm, &status); assert(ierr==0);
        ierr = MPI_Get_count(&status, MPI_INT, &recv_size[proc]); assert(ierr==0);
        recv_buffer[proc] = new index_t[recv_size[proc]];

        MPI_Irecv(recv_buffer[proc], recv_size[proc], MPI_INT, proc,
                  tag, comm, &recv_req[proc]); assert(ierr==0);
      }
    }

    /* Translate received send lists from roots into leaves and assign to send */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc != rank) {
        send[proc].resize( recv_size[proc] );
        for (int i=0; i<recv_size[proc]; i++) {
          send[proc][i] = root_leaf_map.find(recv_buffer[proc][i])->second;
        }
        free(recv_buffer[proc]);
      }
    }

    // Stores the local2global node number map.
    lnn2gnn.resize(NNodes);
    ISLocalToGlobalMapping ltog;
    const PetscInt *ltog_indices;
    ierr = DMGetLocalToGlobalMapping(plex, &ltog); assert(ierr==0);
    ISLocalToGlobalMappingGetIndices(ltog, &ltog_indices); assert(ierr==0);
    for (int i=0; i<NNodes; i++) lnn2gnn[i] = ltog_indices[i];
    ISLocalToGlobalMappingRestoreIndices(ltog, &ltog_indices); assert(ierr==0);

    // Right now we are doing nothing with this. But in the future the plex may contain a special tensor field/section that we will need to import.
    metric.resize(NNodes*msize);


    // This really needs to be revisited...leave as it for now.
    deferred_operations.resize(nthreads);
#pragma omp parallel
    {
      // Each thread allocates nthreads DeferredOperations
      // structs, one for each OMP thread.
      deferred_operations[pragmatic_thread_id()].resize((defOp_scaling_factor*nthreads));

#pragma omp single nowait
      {
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
      }

      // Set the orientation of elements.
#pragma omp single
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

      if(ndims==2){
#pragma omp for schedule(static)
        for(size_t i=1;i<(size_t)NElements;i++){
          const int *n=get_element(i);
          assert(n[0]>=0);

          double volarea = property->area(get_coords(n[0]),
                                          get_coords(n[1]),
                                          get_coords(n[2]));

          if(volarea<0)
            invert_element(i);
        }
      }else{
#pragma omp for schedule(static)
        for(size_t i=1;i<(size_t)NElements;i++){
          const int *n=get_element(i);
          assert(n[0]>=0);

          double volarea = property->volume(get_coords(n[0]),
                                            get_coords(n[1]),
                                            get_coords(n[2]),
                                            get_coords(n[2]));

          if(volarea<0)
            invert_element(i);
        }
      }

      // create_adjacency is meant to be called from inside a parallel region
      create_adjacency();
    }
  }

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

  /// Default destructor.
  ~Mesh(){
    delete property;
  }

  /// Add a new vertex
  index_t append_vertex(const real_t *x, const double *m){
    for(size_t i=0;i<ndims;i++)
      _coords[ndims*NNodes+i] = x[i];

    for(size_t i=0;i<msize;i++)
      metric[msize*NNodes+i] = m[i];

    ++NNodes;

    return get_number_nodes()-1;
  }

  /// Erase a vertex
  void erase_vertex(const index_t nid){
    NNList[nid].clear();
    NEList[nid].clear();
    node_owner[nid] = rank;
    lnn2gnn[nid] = -1;
  }

  /// Add a new element
  index_t append_element(const index_t *n){
    for(size_t i=0;i<nloc;i++)
      _ENList[nloc*NElements+i] = n[i];

    ++NElements;

    return get_number_elements()-1;
  }

  void create_boundary(){
    assert(boundary.size()==0);
    
    size_t NNodes = get_number_nodes();
    size_t NElements = get_number_elements();
    
    if(ndims==2){
      // Initialise the boundary array
      boundary.resize(NElements*3);
      std::fill(boundary.begin(), boundary.end(), -2);
      
      // Create node-element adjancy list.      
      std::vector< std::set<int> > NEList(NNodes);
      for(size_t i=0;i<NElements;i++){
	if(_ENList[i*3]==-1)
	  continue;
	
	for(size_t j=0;j<3;j++)
	  NEList[_ENList[i*3+j]].insert(i);
      }
      
      // Check neighbourhood of each element
      for(int i=0;i<NElements;i++){
	if(_ENList[i*3]==-1)
	  continue;
	
	for(int j=0;j<3;j++){
	  int n1 = _ENList[i*3+(j+1)%3];
	  int n2 = _ENList[i*3+(j+2)%3];
	  
	  if(is_owned_node(n1)||is_owned_node(n2)){
	    std::set<int> neighbours;
	    set_intersection(NEList[n1].begin(), NEList[n1].end(),
			     NEList[n2].begin(), NEList[n2].end(),
			     inserter(neighbours, neighbours.begin()));
	    
	    if(neighbours.size()==2){
	      if(*neighbours.begin()==i)
		boundary[i*3+j] = *neighbours.rbegin();
	      else
		boundary[i*3+j] = *neighbours.begin();
	    }
	  }else{
	    // This is a halo facet.
	    boundary[i*3+j] = -1;
	  }
	}
      }
    }else{ // ndims==3
      // Initialise the boundary array
      boundary.resize(NElements*4);
      std::fill(boundary.begin(), boundary.end(), -2);
      
      // Create node-element adjancy list.      
      std::vector< std::set<int> > NEList(NNodes);
      for(size_t i=0;i<NElements;i++){
	if(_ENList[i*4]==-1)
	  continue;
	
	for(size_t j=0;j<4;j++)
	  NEList[_ENList[i*4+j]].insert(i);
      }
      
      // Check neighbourhood of each element
      for(int i=0;i<NElements;i++){
	if(_ENList[i*4]==-1)
	  continue;
	
	for(int j=0;j<4;j++){
	  int n1 = _ENList[i*4+(j+1)%4];
	  int n2 = _ENList[i*4+(j+2)%4];
	  int n3 = _ENList[i*4+(j+3)%4];

	  if(is_owned_node(n1)||is_owned_node(n2)||is_owned_node(n3)){
	    std::set<int> edge_neighbours;
	    set_intersection(NEList[n1].begin(), NEList[n1].end(),
			     NEList[n2].begin(), NEList[n2].end(),
			     inserter(edge_neighbours, edge_neighbours.begin()));
	    
	    std::set<int> neighbours;
	    set_intersection(NEList[n3].begin(), NEList[n3].end(),
			     edge_neighbours.begin(), edge_neighbours.end(),
			     inserter(neighbours, neighbours.begin()));
	    
	    if(neighbours.size()==2){
	      if(*neighbours.begin()==i)
		boundary[i*4+j] = *neighbours.rbegin();
	      else
		boundary[i*4+j] = *neighbours.begin();
	    }
	  }else{
	    // This is a halo facet.
	    boundary[i*4+j] = -1;
	  }
	}
      }
    }
    for(std::vector<int>::iterator it=boundary.begin();it!=boundary.end();++it)
      if(*it==-2)
        *it = 1;
      else if(*it>=0)
        *it = 0;
  }

  /// Erase an element
  void erase_element(const index_t eid){
    const index_t *n = get_element(eid);

    for(size_t i=0; i<nloc; ++i)
  	  NEList[n[i]].erase(eid);

    _ENList[eid*nloc] = -1;
  }

  /// Flip orientation of element.
  void invert_element(size_t eid){
    int tmp = _ENList[eid*nloc];
    _ENList[eid*nloc] = _ENList[eid*nloc+1];
    _ENList[eid*nloc+1] = tmp;
  }

  /// Return a pointer to the element-node list.
  const index_t *get_element(size_t eid) const{
    return &(_ENList[eid*nloc]);
  }

  /// Return copy of element-node list.
  void get_element(size_t eid, index_t *ele) const{
    for(size_t i=0;i<nloc;i++)
      ele[i] = _ENList[eid*nloc+i];
  }

  /// Return the number of nodes in the mesh.
  size_t get_number_nodes() const{
    return NNodes;
  }

  /// Return the number of elements in the mesh.
  size_t get_number_elements() const{
    return NElements;
  }

  /// Return the number of spatial dimensions.
  size_t get_number_dimensions() const{
    return ndims;
  }

  /// Return positions vector.
  const real_t *get_coords(index_t nid) const{
    return &(_coords[nid*ndims]);
  }

  /// Return copy of the coordinate.
  void get_coords(index_t nid, real_t *x) const{
    for(size_t i=0;i<ndims;i++)
      x[i] = _coords[nid*ndims+i];
    return;
  }

  /// Return metric at that vertex.
  const double *get_metric(index_t nid) const{
    assert(metric.size()>0);
    return &(metric[nid*msize]);
  }

  /// Return copy of metric.
  void get_metric(index_t nid, double *m) const{
    assert(metric.size()>0);
    for(size_t i=0;i<msize;i++)
      m[i] = metric[nid*msize+i];
    return;
  }

  /// Returns true if the node is in any of the partitioned elements.
  bool is_halo_node(index_t nid) const{
    return (node_owner[nid]!= rank || send_halo.count(nid)>0);
  }

  /// Returns true if the node is assigned to the local partition.
  bool is_owned_node(index_t nid) const{
    return node_owner[nid] == rank;
  }

  /// Get the mean edge length metric space.
  double get_lmean(){
    int NNodes = get_number_nodes();
    double total_length=0;
    int nedges=0;
// #pragma omp parallel reduction(+:total_length,nedges)
    {
#pragma omp for schedule(static)
      for(int i=0;i<NNodes;i++){
        if(is_owned_node(i) && (NNList[i].size()>0))
          for(typename std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
            if(i<*it){ // Ensure that every edge length is only calculated once.
              double length = calc_edge_length(i, *it);
#pragma omp atomic
              total_length += length;
#pragma omp atomic
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

  /// Calculate perimeter
  double calculate_perimeter(){
    int NElements = get_number_elements();
    if(ndims==2){
      long double total_length=0;
      
      if(num_processes>1){
	for(int i=0;i<NElements;i++){
	  for(int j=0;j<3;j++){
            int n1 = _ENList[i*nloc+(j+1)%3];
	    int n2 = _ENList[i*nloc+(j+2)%3];

	    if(boundary[i*nloc+j]>0 && (std::min(node_owner[n1], node_owner[n2])==rank)){
	      long double dx = (_coords[n1*2  ]-_coords[n2*2  ]);
	      long double dy = (_coords[n1*2+1]-_coords[n2*2+1]);
	      
	      total_length += sqrt(dx*dx+dy*dy);
	    }
	  }
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total_length, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
      }else{
	for(int i=0;i<NElements;i++){
	  for(int j=0;j<3;j++){
	    if(boundary[i*nloc+j]>0){
	      int n1 = _ENList[i*nloc+(j+1)%3];
	      int n2 = _ENList[i*nloc+(j+2)%3];
	    
              long double dx = (_coords[n1*2  ]-_coords[n2*2  ]);
              long double dy = (_coords[n1*2+1]-_coords[n2*2+1]);
	      
	      total_length += sqrt(dx*dx+dy*dy);
	    }
	  }
	}
      }
      
      return total_length;
    }else{
      std::cerr<<"ERROR: calculate_perimeter() cannot be used for 3D. Use calculate_area() instead if you want the total surface area.\n";
      return -1;
    }
  }


  /// Calculate area
  double calculate_area(){
    int NElements = get_number_elements();
    long double total_area=0;

    if(ndims==2){
      if(num_processes>1){
	for(int i=0;i<NElements;i++){
          const index_t *n=get_element(i);

          // Don't sum if it's not ours
          if(std::min(node_owner[n[0]], std::min(node_owner[n[1]], node_owner[n[2]]))!=rank)
            continue;

          const double *x1 = get_coords(n[0]);
          const double *x2 = get_coords(n[1]);
          const double *x3 = get_coords(n[2]);

          // Use Heron's Formula
          long double a;
          {
            long double dx = (x1[0]-x2[0]);
            long double dy = (x1[1]-x2[1]);
            a = sqrt(dx*dx+dy*dy);
          }
          long double b;
          {
            long double dx = (x1[0]-x3[0]);
            long double dy = (x1[1]-x3[1]);
            b = sqrt(dx*dx+dy*dy);
          }
          long double c;
          {
            long double dx = (x2[0]-x3[0]);
            long double dy = (x2[1]-x3[1]);
            c = sqrt(dx*dx+dy*dy);
          }
          long double s = 0.5*(a+b+c);

          total_area += sqrt(s*(s-a)*(s-b)*(s-c));
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
      }else{
        for(int i=0;i<NElements;i++){
          const index_t *n=get_element(i);

          const double *x1 = get_coords(n[0]);
          const double *x2 = get_coords(n[1]);
          const double *x3 = get_coords(n[2]);
        
          // Use Heron's Formula
          long double a;
          {
            long double dx = (x1[0]-x2[0]);
            long double dy = (x1[1]-x2[1]);
            a = sqrt(dx*dx+dy*dy);
          }
          long double b;
          {
            long double dx = (x1[0]-x3[0]);
            long double dy = (x1[1]-x3[1]);
            b = sqrt(dx*dx+dy*dy);
          }
          long double c;
          {
            long double dx = (x2[0]-x3[0]);
            long double dy = (x2[1]-x3[1]);
            c = sqrt(dx*dx+dy*dy);
          }
          long double s = 0.5*(a+b+c);
            
          total_area += sqrt(s*(s-a)*(s-b)*(s-c));
        }
      }
    }else{ // 3D
      if(num_processes>1){
	for(int i=0;i<NElements;i++){
	  const index_t *n=get_element(i);
	  for(int j=0;j<4;j++){
	    if(boundary[i*nloc+j]<=0)
	      continue;
	    
	    int n1 = n[(j+1)%4];
	    int n2 = n[(j+2)%4];
	    int n3 = n[(j+3)%4];
	    
	    // Don't sum if it's not ours
	    if(std::min(node_owner[n1], std::min(node_owner[n2], node_owner[n3]))!=rank)
	      continue;
	    
	    const double *x1 = get_coords(n1);
	    const double *x2 = get_coords(n2);
	    const double *x3 = get_coords(n3);
	    
	    // Use Heron's Formula
	    long double a;
	    {
	      long double dx = (x1[0]-x2[0]);
	      long double dy = (x1[1]-x2[1]);
	      long double dz = (x1[2]-x2[2]);
	      a = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double b;
	    {
	      long double dx = (x1[0]-x3[0]);
	      long double dy = (x1[1]-x3[1]);
	      long double dz = (x1[2]-x3[2]);
	      b = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double c;
	    {
	      long double dx = (x2[0]-x3[0]);
	      long double dy = (x2[1]-x3[1]);
	      long double dz = (x2[2]-x3[2]);
	      c = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double s = 0.5*(a+b+c);
	    
	    total_area += sqrt(s*(s-a)*(s-b)*(s-c));
	  }
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
      }else{
	for(int i=0;i<NElements;i++){
	  const index_t *n=get_element(i);
	  for(int j=0;j<4;j++){
	    if(boundary[i*nloc+j]<=0)
	      continue;
	    
	    int n1 = n[(j+1)%4];
	    int n2 = n[(j+2)%4];
	    int n3 = n[(j+3)%4];
	    
	    const double *x1 = get_coords(n1);
	    const double *x2 = get_coords(n2);
	    const double *x3 = get_coords(n3);
	    
	    // Use Heron's Formula
	    long double a;
	    {
	      long double dx = (x1[0]-x2[0]);
	      long double dy = (x1[1]-x2[1]);
	      long double dz = (x1[2]-x2[2]);
	      a = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double b;
	    {
	      long double dx = (x1[0]-x3[0]);
	      long double dy = (x1[1]-x3[1]);
	      long double dz = (x1[2]-x3[2]);
	      b = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double c;
	    {
	      long double dx = (x2[0]-x3[0]);
	      long double dy = (x2[1]-x3[1]);
	      long double dz = (x2[2]-x3[2]);
	      c = sqrt(dx*dx+dy*dy+dz*dz);
	    }
	    long double s = 0.5*(a+b+c);
	    
	    total_area += sqrt(s*(s-a)*(s-b)*(s-c));
	  }
	}
      }
    }
    return total_area;
  }

  /// Calculate volume
  double calculate_volume(){
    int NElements = get_number_elements();
    long double total_volume=0;

    if(ndims==2){
      std::cerr<<"ERROR: Cannot calculate volume in 2D\n"; 
    }else{ // 3D
      if(num_processes>1){
	for(int i=0;i<NElements;i++){
	  const index_t *n=get_element(i);
	    
          // Don't sum if it's not ours
          if(std::min(std::min(node_owner[n[0]], node_owner[n[1]]), std::min(node_owner[n[2]], node_owner[n[3]]))!=rank)
            continue;

          const double *x0 = get_coords(n[0]);
          const double *x1 = get_coords(n[1]);
          const double *x2 = get_coords(n[2]);
          const double *x3 = get_coords(n[3]);

          long double x01 = (x0[0] - x1[0]);
          long double x02 = (x0[0] - x2[0]);
          long double x03 = (x0[0] - x3[0]);

          long double y01 = (x0[1] - x1[1]);
          long double y02 = (x0[1] - x2[1]);
          long double y03 = (x0[1] - x3[1]);

          long double z01 = (x0[2] - x1[2]);
          long double z02 = (x0[2] - x2[2]);
          long double z03 = (x0[2] - x3[2]);

          total_volume += (-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - x01*(z03*y02 - z02*y03));
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &total_volume, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
      }else{
	for(int i=0;i<NElements;i++){
	  const index_t *n=get_element(i);

          const double *x0 = get_coords(n[0]);
          const double *x1 = get_coords(n[1]);
          const double *x2 = get_coords(n[2]);
          const double *x3 = get_coords(n[3]);

          long double x01 = (x0[0] - x1[0]);
          long double x02 = (x0[0] - x2[0]);
          long double x03 = (x0[0] - x3[0]);

          long double y01 = (x0[1] - x1[1]);
          long double y02 = (x0[1] - x2[1]);
          long double y03 = (x0[1] - x3[1]);

          long double z01 = (x0[2] - x1[2]);
          long double z02 = (x0[2] - x2[2]);
          long double z03 = (x0[2] - x3[2]);

          total_volume += (-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - x01*(z03*y02 - z02*y03));
        }
      }
    }
    return total_volume;
  }


  /// Get the edge length RMS value in metric space.
  double get_lrms(){
    double mean = get_lmean();

    double rms=0;
    int nedges=0;
#pragma omp parallel reduction(+:rms,nedges)
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)NNodes;i++){
        if(is_owned_node(i) && (NNList[i].size()>0))
          for(typename std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
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
  double get_qmean() const{
    double sum=0;
    int nele=0;

#pragma omp parallel reduction(+:sum, nele)
    {
#pragma omp for schedule(static)
      for(size_t i=0;i<NElements;i++){
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

  /// Print out the qualities. Useful if you want to plot a histogram of element qualities.
  void print_quality() const{
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(size_t i=0;i<NElements;i++){
        const index_t *n=get_element(i);
        if(n[0]<0)
          continue;
        
        double q;
        if(ndims==2){
          q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]),
                                 get_metric(n[0]), get_metric(n[1]), get_metric(n[2]));
        }else{
          q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]),
                                 get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3]));
        }
#pragma omp critical
        std::cout<<"Quality[ele="<<i<<"] = "<<q<<std::endl;
      }
    }
  }

  /// Get the element minimum quality in metric space.
  double get_qmin() const{
    if(ndims==2)
      return get_qmin_2d();
    else
      return get_qmin_3d();
  }

  double get_qmin_2d() const{
    double qmin=1; // Where 1 is ideal.

    for(size_t i=0;i<NElements;i++){
      const index_t *n=get_element(i);
      if(n[0]<0)
        continue;

      qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]),
                                               get_metric(n[0]), get_metric(n[1]), get_metric(n[2])));
    }

    if(num_processes>1)
      MPI_Allreduce(MPI_IN_PLACE, &qmin, 1, MPI_DOUBLE, MPI_MIN, _mpi_comm);
    
    return qmin;
  }

  double get_qmin_3d() const{
    double qmin=1; // Where 1 is ideal.

    for(size_t i=0;i<NElements;i++){
      const index_t *n=get_element(i);
      if(n[0]<0)
        continue;

      qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]),
                                               get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3])));
    }

    if(num_processes>1)
      MPI_Allreduce(MPI_IN_PLACE, &qmin, 1, MPI_DOUBLE, MPI_MIN, _mpi_comm);
    
    return qmin;
  }

#ifdef HAVE_MPI
  /// Return the MPI communicator.
  MPI_Comm get_mpi_comm() const{
    return _mpi_comm;
  }
#endif

  /// Return the node id's connected to the specified node_id
  std::set<index_t> get_node_patch(index_t nid) const{
    assert(nid<(index_t)NNodes);
    std::set<index_t> patch;
    for(typename std::vector<index_t>::const_iterator it=NNList[nid].begin();it!=NNList[nid].end();++it)
      patch.insert(patch.end(), *it);
    return patch;
  }

  /// Grow a node patch around node id's until it reaches a minimum size.
  std::set<index_t> get_node_patch(index_t nid, size_t min_patch_size){
    std::set<index_t> patch = get_node_patch(nid);

    if(patch.size()<min_patch_size){
      std::set<index_t> front = patch, new_front;
      for(;;){
        for(typename std::set<index_t>::const_iterator it=front.begin();it!=front.end();it++){
          for(typename std::vector<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();jt++){
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

  /// Calculates the edge lengths in metric space.
  real_t calc_edge_length(index_t nid0, index_t nid1) const{
    real_t length=-1.0;
    if(ndims==2){
      double m[3];
      m[0] = (metric[nid0*3  ]+metric[nid1*3  ])*0.5;
      m[1] = (metric[nid0*3+1]+metric[nid1*3+1])*0.5;
      m[2] = (metric[nid0*3+2]+metric[nid1*3+2])*0.5;

      length = ElementProperty<real_t>::length2d(get_coords(nid0), get_coords(nid1), m);
    }else{
      double m[6];
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

    for(index_t i=0;i<(index_t) NNodes;i++){
      if(is_owned_node(i) && (NNList[i].size()>0))
        for(typename std::vector<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();++it){
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

  /*! Defragment mesh. This compresses the storage of internal data
    structures. This is useful if the mesh has been significantly
    coarsened. */
  void defragment(){
    // Discover which vertices and elements are active.
    std::vector<index_t> active_vertex_map(NNodes);
    
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<NNodes;i++){
      active_vertex_map[i] = -1;
      NNList[i].clear();
      NEList[i].clear();
    }

    // Identify active elements.
    std::vector<index_t> active_element;

    active_element.reserve(NElements);

    std::map<index_t, std::set<int> > new_send_set, new_recv_set;
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
        active_vertex_map[nid]=0;

        if(halo_element){
          for(int k=0;k<num_processes;k++){
            if(recv_map[k].count(lnn2gnn[nid])){
              new_recv_set[k].insert(nid);
              neigh.insert(k);
            }
          }
        }
      }
      for(size_t j=0;j<nloc;j++){
        nid = _ENList[e*nloc+j];
        for(std::set<int>::iterator kt=neigh.begin();kt!=neigh.end();++kt){
          if(send_map[*kt].count(lnn2gnn[nid]))
            new_send_set[*kt].insert(nid);
        }
      }
    }

    // Create a new numbering.
    index_t cnt=0;
    for(size_t i=0;i<NNodes;i++){
      if(active_vertex_map[i]<0)
        continue;

      active_vertex_map[i] = cnt++;
    }

    int metis_nnodes = cnt;
    int metis_nelements = active_element.size();
    std::vector< std::set<index_t> > graph(metis_nnodes);
    for(typename std::vector<index_t>::iterator ie=active_element.begin();ie!=active_element.end();++ie){
      for(size_t i=0;i<nloc;i++){
        index_t nid0 = active_vertex_map[_ENList[(*ie)*nloc+i]];
        for(size_t j=i+1;j<nloc;j++){
          index_t nid1 = active_vertex_map[_ENList[(*ie)*nloc+j]];
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

    if(metis_nnodes != 0){
      METIS_NodeND(&metis_nnodes, &(xadj[0]), &(adjncy[0]), NULL, NULL, &(norder[0]), &(inorder[0]));

      // Update active_vertex_map
      for(size_t i=0;i<NNodes;i++){
        if(active_vertex_map[i]<0)
          continue;

        active_vertex_map[i] = inorder[active_vertex_map[i]];
      }
    }

    // Renumber elements
    std::map< std::set<index_t>, index_t > ordered_elements;
    for(int i=0;i<metis_nelements;i++){
      index_t old_eid = active_element[i];
      std::set<index_t> sorted_element;
      for(size_t j=0;j<nloc;j++){
        index_t new_nid = active_vertex_map[_ENList[old_eid*nloc+j]];
        sorted_element.insert(new_nid);
      }
      if(ordered_elements.find(sorted_element)==ordered_elements.end()){
        ordered_elements[sorted_element] = old_eid;
      }else{
        std::cerr<<"dup! "
                 <<active_vertex_map[_ENList[old_eid*nloc]]<<" "
                 <<active_vertex_map[_ENList[old_eid*nloc+1]]<<" "
                 <<active_vertex_map[_ENList[old_eid*nloc+2]]<<std::endl;
      }
    }
    std::vector<index_t> metis_element_renumber;
    metis_element_renumber.reserve(metis_nelements);
    for(typename std::map< std::set<index_t>, index_t >::const_iterator it=ordered_elements.begin();it!=ordered_elements.end();++it){
      metis_element_renumber.push_back(it->second);
    }
    // assert(metis_nelements==metis_element_renumber.size());
    metis_nelements=metis_element_renumber.size();
    // end of renumbering

    // Compress data structures.
    assert(cnt == metis_nnodes);
    NNodes = metis_nnodes;
    //NElements = active_element.size();
    NElements = metis_nelements;

    std::vector<index_t> defrag_ENList(NElements*nloc);
    std::vector<real_t> defrag_coords(NNodes*ndims);
    std::vector<double> defrag_metric(NNodes*msize);
    std::vector<int> defrag_boundary(NElements*nloc);

    assert(NElements==(size_t)metis_nelements);

    // This first touch is to bind memory locally.
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<(int)NElements;i++){
        defrag_ENList[i*nloc] = 0;
        defrag_boundary[i*nloc] = 0;
      }

#pragma omp for schedule(static)
      for(int i=0;i<(int)NNodes;i++){
        defrag_coords[i*ndims] = 0.0;
        defrag_metric[i*msize] = 0.0;
      }
    }

    // Second sweep writes elements with new numbering.
    for(int i=0;i<metis_nelements;i++){
      index_t old_eid = metis_element_renumber[i];
      index_t new_eid = i;
      for(size_t j=0;j<nloc;j++){
        index_t new_nid = active_vertex_map[_ENList[old_eid*nloc+j]];
        assert(new_nid<(index_t)NNodes);
        defrag_ENList[new_eid*nloc+j] = new_nid;
        defrag_boundary[new_eid*nloc+j] = boundary[old_eid*nloc+j];
      }
    }

    // Second sweep writes node data with new numbering.
    for(size_t old_nid=0;old_nid<active_vertex_map.size();++old_nid){
      index_t new_nid = active_vertex_map[old_nid];
      if(new_nid<0)
        continue;

      for(size_t j=0;j<ndims;j++)
        defrag_coords[new_nid*ndims+j] = _coords[old_nid*ndims+j];
      for(size_t j=0;j<msize;j++)
        defrag_metric[new_nid*msize+j] = metric[old_nid*msize+j];
    }

    memcpy(&_ENList[0], &defrag_ENList[0], NElements*nloc*sizeof(index_t));
    memcpy(&boundary[0], &defrag_boundary[0], NElements*nloc*sizeof(int));
    memcpy(&_coords[0], &defrag_coords[0], NNodes*ndims*sizeof(real_t));
    memcpy(&metric[0], &defrag_metric[0], NNodes*msize*sizeof(double));

    // Renumber halo, fix lnn2gnn and node_owner.
    if(num_processes>1){
      std::vector<index_t> defrag_lnn2gnn(NNodes);
      std::vector<int> defrag_owner(NNodes);

      for(size_t old_nid=0;old_nid<active_vertex_map.size();++old_nid){
        index_t new_nid = active_vertex_map[old_nid];
        if(new_nid<0)
          continue;

        defrag_lnn2gnn[new_nid] = lnn2gnn[old_nid];
        defrag_owner[new_nid] = node_owner[old_nid];
      }

      lnn2gnn.swap(defrag_lnn2gnn);
      node_owner.swap(defrag_owner);

      for(int k=0;k<num_processes;k++){
        std::vector<int> new_halo;
        send_map[k].clear();
        for(std::vector<int>::iterator jt=send[k].begin();jt!=send[k].end();++jt){
          if(new_send_set[k].count(*jt)){
            index_t new_lnn = active_vertex_map[*jt];
            new_halo.push_back(new_lnn);
            send_map[k][lnn2gnn[new_lnn]] = new_lnn;
          }
        }
        send[k].swap(new_halo);
      }

      for(int k=0;k<num_processes;k++){
        std::vector<int> new_halo;
        recv_map[k].clear();
        for(std::vector<int>::iterator jt=recv[k].begin();jt!=recv[k].end();++jt){
          if(new_recv_set[k].count(*jt)){
            index_t new_lnn = active_vertex_map[*jt];
            new_halo.push_back(new_lnn);
            recv_map[k][lnn2gnn[new_lnn]] = new_lnn;
          }
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
    }else{
      for(size_t i=0; i<NNodes; ++i){
        lnn2gnn[i] = i;
        node_owner[i] = 0;
      }
    }

#pragma omp parallel
    create_adjacency();
  }

  /// This is used to verify that the mesh and its metadata is correct.
  bool verify() const{
    bool state = true;

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
        if(pragmatic_isnan(larea)){
          std::cerr<<"ERROR: Bad element "<<n[0]<<", "<<n[1]<<", "<<n[2]<<std::endl;
        }

        area += larea;
        min_ele_area = std::min(min_ele_area, larea);
        max_ele_area = std::max(max_ele_area, larea);
      }

      MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, get_mpi_comm());
      MPI_Allreduce(MPI_IN_PLACE, &min_ele_area, 1, MPI_DOUBLE, MPI_MIN, get_mpi_comm());
      MPI_Allreduce(MPI_IN_PLACE, &max_ele_area, 1, MPI_DOUBLE, MPI_MAX, get_mpi_comm());

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

      MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, get_mpi_comm());
      MPI_Allreduce(MPI_IN_PLACE, &min_ele_vol, 1, MPI_DOUBLE, MPI_MIN, get_mpi_comm());
      MPI_Allreduce(MPI_IN_PLACE, &max_ele_vol, 1, MPI_DOUBLE, MPI_MAX, get_mpi_comm());

      if(rank==0){
        std::cout<<"VERIFY: total volume.............."<<volume<<std::endl;
        std::cout<<"VERIFY: minimum element volume...."<<min_ele_vol<<std::endl;
        std::cout<<"VERIFY: maximum element volume...."<<max_ele_vol<<std::endl;
      }
    }
    double qmean = get_qmean();
    double qmin = get_qmin();
    if(rank==0){
      std::cout<<"VERIFY: mean quality...."<<qmean<<std::endl;
      std::cout<<"VERIFY: min quality...."<<qmin<<std::endl;
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

  // TODO - This function is here for compatibility with 3D
  void get_global_node_numbering(std::vector<int>& NPNodes, std::vector<int> &lnn2gnn){
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
      for(index_t i=0;i<(index_t)NNodes;i++){
        if(recv_halo.count(i)){
          lnn2gnn[i] = 0;
        }else{
          lnn2gnn[i] = gnn_offset++;
        }
      }

      // Update GNN's for the halo nodes.
      halo_update<int, 1>(_mpi_comm, send, recv, lnn2gnn);
    }else{
      NPNodes[0] = NNodes;
      for(index_t i=0;i<(index_t)NNodes;i++){
        lnn2gnn[i] = i;
      }
    }
  }

  void export_dmplex(DM *plex)
  {
    PetscErrorCode ierr;
    ierr = DMPlexCreateFromCellList(_mpi_comm, ndims, NElements, NNodes, nloc, PETSC_TRUE,
                                    _ENList.data(), ndims, _coords.data(), plex); assert(ierr==0);

    /* Mark boundary faces */
    DMLabel label;
    ierr = DMPlexCreateLabel(*plex, "boundary_faces"); assert(ierr==0);
    ierr = DMPlexGetLabel(*plex, "boundary_faces", &label); assert(ierr==0);
    ierr = DMPlexMarkBoundaryFaces(*plex, label); assert(ierr==0);

    /* Now apply our boundary IDs to the new Plex */
    PetscInt cell, ci, f, v, vStart, vEnd, fStart, fEnd;
    PetscInt bid, vertex, nclosure, *closure=NULL;
    const PetscInt *facets=NULL;
    PetscBool incident;
    ierr = DMPlexCreateLabel(*plex, "boundary_ids"); assert(ierr==0);
    ierr = DMPlexGetLabel(*plex, "boundary_ids", &label); assert(ierr==0);
    ierr = DMPlexGetDepthStratum(*plex, 0, &vStart, &vEnd);assert(ierr==0);
    ierr = DMPlexGetHeightStratum(*plex, 1, &fStart, &fEnd);assert(ierr==0);  // facets
    for (cell=0; cell<NElements; cell++){
      ierr = DMPlexGetCone(*plex, cell, &facets);

      for (v=0; v<nloc; v++) {
        bid = boundary[cell*nloc+v];
        if (bid > 0){
          /* Find the Plex facet that is not included in the "star"
             (inverse closure) of the non-incident vertex */
          vertex = _ENList[cell*nloc+v] + vStart;
          ierr = DMPlexGetTransitiveClosure(*plex, vertex, PETSC_FALSE, &nclosure, &closure);

          for (f=0; f<nloc; f++) {
            incident = PETSC_FALSE;
            for (ci=0; ci<nclosure; ci++) {
              if (facets[f] == closure[2*ci]) {incident=PETSC_TRUE; break;}
            }
            if (!incident) {
              ierr = DMLabelSetValue(label, facets[f], bid); break;
            }
          }
        }
      }
    }

    /* After building the Plex DAG in parallel we need to build the PetscSF
       that maps all local halo points (leaves) to remote points (roots).
    */
    if (num_processes <= 1) return;

    PetscSection point_owners;
    plex_create_point_ownership(*plex, &point_owners);

    PetscSection remote_roots;
    plex_map_remote_roots(*plex, &remote_roots);
  }

  /* This routine establishes the ownership of all points in the DAG
     fom Pragmatic's vertex ownership. The convention is to assign a
     point to the lowest rank of all associated vertex owners. The
     resulting ownership is stored in a PetscSection.
  */
  void plex_create_point_ownership(DM plex, PetscSection *point_owners) {
    PetscErrorCode ierr;
    PetscInt p, pStart, pEnd, cStart, cEnd, v, vStart, vEnd;
    PetscInt vertex, nhalo_points, owner, ci, nclosure, *closure = NULL;
    ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);assert(ierr==0);
    ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);assert(ierr==0);

    /* Build a section to store point owners */
    ierr = DMPlexGetChart(plex, &pStart, &pEnd); assert(ierr==0);
    ierr = PetscSectionCreate(_mpi_comm, point_owners); assert(ierr==0);
    ierr = PetscSectionSetChart(*point_owners, pStart, pEnd); assert(ierr==0);
    ierr = PetscSectionSetUp(*point_owners); assert(ierr==0);
    for (p=pStart; p<pEnd; p++) {
      ierr = PetscSectionSetDof(*point_owners, p, 1); assert(ierr==0);
      ierr = PetscSectionSetOffset(*point_owners, p, rank); assert(ierr==0);
    }

    /* Establish ownership of each Plex point from our halo recv lists */
    std::vector<PetscInt> vertex_star;
    nhalo_points = 0;
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      for (v=0; v<recv[proc].size(); v++) {
        /* Assign ownership of halo vertex */
        vertex = recv[proc][v] + vStart;
        ierr = PetscSectionSetOffset(*point_owners, vertex, proc); assert(ierr==0);
        nhalo_points++;

        /* Store the vertex star (inverse closure) for later traversal */
        ierr = DMPlexGetTransitiveClosure(plex, vertex, PETSC_FALSE, &nclosure, &closure);
        vertex_star.resize(nclosure);
        for (ci=0; ci<nclosure; ci++) vertex_star[ci] = closure[2*ci];

        /* For each point in the vertex star determine the owner as
           the minimum owner of all vertices in the point's closure */
        for (p=0; p<vertex_star.size(); p++) {
          /* Skip the original halo vertex */
          if (vStart <= vertex_star[p] && vertex_star[p] < vEnd) continue;

          owner = std::numeric_limits<int>::max();
          ierr = DMPlexGetTransitiveClosure(plex, vertex_star[p], PETSC_TRUE, &nclosure, &closure);
          for (ci=0; ci<nclosure; ci++) {
            if (vStart <= closure[2*ci] && closure[2*ci] < vEnd) {
              owner = std::min(owner, node_owner[ closure[2*ci]-vStart ]);
            }
          }
          ierr = PetscSectionSetOffset(*point_owners, vertex_star[p], owner); assert(ierr==0);
          if (owner != rank) nhalo_points++;
        }
      }
    }
  }

  /* This routine matches the local plex IDs (leafs) of all points
     in the halo/non-core region with their remote Plex IDs (roots)
     on the sending rank.

     This is done by first matching the local vertex IDs in the
     closure(star(v)), forall v in the "send" halo via Pragmatic's GNN.
     After that we can match entities in higher stratas (edges, facets,
     cells) in turn by communicating the cone of the local entity and
     matching it to the local leaf on the remote rank.
  */
  void plex_map_remote_roots(DM plex, PetscSection *remote_roots) {

    /* Build section to store remote roots */
    PetscErrorCode ierr;
    PetscInt p, pStart, pEnd, cStart, cEnd, vStart, vEnd;
    PetscInt v, ci, root, leaf, depth, size, max_cone_size, max_support_size;
    PetscInt nmatch, njoin, *matches=NULL, nclosure, *closure=NULL;
    const PetscInt *cone=NULL, *join=NULL;
    std::vector< std::set<index_t> > adjacent_points(num_processes);
    std::vector< std::set<index_t> > stratum_points(num_processes);
    std::map<index_t,index_t> root_leaf_map;
    std::map<index_t,index_t>::iterator entry;

    /* Establish Plex sizes */
    ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);assert(ierr==0);
    ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);assert(ierr==0);
    ierr = DMPlexGetMaxSizes(plex, &max_cone_size, &max_support_size); assert(ierr==0);
    ierr = PetscMalloc(max_cone_size, &matches); assert(ierr==0);

    /* Build remote_root section */
    ierr = DMPlexGetChart(plex, &pStart, &pEnd); assert(ierr==0);
    ierr = PetscSectionCreate(_mpi_comm, remote_roots); assert(ierr==0);
    ierr = PetscSectionSetChart(*remote_roots, pStart, pEnd); assert(ierr==0);
    ierr = PetscSectionSetUp(*remote_roots); assert(ierr==0);
    for (p=pStart; p<pEnd; p++) {
      ierr = PetscSectionSetDof(*remote_roots, p, 1); assert(ierr==0);
      ierr = PetscSectionSetOffset(*remote_roots, p, -1); assert(ierr==0);
    }

    /* Communicate send halo vertex IDs */
    std::vector< std::vector<index_t> > send_gnn(num_processes);
    std::vector< std::vector<index_t> > recv_gnn(num_processes);
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      send_gnn[proc].resize( 2*send[proc].size() );
      for (v=0; v<send[proc].size(); v++) {
        /* Send pairs of Plex ID and GNN */
        send_gnn[proc][2*v] = vStart + send[proc][v];
        send_gnn[proc][2*v+1] = lnn2gnn[ send[proc][v] ];
      }
    }
    send_all_to_all(send_gnn, &recv_gnn);

    /* Compute the local leaf -> root mapping for "send" halo */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      for (v=0; v<recv_gnn[proc].size(); v+=2) {
        root = recv_gnn[proc][v];
        leaf = vStart + recv_map[proc][ recv_gnn[proc][v+1] ];
        ierr = PetscSectionSetOffset(*remote_roots, leaf, root); assert(ierr==0);
        root_leaf_map.insert( std::pair<index_t,index_t>(root, leaf) );
      }
    }

    /* Communicate recv halo vertex IDs */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      send_gnn[proc].resize( 2*recv[proc].size() );
      for (v=0; v<recv[proc].size(); v++) {
        /* Send pairs of Plex ID and GNN */
        send_gnn[proc][2*v] = vStart + recv[proc][v];
        send_gnn[proc][2*v+1] = lnn2gnn[ recv[proc][v] ];
      }
    }
    send_all_to_all(send_gnn, &recv_gnn);

    /* Compute the local leaf -> root mapping for "recv" halo */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      for (v=0; v<recv_gnn[proc].size(); v+=2) {
        root = recv_gnn[proc][v];
        leaf = vStart + send_map[proc][ recv_gnn[proc][v+1] ];
        ierr = PetscSectionSetOffset(*remote_roots, leaf, root); assert(ierr==0);
        root_leaf_map.insert( std::pair<index_t,index_t>(root, leaf) );
      }
    }

    /* Now we can to establish the leaf->root mapping for edges, faces, etc.
       For this we first need to derive all points adjacent to our "send"
       vertex list by finding the closure(star(v)), where v is a "send" vertex. */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) continue;

      /* Collect all points in star(v), where v is a "send" vertex */
      for (v=0; v<send[proc].size(); v++) {
        ierr = DMPlexGetTransitiveClosure(plex, vStart+send[proc][v], PETSC_FALSE, &nclosure, &closure);
        for (ci=0; ci<nclosure; ci++)  adjacent_points[proc].insert(closure[2*ci]);
      }

      /* Add the closure of all cells in star(v) */
      for (std::set<index_t>::iterator it=adjacent_points[proc].begin();
           it!= adjacent_points[proc].end(); it++) {
        if (cStart <= *it && *it < cEnd) {
          ierr = DMPlexGetTransitiveClosure(plex, *it, PETSC_TRUE, &nclosure, &closure);
          for (ci=0; ci<nclosure; ci++)  adjacent_points[proc].insert(closure[2*ci]);
        }
      }
    }

    /* Finally, we can step through all layers (strata) above the vertices
       and reconstruct the local_lnn(leaf)->remote_lnn(root) mapping. */
    for (depth = 1; depth < ndims+1; depth++) {
      ierr = DMPlexGetDepthStratum(plex, depth, &pStart, &pEnd);assert(ierr==0);
      for (int proc=0; proc<num_processes; proc++) {
        if (proc == rank) continue;

        /* Go through the adjacent_points and count the entities in this stratum/layer */
        stratum_points[proc].clear();
        for (v=0; v<send[proc].size(); v++) {
          for (std::set<index_t>::iterator it=adjacent_points[proc].begin();
               it!= adjacent_points[proc].end(); it++) {
            if (pStart <= *it && *it < pEnd) stratum_points[proc].insert(*it);
          }
        }

        /* Pack cone of each edge touching a "send" vertex; +1 is the actual cone size */
        send_gnn[proc].resize( stratum_points[proc].size()*(max_cone_size+2) );
        v = 0;
        for (std::set<index_t>::iterator it=stratum_points[proc].begin();
             it!= stratum_points[proc].end(); it++) {
          ierr = DMPlexGetConeSize(plex, *it, &size); assert(ierr==0);
          ierr = DMPlexGetCone(plex, *it, &cone); assert(ierr==0);
          send_gnn[proc][v] = *it;
          send_gnn[proc][v+1] = size;
          for (ci=0; ci<size; ci++)  send_gnn[proc][v+2+ci] = cone[ci];
          v += max_cone_size + 2;
        }
      }
      send_all_to_all(send_gnn, &recv_gnn);

      for (int proc=0; proc<num_processes; proc++) {
        if (proc == rank) continue;

        /* Now we try to match the received cone to one of our local Plex points */
        for (v=0; v<recv_gnn[proc].size(); v+=max_cone_size+2) {
          root = recv_gnn[proc][v];
          size = recv_gnn[proc][v+1];
          nmatch = 0;
          for (ci=2; ci<size+2; ci++) {
            /* Go through the remote cone and find matches with locally know points */
            entry = root_leaf_map.find( recv_gnn[proc][v+ci] );
            if (entry != root_leaf_map.end()) {
              matches[nmatch] = entry->second;
              nmatch++;
            }
          }

          /* If we know all points in the remote cone, we can find the local counterpart */
          if (nmatch == size) {
            ierr = DMPlexGetJoin(plex, nmatch, matches, &njoin, &join); assert(ierr==0);
            if (njoin > 0) {
              assert(njoin == 1);
              ierr = PetscSectionSetOffset(*remote_roots, join[0], root); assert(ierr==0);
              root_leaf_map.insert( std::pair<index_t,index_t>(root, join[0]) );
            }
          }
        }
      }
    }
    ierr = PetscFree(matches); assert(ierr==0);
  }

  void send_all_to_all(std::vector< std::vector<index_t> > send_vec,
                       std::vector< std::vector<index_t> > *recv_vec) {
    int ierr, recv_size, tag = 123456;
    std::vector<MPI_Status> status(num_processes);
    std::vector<MPI_Request> send_req(num_processes);
    std::vector<MPI_Request> recv_req(num_processes);

    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) {send_req[proc] = MPI_REQUEST_NULL; continue;}

      ierr = MPI_Isend(send_vec[proc].data(), send_vec[proc].size(), MPI_INDEX_T,
                       proc, tag, _mpi_comm, &send_req[proc]); assert(ierr==0);
    }

    /* Receive send list from remote proc */
    for (int proc=0; proc<num_processes; proc++) {
      if (proc == rank) {recv_req[proc] = MPI_REQUEST_NULL; continue;}

      ierr = MPI_Probe(proc, tag, _mpi_comm, &(status[proc])); assert(ierr==0);
      ierr = MPI_Get_count(&(status[proc]), MPI_INT, &recv_size); assert(ierr==0);
      (*recv_vec)[proc].resize(recv_size);
      MPI_Irecv((*recv_vec)[proc].data(), recv_size, MPI_INT, proc,
                tag, _mpi_comm, &recv_req[proc]); assert(ierr==0);
    }

    MPI_Waitall(num_processes, &(send_req[0]), &(status[0]));
    MPI_Waitall(num_processes, &(recv_req[0]), &(status[0]));
  }

 private:
  template<typename _real_t> friend class MetricField2D;
  template<typename _real_t> friend class MetricField3D;
  template<typename _real_t> friend class Smooth2D;
  template<typename _real_t> friend class Smooth3D;
  template<typename _real_t> friend class Swapping2D;
  template<typename _real_t> friend class Swapping3D;
  template<typename _real_t> friend class Coarsen2D;
  template<typename _real_t> friend class Coarsen3D;
  template<typename _real_t> friend class Refine2D;
  template<typename _real_t> friend class Refine3D;
  template<typename _real_t> friend class VTKTools;
  template<typename _real_t> friend class CUDATools;

  struct DeferredOperations{
    std::vector<index_t> addNN; // addNN -> [i, n] : Add node n to NNList[i].
    std::vector<index_t> remNN; // remNN -> [i, n] : Remove node n from NNList[i].
    std::vector<index_t> addNE; // addNE -> [i, n] : Add element n to NEList[i].
    std::vector<index_t> remNE; // remNE -> [i, n] : Remove element n from NEList[i].
    std::vector<index_t> propagation_vector; // [i] : Mark Coarseninig::dynamic_vertex[i]=-2.
    std::vector<index_t> propagation_set; // [i, n] : Mark Swapping::marked_edges[i].insert(n).
    std::vector<index_t> reset_colour; // [i] : Set Colouring::node_colour[i]=-1.
  };

  void _init(int _NNodes, int _NElements, const index_t *globalENList,
             const real_t *x, const real_t *y, const real_t *z,
             const index_t *lnn2gnn, const index_t *owner_range){
    num_processes = 1;
    rank=0;

    NElements = _NElements;
    NNodes = _NNodes;

#ifdef HAVE_MPI
    MPI_Comm_size(_mpi_comm, &num_processes);
    MPI_Comm_rank(_mpi_comm, &rank);

    // Assign the correct MPI data type to MPI_INDEX_T
    mpi_type_wrapper<index_t> mpi_index_t_wrapper;
    MPI_INDEX_T = mpi_index_t_wrapper.mpi_type;
#endif

    nthreads = pragmatic_nthreads();

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
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    boost::unordered_map<index_t, index_t> gnn2lnn;
#else
    std::map<index_t, index_t> gnn2lnn;
#endif
    if(num_processes==1){
      ENList = globalENList;
    }else{
#ifdef HAVE_MPI
      assert(lnn2gnn!=NULL);
      for(size_t i=0;i<(size_t)NNodes;i++){
        gnn2lnn[lnn2gnn[i]] = i;
      }

      std::vector< std::set<index_t> > recv_set(num_processes);
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
      recv_map.resize(num_processes);
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
      send_map.resize(num_processes);
      std::vector<MPI_Request> request(num_processes*2);
      for(int i=0;i<num_processes;i++){
        if((i==rank)||(send_size[i]==0)){
          request[i] =  MPI_REQUEST_NULL;
        }else{
          send[i].resize(send_size[i]);
          MPI_Irecv(&(send[i][0]), send_size[i], MPI_INDEX_T, i, 0, _mpi_comm, &(request[i]));
        }
      }

      // Non-blocking sends.
      for(int i=0;i<num_processes;i++){
        if((i==rank)||(recv_size[i]==0)){
          request[num_processes+i] =  MPI_REQUEST_NULL;
        }else{
          MPI_Isend(&(recv[i][0]), recv_size[i], MPI_INDEX_T, i, 0, _mpi_comm, &(request[num_processes+i]));
        }
      }

      std::vector<MPI_Status> status(num_processes*2);
      MPI_Waitall(num_processes, &(request[0]), &(status[0]));
      MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

      for(int j=0;j<num_processes;j++){
        for(int k=0;k<recv_size[j];k++){
          index_t gnn = recv[j][k];
          index_t lnn = gnn2lnn[gnn];
          recv_map[j][gnn] = lnn;
          recv[j][k] = lnn;
        }

        for(int k=0;k<send_size[j];k++){
          index_t gnn = send[j][k];
          index_t lnn = gnn2lnn[gnn];
          send_map[j][gnn] = lnn;
          send[j][k] = lnn;
        }
      }

      ENList = localENList;
#endif
    }

    _ENList.resize(NElements*nloc);
    _coords.resize(NNodes*ndims);
    metric.resize(NNodes*msize);
    NNList.resize(NNodes);
    NEList.resize(NNodes);
    node_owner.resize(NNodes);
    this->lnn2gnn.resize(NNodes);
    deferred_operations.resize(nthreads);

    // TODO I don't know whether this method makes sense anymore.
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
          _coords[i*2  ] = x[i];
          _coords[i*2+1] = y[i];
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0;i<(int)NNodes;i++){
          _coords[i*3  ] = x[i];
          _coords[i*3+1] = y[i];
          _coords[i*3+2] = z[i];
        }
      }

      // Each thread allocates nthreads DeferredOperations
      // structs, one for each OMP thread.
      deferred_operations[pragmatic_thread_id()].resize((defOp_scaling_factor*nthreads));

#pragma omp single nowait
      {
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
      }

      // Set the orientation of elements.
#pragma omp single
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

      if(ndims==2){
#pragma omp for schedule(static)
        for(size_t i=1;i<(size_t)NElements;i++){
          const int *n=get_element(i);
          assert(n[0]>=0);

          double volarea = property->area(get_coords(n[0]),
                                          get_coords(n[1]),
                                          get_coords(n[2]));

          if(volarea<0)
            invert_element(i);
        }
      }else{
#pragma omp for schedule(static)
        for(size_t i=1;i<(size_t)NElements;i++){
          const int *n=get_element(i);
          assert(n[0]>=0);

          double volarea = property->volume(get_coords(n[0]),
                                            get_coords(n[1]),
                                            get_coords(n[2]),
                                            get_coords(n[2]));

          if(volarea<0)
            invert_element(i);
        }
      }

      // create_adjacency is meant to be called from inside a parallel region
      create_adjacency();
    }

    create_global_node_numbering();
  }

  /// Create required adjacency lists.
  void create_adjacency(){
    int tid = pragmatic_thread_id();

    for(size_t i=0; i<NElements; i++){
      if(_ENList[i*nloc]<0)
        continue;

      for(size_t j=0;j<nloc;j++){
        index_t nid_j = _ENList[i*nloc+j];
        if((nid_j%nthreads)==tid){
          NEList[nid_j].insert(NEList[nid_j].end(), i);
          for(size_t k=0;k<nloc;k++){
            if(j!=k){
              index_t nid_k = _ENList[i*nloc+k];
              NNList[nid_j].push_back(nid_k);
            }
          }
        }
      }
    }

#pragma omp barrier

    // Finalise
#pragma omp for schedule(static)
    for(size_t i=0;i<NNodes;i++){
      if(NNList[i].empty())
        continue;

      std::vector<index_t> *nnset = new std::vector<index_t>();

      std::sort(NNList[i].begin(),NNList[i].end());
      std::unique_copy(NNList[i].begin(), NNList[i].end(), std::inserter(*nnset, nnset->begin()));

      NNList[i].swap(*nnset);
      delete nnset;
    }
  }

  /*
   * Park & Miller (aka Lehmer) pseudo-random number generation. Possible bug if
   * index_t is a datatype longer than 32 bits. However, in the context of a single
   * MPI node, it is highly unlikely that index_t will ever need to be longer.
   * A 64-bit datatype makes sense only for global node numbers, not local.
   */
  inline uint32_t hash(const uint32_t id) const{
    return ((uint64_t)id * 279470273UL) % 4294967291UL;
  }

  inline void deferred_addNN(index_t i, index_t n, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNN.push_back(i);
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNN.push_back(n);
  }

  inline void deferred_remNN(index_t i, index_t n, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNN.push_back(i);
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNN.push_back(n);
  }

  inline void deferred_addNE(index_t i, index_t n, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE.push_back(i);
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE.push_back(n);
  }

  inline void deferred_remNE(index_t i, index_t n, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNE.push_back(i);
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNE.push_back(n);
  }

  inline void deferred_propagate_coarsening(index_t i, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].propagation_vector.push_back(i);
  }

  inline void deferred_propagate_swapping(index_t i, index_t n, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].propagation_set.push_back(i);
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].propagation_set.push_back(n);
  }

  inline void deferred_reset_colour(index_t i, size_t tid){
    deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].reset_colour.push_back(i);
  }

  void commit_deferred(size_t vtid){
    for(int i=0; i<nthreads; ++i){
      DeferredOperations& pending = deferred_operations[i][vtid];

      // Commit removals from NNList
      for(typename std::vector<index_t>::const_iterator it=pending.remNN.begin(); it!=pending.remNN.end(); it+=2){
        typename std::vector<index_t>::iterator position;
        position = std::find(NNList[*it].begin(), NNList[*it].end(), *(it+1));
        assert(position != NNList[*it].end());
        NNList[*it].erase(position);
      }
      pending.remNN.clear();

      // Commit additions to NNList
      for(typename std::vector<index_t>::const_iterator it=pending.addNN.begin(); it!=pending.addNN.end(); it+=2){
        NNList[*it].push_back(*(it+1));
      }
      pending.addNN.clear();

      // Commit removals from NEList
      for(typename std::vector<index_t>::const_iterator it=pending.remNE.begin(); it!=pending.remNE.end(); it+=2){
        NEList[*it].erase(*(it+1));
      }
      pending.remNE.clear();

      // Commit additions to NEList
      for(typename std::vector<index_t>::const_iterator it=pending.addNE.begin(); it!=pending.addNE.end(); it+=2){
        NEList[*it].insert(*(it+1));
      }
      pending.addNE.clear();
    }
  }

  void commit_coarsening_propagation(index_t *dynamic_vertex, size_t vtid){
    for(int i=0; i<nthreads; ++i){
      DeferredOperations& pending = deferred_operations[i][vtid];

      for(typename std::vector<index_t>::const_iterator it=pending.propagation_vector.begin(); it!=pending.propagation_vector.end(); ++it)
        dynamic_vertex[*it] = -2;

      pending.propagation_vector.clear();
    }
  }

  void commit_swapping_propagation(std::vector< std::set<index_t> >&marked_edges , size_t vtid){
    for(int i=0; i<nthreads; ++i){
      DeferredOperations& pending = deferred_operations[i][vtid];

      for(typename std::vector<index_t>::const_iterator it=pending.propagation_set.begin(); it!=pending.propagation_set.end(); it+=2)
        marked_edges[*it].insert(*(it+1));

      pending.propagation_set.clear();
    }
  }

  void commit_colour_reset(int *node_colour, size_t vtid){
    for(int i=0; i<nthreads; ++i){
      DeferredOperations& pending = deferred_operations[i][vtid];

      for(typename std::vector<index_t>::const_iterator it=pending.reset_colour.begin(); it!=pending.reset_colour.end(); ++it)
        node_colour[*it] = -1;

      pending.reset_colour.clear();
    }
  }

  void trim_halo(){
    std::set<index_t> recv_halo_temp, send_halo_temp;

    // Traverse all vertices V in all recv[i] vectors. Vertices in send[i] belong by definition to *this* MPI process,
    // so all elements adjacent to them either belong exclusively to *this* process or cross partitions.
    for(int i=0;i<num_processes;i++){
      if(recv[i].size()==0)
        continue;

      std::vector<index_t> recv_temp;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
      boost::unordered_map<index_t, index_t> recv_map_temp;
#else
      std::map<index_t, index_t> recv_map_temp;
#endif

      for(typename std::vector<index_t>::const_iterator vit = recv[i].begin(); vit != recv[i].end(); ++vit){
        // For each vertex, traverse a copy of the vertex's NEList.
        // We need a copy because erase_element modifies the original NEList.
        std::set<index_t> NEList_copy = NEList[*vit];
        for(typename std::set<index_t>::const_iterator eit = NEList_copy.begin(); eit != NEList_copy.end(); ++eit){
          // Check whether all vertices comprising the element belong to another MPI process.
          std::vector<index_t> n(nloc);
          get_element(*eit, &n[0]);
          if(n[0] < 0)
            continue;

          // If one of the vertices belongs to *this* partition, the element should be retained.
          bool to_be_deleted = true;
          for(size_t j=0; j<nloc; ++j)
            if(is_owned_node(n[j])){
              to_be_deleted = false;
              break;
            }

          if(to_be_deleted){
            erase_element(*eit);

            // Now check whether one of the edges must be deleted
            for(size_t j=0; j<nloc; ++j){
              for(size_t k=j+1; k<nloc; ++k){
                std::set<index_t> intersection;
                std::set_intersection(NEList[n[j]].begin(), NEList[n[j]].end(), NEList[n[k]].begin(), NEList[n[k]].end(),
                    std::inserter(intersection, intersection.begin()));

                // If these two vertices have no element in common anymore,
                // then the corresponding edge does not exist, so update NNList.
                if(intersection.empty()){
                  typename std::vector<index_t>::iterator it;
                  it = std::find(NNList[n[j]].begin(), NNList[n[j]].end(), n[k]);
                  NNList[n[j]].erase(it);
                  it = std::find(NNList[n[k]].begin(), NNList[n[k]].end(), n[j]);
                  NNList[n[k]].erase(it);
                }
              }
            }
          }
        }

        // If this vertex is no longer part of any element, then it is safe to be removed.
        if(NEList[*vit].empty()){
          // Update NNList of all neighbours
          for(typename std::vector<index_t>::const_iterator neigh_it = NNList[*vit].begin(); neigh_it != NNList[*vit].end(); ++neigh_it){
            typename std::vector<index_t>::iterator it = std::find(NNList[*neigh_it].begin(), NNList[*neigh_it].end(), *vit);
            NNList[*neigh_it].erase(it);
          }

          erase_vertex(*vit);
        }else{
          // We will keep this vertex, so put it into recv_halo_temp.
          recv_temp.push_back(*vit);
          recv_map_temp[lnn2gnn[*vit]] = *vit;
          recv_halo_temp.insert(*vit);
        }
      }

      recv[i].swap(recv_temp);
      recv_map[i].swap(recv_map_temp);
    }

    // Once all recv[i] have been traversed, update recv_halo.
    recv_halo.swap(recv_halo_temp);

    // Traverse all vertices V in all send[i] vectors.
    // If none of V's neighbours are owned by the i-th MPI process, it means that the i-th process
    // has removed V from its recv_halo, so remove the vertex from send[i].
    for(int i=0;i<num_processes;i++){
      if(send[i].size()==0)
        continue;

      std::vector<index_t> send_temp;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
      boost::unordered_map<index_t, index_t> send_map_temp;
#else
      std::map<index_t, index_t> send_map_temp;
#endif

      for(typename std::vector<index_t>::const_iterator vit = send[i].begin(); vit != send[i].end(); ++vit){
        bool to_be_deleted = true;
        for(typename std::vector<index_t>::const_iterator neigh_it = NNList[*vit].begin(); neigh_it != NNList[*vit].end(); ++neigh_it)
          if(node_owner[*neigh_it] == i){
            to_be_deleted = false;
            break;
          }

        if(!to_be_deleted){
          send_temp.push_back(*vit);
          send_map_temp[lnn2gnn[*vit]] = *vit;
          send_halo_temp.insert(*vit);
        }
      }

      send[i].swap(send_temp);
      send_map[i].swap(send_map_temp);
    }

    // Once all send[i] have been traversed, update send_halo.
    send_halo.swap(send_halo_temp);
  }

  void clear_invisible(std::vector<index_t>& invisible_vertices){
    for(size_t i=0; i<invisible_vertices.size(); ++i){
      index_t v = invisible_vertices[i];

      // Traverse a copy of the vertex's NEList.
      // We need a copy because erase_element modifies the original NEList.
      std::set<index_t> NEList_copy = NEList[v];
      for(typename std::set<index_t>::const_iterator eit = NEList_copy.begin(); eit != NEList_copy.end(); ++eit){
        // If the vertex is invisible, then all elements adjacent to it are also invisible - remove them immediately.
        erase_element(*eit);
      }

      // This vertex is by definition invisible, so remove it immediately. Update NNList of all neighbours.
      for(typename std::vector<index_t>::const_iterator neigh_it = NNList[v].begin(); neigh_it != NNList[v].end(); ++neigh_it){
        typename std::vector<index_t>::iterator it = std::find(NNList[*neigh_it].begin(), NNList[*neigh_it].end(), v);
        NNList[*neigh_it].erase(it);
      }
      erase_vertex(v);
    }
  }

  void create_global_node_numbering(){
    if(num_processes>1){
      // Calculate the global numbering offset for this partition.
      int gnn_offset;
      int NPNodes = NNodes - recv_halo.size();
      MPI_Scan(&NPNodes, &gnn_offset, 1, MPI_INT, MPI_SUM, get_mpi_comm());
      gnn_offset-=NPNodes;

      // Write global node numbering and ownership for nodes assigned to local process.
      for(index_t i=0; i < (index_t) NNodes; i++){
        if(recv_halo.count(i)){
          lnn2gnn[i] = 0;
        }else{
          lnn2gnn[i] = gnn_offset++;
          node_owner[i] = rank;
        }
      }

      // Update GNN's for the halo nodes.
      halo_update<int, 1>(_mpi_comm, send, recv, lnn2gnn);

      // Finish writing node ownerships.
      for(int i=0;i<num_processes;i++){
        for(std::vector<int>::const_iterator it=recv[i].begin();it!=recv[i].end();++it){
          node_owner[*it] = i;
        }
      }
    }else{
      memset(&node_owner[0], 0, NNodes*sizeof(int));
      for(index_t i=0; i < (index_t) NNodes; i++)
        lnn2gnn[i] = i;
    }
  }

  void create_gappy_global_numbering(size_t pNElements){
    // We expect to have NElements_predict/2 nodes in the partition,
    // so let's reserve 10 times more space for global node numbers.
    index_t gnn_reserve = 5*pNElements;
    MPI_Scan(&gnn_reserve, &gnn_offset, 1, MPI_INDEX_T, MPI_SUM, _mpi_comm);
    gnn_offset -= gnn_reserve;

    for(size_t i=0; i<NNodes; ++i){
      if(node_owner[i] == rank)
        lnn2gnn[i] = gnn_offset+i;
      else
        lnn2gnn[i] = -1;
    }

    halo_update<int, 1>(_mpi_comm, send, recv, lnn2gnn);

    for(int i=0;i<num_processes;i++){
      send_map[i].clear();
      for(std::vector<int>::const_iterator it=send[i].begin();it!=send[i].end();++it){
        assert(node_owner[*it]==rank);
        send_map[i][lnn2gnn[*it]] = *it;
      }

      recv_map[i].clear();
      for(std::vector<int>::const_iterator it=recv[i].begin();it!=recv[i].end();++it){
        node_owner[*it] = i;
        recv_map[i][lnn2gnn[*it]] = *it;
      }
    }
  }

  void update_gappy_global_numbering(std::vector<size_t>& recv_cnt, std::vector<size_t>& send_cnt){
#ifdef HAVE_MPI
    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);

    // Setup non-blocking receives.
    std::vector< std::vector<index_t> > recv_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if(recv_cnt[i]==0){
        request[i] =  MPI_REQUEST_NULL;
      }else{
        recv_buff[i].resize(recv_cnt[i]);
        MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_INDEX_T, i, 0, _mpi_comm, &(request[i]));
      }
    }

    // Non-blocking sends.
    std::vector< std::vector<index_t> > send_buff(num_processes);
    for(int i=0;i<num_processes;i++){
      if(send_cnt[i]==0){
        request[num_processes+i] = MPI_REQUEST_NULL;
      }else{
        for(typename std::vector<index_t>::const_iterator it=send[i].end()-send_cnt[i];it!=send[i].end();++it)
          send_buff[i].push_back(lnn2gnn[*it]);

        MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_INDEX_T, i, 0, _mpi_comm, &(request[num_processes+i]));
      }
    }

    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

    for(int i=0;i<num_processes;i++){
      int k=0;
      for(typename std::vector<index_t>::const_iterator it=recv[i].end()-recv_cnt[i];it!=recv[i].end();++it, ++k)
        lnn2gnn[*it] = recv_buff[i][k];
    }
#endif
  }

/* PETSc DMPlex objects do not adhere to the Fenics local numbering convention
   for simplices when deriving the transitive closure of cells. This routine
   performs a one time bulk re-ordering of all cell closures in a DMPlex object
   and store the Plex points for the according vertices and facets in a
   vector<pair<PetscInt, PetscInt>>.

   Input Parameters:
   plex -  The DM object holding mesh topology
   vertex_numbering:
           Section providing the global (universal) vertex numbering
           on which the Fenics local numbering is based
   cell_closure:
           Transitive closures for each cell reordered accordign to Fenics
           conventions. We allocate, but user is responsible for deallocating.
   local_numbering:
           Vector of size(nloc*nlements) with a pair of Plex points <vertex, facet>
           for each nloc verteices/facets in each element according to Fenics convention.
  */
  void derive_fenics_local_numbering(DM plex, PetscSection vertex_numbering, size_t nloc,
                                    std::vector<std::pair<PetscInt,PetscInt>> *local_numbering)
  {
    PetscInt cStart, cEnd, vStart, vEnd, fStart, fEnd;
    PetscInt dim, cell, vertex, v_per_cell, nfacet_vertices, nclosure, *closure = NULL;
    PetscInt f, ci, vi, fi;
    PetscInt *vertices = NULL, *v_global = NULL, *facets = NULL, *facet_vertices = NULL;
    PetscBool incident;
    /* Need to do better error checking in here */
    PetscErrorCode ierr;

    ierr = DMPlexGetDimension(plex, &dim);assert(ierr==0);
    ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);assert(ierr==0);  // vertices
    ierr = DMPlexGetHeightStratum(plex, 1, &fStart, &fEnd);assert(ierr==0);  // facets
    ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);assert(ierr==0);  // cells

    v_per_cell = (PetscInt) nloc;
    ierr = PetscMalloc(v_per_cell, &vertices);assert(ierr==0);
    ierr = PetscMalloc(v_per_cell, &v_global);assert(ierr==0);
    ierr = PetscMalloc(v_per_cell, &facets);assert(ierr==0);
    ierr = PetscMalloc(v_per_cell - 1, &facet_vertices);assert(ierr==0);

    local_numbering->resize((cEnd-cStart) * nloc);

    for (cell=cStart; cell<cEnd; cell++) {
      ierr = DMPlexGetTransitiveClosure(plex, cell, PETSC_TRUE, &nclosure, &closure);

      /* Find vertices and translate universal numbers */
      vi = 0; fi = 0;
      for (ci=0; ci<nclosure; ci++) {
        if (vStart <= closure[2*ci] && closure[2*ci] < vEnd) {
          vertices[vi] = closure[2*ci];
          ierr = PetscSectionGetOffset(vertex_numbering, closure[2*ci], &vertex);

          /* Correct -ve offsets for non-owned vertices */
          if (vertex >= 0) { v_global[vi] = vertex; }
          else { v_global[vi] = -(vertex+1);}
          vi++;
        }

        if (fStart <= closure[2*ci] && closure[2*ci] < fEnd) {
          facets[fi] = closure[2*ci]; fi++;
        }
      }

      /* Sort vertices by universal number */
      ierr = PetscSortIntWithArray(v_per_cell, v_global, vertices);

      for (vi=0; vi<v_per_cell; vi++) {
        (*local_numbering)[cell*nloc + vi].first = vertices[vi];
      }

      for (f=0; f<nloc; f++) {
        /* Get all vertices associated with this facet */
        ierr = DMPlexGetTransitiveClosure(plex, facets[f], PETSC_TRUE, &nclosure, &closure);
        vi = 0;
        for (ci=0; ci<nclosure; ci++) {
          if (vStart <= closure[2*ci] && closure[2*ci] < vEnd) {
            facet_vertices[vi] = closure[2*ci]; vi++;
          }
        }
        nfacet_vertices = vi;

        /* Now we look for the non-incident (opposite) cell vertex */
        for (vi=0; vi<v_per_cell; vi++) {
          incident = PETSC_FALSE;
          for (fi=0; fi<nfacet_vertices; fi++) {
            if (vertices[vi] == facet_vertices[fi]) {
              incident = PETSC_TRUE; break;
            }
          }
          if (!incident) {
            /* We have found the non-incident vertex for this facet */
            (*local_numbering)[cell*nloc + vi].second = facets[f]; break;
          }
        }
      }
    }

    PetscFree(vertices);
    PetscFree(v_global);
    PetscFree(facets);
    PetscFree(facet_vertices);
  }

  // This is used temporarily for 3D - it will be removed in the future.
  inline index_t get_new_vertex(index_t n0, index_t n1,
      std::vector< std::vector<index_t> > &refined_edges, const index_t *lnn2gnn) const{

    if(lnn2gnn[n0]>lnn2gnn[n1]){
      // Needs to be swapped because we want the lesser gnn first.
      index_t tmp_n0=n0;
      n0=n1;
      n1=tmp_n0;
    }

    for(size_t i=0;i<NNList[n0].size();i++){
      if(NNList[n0][i]==n1){
        return refined_edges[n0][i];
      }
    }

    return -1;
  }

  size_t ndims, nloc, msize;
  std::vector<index_t> _ENList;
  std::vector<real_t> _coords;

  size_t NNodes, NElements;

  // Boundary Label
  std::vector<int> boundary;

  // Adjacency lists
  std::vector< std::set<index_t> > NEList;
  std::vector< std::vector<index_t> > NNList;

  ElementProperty<real_t> *property;

  // Metric tensor field.
  std::vector<double> metric;

  // Parallel support.
  int rank, num_processes, nthreads;
  std::vector< std::vector<index_t> > send, recv;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
  std::vector< boost::unordered_map<index_t, index_t> > send_map, recv_map;
#else
  std::vector< std::map<index_t, index_t> > send_map, recv_map;
#endif
  std::set<index_t> send_halo, recv_halo;
  std::vector<int> node_owner;
  std::vector<index_t> lnn2gnn;
  //Deferred operations
  std::vector< std::vector<DeferredOperations> > deferred_operations;
  static const int defOp_scaling_factor = 32;

#ifdef HAVE_MPI
  MPI_Comm _mpi_comm;
  index_t gnn_offset;

  // MPI data type for index_t
  MPI_Datatype MPI_INDEX_T;
#endif
};

#endif
