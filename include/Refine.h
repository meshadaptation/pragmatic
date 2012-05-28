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
#include <deque>
#include <set>
#include <vector>
#include <limits>

#include <string.h>

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

    rank = 0;
    nprocs = 1;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
      MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
    }
#endif
  }
  
  /// Default destructor.
  ~Refine(){
    delete property;
  }

  /*! Perform one level of refinement See Figure 25; X Li et al, Comp
   * Methods Appl Mech Engrg 194 (2005) 4915-4950. The actual
   * templates used for 3D refinement follows Rupak Biswas, Roger
   * C. Strawn, "A new procedure for dynamic adaption of
   * three-dimensional unstructured grids", Applied Numerical
   * Mathematics, Volume 13, Issue 6, February 1994, Pages 437-452.
   */
  int refine(real_t L_max){
    cache_create();

    /* Establish global node numbering. Here it doesn't need to be
       continuous so we take various shortcuts.*/
    index_t NNodes = _mesh->get_number_nodes();
    std::map<index_t, index_t> gnn2lnn;
    std::vector<index_t> lnn2gnn(NNodes);
#ifdef HAVE_MPI
    if(nprocs>1){
      // Calculate the global numbering offset for this partition.
      index_t gnn_offset=NNodes;
      MPI_Scan(&NNodes, &gnn_offset, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
      gnn_offset-=NNodes;
      
      // Initialise the lnn2gnn numbering.
      for(int i=0;i<NNodes;i++)
        lnn2gnn[i] = gnn_offset+i;
      
      // Update halo values.
      _mesh->halo_update(&(lnn2gnn[0]), 1);

      // Create reverse lookup.
      for(int i=0;i<NNodes;i++)
        gnn2lnn[lnn2gnn[i]] = i;
    }
#endif

    /* Identify all halo elements. This can change every time we apply
       a level of refinement, thus, it is calculated here. */
    int NElements = _mesh->get_number_elements();
    std::map< int, std::deque<int> > halo_elements;
#ifdef HAVE_MPI
    if(nprocs>1){
      for(int i=0;i<NElements;i++){
        /* Check if this element has been erased - if so continue to
           next element.*/
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
        
        /* Find how many additional processes this element is resident
           on.*/
        std::set<int> residency;
        for(size_t j=0;j<nloc;j++){
          int owner = get_node_owner(n[j]);
          if(owner!=rank)
            residency.insert(owner);
        }
        
        for(std::set<int>::const_iterator it=residency.begin();it!=residency.end();++it){
          halo_elements[*it].push_back(i);
        }
      }
    }
#endif
    
    std::map<int, std::set< Edge<index_t> > > new_recv_halo;
    int refined_edges_size;
    
#ifdef _OPENMP
    // Initialise a dynamic vertex list
    std::vector< std::vector<index_t> > refined_edges(_mesh->_NNodes);
    std::vector< std::vector<real_t> > newCoords;
    std::vector< std::vector<real_t> > newMetric;
    std::vector< std::vector<index_t> > newElements;
    std::vector<unsigned int> threadIdx, splitCnt;
    unsigned int nthreads_omp;

    #pragma omp parallel reduction(+:refined_edges_size)
    {
      nthreads_omp = omp_get_num_threads();

      #pragma omp master
      {
      	newCoords.resize(nthreads_omp);
      	newMetric.resize(nthreads_omp);
      	newElements.resize(nthreads_omp);
      	threadIdx.resize(nthreads_omp);
      	splitCnt.resize(nthreads_omp);
      }
      #pragma omp barrier

      const unsigned int tid = omp_get_thread_num();
      splitCnt[tid] = 0;

      /* Loop through all edges and select them for refinement if
         it's length is greater than L_max in transformed space. */
      #pragma omp for schedule(dynamic)
      for(int i=0;i<(int)_mesh->NNList.size();++i){
      	for(index_t it = 0; it < (int)_mesh->NNList[i].size(); ++it){
          refined_edges[i].resize(2 * _mesh->NNList[i].size(), -1);
          index_t otherVertex = _mesh->NNList[i][it];
          if(i < otherVertex){
            double length = _mesh->calc_edge_length(i, otherVertex);
            if(length>L_max){
              Edge<index_t> edge(i, otherVertex);
              refined_edges[i][2*it]   = splitCnt[tid]++;
              refined_edges[i][2*it+1] = tid;
              refine_edge_omp(edge, newCoords[tid], newMetric[tid]);
              // Refining the edge invalidates NNList[i].end() in this
              // loop. It is a good idea to break at this point anyhow.
              break;
            }
          }
      	}
      }
      refined_edges_size = splitCnt[tid];
    }
#else
    // Initialise a dynamic vertex list
    std::map< Edge<index_t>, index_t> refined_edges;

    /* Loop through all edges and select them for refinement if
       it's length is greater than L_max in transformed space. */
    for(int i=0;i<(int)_mesh->NNList.size();++i){
      for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[i].begin();it!=_mesh->NNList[i].end();++it){
        if(i<*it){
          double length = _mesh->calc_edge_length(i, *it);
          if(length>L_max){
            Edge<index_t> edge(i, *it);
            refined_edges[edge] = refine_edge(edge);
            // Refining the edge invalidates NNList[i].end() in this
            // loop. It is a good idea to break at this point anyhow.
            break;
          }
        }
      }
    }
    refined_edges_size = refined_edges.size();
#endif

    /* If there are no edges to be refined globally then we can return
       at this point.
    */
#ifdef HAVE_MPI
    if(nprocs>1)
      MPI_Allreduce(MPI_IN_PLACE, &refined_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#endif
    if(refined_edges_size==0)
      return 0;
    
    /* Given the set of refined edges, apply additional edge-refinement
       to get a regular and conforming element refinement throughout
       the domain.*/
    if(ndims==3){
      for(;;){
      	int new_edges_size = 0;
#ifndef _OPENMP
      	typename std::set< Edge<index_t> > new_edges;
#endif
    		#pragma omp parallel for schedule(dynamic) reduction(+:new_edges_size)
      	for(int i=0;i<NElements;i++){
      		// Check if this element has been erased - if so continue to next element.
      		const int *n=_mesh->get_element(i);
      		if(n[0]<0)
      			continue;
                
      		// Find what edges have been split in this element.
#ifdef _OPENMP
      		typename std::vector< Edge<index_t> > split_set;
      		for(size_t j=0;j<nloc;j++){
      			for(size_t k=j+1;k<nloc;k++){
      				if(_mesh->get_new_vertex_omp(n[j], n[k], refined_edges) >= 0)
      					split_set.push_back(Edge<index_t>(n[j], n[k]));
      			}
      		}
#else
      		std::vector<typename std::map< Edge<index_t>, index_t>::const_iterator> split;
      		typename std::set< Edge<index_t> > split_set;
      		for(size_t j=0;j<nloc;j++){
      			for(size_t k=j+1;k<nloc;k++){
      				typename std::map< Edge<index_t>, index_t>::const_iterator it =
      						refined_edges.find(Edge<index_t>(n[j], n[k]));
      				if(it!=refined_edges.end()){
      					split.push_back(it);
      					split_set.insert(it->first);
      				}
      			}
      		}
#endif
					int refine_cnt=split_set.size();

					switch(refine_cnt){
					// case 0: // No refinement
					// case 1: // 1:2 refinement is ok.
					case 2:{
						/* Here there are two possibilities. Either the two split
						 * edges share a vertex (case 1) or they are opposite edges
						 * (case 2). Case 1 results in a 1:3 subdivision and a
						 * possible mismatch on the surface. So we have to split an
						 * additional edge. Case 2 results in a 1:4 with no issues
						 * so it is left as is.*/

#ifdef _OPENMP
						int n0=split_set[0].connected(split_set[1]);
						if(n0>=0){
							// Case 1.
							int n1 = (n0 == split_set[0].edge.first) ? split_set[0].edge.second : split_set[0].edge.first;
							int n2 = (n0 == split_set[1].edge.first) ? split_set[1].edge.second : split_set[1].edge.first;

							mark_edge_omp(n1, n2, refined_edges);
							new_edges_size++;
						}
#else
						int n0=split[0]->first.connected(split[1]->first);
						if(n0>=0){
							// Case 1.
							int n1 = (n0==split[0]->first.edge.first)?split[0]->first.edge.second:split[0]->first.edge.first;
							int n2 = (n0==split[1]->first.edge.first)?split[1]->first.edge.second:split[1]->first.edge.first;
							new_edges.insert(Edge<index_t>(n1, n2));
						}
#endif
						break;
					}
					case 3:{
						/* There are 3 cases that need to be considered. They can
						 * be distinguished by the total number of nodes that are
						 * common between any pair of edges. Only the case there
						 * are 3 different nodes common between pairs of edges do
						 * we get a 1:4 subdivision. Otherwise, we have to refine
						 * the other edges.*/
						std::set<index_t> shared;
						for(int j=0;j<3;j++){
							for(int k=j+1;k<3;k++){
#ifdef _OPENMP
								index_t nid = split_set[j].connected(split_set[k]);
#else
								index_t nid = split[j]->first.connected(split[k]->first);
#endif
								if(nid>=0)
									shared.insert(nid);
							}
						}
						size_t nshared = shared.size();

						if(nshared!=3){
							// Refine unsplit edges.
							for(int j=0;j<4;j++)
								for(int k=j+1;k<4;k++){
									Edge<index_t> test_edge(n[j], n[k]);
#ifdef _OPENMP
									if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end())
									{
										mark_edge_omp(n[j], n[k], refined_edges);
										new_edges_size++;
									}
#else
									if(split_set.count(test_edge)==0)
										new_edges.insert(test_edge);
#endif
								}
						}
						break;
					}
					case 4:{
						// Refine unsplit edges.
						for(int j=0;j<4;j++)
							for(int k=j+1;k<4;k++){
								Edge<index_t> test_edge(n[j], n[k]);
#ifdef _OPENMP
								if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end())
								{
									mark_edge_omp(n[j], n[k], refined_edges);
									new_edges_size++;
								}
#else
								if(split_set.count(test_edge)==0)
									new_edges.insert(test_edge);
#endif
							}
						break;
					}
					case 5:{
						// Refine unsplit edges.
						for(int j=0;j<4;j++)
							for(int k=j+1;k<4;k++){
								Edge<index_t> test_edge(n[j], n[k]);
#ifdef _OPENMP
								if(std::find(split_set.begin(), split_set.end(), test_edge) == split_set.end())
								{
									mark_edge_omp(n[j], n[k], refined_edges);
									new_edges_size++;
								}
#else
								if(split_set.count(test_edge)==0)
									new_edges.insert(test_edge);
#endif
							}
						break;
					}
					// case 6: // All edges split. Nothing to do.
					default:
						break;
					}
				}
      
        // If there are no new edges then we can jump out of here.
#ifdef HAVE_MPI
#ifndef _OPENMP
        new_edges_size = new_edges.size();
#endif
        if(nprocs>1){
          MPI_Allreduce(MPI_IN_PLACE, &new_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
        }
#endif
        if(new_edges_size==0)
          break;
    
        // Add new edges to refined_edges.
#ifdef _OPENMP
        #pragma omp parallel
        {
          const unsigned int tid = omp_get_thread_num();

          // Loop through all edges and refine those which have been marked
          #pragma omp for schedule(dynamic)
          for(int i=0;i<(int)_mesh->NNList.size();++i){
          	for(index_t it = 0; it < (int)_mesh->NNList[i].size(); ++it){
          	  if(refined_edges[i][2*it] == std::numeric_limits<index_t>::max()){
          	  	index_t otherVertex = _mesh->NNList[i][it];
                Edge<index_t> edge(i, otherVertex);
                refined_edges[i][2*it]   = splitCnt[tid]++;
                refined_edges[i][2*it+1] = tid;
                refine_edge_omp(edge, newCoords[tid], newMetric[tid]);
          	  }
          	}
          }
        }
#else
        for(typename std::set< Edge<index_t> >::const_iterator it=new_edges.begin();it!=new_edges.end();++it)
          refined_edges[*it] = refine_edge(*it);
#endif

#ifdef HAVE_MPI
        if(nprocs>1){
          // Communicate edges within halo elements which are to be refined. First create the send buffers.
          std::map<int, std::vector<int> > send_buffer;
          for(int p=0;p<nprocs;p++){
            // Identify all the edges to be sent
            typename std::set< std::pair<index_t, index_t> > send_edges;
            for(std::deque<int>::const_iterator he=halo_elements[p].begin();he!=halo_elements[p].end();++he){
              const int *n=_mesh->get_element(*he);
            
              for(size_t j=0;j<nloc;j++){
                for(size_t k=j+1;k<nloc;k++){
                  Edge<index_t> edge(n[j], n[k]);
#ifdef _OPENMP
                  if(_mesh->get_new_vertex_omp(n[j], n[k], refined_edges) >= 0)
#else
                  if(refined_edges.count(edge))
#endif
                    send_edges.insert(std::pair<index_t, index_t>(edge.edge.first, edge.edge.second));
                }
              }
            }
          
            // Stuff these edges into the send buffer for p and add them to the new halo.
            for(typename std::set<std::pair<index_t,index_t> >::const_iterator se=send_edges.begin();se!=send_edges.end();++se){
              index_t lnn0 = se->first;
              index_t lnn1 = se->second;
            
              index_t gnn0 = lnn2gnn[lnn0];
              index_t gnn1 = lnn2gnn[lnn1];
            
              send_buffer[p].push_back(gnn0);
              send_buffer[p].push_back(gnn1);
            
              // Edge owner is defined as its minimum node owner.
              int owner = std::min(get_node_owner(lnn0), get_node_owner(lnn1));
            
              // Add edge into the new halo.
              if(owner!=rank)
                new_recv_halo[owner].insert(Edge<index_t>(gnn0, gnn1));
            }
          }
        
          // Set up the receive buffer.
          std::map<int, std::vector<int> > recv_buffer;
          std::vector<int> send_buffer_size(nprocs, 0), recv_buffer_size(nprocs);
          for(std::map<int, std::vector<int> >::const_iterator sb=send_buffer.begin();sb!=send_buffer.end();++sb){
            send_buffer_size[sb->first] = sb->second.size();
          }
          MPI_Alltoall(&(send_buffer_size[0]), 1, MPI_INT,
                       &(recv_buffer_size[0]), 1, MPI_INT, _mesh->get_mpi_comm());
        
          // Exchange data using non-blocking communication.
        
          // Setup non-blocking receives
          std::vector<MPI_Request> request(nprocs*2);
          for(int i=0;i<nprocs;i++){
            if(recv_buffer_size[i]==0){
              request[i] =  MPI_REQUEST_NULL;
            }else{
              recv_buffer[i].resize(recv_buffer_size[i]);
              MPI_Irecv(&(recv_buffer[i][0]), recv_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[i]));
            }
          }
        
          // Non-blocking sends.
          for(int i=0;i<nprocs;i++){
            if(send_buffer_size[i]==0){
              request[nprocs+i] =  MPI_REQUEST_NULL;
            }else{
              MPI_Isend(&(send_buffer[i][0]), send_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[nprocs+i]));
            }
          }
        
          // Wait for all communication to finish.
          std::vector<MPI_Status> status(nprocs*2);
          MPI_Waitall(nprocs, &(request[0]), &(status[0]));
          MPI_Waitall(nprocs, &(request[nprocs]), &(status[nprocs]));
        
          // Need to unpack and decode data.
          for(int i=0;i<nprocs;i++){
            for(int j=0;j<recv_buffer_size[i];j+=2){
              // Edge in terms of its global node numbering.
              int gnn0 = recv_buffer[i][j];
              int gnn1 = recv_buffer[i][j+1];
              Edge<index_t> global_edge(gnn0, gnn1);
            
              // Edge in terms of its local node numbering.
              assert(gnn2lnn.count(gnn0));
              assert(gnn2lnn.count(gnn1));
              int nid0 = gnn2lnn[gnn0];
              int nid1 = gnn2lnn[gnn1];
              Edge<index_t> local_edge(nid0, nid1);
            
              // Edge owner is defined as its minimum node owner.
              int owner = std::min(get_node_owner(nid0), get_node_owner(nid1));
            
              // Add edge into the new halo.
              if(owner!=rank)
                new_recv_halo[owner].insert(global_edge);
            
              // Add this to refined_edges if it's not already known.
#ifdef _OPENMP
              index_t minID = std::min(nid0, nid1);
              index_t maxID = std::max(nid0, nid1);
              index_t pos = 0;
              while(_mesh->NNList[minID][pos] != maxID) ++pos;

              if(refined_edges[minID][2*pos] == -1){
                refined_edges[minID][2*pos]   = splitCnt[0]++;
                refined_edges[minID][2*pos+1] = 0;
                refine_edge_omp(local_edge, newCoords[0], newMetric[0]);
              }
#else
              if(refined_edges.count(local_edge)==0){
                refined_edges[local_edge] = refine_edge(local_edge);
              }
#endif
            }
          }
        }
#endif
      }
    }

#ifdef _OPENMP
    // Insert new vertices into mesh
    #pragma omp parallel
    {
    	const unsigned int tid = omp_get_thread_num();

    	// Perform parallel prefix sum to find (for each OMP thread) the starting position
    	// in mesh._coords and mesh.metric at which new coords and metric should be appended.
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

    		blockSize <<= 1;
    	}

    	threadIdx[tid] += _mesh->_NNodes - splitCnt[tid];

    	#pragma omp barrier

    	// Resize mesh containers
    	#pragma omp master
    	{
    		const int newSize = threadIdx[nthreads_omp - 1] + splitCnt[nthreads_omp - 1];
    		refined_edges_size = newSize - _mesh->_NNodes;

    		_mesh->_coords.resize(ndims * newSize);
    		_mesh->metric.resize(ndims * ndims * newSize);
    		_mesh->node_towner.resize(newSize);
    		_mesh->NEList.resize(newSize);
    		_mesh->NNList.resize(newSize);

    		memset(&_mesh->node_towner[_mesh->_NNodes], 0, (sizeof(int) / sizeof(char)) * refined_edges_size);

    		_mesh->_NNodes = newSize;
    	}
    	#pragma omp barrier

    	// Append new coords and metric to the mesh
    	memcpy(&_mesh->_coords[ndims*threadIdx[tid]], &newCoords[tid][0], ndims*splitCnt[tid]*sizeof(real_t));
    	memcpy(&_mesh->metric[ndims*ndims*threadIdx[tid]], &newMetric[tid][0], ndims*ndims*splitCnt[tid]*sizeof(real_t));

    	// Fix IDs of new vertices in refined_edges
    	#pragma omp for schedule(dynamic)
    	for(unsigned int i = 0; i < refined_edges.size(); ++i){
    		for(typename std::vector<index_t>::iterator it=refined_edges[i].begin(); it!=refined_edges[i].end(); it+=2)
    			if(*it != -1)
    				*it += threadIdx[*(it+1)];
    	}
    }
#endif

#ifdef HAVE_MPI
    // All edges have been refined. Time to reconstruct the halo.
    if(nprocs>1){
      typename std::vector< std::vector<int> > send_buffer(nprocs), recv_buffer(nprocs);      
      for(typename std::map<int, std::set< Edge<index_t> > >::const_iterator rh=new_recv_halo.begin();rh!=new_recv_halo.end();++rh){
        int proc = rh->first;
        for(typename std::set< Edge<index_t> >::const_iterator ed=rh->second.begin();ed!=rh->second.end();++ed){
          index_t gnn0 = ed->edge.first;
          index_t gnn1 = ed->edge.second;
          
          send_buffer[proc].push_back(gnn0);
          send_buffer[proc].push_back(gnn1);

          index_t lnn0 = gnn2lnn[gnn0];
          index_t lnn1 = gnn2lnn[gnn1];
          
#ifdef _OPENMP
          index_t lnn = _mesh->get_new_vertex_omp(lnn0, lnn1, refined_edges);
#else
          index_t lnn = refined_edges[Edge<index_t>(lnn0, lnn1)];
#endif
          
          _mesh->recv[proc].push_back(lnn);
          _mesh->recv_halo.insert(lnn);

          node_owner[lnn] = proc;
        }
      }
      std::vector<int> send_buffer_size(nprocs), recv_buffer_size(nprocs);
      for(int i=0;i<nprocs;i++)
        send_buffer_size[i] = send_buffer[i].size();

      MPI_Alltoall(&(send_buffer_size[0]), 1, MPI_INT,
                   &(recv_buffer_size[0]), 1, MPI_INT, _mesh->get_mpi_comm()); 
      
      // Setup non-blocking receives
      std::vector<MPI_Request> request(nprocs*2);
      for(int i=0;i<nprocs;i++){
        if(recv_buffer_size[i]==0){
          request[i] =  MPI_REQUEST_NULL;
        }else{
          recv_buffer[i].resize(recv_buffer_size[i]);
          MPI_Irecv(&(recv_buffer[i][0]), recv_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[i]));
        }
      }

      // Non-blocking sends.
      for(int i=0;i<nprocs;i++){
        if(send_buffer_size[i]==0){
          request[nprocs+i] =  MPI_REQUEST_NULL;
        }else{
          MPI_Isend(&(send_buffer[i][0]), send_buffer_size[i], MPI_INT, i, 0, _mesh->get_mpi_comm(), &(request[nprocs+i]));
        }
      }

      std::vector<MPI_Status> status(nprocs*2);
      MPI_Waitall(nprocs, &(request[0]), &(status[0]));
      MPI_Waitall(nprocs, &(request[nprocs]), &(status[nprocs]));

      // Unpack halo's
      for(int i=0;i<nprocs;i++){
        for(int j=0;j<recv_buffer_size[i];j+=2){
          assert(gnn2lnn.count(recv_buffer[i][j]));
          index_t lnn0 = gnn2lnn[recv_buffer[i][j]];
          
          assert(gnn2lnn.count(recv_buffer[i][j+1]));
          index_t lnn1 = gnn2lnn[recv_buffer[i][j+1]];

#ifdef _OPENMP
          index_t lnn = _mesh->get_new_vertex_omp(lnn0, lnn1, refined_edges);
#else
          index_t lnn = refined_edges[Edge<index_t>(lnn0, lnn1)];
#endif
          _mesh->send[i].push_back(lnn);
          _mesh->send_halo.insert(lnn);
        }
      }
      _mesh->halo_update(&(_mesh->_coords[0]), ndims);
      _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
    }
#endif

    // Perform refinement.
#ifdef _OPENMP
    #pragma omp parallel
    {
      const unsigned int tid = omp_get_thread_num();
      splitCnt[tid] = 0;

      #pragma omp for schedule(dynamic)
      for(int i=0;i<NElements;i++){
				// Check if this element has been erased - if so continue to next element.
				const int *n=_mesh->get_element(i);
				if(n[0]<0)
				continue;

				if(ndims==2){
					// Note the order of the edges - the i'th edge is opposite the i'th node in the element.
					index_t newVertex[3];
					newVertex[0] = _mesh->get_new_vertex_omp(n[1], n[2], refined_edges);
					newVertex[1] = _mesh->get_new_vertex_omp(n[2], n[0], refined_edges);
					newVertex[2] = _mesh->get_new_vertex_omp(n[0], n[1], refined_edges);

					int refine_cnt=0;
					for(int j=0;j<3;j++)
						if(newVertex[j] >= 0)
						refine_cnt++;

					if(refine_cnt==0){
						// No refinement - continue to next element.
						continue;
					}else if(refine_cnt==1){
						// Single edge split.
						int rotated_ele[3] = {-1, -1, -1};
						index_t vertexID;
						for(int j=0;j<3;j++)
						if(newVertex[j] >= 0){
							vertexID = newVertex[j];
							for(int k=0;k<3;k++)
							rotated_ele[k] = n[(j+k)%3];
							break;
						}

						const int ele0[] = {rotated_ele[0], rotated_ele[1], vertexID};
						const int ele1[] = {rotated_ele[0], vertexID, rotated_ele[2]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						splitCnt[tid] += 2;
					}else if(refine_cnt==2){
						int rotated_ele[3];
						index_t vertexID[2];
						for(int j=0;j<3;j++){
							if(newVertex[j] < 0){
								vertexID[0] = newVertex[(j+1)%3];
								vertexID[1] = newVertex[(j+2)%3];
								for(int k=0;k<3;k++)
									rotated_ele[k] = n[(j+k)%3];
								break;
							}
						}

						real_t ldiag0 = _mesh->calc_edge_length(vertexID[0], rotated_ele[1]);
						real_t ldiag1 = _mesh->calc_edge_length(vertexID[1], rotated_ele[2]);
						const int offset = ldiag0 < ldiag1 ? 0 : 1;

						const int ele0[] = {rotated_ele[0], vertexID[1], vertexID[0]};
						const int ele1[] = {vertexID[offset], rotated_ele[1], rotated_ele[2]};
						const int ele2[] = {vertexID[0], vertexID[1], rotated_ele[offset+1]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						append_element_omp(ele2, newElements[tid]);
						splitCnt[tid] += 3;
					}else if(refine_cnt==3){
						const int ele0[] = {n[0], newVertex[2], newVertex[1]};
						const int ele1[] = {n[1], newVertex[0], newVertex[2]};
						const int ele2[] = {n[2], newVertex[1], newVertex[0]};
						const int ele3[] = {newVertex[0], newVertex[1], newVertex[2]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						append_element_omp(ele2, newElements[tid]);
						append_element_omp(ele3, newElements[tid]);
						splitCnt[tid] += 4;
					}
				}else{ // 3D
					std::vector<index_t> newVertex;
					std::vector< Edge<index_t> > splitEdges;
					index_t vertexID;
					for(size_t j=0;j<4;j++)
						for(size_t k=j+1;k<4;k++){
							vertexID = _mesh->get_new_vertex_omp(n[j], n[k], refined_edges);
							if(vertexID >= 0){
								newVertex.push_back(vertexID);
								splitEdges.push_back(Edge<index_t>(n[j], n[k]));
							}
						}
					int refine_cnt=newVertex.size();

					// Apply refinement templates.
					if(refine_cnt==0){
						// No refinement - continue to next element.
						continue;
					}else if(refine_cnt==1){
						// Find the opposite edge
						int oe[2];
						for(int j=0, pos=0;j<4;j++)
							if(!splitEdges[0].contains(n[j]))
								oe[pos++] = n[j];

						// Form and add two new edges.
						const int ele0[] = {splitEdges[0].edge.first, newVertex[0], oe[0], oe[1]};
						const int ele1[] = {splitEdges[0].edge.second, newVertex[0], oe[0], oe[1]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						splitCnt[tid] += 2;
					}else if(refine_cnt==2){
						const int ele0[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
						const int ele1[] = {splitEdges[0].edge.first, newVertex[0], splitEdges[1].edge.second, newVertex[1]};
						const int ele2[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.first, newVertex[1]};
						const int ele3[] = {splitEdges[0].edge.second, newVertex[0], splitEdges[1].edge.second, newVertex[1]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						append_element_omp(ele2, newElements[tid]);
						append_element_omp(ele3, newElements[tid]);
						splitCnt[tid] += 4;
					}else if(refine_cnt==3){
						index_t m[] = {-1, -1, -1, -1, -1, -1, -1};
						m[0] = splitEdges[0].edge.first;
						m[1] = newVertex[0];
						m[2] = splitEdges[0].edge.second;
						if(splitEdges[1].contains(m[2])){
							m[3] = newVertex[1];
							if(splitEdges[1].edge.first!=m[2])
								m[4] = splitEdges[1].edge.first;
							else
								m[4] = splitEdges[1].edge.second;
							m[5] = newVertex[2];
						}else{
							m[3] = newVertex[2];
							if(splitEdges[2].edge.first!=m[2])
								m[4] = splitEdges[2].edge.first;
							else
								m[4] = splitEdges[2].edge.second;
							m[5] = newVertex[1];
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

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						append_element_omp(ele2, newElements[tid]);
						append_element_omp(ele3, newElements[tid]);
						splitCnt[tid] += 4;
					}else if(refine_cnt==6){
						const int ele0[] = {n[0], newVertex[0], newVertex[1], newVertex[2]};
						const int ele1[] = {n[1], newVertex[3], newVertex[0], newVertex[4]};
						const int ele2[] = {n[2], newVertex[1], newVertex[3], newVertex[5]};
						const int ele3[] = {newVertex[0], newVertex[3], newVertex[1], newVertex[4]};
						const int ele4[] = {newVertex[0], newVertex[4], newVertex[1], newVertex[2]};
						const int ele5[] = {newVertex[1], newVertex[3], newVertex[5], newVertex[4]};
						const int ele6[] = {newVertex[1], newVertex[4], newVertex[5], newVertex[2]};
						const int ele7[] = {newVertex[2], newVertex[4], newVertex[5], n[3]};

						append_element_omp(ele0, newElements[tid]);
						append_element_omp(ele1, newElements[tid]);
						append_element_omp(ele2, newElements[tid]);
						append_element_omp(ele3, newElements[tid]);
						append_element_omp(ele4, newElements[tid]);
						append_element_omp(ele5, newElements[tid]);
						append_element_omp(ele6, newElements[tid]);
						append_element_omp(ele7, newElements[tid]);
						splitCnt[tid] += 8;
					}
				}

				// Remove parent element.
				_mesh->erase_element_no_recycle(i);
			}

      // Perform parallel prefix sum to find (for each OMP thread) the starting position
      // in mesh._ENList at which new elements should be appended.
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

      threadIdx[tid] += _mesh->_NElements - splitCnt[tid];

      #pragma omp barrier

      // Resize mesh containers
      #pragma omp master
      {
      	const int newSize = threadIdx[nthreads_omp - 1] + splitCnt[nthreads_omp - 1];

      	_mesh->_ENList.resize(nloc*newSize);
        _mesh->element_towner.resize(newSize);

        memset(&_mesh->element_towner[_mesh->_NElements], 0, (sizeof(int) / sizeof(char)) * (newSize - _mesh->_NElements));

        _mesh->_NElements = newSize;
      }
      #pragma omp barrier

      // Append new elements to the mesh
      memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
    }
#else
    for(int i=0;i<NElements;i++){
      // Check if this element has been erased - if so continue to next element.
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      if(ndims==2){
        // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
        typename std::map< Edge<index_t>, index_t>::const_iterator edge[3];
        edge[0] = refined_edges.find(Edge<index_t>(n[1], n[2]));
        edge[1] = refined_edges.find(Edge<index_t>(n[2], n[0]));
        edge[2] = refined_edges.find(Edge<index_t>(n[0], n[1]));

        int refine_cnt=0;
        for(int j=0;j<3;j++)
          if(edge[j]!=refined_edges.end())
            refine_cnt++;

        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
          // Single edge split.
          typename std::map< Edge<index_t>, index_t>::const_iterator split;
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
          typename std::map< Edge<index_t>, index_t>::const_iterator split[2];
          int rotated_ele[3];
          for(int j=0;j<3;j++){
            if(edge[j]==refined_edges.end()){
              split[0] = edge[(j+1)%3];
              split[1] = edge[(j+2)%3];
              for(int k=0;k<3;k++)
            	rotated_ele[k] = n[(j+k)%3];
              break;
            }
          }

          real_t ldiag0 = _mesh->calc_edge_length(split[0]->second, rotated_ele[1]);
          real_t ldiag1 = _mesh->calc_edge_length(split[1]->second, rotated_ele[2]);
          const int offset = ldiag0 < ldiag1 ? 0 : 1;

          const int ele0[] = {rotated_ele[0], split[1]->second, split[0]->second};
          const int ele1[] = {split[offset]->second, rotated_ele[1], rotated_ele[2]};
          const int ele2[] = {split[0]->second, split[1]->second, rotated_ele[offset+1]};

          _mesh->append_element(ele0);
          _mesh->append_element(ele1);
          _mesh->append_element(ele2);
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
      }else{ // 3D
        std::vector<typename std::map< Edge<index_t>, index_t>::const_iterator> split;
        for(size_t j=0;j<4;j++)
          for(size_t k=j+1;k<4;k++){
            typename std::map< Edge<index_t>, index_t>::const_iterator it =
              refined_edges.find(Edge<index_t>(n[j], n[k]));
            if(it!=refined_edges.end())
              split.push_back(it);
          }
        int refine_cnt=split.size();

        // Apply refinement templates.
        if(refine_cnt==0){
          // No refinement - continue to next element.
          continue;
        }else if(refine_cnt==1){
          // Find the opposite edge
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
      }

      // Remove parent element.
      _mesh->erase_element(i);
    }
#endif

    // Remove any elements that are no longer resident
#ifdef HAVE_MPI
    if(nprocs>1){
      size_t lNElements = _mesh->get_number_elements();
      #pragma omp parallel for schedule(dynamic)
      for(size_t e=0;e<lNElements;e++){
        const int* n=_mesh->get_element(e);
        if(n[0]<0)
          continue;
        
        bool needed=false;
        for(size_t j=0;j<nloc;j++){
          if(get_node_owner(n[j])==rank){
            needed = true;
            break;
          }
        }

        if(!needed)
#ifdef _OPENMP
        	_mesh->erase_element_no_recycle(e);
#else
          _mesh->erase_element(e);
#endif
      }
    }
#endif

    // Fix orientations of new elements.
    #pragma omp parallel for schedule(dynamic)
    for(int i=NElements;i<_mesh->get_number_elements();i++){
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
      if(av<0){
        // Flip element
        int ntmp = n[0];
        n[0] = n[1];
        n[1] = ntmp;
      }
    }

    real_t total_volume=0;
    #pragma omp parallel for schedule(dynamic) reduction(+:total_volume)
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
#ifdef _OPENMP
    if(refined_edges_size)
    	_surface->refine(refined_edges);
#else
    _surface->refine(refined_edges);
#endif

#ifdef HAVE_MPI
#ifndef _OPENMP
    refined_edges_size = refined_edges.size();
#endif
    if(nprocs>1)
      MPI_Allreduce(MPI_IN_PLACE, &refined_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#endif

    // Tidy up. Need to look at efficiencies here.
    _mesh->create_adjancy();
    cache_clear();
  
    return refined_edges_size;
  }

 private:
  void cache_create(){
    if(nprocs<2)
      return;

    for(int i=0;i<nprocs;i++){
      for(std::vector<int>::const_iterator it=_mesh->recv[i].begin();it!=_mesh->recv[i].end();++it){
        node_owner[*it] = i;
      }
    }
  }
  
  void cache_clear(){
    node_owner.clear();
  }

  int get_node_owner(index_t nid){
    int owner = rank;
    if(node_owner.count(nid))
      owner = node_owner[nid];
    return owner;
  }

  int refine_edge(const Edge<index_t> &edge){
    // Calculate the position of the new point. From equation 16 in
    // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
    real_t x[3], m[9];
    index_t n0 = edge.edge.first;
    const real_t *x0 = _mesh->get_coords(n0);
    const real_t *m0 = _mesh->get_metric(n0);
    
    index_t n1 = edge.edge.second;
    const real_t *x1 = _mesh->get_coords(n1);
    const real_t *m1 = _mesh->get_metric(n1);
    
    real_t weight = 1.0/(1.0 + sqrt(property->length(x0, x1, m0)/
                                    property->length(x0, x1, m1)));
    
    // Calculate position of new vertex
    for(size_t i=0;i<ndims;i++)
      x[i] = x0[i]+weight*(x1[i] - x0[i]);
    
    // Interpolate new metric
    for(size_t i=0;i<ndims*ndims;i++){
      m[i] = m0[i]+weight*(m1[i] - m0[i]);
      if(isnan(m[i]))
        std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                 <<"m0[i] = "<<m0[i]<<std::endl
                 <<"m1[i] = "<<m1[i]<<std::endl
                 <<"weight = "<<weight<<std::endl;
    }

    // Append this new vertex and metric into mesh data structures.
    index_t nid = _mesh->append_vertex(x, m);
    
    return nid;
  }

  void refine_edge_omp(const Edge<index_t> &edge, std::vector<real_t> &coords, std::vector<real_t> &metric){
    // Calculate the position of the new point. From equation 16 in
    // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
    real_t x, m;
    index_t n0 = edge.edge.first;
    const real_t *x0 = _mesh->get_coords(n0);
    const real_t *m0 = _mesh->get_metric(n0);

    index_t n1 = edge.edge.second;
    const real_t *x1 = _mesh->get_coords(n1);
    const real_t *m1 = _mesh->get_metric(n1);

    real_t weight = 1.0/(1.0 + sqrt(property->length(x0, x1, m0)/
                                    property->length(x0, x1, m1)));

    // Calculate position of new vertex and append it to OMP thread's temp storage
    for(size_t i=0;i<ndims;i++){
      x = x0[i]+weight*(x1[i] - x0[i]);
      coords.push_back(x);
    }

    // Interpolate new metric and append it to OMP thread's temp storage
    for(size_t i=0;i<ndims*ndims;i++){
      m = m0[i]+weight*(m1[i] - m0[i]);
      metric.push_back(m);
      if(isnan(m))
        std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                 <<"m0[i] = "<<m0[i]<<std::endl
                 <<"m1[i] = "<<m1[i]<<std::endl
                 <<"weight = "<<weight<<std::endl;
    }
  }

  inline void mark_edge_omp(index_t n0, index_t n1, std::vector< std::vector<index_t> > &refined_edges){
  	index_t minID = std::min(n0, n1);
  	index_t maxID = std::max(n0, n1);
  	index_t pos = 0;
    while(_mesh->NNList[minID][pos] != maxID) ++pos;

		/*
		 * WARNING! Code analysis tools may warn about a race condition
		 * (write-after-write) for the following line. This is not really
		 * a problem, since any thread accessing this place in memory will
		 * write the same value (MAX_INT).
		 */
		refined_edges[minID][2*pos] = std::numeric_limits<index_t>::max();
  }

  inline void append_element_omp(const index_t *elem, std::vector<index_t> &ENList){
  	for(size_t i=0; i<nloc; ++i)
  		ENList.push_back(elem[i]);
  }

  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  size_t ndims, nloc;
  std::map<index_t, index_t> node_owner;
  int nprocs, rank;
};

#endif
