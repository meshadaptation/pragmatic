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
    
    // Initialise a dynamic vertex list
    std::map< Edge<index_t>, index_t> refined_edges;
    std::map<int, std::set< Edge<index_t> > > new_recv_halo;
    
    /* Loop through all edges and select them for refinement is
       it's length is greater than L_max in transformed space. */
    for(int i=0;i<(int)_mesh->NNList.size();++i){
      for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[i].begin();it!=_mesh->NNList[i].end();++it){
        if(i<*it){
          Edge<index_t> edge(i, *it);
          double length = _mesh->calc_edge_length(i, *it);
          if(length>L_max)
            refined_edges[edge] = refine_edge(edge);
        }
      }
    }

    /* If there are no edges to be refined globally then we can return
       at this point.
     */
    int refined_edges_size = refined_edges.size();
#ifdef HAVE_MPI
    if(nprocs>1)
      MPI_Allreduce(MPI_IN_PLACE, &refined_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#endif
    if(refined_edges_size==0)
      return 0;
    
    /* Given the set of refined edge, apply additional edge-refinement
       to get a regular and conforming element refinement throughout
       the domain.*/
    for(;;){
      typename std::set< Edge<index_t> > new_edges;
      for(int i=0;i<NElements;i++){
        // Check if this element has been erased - if so continue to next element.
        const int *n=_mesh->get_element(i);
        if(n[0]<0)
          continue;
                
        // Find what edges have been split in this element.
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
        int refine_cnt=split.size();
        
        if(ndims==2){
          switch(refine_cnt){
            // case 0: // No refinement - continue to next element.
            // case 1: // 1:2 refinement is ok.
          case 2:{
            /* Here there are two possibilities when splitting the
               remaining quad. While this would be ok for generating
               conformal refinements in serial it is an additional
               complication for MPI parallel. Therefore we change this
               to a 1:4 subdivision. We can later improve this by only
               having this restriction on the halo elements.*/
            int n0=split[0]->first.connected(split[1]->first);
            assert(n0>=0);
            
            int n1 = (n0==split[0]->first.edge.first)?split[0]->first.edge.second:split[0]->first.edge.first;
            int n2 = (n0==split[1]->first.edge.first)?split[1]->first.edge.second:split[1]->first.edge.first;
            new_edges.insert(Edge<index_t>(n1, n2));
            break;}
            // case 3: // 1:4 refinement is ok.
          default:
            break;
          }
        }else{ // 3d case
          switch(refine_cnt){
            // case 0: // No refinement
            // case 1: // 1:2 refinement is ok.
          case 2:{
            /* Here there are two possibilities. Either the two split
               edges share a vertex (case 1) or there are opposit
               (case 2). Case 1 results in a 1:3 subdivision and a
               possible mismatch on the surface. So we have to spit an
               additional edge. Case 2 results in a 1:4 with no issues
               so is left as is.*/
            
            int n0=split[0]->first.connected(split[1]->first);
            if(n0>=0){
              // Case 1.
              int n1 = (n0==split[0]->first.edge.first)?split[0]->first.edge.second:split[0]->first.edge.first;
              int n2 = (n0==split[1]->first.edge.first)?split[1]->first.edge.second:split[1]->first.edge.first;
              new_edges.insert(Edge<index_t>(n1, n2));
            }
            break;
          }
          case 3:{
            /* There are 3 cases that need to be considered. They can
               be distinguished by the total number of nodes that are
               common between any pair of edges. Only the case there
               are 3 different nodes common between pairs of edges do
               we get a 1:4 subdivision. Otherwhile, we have to refine
               the other edges.*/
            std::set<index_t> shared;
            for(int j=0;j<refine_cnt;j++){
              for(int k=j+1;k<refine_cnt;k++){
                index_t nid = split[j]->first.connected(split[k]->first);
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
                  if(split_set.count(test_edge)==0)
                    new_edges.insert(test_edge);
                }
            }
            break;
          }
          case 4:{
            // Refine unsplit edges.
            for(int j=0;j<4;j++)
              for(int k=j+1;k<4;k++){
                Edge<index_t> test_edge(n[j], n[k]);
                if(split_set.count(test_edge)==0)
                  new_edges.insert(test_edge);
              }
            break;
          }
          case 5:{
            // Refine unsplit edges.
            for(int j=0;j<4;j++)
              for(int k=j+1;k<4;k++){
                Edge<index_t> test_edge(n[j], n[k]);
                if(split_set.count(test_edge)==0)
                  new_edges.insert(test_edge);
              }
            break;
          }
            // case 6: // All edges spit. Nothing to do.
          default:
            break;
          }
        }
      }
      
      // If there are no new edges then we can jump out of here.
      int new_edges_size = new_edges.size();
#ifdef HAVE_MPI
      if(nprocs>1){
        MPI_Allreduce(MPI_IN_PLACE, &new_edges_size, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
      }
#endif
      if(new_edges_size==0)
        break;
    
      // Add new edges to refined_edges.
      for(typename std::set< Edge<index_t> >::const_iterator it=new_edges.begin();it!=new_edges.end();++it)
        refined_edges[*it] = refine_edge(*it);
      
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
                if(refined_edges.count(edge))
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
            if(refined_edges.count(local_edge)==0){
              refined_edges[local_edge] = refine_edge(local_edge);
            }
          }
        }
      }
#endif
    }
    
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
          
          index_t lnn = refined_edges[Edge<index_t>(lnn0, lnn1)];
          
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

          index_t lnn = refined_edges[Edge<index_t>(lnn0, lnn1)];
          _mesh->send[i].push_back(lnn);
          _mesh->send_halo.insert(lnn);
        }
      }
      _mesh->halo_update(&(_mesh->_coords[0]), ndims);
      _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
    }
#endif

    // Perform refinement.
    for(int i=0;i<NElements;i++){
      // Check if this element has been erased - if so continue to next element.
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      if(ndims==2){
        // Note the order of the edges - the i'th edge is opposit the i'th node in the element. 
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
      }

      // Remove parent element.
      _mesh->erase_element(i);
    }

    // Remove any elements that are no longer resident
#ifdef HAVE_MPI
    if(nprocs>1){
      size_t lNElements = _mesh->get_number_elements();
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
          _mesh->erase_element(e);
      }
    }
#endif

    // Fix orientations of new elements.
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

#ifdef HAVE_MPI
    refined_edges_size = refined_edges.size();
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

  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  size_t ndims, nloc;
  std::map<index_t, index_t> node_owner;
  int nprocs, rank;
};

#endif
