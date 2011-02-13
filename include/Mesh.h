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

#ifndef MESH_H
#define MESH_H

#include <deque>
#include <vector>
#include <set>

#include <omp.h>

#include "Metis.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t, typename index_t>
  class Mesh{
 public:
  
  /*! 2D triangular mesh constructor.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param comm MPI communicator. (MPI only)
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y
#ifdef HAVE_MPI
       , const MPI_Comm *comm=NULL
#endif
       ){
    _init(NNodes, NElements, ENList,
          x, y, NULL
#ifdef HAVE_MPI
          ,comm
#endif
          );
  }

  /*! 3D tetrahedra mesh constructor.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   * @param comm MPI communicator. (MPI only)
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y, const real_t *z
#ifdef HAVE_MPI
       , const MPI_Comm *comm=NULL
#endif
       ){
    _init(NNodes, NElements, ENList,
          x, y, z
#ifdef HAVE_MPI
          ,comm
#endif
          );
  }
  
  /// Return the number of nodes in the mesh.
  int get_number_nodes() const{
    return _NNodes;
  }

  /// Return the number of elements in the mesh.
  int get_number_elements() const{
    return _NElements;
  }

  /// Return the number of spatial dimensions.
  int get_number_dimensions() const{
    return _ndims;
  }

  /// Return a pointer to the element-node list.
  const index_t *get_enlist() const{
    return _ENList;
  }

  /// Return a pointer to the element-node list.
  const index_t *get_enlist(size_t eid) const{
    return _ENList + eid*_nloc;
  }

  /// Return the node id's connected to the specified node_id
  void get_node_patch(index_t nid, std::set<index_t> &patch){
    for(typename std::vector<index_t>::const_iterator it=_NNList[nid];it!=_NNList[nid];it++)
      patch.insert(*it);
    return;
  }

  /// Grow a node patch around node id's until it reaches a minimum size.
  void get_node_patch(index_t nid, int min_patch_size, std::set<index_t> &patch){
    for(typename std::vector<index_t>::const_iterator it=_NNList[nid].begin();it!=_NNList[nid].end();it++)
      patch.insert(*it);
    
    if(patch.size()<(size_t)min_patch_size){
      std::set<index_t> front = patch, new_front;
      for(;;){
        for(typename std::set<index_t>::const_iterator it=front.begin();it!=front.end();it++){
          for(typename std::vector<index_t>::const_iterator jt=_NNList[*it].begin();jt!=_NNList[*it].end();jt++){
            if(patch.find(*jt)==patch.end()){
              new_front.insert(*jt);
              patch.insert(*jt);
            }
          }
        }
        
        if(patch.size()>=(size_t)min_patch_size)
          break;
        
        front.swap(new_front);
      }
    }
    
    return;
  }

  /// Return positions vector.
  real_t *get_coords(){
    return _coords;
  }

  /// Return positions vector.
  const real_t *get_coords(size_t nid) const{
    return _coords + nid*_ndims;
  }

  /// Return new local node number given on original node number.
  int new2old(int nid){
    return nid_new2old[nid];
  }

#ifdef HAVE_MPI
  /// Return mpi communicator
  const MPI_Comm * get_mpi_comm(){
    return_comm;
  }
#endif

  /// Default destructor.
  ~Mesh(){
    delete [] _ENList;
    delete [] _coords;
  }
  
 private:
  void _init(int NNodes, int NElements, const index_t *ENList,
             const real_t *x, const real_t *y, const real_t *z
#ifdef HAVE_MPI
             ,const MPI_Comm *comm
#endif
             ){
    int etype;
    if(z==NULL){
      _nloc = 3;
      _ndims = 2;
      etype = 1; // triangles
    }else{
      _nloc = 4;
      _ndims = 3;
      etype = 2; // tetrahedra
    }

    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = new index_t[_NElements*_nloc];
    _coords = new real_t[_NNodes*_ndims];

    // Partition the nodes and elements so that the mesh can be
    // topologically mapped to the computer node topology.
    int nparts = omp_get_max_threads();

    std::vector<int> eid_new2old;
    if(nparts>1){
      std::vector<idxtype> epart(NElements, 0);
      std::vector<idxtype> npart(NNodes, 0);
      int numflag = 0;
      int edgecut;
      
      std::vector<idxtype> metis_ENList(_NElements*_nloc);
      for(int i=0;i<NElements*_nloc;i++)
        metis_ENList[i] = ENList[i];
      METIS_PartMeshNodal(&NElements, &NNodes, &(metis_ENList[0]), &etype, &numflag, &nparts,
                          &edgecut, &(epart[0]), &(npart[0]));
      metis_ENList.clear();
      
      // Create sets of nodes and elements in each partition
      std::vector< std::deque<int> > nodes(nparts), elements(nparts);
      for(int i=0;i<NNodes;i++)
        nodes[npart[i]].push_back(i);
      for(int i=0;i<NElements;i++)
        elements[epart[i]].push_back(i);
      
      std::vector< std::set<int> > edomains(nparts);
      for(int i=0; i<_NElements; i++){
        edomains[epart[i]].insert(i);
      }
      
      // Create element renumbering
      for(int i=0;i<nparts;i++){
        for(std::set<int>::const_iterator it=edomains[i].begin();it!=edomains[i].end();++it){
          eid_new2old.push_back(*it);
        }
      }
      
      // Create seperate graphs for each partition.
      std::vector< std::map<index_t, std::set<index_t> > > pNNList(nparts);
      for(int i=0; i<_NElements; i++){
        for(int j=0;j<_nloc;j++){
          int jnid = ENList[i*_nloc+j];
          int jpart = npart[jnid];
          for(int k=j+1;k<_nloc;k++){
            int knid = ENList[i*_nloc+k];
            int kpart = npart[knid];
            if(jpart!=kpart)
              continue;
            pNNList[jpart][jnid].insert(knid);
            pNNList[jpart][knid].insert(jnid);
          }
        }
      }
      
      // Renumber nodes within each partition.
      for(int p=0;p<nparts;p++){
        // Create mapping from node numbering to local thread partition numbering, and it's inverse.
        std::map<index_t, index_t> nid2tnid;
        std::deque<index_t> tnid2nid(pNNList[p].size());
        index_t loc=0;
        for(typename std::map<index_t, std::set<index_t> >::const_iterator it=pNNList[p].begin();it!=pNNList[p].end();++it){
          tnid2nid[loc] = it->first;
          nid2tnid[it->first] = loc++;
        }
        
        std::vector< std::set<index_t> > pgraph(nid2tnid.size());
        for(typename std::map<index_t, std::set<index_t> >::const_iterator it=pNNList[p].begin();it!=pNNList[p].end();++it){
          for(typename std::set<index_t>::const_iterator jt=it->second.begin();jt!=it->second.end();++jt){
            pgraph[nid2tnid[it->first]].insert(nid2tnid[*jt]);
          }
        }
        
        std::vector<int> porder;
        Metis<index_t>::reorder(pgraph, porder);
        
        for(typename std::vector<index_t>::const_iterator it=porder.begin();it!=porder.end();++it){
          nid_new2old.push_back(tnid2nid[*it]);
        }
      }
    }else{
      // Create an optimised node ordering.
      std::vector< std::set<index_t> > NNList(_NNodes);
      for(int i=0; i<_NElements; i++){
        for(int j=0;j<_nloc;j++){
          for(int k=j+1;k<_nloc;k++){
            NNList[ENList[i*_nloc+j]].insert(ENList[i*_nloc+k]);
            NNList[ENList[i*_nloc+k]].insert(ENList[i*_nloc+j]);
          }
        }
      }
      
      Metis<index_t>::reorder(NNList, nid_new2old);
      
      eid_new2old.resize(_NElements);
      for(int e=0;e<_NElements;e++)
        eid_new2old[e] = e;
    }
    
    // Reverse mapping of renumbering.
    std::vector<index_t> nid_old2new(_NNodes);
    for(int i=0;i<_NNodes;i++){
      nid_old2new[nid_new2old[i]] = i;
    }
    
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NElements;i++){
        for(int j=0;j<_nloc;j++){
          _ENList[i*_nloc+j] = nid_old2new[ENList[eid_new2old[i]*_nloc+j]];
        }
      }
      if(_ndims==2){
#pragma omp for schedule(static)
        for(int i=0;i<_NNodes;i++){
          _coords[i*_ndims  ] = x[nid_new2old[i]];
          _coords[i*_ndims+1] = y[nid_new2old[i]];
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0;i<_NNodes;i++){
          _coords[i*_ndims  ] = x[nid_new2old[i]];
          _coords[i*_ndims+1] = y[nid_new2old[i]];
          _coords[i*_ndims+2] = z[nid_new2old[i]];
        }
      }
    }

    // Create new NNList based on new numbering.
    std::vector< std::set<index_t> > NNList(_NNodes);
    for(int i=0; i<_NElements; i++){
      for(int j=0;j<_nloc;j++){
        for(int k=j+1;k<_nloc;k++){
          NNList[_ENList[i*_nloc+j]].insert(_ENList[i*_nloc+k]);
          NNList[_ENList[i*_nloc+k]].insert(_ENList[i*_nloc+j]);
        }
      }
    }

    // Compress NNList enforcing first-touch policy
    _NNList.resize(_NNodes);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        for(typename std::set<index_t>::const_iterator it=NNList[i].begin();it!=NNList[i].end();it++){
          _NNList[i].push_back(*it);
        }
      }
    }

#ifdef HAVE_MPI
    _comm = comm;
#endif
  }

  int _NNodes, _NElements, _ndims, _nloc;
  index_t *_ENList;
  real_t *_coords;
  std::vector< std::vector<index_t> > _NNList;
  std::vector<index_t> nid_new2old;
#ifdef HAVE_MPI
  const MPI_Comm *_comm;
#endif
};
#endif
