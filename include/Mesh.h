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

#include "confdefs.h"

#include <deque>
#include <vector>
#include <set>

#include <omp.h>

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

#include "Metis.h"
#include "Edge.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t, typename index_t> class Mesh{
 public:
  
  /*! 2D triangular mesh constructor.
   * 
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   */
  Mesh(int NNodes, int NElements, const index_t *ENList,
       const real_t *x, const real_t *y){
#ifdef HAVE_MPI
    _comm = NULL;
#endif
    _init(NNodes, NElements, ENList, x, y, NULL);
  }

  /*! 3D tetrahedra mesh constructor.
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
    _init(NNodes, NElements, ENList,
          x, y, z);
  }

  /*! Defragment mesh. This compresses the storage of internal data
    structures. This is useful if the mesh has been significently
    coarsened. */
  void defragment(){
    // Discover which verticies and elements are active.
    std::map<index_t, index_t> active_vertex_map;
    std::deque<index_t> active_vertex, active_element;
    for(size_t e=0;e<_NElements;e++){
      index_t nid = _ENList[e*_nloc];
      if(nid<0)
        continue;
      active_element.push_back(e);

      active_vertex_map[nid] = 0;
      for(size_t j=1;j<_nloc;j++){
        nid = _ENList[e*_nloc+j];
        active_vertex_map[nid]=0;
      }
    }
    index_t cnt=0;
    for(typename std::map<index_t, index_t>::iterator it=active_vertex_map.begin();it!=active_vertex_map.end();++it){
      it->second = cnt++;
      active_vertex.push_back(it->first);
    }

    // Compress data structures.
    _NNodes = active_vertex.size();
    node_towner.resize(_NNodes);
    _NElements = active_element.size();
    element_towner.resize(_NElements);
    index_t *defrag_ENList = new index_t[_NElements*_nloc];
    real_t *defrag_coords = new real_t[_NNodes*_ndims];
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NElements;i++){
        index_t eid = active_element[i];
        for(size_t j=0;j<_nloc;j++){
          defrag_ENList[i*_nloc+j] = active_vertex_map[_ENList[eid*_nloc+j]];
        }
#ifdef _OPENMP
        element_towner[i] = omp_get_thread_num();
#else
        element_towner[i] = 0;
#endif
      }

      node_towner.resize(_NNodes);
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        index_t nid=active_vertex[i];
        for(size_t j=0;j<_ndims;j++)
          defrag_coords[i*_ndims+j] = _coords[nid*_ndims+j];
#ifdef _OPENMP
        node_towner[i] = omp_get_thread_num();
#else
        node_towner[i] = 0;
#endif 
      }
    }

    delete [] _ENList;
    _ENList = defrag_ENList;

    delete [] _coords;
    _coords = defrag_coords;
    
    create_adjancy();
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

  /// Return the thread that ownes the node.
  int get_node_towner(int i) const{
    if(node_towner.size())
      return node_towner[i];
    return 0;
  }

  /// Return the thread that ownes the element.
  int get_element_towner(int i) const{
    if(element_towner.size())
      return element_towner[i];
    return 0;
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
  std::set<index_t> get_node_patch(index_t nid) const{
    std::set<index_t> patch;
    for(typename std::deque<index_t>::const_iterator it=NNList[nid].begin();it!=NNList[nid].end();++it)
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
          for(typename std::deque<index_t>::const_iterator jt=NNList[*it].begin();jt!=NNList[*it].end();jt++){
            if(patch.find(*jt)==patch.end()){
              new_front.insert(*jt);
              patch.insert(*jt);
            }
          }
        }
        
        if(patch.size()>=std::min(min_patch_size, _NNodes))
          break;
        
        front.swap(new_front);
      }
    }
    
    return patch;
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
    return _comm;
  }
#endif

  /// Default destructor.
  ~Mesh(){
    delete [] _ENList;
    delete [] _coords;
  }

  /// Calculates the edge lengths in metric space.
  void calc_edge_lengths(){
    assert(Edges.size());
    
    for(typename std::set< Edge<real_t, index_t> >::iterator it=Edges.begin();it!=Edges.end();){
      typename std::set< Edge<real_t, index_t> >::iterator current_edge = it++;
      
      Edge<real_t, index_t> edge = *current_edge;
      Edges.erase(current_edge);
      
      index_t nid0 = edge.edge.first;
      index_t nid1 = edge.edge.second;
      
      edge.length = calc_edge_length(nid0, nid1);
      Edges.insert(edge);
    }
  }

  /// Calculates the edge lengths in metric space.
  real_t calc_edge_length(index_t nid0, index_t nid1){
    real_t length=-1.0;
    if(_ndims==2){
      real_t ml00 = (metric[nid0*_ndims*_ndims]+metric[nid1*_ndims*_ndims])*0.5;
      real_t ml01 = (metric[nid0*_ndims*_ndims+1]+metric[nid1*_ndims*_ndims+1])*0.5;
      real_t ml11 = (metric[nid0*_ndims*_ndims+3]+metric[nid1*_ndims*_ndims+3])*0.5;
      
      real_t x=_coords[nid1*_ndims]-_coords[nid0*_ndims];
      real_t y=_coords[nid1*_ndims+1]-_coords[nid0*_ndims+1];
      
      length = sqrt((ml01*x + ml11*y)*y + 
                    (ml00*x + ml01*y)*x);
    }else{
      real_t ml00 = (metric[nid0*_ndims*_ndims  ]+metric[nid1*_ndims*_ndims  ])*0.5;
      real_t ml01 = (metric[nid0*_ndims*_ndims+1]+metric[nid1*_ndims*_ndims+1])*0.5;
      real_t ml02 = (metric[nid0*_ndims*_ndims+2]+metric[nid1*_ndims*_ndims+2])*0.5;
      
      real_t ml11 = (metric[nid0*_ndims*_ndims+4]+metric[nid1*_ndims*_ndims+4])*0.5;
      real_t ml12 = (metric[nid0*_ndims*_ndims+5]+metric[nid1*_ndims*_ndims+5])*0.5;
      
      real_t ml22 = (metric[nid0*_ndims*_ndims+8]+metric[nid1*_ndims*_ndims+8])*0.5;
      
      real_t x=_coords[nid1*_ndims]-_coords[nid0*_ndims];
      real_t y=_coords[nid1*_ndims+1]-_coords[nid0*_ndims+1];
      real_t z=_coords[nid1*_ndims+2]-_coords[nid0*_ndims+2];
      
      length = sqrt((ml02*x + ml12*y + ml22*z)*z + 
                    (ml01*x + ml11*y + ml12*z)*y + 
                    (ml00*x + ml01*y + ml02*z)*x);
    }
    return length;
  }

 private:
  template<typename _real_t, typename _index_t> friend class MetricField;
  template<typename _real_t, typename _index_t> friend class Smooth;
  template<typename _real_t, typename _index_t> friend class Coarsen;
  template<typename _real_t, typename _index_t> friend class Surface;
  template<typename _real_t, typename _index_t> friend void export_vtu(const char *, const Mesh<_real_t, _index_t> *, const _real_t *);

  void _init(int NNodes, int NElements, const index_t *ENList,
             const real_t *x, const real_t *y, const real_t *z){
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
    // topologically mapped to the computer node topology. If we have
    // NUMA library dev support then we use the number of memory
    // nodes. Otherwise, play it save and use the number of threads.
#ifdef HAVE_LIBNUMA
    int nparts =numa_max_node()+1;
#else
    int nparts = 1;
#endif

    std::vector<int> eid_new2old;
    std::vector<idxtype> epart(NElements, 0), npart(NNodes, 0);
    if(nparts>1){
      int numflag = 0;
      int edgecut;
      
      std::vector<idxtype> metis_ENList(_NElements*_nloc);
      for(size_t i=0;i<NElements*_nloc;i++)
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
      for(size_t i=0; i<_NElements; i++){
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
      for(size_t i=0; i<_NElements; i++){
        for(size_t j=0;j<_nloc;j++){
          int jnid = ENList[i*_nloc+j];
          int jpart = npart[jnid];
          for(size_t k=j+1;k<_nloc;k++){
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
      std::vector< std::set<index_t> > lNNList(_NNodes);
      for(size_t i=0; i<_NElements; i++){
        for(size_t j=0;j<_nloc;j++){
          index_t nid_j = ENList[i*_nloc+j];
          for(size_t k=j+1;k<_nloc;k++){
            index_t nid_k = ENList[i*_nloc+k];
            lNNList[nid_j].insert(nid_k);
            lNNList[nid_k].insert(nid_j);
          }
        }
      }
      Metis<index_t>::reorder(lNNList, nid_new2old);
      
      eid_new2old.resize(_NElements);
      for(size_t e=0;e<_NElements;e++)
        eid_new2old[e] = e;
    }
    
    // Reverse mapping of renumbering.
    std::vector<index_t> nid_old2new(_NNodes);
    for(size_t i=0;i<_NNodes;i++){
      nid_old2new[nid_new2old[i]] = i;
    }
    
    // Enforce first-touch policy
    element_towner.resize(_NElements);
    node_towner.resize(_NNodes);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NElements;i++){
        for(size_t j=0;j<_nloc;j++){
          _ENList[i*_nloc+j] = nid_old2new[ENList[eid_new2old[i]*_nloc+j]];
        }
        element_towner[i] = epart[eid_new2old[i]];
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
      for(size_t i=0;i<_NNodes;i++)
        node_towner[i] = npart[nid_new2old[i]];
    }

    create_adjancy();
  }

  /// Create required adjancy lists.
  void create_adjancy(){
    // Create new NNList, NEList and edges
    std::vector< std::set<index_t> > NNList_set(_NNodes);
    NEList.clear();
    NEList.resize(_NNodes);
    Edges.clear();
    for(size_t i=0; i<_NElements; i++){
      for(size_t j=0;j<_nloc;j++){
        index_t nid_j = _ENList[i*_nloc+j];
        if(nid_j<0)
          break;
        NEList[nid_j].insert(i);
        for(size_t k=j+1;k<_nloc;k++){
          index_t nid_k = _ENList[i*_nloc+k];
          NNList_set[nid_j].insert(nid_k);
          NNList_set[nid_k].insert(nid_j);
          
          Edge<real_t, index_t> edge(nid_j, nid_k);
          typename std::set< Edge<real_t, index_t> >::iterator edge_ptr = Edges.find(edge);
          if(edge_ptr!=Edges.end()){
            edge.adjacent_elements = edge_ptr->adjacent_elements;
            Edges.erase(edge_ptr);
          }
          edge.adjacent_elements.insert(i);
          Edges.insert(edge);
        }
      }
    }
    
    // Compress NNList
    NNList.clear();
    NNList.resize(_NNodes);
    for(size_t i=0;i<_NNodes;i++){
      for(typename std::set<index_t>::const_iterator it=NNList_set[i].begin();it!=NNList_set[i].end();++it){
        NNList[i].push_back(*it);
      }
    }
  }

  size_t _NNodes, _NElements, _ndims, _nloc;
  index_t *_ENList;
  real_t *_coords;
  
  std::vector<index_t> nid_new2old;
  std::vector<int> element_towner, node_towner;

  // Adjancy lists
  std::vector< std::set<index_t> > NEList;
  std::vector< std::deque<index_t> > NNList;
  std::set< Edge<real_t, index_t> > Edges;

  // Metric tensor field.
  std::vector<real_t> metric;

#ifdef HAVE_MPI
  const MPI_Comm *_comm;
#endif
};
#endif
