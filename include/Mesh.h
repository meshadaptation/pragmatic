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

#include <vector>
#include <set>

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
  int get_number_nodes(){
    return _NNodes;
  }

  /// Return the number of elements in the mesh.
  int get_number_elements(){
    return _NElements;
  }

  /// Return the number of spatial dimensions.
  int get_number_dimensions(){
    return _ndims;
  }

  /// Return a pointer to the element-node list.
  index_t *get_enlist(){
    return _ENList;
  }
  
  /// Return positions vector.
  real_t *get_coords(){
    return _coords;
  }

#ifdef HAVE_MPI
  /// Return mpi communicator
  const MPI_Comm * get_mpi_comm(){
    return_comm;
  }
#endif

  /// Default destructor.
  ~Mesh(){
  }
  
 private:
  void _init(int NNodes, int NElements, const index_t *ENList,
             const real_t *x, const real_t *y, const real_t *z
#ifdef HAVE_MPI
             ,const MPI_Comm *comm
#endif
             ){
    if(z==NULL){
      _nloc = 3;
      _ndims = 2;
    }else{
      _nloc = 4;
      _ndims = 3;
    }
    
    _NNodes = NNodes;
    _NElements = NElements;
    
    // Allocate space.
    _ENList = new index_t[_NElements*_nloc];
    _coords = new real_t[_NNodes*_ndims];

    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NElements;i++){
        for(int j=0;j<_nloc;j++){
          _ENList[i*_nloc+j] = ENList[i*_nloc+j];
        }
      }
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _coords[i*_ndims  ] = x[i];
        _coords[i*_ndims+1] = y[i];
        if(_ndims==3)
          _coords[i*_ndims+2] = z[i];
      }
    }

#ifdef HAVE_MPI
    _comm = comm;
#endif
  }

  int _NNodes, _NElements, _ndims, _nloc;
  index_t *_ENList;
  real_t *_coords;
#ifdef HAVE_MPI
  const MPI_Comm *_comm;
#endif
};
#endif
