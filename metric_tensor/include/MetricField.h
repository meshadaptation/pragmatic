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

#ifndef METRICFIELD_H
#define METRICFIELD_H

#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "MetricTensor.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

/*! \brief Constructs metric tensor field which encodes anisotropic
 *  edge size information.
 * 
 * Use this class to create a metric tensor field using desired error
 * tolerences and curvature of linear fields subject to a set of constraints.
 * index_t should be set as int32_t for problems with less than 2 billion nodes.
 * For larger meshes int64_t should be used.
 */
template<typename real_t, typename index_t>
  class MetricField{
 public:
  
  /// Default constructor.
  MetricField(){
    _NNodes = 0;
    _NElements = 0;
    _ndims = 0;
    _ENList = NULL;
    _node_distribution = NULL;
    _x = NULL;
    _y = NULL;
    _z = NULL;
#ifdef HAVE_MPI
    _mesh_comm = NULL;
#endif
    _metric = NULL;
  }

  /// Default destructor.
  ~MetricField(){
    if(_metric!=NULL)
      delete [] _metric;
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
    _ndims = 2;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _x = x;
    _y = y;
    _z = NULL;
    _node_distribution = node_distribution;
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
    _ndims = 3;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _x = x;
    _y = y;
    _z = z;
    _node_distribution = node_distribution;
  }
  
#ifdef HAVE_MPI
  /*! Set the MPI communicator to be used.
   * @param mesh_comm is the MPI communicator.
   */
  void set_mpi_communicator(const MPI_Comm *mesh_comm){
    _mesh_comm = mesh_comm;
  }
#endif

  /*! Add the contribution from the metric field from a new field with a target linear interpolation error.
   * @param psi is field while curvature is to be considered.
   * @param target_error is the target interpolation error.
   * @param sigma_psi should be set if a relative interpolation error is specified. It is the minimum value for psi used when applying the relative error. If the argument is not specified or less than 0 then an absolute error is assumed.
   */
  void add_field(const real_t *psi, const real_t target_error, real_t sigma_psi=-1.0){
    bool relative = sigma_psi>0.0;

    int ndims2 = _ndims*_ndims;
    
    real_t *Hessian = new real_t[_NNodes*ndims2];
    
    get_hessian(psi, Hessian);
    
    if(relative){
      for(size_t i=0;i<_NNodes;i++){
        real_t eta = 1.0/max(target_error*psi[i], sigma_psi);
        for(size_t j=0;j<ndims2;j++)
          Hessian[i*ndims2+j]*=eta;
      }
    }else{
      for(size_t i=0;i<_NNodes;i++){
        real_t eta = 1.0/target_error;
        for(size_t j=0;j<ndims2;j++)
          Hessian[i*ndims2+j]*=eta;
      }
    }
    
    if(_metric==NULL){
      _metric = new MetricTensor<real_t>[_NNodes];
      for(size_t i=0;i<_NNodes;i++){
        _metric[i].set(_ndims, Hessian+i*ndims2);
      }
    }else{
      for(size_t i=0;i<_NNodes;i++)
        _metric[i].constrain(MetricTensor<real_t>(_ndims, Hessian+i*ndims2));
    }
    delete [] Hessian;
  }
  
  /*! Apply maximum edge length constraint.
   */
  void apply_max_edge_length(real_t max_len){
    real_t M[_ndims*_ndims];
    for(size_t i=0;i<_ndims;i++)
      for(size_t j=0;j<_ndims;j++)
        if(i==j)
          M[i*_ndims+j] = 1.0/(max_len*max_len);
        else
          M[i*_ndims+j] = 0.0;
    
    MetricTensor<real_t> constraint(_ndims, M);
    
    for(size_t i=0;i<_NNodes;i++)
      _metric[i].constrain(constraint);
  }
  
  /*! Apply minimum edge length constraint.
   */
  void apply_min_edge_length(real_t min_len){
    real_t M[_ndims*_ndims];
    for(size_t i=0;i<_ndims;i++)
      for(size_t j=0;j<_ndims;j++)
        if(i==j)
          M[i*_ndims+j] = 1.0/(min_len*min_len);
        else
          M[i*_ndims+j] = 0.0;
    
    MetricTensor<real_t> constraint(_ndims, M);
    
    for(size_t i=0;i<_NNodes;i++)
      _metric[i].constrain(constraint, false);
  }
  
  /*! Apply maximum aspect ratio constraint.
   */
  void apply_max_aspect_ratio(real_t max_aspect_ratio);

  /*! Apply maximum number of elements constraint.
   */
  void apply_max_nelements(real_t max_nelements);

  /*! Apply minimum number of elements constraint.
   */
  void apply_min_nelements(real_t min_nelements);

  /*! Apply required number of elements.
   */
  void apply_nelements(real_t nelements);

 private:
  /*! Apply required number of elements.
   */
  void get_hessian(const real_t *psi, real_t *Hessian){
    
    // Create node-node list
    if(NNList.empty()){
      NNList.resize(_NNodes);
      size_t nloc = (_ndims==2)?3:4;
      for(size_t i=0; i<_NElements; i++){
        for(size_t j=0;j<nloc;j++){
          for(size_t k=j+1;k<nloc;k++){
            NNList[_ENList[i*nloc+j]].insert(_ENList[i*nloc+k]);
            NNList[_ENList[i*nloc+k]].insert(_ENList[i*nloc+j]);
          }
        }
      }
    }
    
    // Calculate Hessian at each point.
    int min_patch_size = (_ndims==2)?6:10;
    for(size_t i=0; i<_NNodes; i++){
      std::set<index_t> patch = NNList[i];
      
      if(patch.size()<min_patch_size){
        std::set<index_t> front = NNList[i];
        for(typename std::set<index_t>::const_iterator it=front.begin();it!=front.end();it++){
          patch.insert(NNList[*it].begin(), NNList[*it].end());
        }
        if(patch.size()<min_patch_size){
          cerr<<"WARNING: Small mesh patch detected.\n";  
        }
      }

      if(_ndims==2){
        // Form quadratic system to be solved. The quadratic fit is:
        // P = 1 + x + y + x^2 + xy + y^2
        // A = P^TP
        real_t x=_x[i], y=_y[i], z=_z[i];
        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);
        
        A[0]+=1; A[1]+=x; A[2]+=y; A[3]+=pow(x,2); A[4]+=x*y; A[5]+=pow(y,2);
        A[6]+=x; A[7]+=pow(x,2); A[8]+=x*y; A[9]+=pow(x,3); A[10]+=pow(x,2)*y; A[11]+=x*pow(y,2);
        A[12]+=y; A[13]+=x*y; A[14]+=pow(y,2); A[15]+=pow(x,2)*y; A[16]+=x*pow(y,2); A[17]+=pow(y,3);
        A[18]+=pow(x,2); A[19]+=pow(x,3); A[20]+=pow(x,2)*y; A[21]+=pow(x,4); A[22]+=pow(x,3)*y; A[23]+=pow(x,2)*pow(y,2);
        A[24]+=x*y; A[25]+=pow(x,2)*y; A[26]+=x*pow(y,2); A[27]+=pow(x,3)*y; A[28]+=pow(x,2)*pow(y,2); A[29]+=x*pow(y,3);
        A[30]+=pow(y,2); A[31]+=x*pow(y,2); A[32]+=pow(y,3); A[33]+=pow(x,2)*pow(y,2); A[34]+=x*pow(y,3); A[35]+=pow(y,4);

        Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
        b[0]+=psi[i]*1;
        b[1]+=psi[i]*x;
        b[2]+=psi[i]*y;
        b[3]+=psi[i]*pow(x,2);
        b[4]+=psi[i]*x*y;
        b[5]+=psi[i]*pow(y,2);

        for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
          x=_x[*n], y=_y[*n], z=_z[*n];
          
          A[0]+=1; A[1]+=x; A[2]+=y; A[3]+=pow(x,2); A[4]+=x*y; A[5]+=pow(y,2);
          A[6]+=x; A[7]+=pow(x,2); A[8]+=x*y; A[9]+=pow(x,3); A[10]+=pow(x,2)*y; A[11]+=x*pow(y,2);
          A[12]+=y; A[13]+=x*y; A[14]+=pow(y,2); A[15]+=pow(x,2)*y; A[16]+=x*pow(y,2); A[17]+=pow(y,3);
          A[18]+=pow(x,2); A[19]+=pow(x,3); A[20]+=pow(x,2)*y; A[21]+=pow(x,4); A[22]+=pow(x,3)*y; A[23]+=pow(x,2)*pow(y,2);
          A[24]+=x*y; A[25]+=pow(x,2)*y; A[26]+=x*pow(y,2); A[27]+=pow(x,3)*y; A[28]+=pow(x,2)*pow(y,2); A[29]+=x*pow(y,3);
          A[30]+=pow(y,2); A[31]+=x*pow(y,2); A[32]+=pow(y,3); A[33]+=pow(x,2)*pow(y,2); A[34]+=x*pow(y,3); A[35]+=pow(y,4);
          
          b[0]+=psi[*n]*1;
          b[1]+=psi[*n]*x;
          b[2]+=psi[*n]*y;
          b[3]+=psi[*n]*pow(x,2);
          b[4]+=psi[*n]*x*y;
          b[5]+=psi[*n]*pow(y,2);
        }
        
        // Eigen::Matrix<real_t, Eigen::Dynamic, 1> coeffs; // 
        A.ldlt().solveInPlace(b);
        
        Hessian[i*4  ] = b[3]*2.0; // d2/dx2
        Hessian[i*4+1] = b[4];     // d3/dxdy
        Hessian[i*4+2] = b[4];     // d3/dxdy
        Hessian[i*4+3] = b[5]*2.0; // d2/dy2
      }else{
        std::cerr<<"ERROR: 3D not implemented yet\n";
      }
    }
  }
  
  int _NNodes, _NElements, _ndims;
  const index_t *_ENList, *_node_distribution;
  const real_t *_x, *_y, *_z;
  std::vector< std::set<index_t> > NNList;
#ifdef HAVE_MPI
  const MPI_Comm *_mesh_comm;
#endif
  MetricTensor<real_t> *_metric;
};

#endif
