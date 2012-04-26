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

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cfloat>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "MetricTensor.h"
#include "Surface.h"
#include "Mesh.h"
#include "ElementProperty.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
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

  /*! Default constructor.
   */
  MetricField(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
    _NNodes = mesh.get_number_nodes();
    _NElements = mesh.get_number_elements();
    _ndims = mesh.get_number_dimensions();
    _nloc = (_ndims==2)?3:4;
    _surface = &surface;
    _mesh = &mesh;

    set_hessian_method("qls");
    
    rank = 0;
    nprocs = 1;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
      MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
    }
#endif

    _metric = NULL;
  }

  /*! Default destructor.
   */
  ~MetricField(){
    if(_metric!=NULL)
      delete [] _metric;
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(real_t *metric){
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(metric+i*_ndims*_ndims);
      }
    }
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const real_t *metric){
    if(_metric==NULL)
      _metric = new MetricTensor<real_t>[_NNodes];
    
    const size_t stride = _ndims*_ndims;
    for(int i=0;i<_NNodes;i++)
      _metric[i].set_metric(_ndims, metric+stride*i);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   * @param id is the node index of the metric field being set.
   */
  void set_metric(const real_t *metric, int id){
    if(_metric==NULL)
      _metric = new MetricTensor<real_t>[_NNodes];

    _metric[id].set_metric(_ndims, metric);
  }

  /// Update the metric field on the mesh.
  void update_mesh(){
    assert(_metric!=NULL);

    _mesh->metric.clear();
    _mesh->metric.resize(_NNodes*_ndims*_ndims);
    
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(&(_mesh->metric[i*_ndims*_ndims]));
      }
    }

    // Halo update if parallel
    _mesh->halo_update(&(_mesh->metric[0]), _ndims*_ndims);
  }

  /*! Add the contribution from the metric field from a new field with a target linear interpolation error. 
   * @param psi is field while curvature is to be considered.
   * @param target_error is the user target error for a given norm.
   * @param p_norm Set this optional argument to a positive integer to
    apply the p-norm scaling to the metric, as in Chen, Sun and Xu,
    Mathematics of Computation, Volume 76, Number 257, January 2007,
    pp. 179-204.
   */
  void add_field(const real_t *psi, const real_t target_error, int p_norm=-1){
    int ndims2=_ndims*_ndims;
    real_t *Hessian = new real_t[_NNodes*ndims2];
    
    bool add_to=true;
    if(_metric==NULL){
      add_to = false;
      _metric = new MetricTensor<real_t>[_NNodes];
    }

#pragma omp parallel
    {
      // Calculate Hessian at each point.
#pragma omp for schedule(static)
      for(int i=0; i<_NNodes; i++){
        (this->*hessian_kernel)(psi, i, Hessian);
        
        if(p_norm>0){
          double *h=Hessian+i*ndims2, m_det;
          if(_ndims==2){
            /*|h[0] h[1]| 
              |h[2] h[3]|*/
            m_det = fabs(h[0]*h[3]-h[1]*h[2]);
          }else{
            /*|h[0] h[1] h[2]| 
              |h[3] h[4] h[5]| 
              |h[6] h[7] h[8]|*/
            m_det = h[0]*(h[4]*h[8]-h[5]*h[7]) - h[1]*(h[3]*h[8]-h[5]*h[6]) + h[2]*(h[3]*h[7]-h[4]*h[6]);
          }
          
          double scaling_factor = pow(m_det, -1.0 / (2.0 * p_norm + _ndims));  
          for(int j=0;j<ndims2;j++)
            h[j] = scaling_factor*h[j];
        }
        
        real_t eta = 1.0/target_error;
        for(int j=0;j<ndims2;j++)
          Hessian[i*ndims2+j]*=eta;
        
        // Merge this metric with the existing metric field.
        if(add_to)
          _metric[i].constrain(Hessian+i*ndims2);
        else
          _metric[i].set_metric(_ndims, Hessian+i*ndims2);
      }
    }
    delete [] Hessian;
  }

  // For this to work, need to replace calls to vtk objects with calls to new mesh class
  void apply_gradation(double gradation)
  {
    // Form NNlist.
    std::deque< std::set<index_t> > NNList( _NNodes );
    for(int e=0; e<_NElements; e++){
      if(_ndims == 2)
      {
        const index_t *n=_mesh->get_element(e);           // indices for element e start at _ENList[ nloc*e ]
        for(index_t i=0; i<3; i++){
          for(index_t j=i+1; j<3; j++)
          {
            NNList[n[i]].insert(n[j]);
            NNList[n[j]].insert(n[i]);
          }
        }
      }
      else if( _ndims == 3 )
      {
        const index_t *n=_mesh->get_element(e);           // indices for element e start at _ENList[ nloc*e ]
        for(index_t i=0; i<4; i++)
        {
          for(index_t j=i+1; j<4; j++)
          {
            NNList[n[i]].insert(n[j]);
            NNList[n[j]].insert(n[i]);
          }
        }
      }
      else
      {
        std::cerr<<"ERROR: unsupported dimension: " << _ndims << " (must be 2 or 3)" << std::endl;
      }
    } 

    double log_gradation = log(gradation);

    // This is used to ensure we don't revisit parts of the mesh that
    // are known to have converged.
    std::set<int> hits;
    for(size_t cnt=0; cnt<10; cnt++)
    {
      std::multimap<double, size_t> ordered_edges;
      if(cnt==0)
      {
        // Iterate over everything.
        for(int n=0; n<_NNodes; n++)
        {
          double l = _metric[n].average_length();
          ordered_edges.insert(std::pair<double, size_t>(l, n));
        }
      }
      else
      {
        // Iterate over only nodes which were modified in the previous iteration.
        for(std::set<int>::const_iterator n=hits.begin(); n!=hits.end(); ++n)
        {
          double l = _metric[*n].average_length();
          ordered_edges.insert(std::pair<double, size_t>(l, *n));
        }
        hits.clear();
      }

      for(std::multimap<double, size_t>::const_iterator n=ordered_edges.begin(); n!=ordered_edges.end(); n++)
      {
        // Used to ensure that the front cannot go back on itself.
        std::set<size_t> swept;

        // Start the new front
        std::set<size_t> front;
        front.insert(n->second);

        while(!front.empty())
        {
          index_t p=*(front.begin());
          front.erase(p);
          swept.insert(p);

          std::vector<double> Dp(_ndims), Vp(_ndims*_ndims);
          std::vector<double> Dq(_ndims), Vq(_ndims*_ndims);

          std::set<index_t> adjacent_nodes = _mesh->get_node_patch(p);

          for(typename std::set<index_t>::const_iterator it=adjacent_nodes.begin(); 
                it!=adjacent_nodes.end(); 
                it++)
          {
            index_t q=*it;

            if(swept.count(q))
              continue;
            else
              swept.insert(q);

            MetricTensor<real_t> Mp(_ndims, _metric[p].get_metric());
            Mp.eigen_decomp(Dp, Vp);

            MetricTensor<real_t> Mq(_ndims, _metric[q].get_metric());
            Mq.eigen_decomp(Dq, Vq);

            // Pair the eigenvectors between p and q by minimising the angle between them.
            std::vector<int> pairs(_ndims, -1);
            std::vector<bool> paired(_ndims, false);
            for(index_t d=0; d<_ndims; d++)
            {
              std::vector<double> angle(_ndims);
              for(index_t k=0; k<_ndims; k++)
              {
                if(paired[k])
                  continue;
                angle[k] = Vp[d*_ndims]*Vq[k*_ndims];
                for(index_t l=1; l<_ndims; l++)
                  angle[k] += Vp[d*_ndims+l]*Vq[k*_ndims+l];
                angle[k] = acos(fabs(angle[k]));
              }

              index_t r=0;
              for(;r<_ndims;r++)
              {
                if(!paired[r])
                {
                  pairs[d] = r;
                  break;
                }
              }
              r++;

              for(;r<_ndims;r++)
              {
                if(angle[pairs[d]]<angle[r]){
                  pairs[d] = r;
                }
              }

              paired[pairs[d]] = true;

              assert(pairs[d]!=-1);
            }

            // Resize eigenvalues if necessary
            double Lpq=length(p, q);
            double dh=Lpq*log_gradation;
            bool add_p=false, add_q=false;
            for(index_t k=0; k<_ndims; k++)
            {
              double hp = 1.0/sqrt(Dp[k]);
              double hq = 1.0/sqrt(Dq[pairs[k]]);
              double gamma = exp(fabs(hp - hq)/Lpq);

              if(isinf(gamma))
                gamma = DBL_MAX;
              if(gamma>(1.05*gradation))
              {
                if(hp>hq)
                {
                  hp = hq + dh;
                  Dp[k] = 1.0/(hp*hp);
                  add_p = true;
                }
                else
                {
                  hq = hp + dh;
                  Dq[pairs[k]] = 1.0/(hq*hq);
                  add_q = true;
                }
              }
            }


            // Reform metrics if modified
            if(add_p)
            {
              front.insert(p);

              Mp.eigen_undecomp(&(Dp[0]), &(Vp[0]));
              _metric[p].set_metric(_ndims, Mp.get_metric());
              hits.insert(p);
            }
            if(add_q)
            {
              front.insert(q);

              Mq.eigen_undecomp(&(Dq[0]), &(Vq[0]));
              _metric[q].set_metric(_ndims, Mq.get_metric());
              hits.insert(p);
            } 
          }
        }
      }

      if(hits.empty())
        break;
    }

    return;
  }

  /*! Apply maximum edge length constraint.
   * @param max_len specifies the maximum allowed edge length.
   */
  void apply_max_edge_length(real_t max_len){
    std::vector<real_t> M(_ndims*_ndims);
    for(int i=0;i<_ndims;i++)
      for(int j=0;j<_ndims;j++)
        if(i==j)
          M[i*_ndims+j] = 1.0/(max_len*max_len);
        else
          M[i*_ndims+j] = 0.0;

    for(int i=0;i<_NNodes;i++)
      _metric[i].constrain(&(M[0]));
  }

  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length globally.
   */
  void apply_min_edge_length(real_t min_len){
    std::vector<real_t> M(_ndims*_ndims);
    for(int i=0;i<_ndims;i++)
      for(int j=0;j<_ndims;j++)
        if(i==j)
          M[i*_ndims+j] = 1.0/(min_len*min_len);
        else
          M[i*_ndims+j] = 0.0;

    for(int i=0;i<_NNodes;i++)
      _metric[i].constrain(&(M[0]), false);
  }

  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length locally at each vertex.
   */
  void apply_min_edge_length(const real_t *min_len){
    std::vector<real_t> M(_ndims*_ndims);
    for(int n=0;n<_NNodes;n++){
      for(int i=0;i<_ndims;i++){
        for(int j=0;j<_ndims;j++){
          if(i==j)
            M[i*_ndims+j] = 1.0/(min_len[n]*min_len[n]);
          else
            M[i*_ndims+j] = 0.0;
        }
      }
      
      _metric[n].constrain(M, false);
    }
  }

  /*! Apply maximum aspect ratio constraint.
   * @param max_aspect_ratio maximum aspect ratio for elements.
   */
  void apply_max_aspect_ratio(real_t max_aspect_ratio){
    std::cerr<<"ERROR: Not yet implemented\n";
  }

  /*! Apply maximum number of elements constraint.
   * @param nelements the maximum number of elements desired.
   */
  void apply_max_nelements(real_t nelements){
    int predicted = predict_nelements();
    if(predicted>nelements)
      apply_nelements(nelements);
  }

  /*! Apply minimum number of elements constraint.
   * @param nelements the minimum number of elements desired.
   */
  void apply_min_nelements(real_t nelements){
    int predicted = predict_nelements();
    if(predicted<nelements)
      apply_nelements(nelements);
  }

  /*! Apply required number of elements.
   * @param nelements is the required number of elements after adapting.
   */
  void apply_nelements(real_t nelements){
    real_t scale_factor = pow((nelements/predict_nelements()), ((real_t)2.0/_ndims));

    for(int i=0;i<_NNodes;i++)
      _metric[i].scale(scale_factor);
  }

  /*! Predict the number of elements when mesh satisifies metric tensor field.
   */
  real_t predict_nelements(){
    double predicted=0;

    if(_NElements>0){
      if(_ndims==2){
        const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
        const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
        const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
        ElementProperty<real_t> property(refx0, refx1, refx2);
        
        real_t total_area_metric = 0.0;
        for(int i=0;i<_NElements;i++){
          const index_t *n=_mesh->get_element(i);
          if(n[0]<0)
            continue;

          const real_t *x0 = _mesh->get_coords(n[0]);
          const real_t *x1 = _mesh->get_coords(n[1]);
          const real_t *x2 = _mesh->get_coords(n[2]);
          real_t area = property.area(x0, x1, x2);
          
          const real_t *m0=_metric[n[0]].get_metric();
          const real_t *m1=_metric[n[1]].get_metric();
          const real_t *m2=_metric[n[2]].get_metric();
          
          real_t m00 = (m0[0]+m1[0]+m2[0])/3;
          real_t m01 = (m0[1]+m1[1]+m2[1])/3;
          real_t m11 = (m0[3]+m1[3]+m2[3])/3;
          
          real_t det = m00*m11-m01*m01;
          total_area_metric += area*sqrt(det);
        }
        
        // Ideal area of triangle in metric space.
        double ideal_area = sqrt(3.0)/4.0;
        
        predicted = total_area_metric/ideal_area;
      }else{
        const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
        const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
        const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
        const real_t *refx3 = _mesh->get_coords(_mesh->get_element(0)[3]);
        ElementProperty<real_t> property(refx0, refx1, refx2, refx3);
        
        real_t total_volume_metric = 0.0;
        for(int i=0;i<_NElements;i++){
          const index_t *n=_mesh->get_element(i);
          if(n[0]<0)
            continue;

          const real_t *x0 = _mesh->get_coords(n[0]);
          const real_t *x1 = _mesh->get_coords(n[1]);
          const real_t *x2 = _mesh->get_coords(n[2]);
          const real_t *x3 = _mesh->get_coords(n[3]);
          real_t volume = property.volume(x0, x1, x2, x3);
          
          const real_t *m0=_metric[n[0]].get_metric();
          const real_t *m1=_metric[n[1]].get_metric();
          const real_t *m2=_metric[n[2]].get_metric();
          const real_t *m3=_metric[n[3]].get_metric();
          
          real_t m00 = (m0[0]+m1[0]+m2[0]+m3[0])/4;
          real_t m01 = (m0[1]+m1[1]+m2[1]+m3[1])/4;
          real_t m02 = (m0[2]+m1[2]+m2[2]+m3[2])/4;
          real_t m11 = (m0[4]+m1[4]+m2[4]+m3[4])/4;
          real_t m12 = (m0[5]+m1[5]+m2[5]+m3[5])/4;
          real_t m22 = (m0[8]+m1[8]+m2[8]+m3[8])/4;
          
          real_t det = (m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02;
          total_volume_metric += volume*sqrt(det);
        }
        
        // Ideal volume of triangle in metric space.
        double ideal_volume = 1.0/sqrt(72.0);
        
        predicted = total_volume_metric/ideal_volume;
      }
    }

#ifdef HAVE_MPI
    if(nprocs>1){
      MPI_Allreduce(MPI_IN_PLACE, &predicted, 1, MPI_DOUBLE, MPI_SUM, _mesh->get_mpi_comm());
    }
#endif
    
    return predicted;
  }

  /*! Choose method for evaluating Hessian of field.
   * @param method - valid values are "qls", "qls2".
   */
  void set_hessian_method(const char *method){
    if(_ndims==2){
      if(std::string(method)=="qls"){
        min_patch_size = 6;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_2d;
      }else if(std::string(method)=="qls2"){
        min_patch_size = 12;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_2d;
      }else{
        std::cerr<<"WARNING: unknown Hessian recovery method specified. Using default.\n";
        min_patch_size = 6;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_2d;
      }
    }else{
      if(std::string(method)=="qls"){
        min_patch_size = 10;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_3d;
      }else if(std::string(method)=="qls2"){
        min_patch_size = 20;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_3d;
      }else{
        std::cerr<<"WARNING: unknown Hessian recovery method specified. Using default.\n";
        min_patch_size = 10;
        hessian_kernel = &MetricField<real_t, index_t>::hessian_qls_kernel_3d;
      }
    }
  }

 private:

  inline real_t get_x(index_t nid){
    return _mesh->get_coords(nid)[0];
  }

  inline real_t get_y(index_t nid){
    return _mesh->get_coords(nid)[1];
  }

  inline real_t get_z(index_t nid){
    return _mesh->get_coords(nid)[2];
  }

  inline double length(index_t n0, index_t n1) const{
    assert(n0!=n1);

    double l = 0.0;
    for(index_t i=0; i<_ndims; i++)
      l += pow(_mesh->get_coords(n0)[i]-_mesh->get_coords(n1)[i], 2);
//    l +=  (      x_i|n_0       -      x_1|n_1       )^2
    return sqrt(l);
  }
  
  /// Least squared Hessian recovery.
  void hessian_qls_kernel_2d(const real_t *psi, int i, real_t *Hessian){
    std::set<index_t> patch = _mesh->get_node_patch(i, min_patch_size);
    patch.insert(i);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = a0*y^2+a1*x^2+a2*x*y+a3*y+a4*x+a5
    // A = P^TP
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    
    double x0=get_x(i), y0=get_y(i);

    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      double x=get_x(*n)-x0, y=get_y(*n)-y0;
      
      A[0]+=y*y*y*y;
      A[6]+=x*x*y*y;  A[7]+=x*x*x*x;  
      A[12]+=x*y*y*y; A[13]+=x*x*x*y; A[14]+=x*x*y*y;
      A[18]+=y*y*y;   A[19]+=x*x*y;   A[20]+=x*y*y;   A[21]+=y*y;
      A[24]+=x*y*y;   A[25]+=x*x*x;   A[26]+=x*x*y;   A[27]+=x*y; A[28]+=x*x;
      A[30]+=y*y;     A[31]+=x*x;     A[32]+=x*y;     A[33]+=y;   A[34]+=x;   A[35]+=1;
      
      b[0]+=psi[*n]*y*y; b[1]+=psi[*n]*x*x; b[2]+=psi[*n]*x*y; b[3]+=psi[*n]*y; b[4]+=psi[*n]*x; b[5]+=psi[*n];
    }
    A[1] = A[6]; A[2] = A[12]; A[3] = A[18]; A[4] = A[24]; A[5] = A[30];
                 A[8] = A[13]; A[9] = A[19]; A[10]= A[25]; A[11]= A[31];
                               A[15]= A[20]; A[16]= A[26]; A[17]= A[32];
                                             A[22]= A[27]; A[23]= A[33];
                                                           A[29]= A[34];

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.svd().solve(b, &a);

    Hessian[i*4  ] = 2*a[1]; // d2/dx2
    Hessian[i*4+1] = a[2];   // d2/dxdy
    Hessian[i*4+2] = a[2];   // d2/dxdy
    Hessian[i*4+3] = 2*a[0]; // d2/dy2
  }
  
  /// Least squared Hessian recovery.
  void hessian_qls_kernel_3d(const real_t *psi, int i, real_t *Hessian){
    std::set<index_t> patch;
    if(_surface->contains_node(i))
      patch = _mesh->get_node_patch(i, 2*min_patch_size);
    else
      patch = _mesh->get_node_patch(i, min_patch_size);
    patch.insert(i);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = 1 + x + y + z + x^2 + y^2 + z^2 + xy + xz + yz
    // A = P^TP
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(10,10);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(10);

    double x0=get_x(i), y0=get_y(i), z0=get_z(i);
    
    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      double x=get_x(*n)-x0, y=get_y(*n)-y0, z=get_z(*n)-z0;
      
      A[0]+=1;
      A[10]+=x;   A[11]+=x*x;
      A[20]+=y;   A[21]+=x*y;   A[22]+=y*y;
      A[30]+=z;   A[31]+=x*z;   A[32]+=y*z;   A[33]+=z*z;
      A[40]+=x*x; A[41]+=x*x*x; A[42]+=x*x*y; A[43]+=x*x*z; A[44]+=x*x*x*x;
      A[50]+=x*y; A[51]+=x*x*y; A[52]+=x*y*y; A[53]+=x*y*z; A[54]+=x*x*x*y; A[55]+=x*x*y*y;
      A[60]+=x*z; A[61]+=x*x*z; A[62]+=x*y*z; A[63]+=x*z*z; A[64]+=x*x*x*z; A[65]+=x*x*y*z; A[66]+=x*x*z*z;
      A[70]+=y*y; A[71]+=x*y*y; A[72]+=y*y*y; A[73]+=y*y*z; A[74]+=x*x*y*y; A[75]+=x*y*y*y; A[76]+=x*y*y*z; A[77]+=y*y*y*y;
      A[80]+=y*z; A[81]+=x*y*z; A[82]+=y*y*z; A[83]+=y*z*z; A[84]+=x*x*y*z; A[85]+=x*y*y*z; A[86]+=x*y*z*z; A[87]+=y*y*y*z; A[88]+=y*y*z*z;
      A[90]+=z*z; A[91]+=x*z*z; A[92]+=y*z*z; A[93]+=z*z*z; A[94]+=x*x*z*z; A[95]+=x*y*z*z; A[96]+=x*z*z*z; A[97]+=y*y*z*z; A[98]+=y*z*z*z; A[99]+=z*z*z*z;
      
      b[0]+=psi[*n]*1; b[1]+=psi[*n]*x; b[2]+=psi[*n]*y; b[3]+=psi[*n]*z; b[4]+=psi[*n]*x*x; b[5]+=psi[*n]*x*y; b[6]+=psi[*n]*x*z; b[7]+=psi[*n]*y*y; b[8]+=psi[*n]*y*z; b[9]+=psi[*n]*z*z;
    }
    
    A[1] = A[10]; A[2]  = A[20]; A[3]  = A[30]; A[4]  = A[40]; A[5]  = A[50]; A[6]  = A[60]; A[7]  = A[70]; A[8]  = A[80]; A[9]  = A[90];
                  A[12] = A[21]; A[13] = A[31]; A[14] = A[41]; A[15] = A[51]; A[16] = A[61]; A[17] = A[71]; A[18] = A[81]; A[19] = A[91];
                                 A[23] = A[32]; A[24] = A[42]; A[25] = A[52]; A[26] = A[62]; A[27] = A[72]; A[28] = A[82]; A[29] = A[92];
                                                A[34] = A[43]; A[35] = A[53]; A[36] = A[63]; A[37] = A[73]; A[38] = A[83]; A[39] = A[93];
                                                               A[45] = A[54]; A[46] = A[64]; A[47] = A[74]; A[48] = A[84]; A[49] = A[94];
                                                                              A[56] = A[65]; A[57] = A[75]; A[58] = A[85]; A[59] = A[95];
                                                                                             A[67] = A[76]; A[68] = A[86]; A[69] = A[96];
                                                                                                            A[78] = A[87]; A[79] = A[97];
                                                                                                                           A[89] = A[98];
                  
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(10);
    A.svd().solve(b, &a);

    Hessian[i*9  ] = a[4]*2.0; // d2/dx2
    Hessian[i*9+1] = a[5];     // d2/dxdy
    Hessian[i*9+2] = a[6];     // d2/dxdz
    Hessian[i*9+3] = a[5];     // d2/dydx
    Hessian[i*9+4] = a[7]*2.0; // d2/dy2
    Hessian[i*9+5] = a[8];     // d2/dydz
    Hessian[i*9+6] = a[6];     // d2/dzdx
    Hessian[i*9+7] = a[8];     // d2/dzdy
    Hessian[i*9+8] = a[9]*2.0; // d2/dz2
  }

  int rank, nprocs;
  int _NNodes, _NElements, _ndims, _nloc;
  size_t min_patch_size;
  MetricTensor<real_t> *_metric;
  Surface<real_t, index_t> *_surface;
  Mesh<real_t, index_t> *_mesh;

  void (MetricField<real_t, index_t>::*hessian_kernel)(const real_t *psi, int i, real_t *Hessian);
};

#endif
