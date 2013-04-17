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
  class MetricField2D{
 public:
  
  /*! Default constructor.
   */
  MetricField2D(Mesh<real_t, index_t> &mesh, Surface2D<real_t, index_t> &surface){
    _NNodes = mesh.get_number_nodes();
    _NElements = mesh.get_number_elements();
    _surface = &surface;
    _mesh = &mesh;
    
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
  ~MetricField2D(){
    if(_metric!=NULL)
      delete [] _metric;
  }
  
  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(float *metric){
    // Enforce first-touch policy.
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(metric+i*3);
      }
    }
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(double *metric){
    // Enforce first-touch policy.
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(metric+i*3);
      }
    }
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const float *metric){
    if(_metric==NULL)
      _metric = new MetricTensor2D<float>[_NNodes];
    
    for(int i=0;i<_NNodes;i++)
      _metric[i].set_metric(metric+3*i);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const double *metric){
    if(_metric==NULL)
      _metric = new MetricTensor2D<float>[_NNodes];
    
    for(int i=0;i<_NNodes;i++)
      _metric[i].set_metric(metric+3*i);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   * @param id is the node index of the metric field being set.
   */
  void set_metric(const float *metric, int id){
    if(_metric==NULL)
      _metric = new MetricTensor2D<float>[_NNodes];

    _metric[id].set_metric(metric);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   * @param id is the node index of the metric field being set.
   */
  void set_metric(const double *metric, int id){
    if(_metric==NULL)
      _metric = new MetricTensor2D<float>[_NNodes];

    _metric[id].set_metric(metric);
  }

  /// Update the metric field on the mesh.
  void update_mesh(){
    assert(_metric!=NULL);
    
    size_t pNElements = (size_t) predict_nelements_part();

    if(pNElements > _mesh->NElements){
      // Let's leave a safety margin.
      pNElements *= 3;
    }else{
      /* The mesh can contain more elements than the predicted number, however
       * some elements may still need to be refined, therefore until the mesh
       * is coarsened and defraged we need extra space for the new vertices and
       * elements that will be created during refinement.
       */
      pNElements = _mesh->NElements * 3;
    }

    // In 2D, the number of nodes is ~ 1/2 the number of elements.
    _mesh->_ENList.resize(pNElements*3);
    _mesh->_coords.resize(pNElements);
    _mesh->metric.resize(pNElements*1.5);
    _mesh->NNList.resize(pNElements/2);
    _mesh->NEList.resize(pNElements/2);
    _mesh->node_owner.resize(pNElements/2, -1);
    _mesh->lnn2gnn.resize(pNElements/2, -1);

#ifdef HAVE_MPI
    // At this point we can establish a new, gappy global numbering system
    if(nprocs>1)
      _mesh->create_gappy_global_numbering(pNElements);
#endif

    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(&(_mesh->metric[i*3]));
      }
    }
    
    // Halo update if parallel
    _mesh->halo_update(&(_mesh->metric[0]), 3);
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
    float *Hessian = new float[_NNodes*3];
    
    bool add_to=true;
    if(_metric==NULL){
      add_to = false;
      _metric = new MetricTensor2D<float>[_NNodes];
    }

    real_t eta = 1.0/target_error;
#pragma omp parallel
    {
      // Calculate Hessian at each point.
      if(p_norm>0){
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          hessian_qls_kernel(psi, i, Hessian);
          
          float *h=Hessian+i*3, m_det;
          /*|h[0] h[1]| 
            |h[1] h[2]|*/
          m_det = fabs(h[0]*h[2]-h[1]*h[1]);
          
          float scaling_factor = eta*pow(m_det, -1.0 / (2.0 * p_norm + 2));  
          for(int j=0;j<3;j++)
            h[j] *= scaling_factor;
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          hessian_qls_kernel(psi, i, Hessian);
          
          for(int j=0;j<3;j++)
            Hessian[i*3+j] *= eta;
        }
      }
      
      // Store metric.
      if(add_to){
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          // Merge this metric with the existing metric field.
          _metric[i].constrain(Hessian+i*3);
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          _metric[i].set_metric(Hessian+i*3);
        }
      }
    }

    delete [] Hessian;
  }

  /*! Apply gradation to the metric field.
   * @param gradation specifies the required gradation factor (<=1.3 recommended).
   */
  void apply_gradation(double gradation){
    // Form NNlist.
    std::deque< std::set<index_t> > NNList( _NNodes );
    for(int e=0; e<_NElements; e++){
      const index_t *n=_mesh->get_element(e);           // indices for element e start at _ENList[ nloc*e ]
      for(index_t i=0; i<3; i++){
        for(index_t j=i+1; j<3; j++){
          NNList[n[i]].insert(n[j]);
          NNList[n[j]].insert(n[i]);
        }
      }
    }

    // float log_gradation = logf(gradation);

    // This is used to ensure we don't revisit parts of the mesh that
    // are known to have converged.
    // std::vector<bool> active(_NNodes, true);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(size_t cnt=0; cnt<100; cnt++){
        
        for(index_t p=0;p<_NNodes;p++){
          
          float Dp[2], Vp[4], hp[2];
          
          std::set<index_t> adjacent_nodes = _mesh->get_node_patch(p);
          
          for(typename std::set<index_t>::const_iterator it=adjacent_nodes.begin(); it!=adjacent_nodes.end(); it++){
            index_t q=*it;
            
            // Resize eigenvalues if necessary
            /*
            float m[3];
            m[0] = (_metric[p].get_metric()[0]+_metric[q].get_metric()[0])*0.5;
            m[1] = (_metric[p].get_metric()[1]+_metric[q].get_metric()[1])*0.5;
            m[2] = (_metric[p].get_metric()[2]+_metric[q].get_metric()[2])*0.5;
            
            float Lpq = ElementProperty<real_t>::length2d(&(_mesh->_coords[p*2]), &(_mesh->_coords[q*2]), m);
            
            if(Lpq==0)
              continue;
            
            float dh=Lpq*log_gradation;
            */
            float dx = _mesh->_coords[p*2] - _mesh->_coords[q*2];
            float dy = _mesh->_coords[p*2+1] - _mesh->_coords[q*2+1];

            float Lpq = sqrtf(dx*dx+dy*dy);
            float dh=Lpq*gradation;

            MetricTensor2D<float> Mq(_metric[q]);
            Mq.eigen_decomp(Dp, Vp);
            
            hp[0] = 1.0/sqrt(Dp[0]) + dh;
            hp[1] = 1.0/sqrt(Dp[1]) + dh;
            
            Dp[0] = 1.0/(hp[0]*hp[0]); 
            Dp[1] = 1.0/(hp[1]*hp[1]); 
            
            Mq.eigen_undecomp(Dp, Vp);
            
            _metric[p].constrain(Mq.get_metric());
          }
        }
      }
    }

    return;
  }

  /*! Apply maximum edge length constraint.
   * @param max_len specifies the maximum allowed edge length.
   */
  void apply_max_edge_length(real_t max_len){
    float M[3];
    M[0] = 1.0/(max_len*max_len);
    M[1] = 0.0;
    M[2] = M[0];

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].constrain(&(M[0]));
    }
  }

  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length globally.
   */
  void apply_min_edge_length(real_t min_len){
    float M[3];
    M[0] = 1.0/(min_len*min_len);
    M[1] = 0.0;
    M[2] = M[0];
    
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].constrain(&(M[0]), false);
    }
  }

  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length locally at each vertex.
   */
  void apply_min_edge_length(const real_t *min_len){
    float M[3];
    
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int n=0;n<_NNodes;n++){
        M[0] = 1.0/(min_len[n]*min_len[n]);
        M[1] = 0.0;
        M[2] = M[0];
        
        _metric[n].constrain(M, false);
      }
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
    float scale_factor = nelements/predict_nelements();
    
    std::cerr<<"here\n";

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].scale(scale_factor);
    }
  }

  /*! Predict the number of elements in this partition when mesh satisfies metric tensor field.
   */
  real_t predict_nelements_part(){
    float predicted=0;
    float inv3=1.0/3.0;

    if(_NElements>0){
      const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
      const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
      const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
      ElementProperty<real_t> property(refx0, refx1, refx2);

      real_t total_area_metric = 0.0;
      for(int i=0;i<_NElements;i++){
        const index_t *n=_mesh->get_element(i);

        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        real_t area = property.area(x0, x1, x2);

        const float *m0=_metric[n[0]].get_metric();
        const float *m1=_metric[n[1]].get_metric();
        const float *m2=_metric[n[2]].get_metric();

        real_t m00 = (m0[0]+m1[0]+m2[0])*inv3;
        real_t m01 = (m0[1]+m1[1]+m2[1])*inv3;
        real_t m11 = (m0[2]+m1[2]+m2[2])*inv3;

        real_t det = m00*m11-m01*m01;
        total_area_metric += area*sqrt(det);
      }

      // Ideal area of triangle in metric space.
      double ideal_area = sqrt(3.0)/4.0;

      predicted = total_area_metric/ideal_area;
    }

    return predicted;
  }

  /*! Predict the number of elements when mesh satisfies metric tensor field.
   */
  real_t predict_nelements(){
    float predicted=predict_nelements_part();

#ifdef HAVE_MPI
    if(nprocs>1){
      MPI_Allreduce(MPI_IN_PLACE, &predicted, 1, MPI_FLOAT, MPI_SUM, _mesh->get_mpi_comm());
    }
#endif
    
    return predicted;
  }

 private:
  
  /// Least squared Hessian recovery.
  void hessian_qls_kernel(const real_t *psi, int i, float *Hessian){
    int min_patch_size = 6;

    std::set<index_t> patch = _mesh->get_node_patch(i, min_patch_size);
    patch.insert(i);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = a0*y^2+a1*x^2+a2*x*y+a3*y+a4*x+a5
    // A = P^TP
    Eigen::Matrix<real_t, 6, 6> A = Eigen::Matrix<real_t, 6, 6>::Zero(6,6);
    Eigen::Matrix<real_t, 6, 1> b = Eigen::Matrix<real_t, 6, 1>::Zero(6);
    
    real_t x0=_mesh->_coords[i*2], y0=_mesh->_coords[i*2+1];
    
    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      real_t x=_mesh->_coords[(*n)*2]-x0, y=_mesh->_coords[(*n)*2+1]-y0;

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

    Eigen::Matrix<real_t, 6, 1> a = Eigen::Matrix<real_t, 6, 1>::Zero(6);
    A.svd().solve(b, &a);

    Hessian[i*3  ] = 2*a[1]; // d2/dx2
    Hessian[i*3+1] = a[2];   // d2/dxdy
    Hessian[i*3+2] = 2*a[0]; // d2/dy2
  }

  int rank, nprocs;
  int _NNodes, _NElements;
  MetricTensor2D<float> *_metric;
  Surface2D<real_t, index_t> *_surface;
  Mesh<real_t, index_t> *_mesh;
};

// 3D implementation

/*! \brief Constructs metric tensor field which encodes anisotropic
 *  edge size information.
 *
 * Use this class to create a metric tensor field using desired error
 * tolerences and curvature of linear fields subject to a set of constraints.
 * index_t should be set as int32_t for problems with less than 2 billion nodes.
 * For larger meshes int64_t should be used.
 */
template<typename real_t, typename index_t>
  class MetricField3D{
 public:

  /*! Default constructor.
   */
  MetricField3D(Mesh<real_t, index_t> &mesh, Surface3D<real_t, index_t> &surface){
    _NNodes = mesh.get_number_nodes();
    _NElements = mesh.get_number_elements();
    _surface = &surface;
    _mesh = &mesh;
    
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
  ~MetricField3D(){
    if(_metric!=NULL)
      delete [] _metric;
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(float *metric){
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(metric+i*6);
      }
    }
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(double *metric){
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(metric+i*6);
      }
    }
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const float *metric){
    if(_metric==NULL)
      _metric = new MetricTensor3D<float>[_NNodes];
    
    for(int i=0;i<_NNodes;i++)
      _metric[i].set_metric(metric+i*6);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const double *metric){
    if(_metric==NULL)
      _metric = new MetricTensor3D<float>[_NNodes];
    
    for(int i=0;i<_NNodes;i++)
      _metric[i].set_metric(metric+i*6);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   * @param id is the node index of the metric field being set.
   */
  void set_metric(const float *metric, int id){
    if(_metric==NULL)
      _metric = new MetricTensor3D<float>[_NNodes];

    _metric[id].set_metric(metric);
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   * @param id is the node index of the metric field being set.
   */
  void set_metric(const double *metric, int id){
    if(_metric==NULL)
      _metric = new MetricTensor3D<float>[_NNodes];

    _metric[id].set_metric(metric);
  }

  /// Update the metric field on the mesh.
  void update_mesh(){
    assert(_metric!=NULL);

    _mesh->metric.clear();
    _mesh->metric.resize(_NNodes*6);
    
    // Enforce first-touch policy
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++){
        _metric[i].get_metric(&(_mesh->metric[i*6]));
      }
    }

    // Halo update if parallel
    _mesh->halo_update(&(_mesh->metric[0]), 6);
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
    float *Hessian = new float[_NNodes*6];
    
    bool add_to=true;
    if(_metric==NULL){
      add_to = false;
      _metric = new MetricTensor3D<float>[_NNodes];
    }

#pragma omp parallel
    {
      // Calculate Hessian at each point.
      float eta = 1.0/target_error;
      
      if(p_norm>0){
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          hessian_qls_kernel(psi, i, Hessian);
          
          float *h=Hessian+i*6, m_det;
          
          /*|h[0] h[1] h[2]| 
            |h[3] h[4] h[5]| 
            |h[6] h[7] h[8]|*/
          m_det = h[0]*(h[4]*h[8]-h[5]*h[7]) - h[1]*(h[3]*h[8]-h[5]*h[6]) + h[2]*(h[3]*h[7]-h[4]*h[6]);
          
          float scaling_factor = eta*pow(m_det, -1.0 / (2.0 * p_norm + 3));  
          for(int j=0;j<6;j++)
            h[j] = scaling_factor*h[j];
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){
          hessian_qls_kernel(psi, i, Hessian);
          
          for(int j=0;j<6;j++)
            Hessian[i*6+j]*=eta;
        }
      }
      
      // Store metric
      if(add_to){
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){ 
          // Merge this metric with the existing metric field.      
          _metric[i].constrain(Hessian+i*6);
        }
      }else{
#pragma omp for schedule(static)
        for(int i=0; i<_NNodes; i++){ 
          _metric[i].set_metric(Hessian+i*6);
        }
      }
    }
    delete [] Hessian;
  }

  /*! Apply maximum edge length constraint.
   * @param max_len specifies the maximum allowed edge length.
   */
  void apply_max_edge_length(float max_len){
    float M[6];
    float m = 1.0/(max_len*max_len);
    M[0] = m;
    M[1] = 0.0;
    M[2] = 0.0;
    M[3] = m;
    M[4] = 0.0; 
    M[5] = m;
    
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].constrain(M);
    }
  }
  
  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length globally.
   */
  void apply_min_edge_length(real_t min_len){
     float M[6];
    float m = 1.0/(min_len*min_len);
    M[0] = m;
    M[1] = 0.0;
    M[2] = 0.0;
    M[3] = m;
    M[4] = 0.0; 
    M[5] = m;
    
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].constrain(M, false);
    }
  }

  /*! Apply minimum edge length constraint.
   * @param min_len specifies the minimum allowed edge length locally at each vertex.
   */
  void apply_min_edge_length(const real_t *min_len){
#pragma omp parallel
    {
      float M[6];
#pragma omp for schedule(static)
      for(int n=0;n<_NNodes;n++){
        float m = 1.0/(min_len[n]*min_len[n]);
        M[0] = m;
        M[1] = 0.0;
        M[2] = 0.0;
        M[3] = m;
        M[4] = 0.0; 
        M[5] = m;
        
        _metric[n].constrain(M, false);
      }
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
    float scale_factor = pow((nelements/predict_nelements()), 2.0/3.0);
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for(int i=0;i<_NNodes;i++)
        _metric[i].scale(scale_factor);
    }
  }

  /*! Predict the number of elements when mesh satisifies metric tensor field.
   */
  real_t predict_nelements(){
    double predicted=0;

    if(_NElements>0){
      const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
      const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
      const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
      const real_t *refx3 = _mesh->get_coords(_mesh->get_element(0)[3]);
      ElementProperty<real_t> property(refx0, refx1, refx2, refx3);
      
      float total_volume_metric = 0.0;
      for(int i=0;i<_NElements;i++){
        const index_t *n=_mesh->get_element(i);
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        const real_t *x3 = _mesh->get_coords(n[3]);
        float volume = property.volume(x0, x1, x2, x3);
        
        const float *m0=_metric[n[0]].get_metric();
        const float *m1=_metric[n[1]].get_metric();
        const float *m2=_metric[n[2]].get_metric();
        const float *m3=_metric[n[3]].get_metric();
        
        float m00 = (m0[0]+m1[0]+m2[0]+m3[0])*0.25;
        float m01 = (m0[1]+m1[1]+m2[1]+m3[1])*0.25;
        float m02 = (m0[2]+m1[2]+m2[2]+m3[2])*0.25;
        float m11 = (m0[3]+m1[3]+m2[3]+m3[3])*0.25;
        float m12 = (m0[4]+m1[4]+m2[4]+m3[4])*0.25;
        float m22 = (m0[5]+m1[5]+m2[5]+m3[5])*0.25;
        
        real_t det = (m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02;
        total_volume_metric += volume*sqrt(det);
      }
        
      // Ideal volume of triangle in metric space.
      double ideal_volume = 1.0/sqrt(72.0);
      
      predicted = total_volume_metric/ideal_volume;
    }
    
#ifdef HAVE_MPI
    if(nprocs>1){
      MPI_Allreduce(MPI_IN_PLACE, &predicted, 1, MPI_FLOAT, MPI_SUM, _mesh->get_mpi_comm());
    }
#endif
    
    return predicted;
  }

 private:
  
  /// Least squared Hessian recovery.
  void hessian_qls_kernel(const real_t *psi, int i, float *Hessian){
    size_t min_patch_size=10;

    std::set<index_t> patch;
    if(_surface->contains_node(i))
      patch = _mesh->get_node_patch(i, 2*min_patch_size);
    else
      patch = _mesh->get_node_patch(i, min_patch_size);
    patch.insert(i);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = 1 + x + y + z + x^2 + y^2 + z^2 + xy + xz + yz
    // A = P^TP
    Eigen::Matrix<real_t, 10, 10> A = Eigen::Matrix<real_t, 10, 10>::Zero(10,10);
    Eigen::Matrix<real_t, 10, 1> b = Eigen::Matrix<real_t, 10, 1>::Zero(10);

    real_t x0=_mesh->_coords[i*3], y0=_mesh->_coords[i*3+1], z0=_mesh->_coords[i*3+2];
    
    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      real_t x=_mesh->_coords[(*n)*3]-x0, y=_mesh->_coords[(*n)*3+1]-y0, z=_mesh->_coords[(*n)*3+2]-z0;
      
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
                  
    Eigen::Matrix<real_t, 10, 1> a = Eigen::Matrix<real_t, 10, 1>::Zero(10);
    A.svd().solve(b, &a);

    Hessian[i*6  ] = a[4]*2.0; // d2/dx2
    Hessian[i*6+1] = a[5];     // d2/dxdy
    Hessian[i*6+2] = a[6];     // d2/dxdz
    Hessian[i*6+3] = a[7]*2.0; // d2/dy2
    Hessian[i*6+4] = a[8];     // d2/dydz
    Hessian[i*6+5] = a[9]*2.0; // d2/dz2
  }

  int rank, nprocs;
  int _NNodes, _NElements;
  MetricTensor3D<float> *_metric;
  Surface3D<real_t, index_t> *_surface;
  Mesh<real_t, index_t> *_mesh;
};

#endif
