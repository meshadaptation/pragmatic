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

#ifndef METRICTENSOR_H
#define METRICTENSOR_H

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

// 2D version
template<typename treal_t> class MetricTensor2D;

template<typename treal_t>
  std::ostream& operator<<(std::ostream& out, const MetricTensor2D<treal_t>& in);

/*! \brief 2D symmetric metic tensor class.
 *
 * Use to store and operate on metric tensors for 2D.
 * The upper triangle is stored, ie:
 * m[0] m[1]
 * .... m[2]
 */
template<typename treal_t> class MetricTensor2D{
 public:
  /// Default constructor.
  MetricTensor2D(){
  }
  
  /// Default destructor.
  ~MetricTensor2D(){
  }
  
  /*! Constructor.
   * The upper triangle is expected, ie:
   * m[0] m[1]
   * .... m[2]
   * @param metric points to the upper triangle of the 2D tensor.
   */
  MetricTensor2D(const treal_t *metric){
    set_metric(metric);
  }

  /*! Copy constructor.
   * @param metric is a reference to a MetricTensor object.
   */
  MetricTensor2D(const MetricTensor2D& metric){
    *this = metric;
  }

  /*! Assignment operator.
   * @param metric is a reference to a MetricTensor2D object.
   */
  const MetricTensor2D& operator=(const MetricTensor2D &metric){
    for(size_t i=0;i<3;i++)
      _metric[i] = metric._metric[i];
    return *this;
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(double *metric) const{
    for(size_t i=0;i<3;i++)
      metric[i] = _metric[i];
  }

  /// Give const pointer to metric tensor.
  const treal_t* get_metric() const{
    return _metric;
  }

  /*! Set the metric tensor field.
   * The upper triangle is expected, ie:
   * m[0] m[1]
   * .... m[2]
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const double *metric){
    for(size_t i=0;i<3;i++)
      _metric[i] = metric[i];
    positive_definiteness(_metric);
  }

  // Enforce positive definiteness
  static void positive_definiteness(treal_t *metric){
    Eigen::Matrix<double, 2, 2> M;
    M <<
      metric[0], metric[1],
      metric[1], metric[2];
    
    if(M.isZero())
      return;
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver(M);
    
    Eigen::Matrix<double, 2, 1> evalues = solver.eigenvalues().real().cwise().abs();
    Eigen::Matrix<double, 2, 2> evectors = solver.eigenvectors().real();
    
    Eigen::Matrix<double, 2, 2> Mp = evectors*evalues.asDiagonal()*evectors.transpose();
    
    metric[0] = Mp[0];
    metric[1] = Mp[1];
    metric[2] = Mp[3];

    return;
  }

  /*! By default this calculates the superposition of two metrics where by default small
   * edge lengths are preserved. If the optional argument perserved_small_edges==false
   * then large edge lengths are perserved instead.
   * @param M is a reference to a MetricTensor object.
   * @param perserved_small_edges when true causes small edge lengths to be preserved (default). Otherwise long edge are perserved.
   */
  void constrain(const treal_t *M_in, bool perserved_small_edges=true){
    for(size_t i=0;i<3;i++)
      if(isnan(M_in[i]))
        return;
    
    MetricTensor2D<treal_t> metric(M_in);

    // Make the tensor with the smallest aspect ratio the reference space Mr.
    const treal_t *Mr=_metric, *Mi=metric._metric;
    Eigen::Matrix<double, 2, 2> M1;
    M1 <<
      _metric[0], _metric[1],
      _metric[1], _metric[2];
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver1(M1);
    
    Eigen::Matrix<double, 2, 1> evalues1 = solver1.eigenvalues().real().cwise().abs();
    
    treal_t aspect_r = std::min(evalues1[0], evalues1[1])/
      std::max(evalues1[0], evalues1[1]);
    
    Eigen::Matrix<double, 2, 2> M2;
    M2 <<
      metric._metric[0], metric._metric[1],
      metric._metric[1], metric._metric[2];
    
    // The input matrix could be zero if there is zero curvature in the local solution.
    if(M2.isZero())
      return;
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver2(M2);
    
    Eigen::Matrix<double, 2, 1> evalues2 = solver2.eigenvalues().real().cwise().abs();
    
    treal_t aspect_i = std::min(evalues2[0], evalues2[1])/
      std::max(evalues2[0], evalues2[1]);
    
    if(aspect_i>aspect_r){
      Mi=_metric;
      Mr=metric._metric;
    }
    
    // Map Mi to the reference space where Mr==I
    M1 <<
      Mr[0], Mr[1],
      Mr[1], Mr[2];
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver(M1);
    
    Eigen::Matrix<double, 2, 2> F =
      solver.eigenvalues().real().cwise().abs().cwise().sqrt().asDiagonal()*
      solver.eigenvectors().real();
    
    M2 <<
      Mi[0], Mi[1],
      Mi[1], Mi[2];
    Eigen::Matrix<double, 2, 2> M = F.inverse().transpose()*M2*F.inverse();
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver3(M);
    Eigen::Matrix<double, 2, 1> evalues = solver3.eigenvalues().real().cwise().abs();
    Eigen::Matrix<double, 2, 2> evectors = solver3.eigenvectors().real();
    
    if(perserved_small_edges)
      for(size_t i=0;i<2;i++)
        evalues[i] = std::max(1.0, evalues[i]);
    else
      for(size_t i=0;i<2;i++)
        evalues[i] = std::min(1.0, evalues[i]);
    
    Eigen::Matrix<double, 2, 2> Mc = F.transpose()*evectors*evalues.asDiagonal()*evectors.transpose()*F;
    
    _metric[0] = Mc[0];
    _metric[1] = Mc[1];
    _metric[2] = Mc[3];

    return;
  }

  /*! Limits the ratio of the edge lengths.
   * @param max_ratio The maximum allowed ratio between edge lengths in the orthogonal
   */
  void limit_aspect_ratio(double max_ratio){
    Eigen::Matrix<double, 2, 2> M1;
    M1 <<
      _metric[0], _metric[1],
      _metric[1], _metric[2];
    
    Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver1(M1);
    
    Eigen::Matrix<double, 2, 1> evalues = solver1.eigenvalues().real().cwise().abs();
    Eigen::Matrix<double, 2, 2> evectors = solver1.eigenvectors().real();
    
    if(evalues[0]<evalues[1]){
	evalues[0] = std::max(evalues[0], evalues[1]/(max_ratio*max_ratio));
    }else{
	evalues[1] = std::max(evalues[1], evalues[0]/(max_ratio*max_ratio));
    }
    
    Eigen::Matrix<double, 2, 2> Mc = evectors*evalues.asDiagonal()*evectors.transpose();
    
    _metric[0] = Mc[0];
    _metric[1] = Mc[1];
    _metric[2] = Mc[3];

    return;
  }


  /*! Stream operator.
   */
  friend std::ostream &operator<< <>(std::ostream& out, const MetricTensor2D<treal_t>& metric);
  
  void scale(treal_t scale_factor){
    for(size_t i=0;i<3;i++)
      _metric[i] *= scale_factor;
  }

  treal_t average_length() const{
    treal_t D[2];
    treal_t V[4];
    
    eigen_decomp(D, V);
    
    treal_t l0 = sqrt(1.0/D[0]);
    treal_t l1 = sqrt(1.0/D[1]);

    return (l0+l1)*0.5;
  }

  double max_length() const{
    treal_t D[2];
    treal_t V[4];

    eigen_decomp(D, V);

    treal_t min_d = std::min(D[0], D[1]);

    return sqrt(1.0/min_d); // ie, the max
  }

  double min_length() const{
    treal_t D[2];
    treal_t V[4];
    
    eigen_decomp(D, V);

    treal_t max_d = std::max(D[0], D[1]);

    return sqrt(1.0/max_d); // ie, the min
  }

  void eigen_decomp(treal_t *eigenvalues, treal_t *eigenvectors) const{
    Eigen::Matrix<double, 2, 2> M;
    M <<
      _metric[0], _metric[1],
      _metric[1], _metric[2];
    
    if(M.isZero()){
      for (size_t i=0;i<2;i++)
        eigenvalues[i] = 0.0;
      
      for(size_t i=0;i<4;i++)
        eigenvectors[i] = 0.0;
    }else{
      Eigen::EigenSolver< Eigen::Matrix<double, 2, 2> > solver(M);
      
      Eigen::Matrix<double, 2, 1> evalues = solver.eigenvalues().real().cwise().abs();
      Eigen::Matrix<double, 2, 2> evectors = solver.eigenvectors().real();
      
      for (size_t i=0;i<2;i++)
        eigenvalues[i] = evalues[i];
      
      Eigen::Matrix<double, 2, 2> Mp = evectors.transpose();
      for(size_t i=0;i<4;i++)
        eigenvectors[i] = Mp[i];
    }
  }
  
  int eigen_undecomp(const treal_t *D, const treal_t *V){
    // Insure eigenvalues are positive
    treal_t eigenvalues[2];
    for(size_t i=0;i<2;i++)
      eigenvalues[i] = fabs(D[i]);

    // Rubbish code - fixme.
    treal_t M[4];
    for(size_t i=0;i<2;i++){
      for(size_t j=0;j<2;j++){
        int ii = (i*2) + j;
        M[ii] = 0.0;
        for(size_t k=0;k<2;k++)
          M[ii]+=eigenvalues[k]*V[k*2+i]*V[k*2+j];
      }
    }
    
    _metric[0] = M[0];
    _metric[1] = M[1];
    _metric[2] = M[3];
    
    return 0;
  }
  
 private:
  treal_t _metric[3];
};

template<typename treal_t>
std::ostream &operator<<(std::ostream& out, const MetricTensor2D<treal_t>& in){
  out<<in._metric[0]<<" "<<in._metric[1]<<std::endl
     <<in._metric[1]<<" "<<in._metric[2]<<std::endl;
  
  return out;
}

// 3D version
template<typename treal_t> class MetricTensor3D;

template<typename treal_t>
std::ostream& operator<<(std::ostream& out, const MetricTensor3D<treal_t>& in);

/*! \brief 3D symmetric metic tensor class.
 *
 * Use to store and operate on metric tensors for 3D.
 * The upper triangle is stored, ie:
 * m[0] m[1] m[2]
 * .... m[3] m[4]
 * .... .... m[5] 
 */
template<typename treal_t> class MetricTensor3D{
 public:
  /// Default constructor.
  MetricTensor3D(){
  }
  
  /// Default destructor.
  ~MetricTensor3D(){
  }
  
  /*! Constructor.
   * The upper triangle is expected, ie:
   * m[0] m[1] m[2]
   * .... m[3] m[4]
   * .... .... m[5] 
   * @param metric points to the upper triangle of the 3D tensor.
   */
  MetricTensor3D(const treal_t *metric){
    set_metric(metric);
  }

  /*! Copy constructor.
   * @param metric is a reference to a MetricTensor object.
   */
  MetricTensor3D(const MetricTensor3D& metric){
    *this = metric;
  }

  /*! Assignment operator.
   * @param metric is a reference to a MetricTensor3D object.
   */
  const MetricTensor3D& operator=(const MetricTensor3D &metric){
    for(size_t i=0;i<6;i++)
      _metric[i] = metric._metric[i];
    return *this;
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(double *metric){
    for(size_t i=0;i<6;i++)
      metric[i] = _metric[i];
  }

  /// Give const pointer to metric tensor.
  const treal_t* get_metric(){
    return _metric;
  }

  /*! Set the metric tensor field.
   * The upper triangle is expected, ie:
   * m[0] m[1] m[2]
   * .... m[3] m[4]
   * .... .... m[5] 
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const double *metric){
    for(size_t i=0;i<6;i++)
      _metric[i] = metric[i];
    positive_definiteness(_metric);
  }

  // Enforce positive definiteness
  static void positive_definiteness(treal_t *metric){
    Eigen::Matrix<double, 3, 3> M;
    M <<
      metric[0], metric[1], metric[2],
      metric[1], metric[3], metric[4],
      metric[2], metric[4], metric[5];
    
    if(M.isZero())
      return;
    
    Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver(M);
    
    Eigen::Matrix<double, 3, 1> evalues = solver.eigenvalues().real().cwise().abs();
    Eigen::Matrix<double, 3, 3> evectors = solver.eigenvectors().real();
    
    Eigen::Matrix<double, 3, 3> Mp = evectors*evalues.asDiagonal()*evectors.transpose();
    
    metric[0] = Mp[0];
    metric[1] = Mp[1];
    metric[2] = Mp[2];
    metric[3] = Mp[4];
    metric[4] = Mp[5];
    metric[5] = Mp[8];

    return;
  }

  /*! By default this calculates the superposition of two metrics where by default small
   * edge lengths are preserved. If the optional argument perserved_small_edges==false
   * then large edge lengths are perserved instead.
   * @param M is a reference to a MetricTensor object.
   * @param perserved_small_edges when true causes small edge lengths to be preserved (default). Otherwise long edge are perserved.
   */
  void constrain(const treal_t *M_in, bool perserved_small_edges=true){
    for(size_t i=0;i<6;i++)
      if(isnan(M_in[i]))
        return;
    
    MetricTensor3D<treal_t> metric(M_in);
    
    // Make the tensor with the smallest aspect ratio the reference space Mr.
    const treal_t *Mr=_metric, *Mi=metric._metric;
    Eigen::Matrix<double, 3, 3> M1;
    M1[0] = _metric[0]; M1[1] = _metric[1]; M1[2] = _metric[2];
    M1[3] = _metric[1]; M1[4] = _metric[3]; M1[5] = _metric[4];
    M1[6] = _metric[2]; M1[7] = _metric[4]; M1[8] = _metric[5];
    
    Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver1(M1);
    
    Eigen::Matrix<double, 3, 1> evalues1 = solver1.eigenvalues().real().cwise().abs();
    
    treal_t aspect_r = std::min(std::min(evalues1[0], evalues1[1]), evalues1[2])/
      std::max(std::max(evalues1[0], evalues1[1]), evalues1[2]);
    
    Eigen::Matrix<double, 3, 3> M2;
    M2[0] = metric._metric[0]; M2[1] = metric._metric[1]; M2[2] = metric._metric[2];
    M2[3] = metric._metric[1]; M2[4] = metric._metric[3]; M2[5] = metric._metric[4];
    M2[6] = metric._metric[2]; M2[7] = metric._metric[4]; M2[8] = metric._metric[5];
    
    // The input matrix could be zero if there is zero curvature in the local solution.
    if(M2.isZero())
      return;

    Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver2(M2);
    
    Eigen::Matrix<double, 3, 1> evalues2 = solver2.eigenvalues().real().cwise().abs();
    
    treal_t aspect_i = std::min(std::min(evalues2[0], evalues2[1]), evalues2[2])/
      std::max(std::max(evalues2[0], evalues2[1]), evalues2[2]);
    
    if(aspect_i>aspect_r){
      Mi=_metric;
      Mr=metric._metric;
    }
    
    // Map Mi to the reference space where Mr==I
    M1 <<
      Mr[0], Mr[1], Mr[2],
      Mr[1], Mr[3], Mr[4],
      Mr[2], Mr[4], Mr[5];
    
    Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver(M1);
    
    Eigen::Matrix<double, 3, 3> F =
      solver.eigenvalues().real().cwise().abs().cwise().sqrt().asDiagonal()*
      solver.eigenvectors().real();
    
    M2 <<
      Mi[0], Mi[1], Mi[2],
      Mi[1], Mi[3], Mi[4],
      Mi[2], Mi[4], Mi[5];
    Eigen::Matrix<double, 3, 3> M = F.inverse().transpose()*M2*F.inverse();
    
    Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver3(M);
    Eigen::Matrix<double, 3, 1> evalues = solver3.eigenvalues().real().cwise().abs();
    Eigen::Matrix<double, 3, 3> evectors = solver3.eigenvectors().real();
    
    if(perserved_small_edges)
      for(size_t i=0;i<3;i++)
        evalues[i] = std::max((double)1.0, evalues[i]);
    else
      for(size_t i=0;i<3;i++)
        evalues[i] = std::min((double)1.0, evalues[i]);
    
    Eigen::Matrix<double, 3, 3> Mc = F.transpose()*evectors*evalues.asDiagonal()*evectors.transpose()*F;
  
    _metric[0] = Mc[0];
    _metric[1] = Mc[1];
    _metric[2] = Mc[2];
    _metric[3] = Mc[4];
    _metric[4] = Mc[5];
    _metric[5] = Mc[8];
      
    return;
  }

  /*! Stream operator.
   */
  friend std::ostream &operator<< <>(std::ostream& out, const MetricTensor3D<treal_t>& metric);
  
  void scale(treal_t scale_factor){
    for(size_t i=0;i<6;i++)
      _metric[i] *= scale_factor;
  }
  
  treal_t average_length() const{
    treal_t D[3], V[9];
    eigen_decomp(D, V);
    
    treal_t l0 = sqrt(1.0/D[0]);
    treal_t l1 = sqrt(1.0/D[1]);
    treal_t l2 = sqrt(1.0/D[2]);
    
    return (l0+l1+l2)/3.0;
  }

  double max_length() const{
    treal_t D[3], V[9];
    eigen_decomp(D, V);

    treal_t min_d = std::min(std::min(D[0], D[1]), D[2]);
    
    return sqrt(1.0/min_d); // ie, the max
  }

  double min_length() const{
    treal_t D[3], V[9];
    eigen_decomp(D, V);

    treal_t max_d = std::max(std::max(D[0], D[1]), D[2]);

    return sqrt(1.0/max_d); // ie, the min
  }

  void eigen_decomp(treal_t *eigenvalues, treal_t *eigenvectors) const{
    Eigen::Matrix<double, 3, 3> M;
    M <<
      _metric[0], _metric[1], _metric[2],
      _metric[1], _metric[3], _metric[4],
      _metric[2], _metric[4], _metric[5];
    
    if(M.isZero()){
      for (size_t i=0;i<3;i++)
        eigenvalues[i] = 0.0;
      
      for(size_t i=0;i<9;i++)
        eigenvectors[i] = 0.0;
    }else{
      Eigen::EigenSolver< Eigen::Matrix<double, 3, 3> > solver(M);
      
      Eigen::Matrix<double, 3, 1> evalues = solver.eigenvalues().real().cwise().abs();
      Eigen::Matrix<double, 3, 3> evectors = solver.eigenvectors().real();
      
      for (size_t i=0;i<3;i++)
        eigenvalues[i] = evalues[i];
      
      Eigen::Matrix<double, 3, 3> Mp = evectors.transpose();
      for(size_t i=0;i<9;i++)
        eigenvectors[i] = Mp[i];
    }
  }

  int eigen_undecomp(const double *D, const double *V){
    // Insure eigenvalues are positive
    treal_t eigenvalues[3];
    for(size_t i=0;i<3;i++)
      eigenvalues[i] = fabs(D[i]);

    // Rubbish code - fixme.
    treal_t M[9];
    for(size_t i=0; i<3; i++){
      for(size_t j=0;j<3;j++){
        int ii = i*3 + j;
        M[ii] = 0.0;
        for(size_t k=0;k<3;k++)
          M[ii]+=eigenvalues[k]*V[k*3+i]*V[k*3+j];
      }
    }
    
    _metric[0] = M[0];
    _metric[1] = M[1];
    _metric[2] = M[2];
    _metric[3] = M[4];
    _metric[4] = M[5];
    _metric[5] = M[8];

    return 0;
  }

 private:
  treal_t _metric[6];
};

template<typename treal_t>
std::ostream &operator<<(std::ostream& out, const MetricTensor3D<treal_t>& in){
  out<<in._metric[0]<<" "<<in._metric[1]<<" "<<in._metric[2]<<std::endl
     <<in._metric[1]<<" "<<in._metric[3]<<" "<<in._metric[4]<<std::endl
     <<in._metric[2]<<" "<<in._metric[4]<<" "<<in._metric[5]<<std::endl;

  return out;
}

#endif
