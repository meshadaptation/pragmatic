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
#ifndef METRICTENSOR_H
#define METRICTENSOR_H

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

extern"C"
{
//    void test_calling_fortran_function_(int int_to_c[]);

    void dspev_(char*, char*, size_t*, double*, double*, double*, size_t*, double*, int*);
}

template<typename real_t>
class MetricTensor;

template<typename real_t>
std::ostream& operator<<(std::ostream& out, const MetricTensor<real_t>& in);

/*! \brief Symmetric metic tensor class.
 *
 * Use to store and operate on metric tensors.
 */
template<typename real_t>
class MetricTensor{
 public:
  /// Default constructor.
  MetricTensor(){
    _dimension = 0;
    _metric = NULL;
  }

  /// Default destructor.
  ~MetricTensor(){
    if(_metric != NULL){
      delete [] _metric;
    }
  }

  /*! Constructor.
   * @param dimension of tensor.
   * @param metric points to the dimension*dimension tensor.
   */
  MetricTensor(int dimension, const real_t *metric){
    set(dimension, metric);
  }

  /*! Copy constructor.
   * @param metric is a reference to a MetricTensor object.
   */
  MetricTensor(const MetricTensor& metric){
    *this = metric;
  }

  /*! Assignment operator.
   * @param metric is a reference to a MetricTensor object.
   */
  const MetricTensor& operator=(const MetricTensor &metric){
    _dimension = metric._dimension;
    if(_metric==NULL)
      _metric = new real_t[_dimension*_dimension];
    for(size_t i=0;i<_dimension*_dimension;i++)
      _metric[i] = metric._metric[i];
    return *this;
  }

  /*! Copy back the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field can be copied.
   */
  void get_metric(real_t *metric){
    for(size_t i=0;i<_dimension*_dimension;i++)
      metric[i] = _metric[i];
  }

  /// Give const pointer to metric tensor.
  const real_t* get_metric(){
    return _metric;
  }

  /*! Set the metric tensor field.
   * @param metric is a pointer to the buffer where the metric field is to be copied from.
   */
  void set_metric(const real_t *metric){
    for(size_t i=0;i<_dimension*_dimension;i++)
      _metric[i] = metric[i];
  }

  // Enforce positive definiteness
  static void positive_definiteness(int dimension, real_t *metric){
    if(dimension==2){
      Eigen::Matrix<real_t, 2, 2> M;
      M <<
        metric[0], metric[1],
        metric[2], metric[3];

      if(M2.isZero())
        return;

      Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver(M);

      Eigen::Matrix<real_t, 2, 1> evalues = solver.eigenvalues().real().cwise().abs();
      Eigen::Matrix<real_t, 2, 2> evectors = solver.eigenvectors().real();

      Eigen::Matrix<real_t, 2, 2> Mp = evectors.transpose()*evalues.asDiagonal()*evectors;
      for(size_t i=0;i<4;i++)
        metric[i] = Mp[i];
    }else{
      Eigen::Matrix<real_t, 3, 3> M;
      M <<
        metric[0], metric[1], metric[2],
        metric[3], metric[4], metric[5],
        metric[6], metric[7], metric[8];

      if(M2.isZero())
        return;

      Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver(M);

      Eigen::Matrix<real_t, 3, 1> evalues = solver.eigenvalues().real().cwise().abs();
      Eigen::Matrix<real_t, 3, 3> evectors = solver.eigenvectors().real();

      Eigen::Matrix<real_t, 3, 3> Mp = evectors.transpose()*evalues.asDiagonal()*evectors;
      for(size_t i=0;i<9;i++)
        metric[i] = Mp[i];
    }
    return;
  }

  /*! By default this calculates the superposition of two metrics where by default small
   * edge lengths are preserved. If the optional argument perserved_small_edges==false
   * then large edge lengths are perserved instead.
   * @param M is a reference to a MetricTensor object.
   * @param perserved_small_edges when true causes small edge lengths to be preserved (default). Otherwise long edge are perserved.
   */
  void constrain(const real_t *M, bool perserved_small_edges=true){
    MetricTensor<real_t> metric(_dimension, M);

    // Make the tensor with the smallest aspect ratio the reference space Mr.
    const real_t *Mr=_metric, *Mi=metric._metric;
    if(_dimension==2){
      Eigen::Matrix<real_t, 2, 2> M1;
      M1 <<
        _metric[0], _metric[1],
        _metric[2], _metric[3];

      Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver1(M1);

      Eigen::Matrix<real_t, 2, 1> evalues1 = solver1.eigenvalues().real().cwise().abs();

      real_t aspect_r = std::min(evalues1[0], evalues1[1])/
        std::max(evalues1[0], evalues1[1]);

      Eigen::Matrix<real_t, 2, 2> M2;
      M2 <<
        metric._metric[0], metric._metric[1],
        metric._metric[2], metric._metric[3];
      // The input matrix could be zero if there is zero curvature in the local solution.
      if(M2.isZero())
        return;

      Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver2(M2);

      Eigen::Matrix<real_t, 2, 1> evalues2 = solver2.eigenvalues().real().cwise().abs();

      real_t aspect_i = std::min(evalues2[0], evalues2[1])/
        std::max(evalues2[0], evalues2[1]);

      if(aspect_i>aspect_r){
        Mi=_metric;
        Mr=metric._metric;
      }
    }else{
      Eigen::Matrix<real_t, 3, 3> M1;
      for(size_t i=0;i<9;i++)
                M1[i] = _metric[i];

      Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver1(M1);

      Eigen::Matrix<real_t, 3, 1> evalues1 = solver1.eigenvalues().real().cwise().abs();

      real_t aspect_r = std::min(std::min(evalues1[0], evalues1[1]), evalues1[2])/
        std::max(std::max(evalues1[0], evalues1[1]), evalues1[2]);

      Eigen::Matrix<real_t, 3, 3> M2;
      for(size_t i=0;i<9;i++)
        M2[i] = metric._metric[i];

      // The input matrix could be zero if there is zero curvature in the local solution.
      if(M2.isZero())
        return;

      Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver2(M2);

      Eigen::Matrix<real_t, 3, 1> evalues2 = solver2.eigenvalues().real().cwise().abs();

      real_t aspect_i = std::min(std::min(evalues2[0], evalues2[1]), evalues2[2])/
        std::max(std::max(evalues2[0], evalues2[1]), evalues2[2]);

      if(aspect_i>aspect_r){
        Mi=_metric;
        Mr=metric._metric;
      }
    }

    // Map Mi to the reference space where Mr==I
    if(_dimension==2){
      Eigen::Matrix<real_t, 2, 2> M1;
      M1 <<
        Mr[0], Mr[1],
        Mr[2], Mr[3];

      Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver(M1);

      Eigen::Matrix<real_t, 2, 2> F =
        solver.eigenvalues().real().cwise().abs().cwise().sqrt().asDiagonal()*
        solver.eigenvectors().real();

      Eigen::Matrix<real_t, 2, 2> M2;
      M2 <<
        Mi[0], Mi[1],
        Mi[2], Mi[3];
      Eigen::Matrix<real_t, 2, 2> M = F.inverse().transpose()*M2*F.inverse();

      Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver2(M);
      Eigen::Matrix<real_t, 2, 1> evalues = solver2.eigenvalues().real().cwise().abs();
      Eigen::Matrix<real_t, 2, 2> evectors = solver2.eigenvectors().real();

      if(perserved_small_edges)
        for(size_t i=0;i<2;i++)
          evalues[i] = std::max((real_t)1.0, evalues[i]);
      else
        for(size_t i=0;i<2;i++)
          evalues[i] = std::min((real_t)1.0, evalues[i]);

      Eigen::Matrix<real_t, 2, 2> Mc = F.transpose()*evectors.transpose()*evalues.asDiagonal()*evectors*F;

      for(size_t i=0;i<_dimension*_dimension;i++)
        _metric[i] = Mc[i];
    }else{
      Eigen::Matrix<real_t, 3, 3> M1;
      M1 <<
        Mr[0], Mr[1], Mr[2],
        Mr[3], Mr[4], Mr[5],
        Mr[6], Mr[7], Mr[8];

      Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver(M1);

      Eigen::Matrix<real_t, 3, 3> F =
        solver.eigenvalues().real().cwise().abs().cwise().sqrt().asDiagonal()*
        solver.eigenvectors().real();

      Eigen::Matrix<real_t, 3, 3> M2;
      M2 <<
        Mi[0], Mi[1], Mi[2],
        Mi[3], Mi[4], Mi[5],
        Mi[6], Mi[7], Mi[8];
      Eigen::Matrix<real_t, 3, 3> M = F.inverse().transpose()*M2*F.inverse();

      Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver2(M);
      Eigen::Matrix<real_t, 3, 1> evalues = solver2.eigenvalues().real().cwise().abs();
      Eigen::Matrix<real_t, 3, 3> evectors = solver2.eigenvectors().real();

      if(perserved_small_edges)
        for(size_t i=0;i<3;i++)
          evalues[i] = std::max((real_t)1.0, evalues[i]);
      else
        for(size_t i=0;i<3;i++)
          evalues[i] = std::min((real_t)1.0, evalues[i]);

      Eigen::Matrix<real_t, 3, 3> Mc = F.transpose()*evectors.transpose()*evalues.asDiagonal()*evectors*F;

      for(size_t i=0;i<_dimension*_dimension;i++)
        _metric[i] = Mc[i];
    }

    return;
  }

  /*! Stream operator.
   */
  friend std::ostream &operator<< <>(std::ostream& out, const MetricTensor<real_t>& metric);

  /*! Set values from input.
   * @param dimension of tensor.
   * @param metric points to the dimension*dimension tensor.
   */
  void set(int dimension, const real_t *metric){
    _dimension = dimension;
    _metric = new real_t[_dimension*_dimension];
    for(size_t i=0;i<_dimension*_dimension;i++)
      _metric[i] = metric[i];
  }

  void scale(real_t scale_factor){
    for(size_t i=0;i<_dimension*_dimension;i++)
      _metric[i] *= scale_factor;
  }

  double average_length() const
  {
    double D[_dimension], V[_dimension*_dimension];
    eigen_decomp(D, V);

    double sum = D[0];
    for(size_t i=1; i<_dimension; i++)
      sum+=D[i];
    double average = sum/_dimension;

    return sqrt(1.0/average);
  }

  void eigen_decomp(double *eigenvalues, double *eigenvectors) const
  {
    if(_dimension==2)
    {
        Eigen::Matrix<real_t, 2, 2> M;
        M <<
          _metric[0], _metric[1],
          _metric[2], _metric[3];

        Eigen::EigenSolver< Eigen::Matrix<real_t, 2, 2> > solver(M);

        Eigen::Matrix<real_t, 2, 1> evalues = solver.eigenvalues().real().cwise().abs();
        Eigen::Matrix<real_t, 2, 2> evectors = solver.eigenvectors().real();

        for (size_t i=0;i<2;i++)
            eigenvalues[i] = evalues[i];

        Eigen::Matrix<real_t, 2, 2> Mp = evectors.transpose();
        for(size_t i=0;i<4;i++)
            eigenvectors[i] = Mp[i];
    }
    else if (_dimension==3)
    {
        Eigen::Matrix<real_t, 3, 3> M;
        M <<
          _metric[0], _metric[1], _metric[2],
          _metric[3], _metric[4], _metric[5],
          _metric[6], _metric[7], _metric[8];

        Eigen::EigenSolver< Eigen::Matrix<real_t, 3, 3> > solver(M);

        Eigen::Matrix<real_t, 3, 1> evalues = solver.eigenvalues().real().cwise().abs();
        Eigen::Matrix<real_t, 3, 3> evectors = solver.eigenvectors().real();

        for (size_t i=0;i<3;i++)
            eigenvalues[i] = evalues[i];

        Eigen::Matrix<real_t, 3, 3> Mp = evectors.transpose();
        for(size_t i=0;i<9;i++)
          eigenvectors[i] = Mp[i];
    }
    else
    {
        std::cerr<<"ERROR: unsupported dimension: " << _dimension << " (must be 2 or 3)" << endl;
    }

  }

  int eigen_undecomp(const double *D, const double *V){
    // Insure eigenvalues are positive
    double eigenvalues[_dimension];
    for(size_t i=0;i<_dimension;i++)
      eigenvalues[i] = fabs(D[i]);

    for(size_t i=0; i<_dimension; i++)
    {
      for(size_t j=0; j<_dimension; j++)
      {
        int ii = (i*_dimension) + j;
        _metric[ii] = 0.0;
        for(size_t k=0;k<_dimension;k++)
          _metric[ii]+=eigenvalues[k]*V[k*_dimension+i]*V[k*_dimension+j];
      }
    }

    return 0;
  }

 private:
  real_t *_metric;
  size_t _dimension;
};

template<typename real_t>
std::ostream &operator<<(std::ostream& out, const MetricTensor<real_t>& in){
  const real_t *g = in._metric;
  for(size_t i=0; i<in._dimension; i++){
    for(size_t j=0; j<in._dimension; j++){
      out<<*g<<" "; g++;
    }
    out<<"\n";
  }
  return out;
}

#endif
