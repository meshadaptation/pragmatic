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

#ifndef ELEMENTPROPERTY_H
#define ELEMENTPROPERTY_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <cmath>

#include <cfloat>

/*! \brief Calculates a number of element properties.
 *
 * The constructor for this class requires a reference element so
 * that the orientation convention can be established. After the
 * orientation has been established a negative area or volume
 * indicated an inverted element. Properties calculated are:
 * \li Euclidean area (if 2D element).
 * \li Euclidean Volume (if 3D element).
 * \li Non-Euclidean edge length.
 * \li 2D/3D Lipnikov quality functional.
 * \li 3D sliver functional.
 */
template<typename real_t>
class ElementProperty{
 public:
  /*! Constructor for 2D triangular elements.
   * @param x0 pointer to 2D position for first point in triangle.
   * @param x1 pointer to 2D position for second point in triangle.
   * @param x2 pointer to 2D position for third point in triangle.
   */
 ElementProperty(const real_t *x0, const real_t *x1, const real_t *x2): dimension(2), inv2(0.5), inv3(1.0/3.0), inv4(0.25), inv6(1.0/6.0), lipnikov_const2d(20.784609690826528), lipnikov_const3d(1832.8207768355312){
   double A = area(x0, x1, x2);
   if(A<0)
     orientation = -1;
   else
     orientation = 1;
  }

  /*! Constructor for 3D tetrahedral elements.
   * @param x0 pointer to 3D position for first point in triangle.
   * @param x1 pointer to 3D position for second point in triangle.
   * @param x2 pointer to 3D position for third point in triangle.
   * @param x3 pointer to 3D position for forth point in triangle.
   */
 ElementProperty(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3) : dimension(3), inv2(0.5), inv3(1.0/3.0), inv4(0.25), inv6(1.0/6.0), lipnikov_const2d(20.784609690826528), lipnikov_const3d(1832.8207768355312){
    double V = volume(x0, x1, x2, x3);
    if(V<0)
      orientation = -1;
  }

  /*! Calculate area of 2D triangle.
   * @param x0 pointer to 2D position for first point in triangle.
   * @param x1 pointer to 2D position for second point in triangle.
   * @param x2 pointer to 2D position for third point in triangle.
   */
  real_t area(const real_t *x0, const real_t *x1, const real_t *x2) const{
    real_t x01 = (x0[0] - x1[0]);
    real_t y01 = (x0[1] - x1[1]);
    
    real_t x02 = (x0[0] - x2[0]);
    real_t y02 = (x0[1] - x2[1]);
    
    return orientation*inv2*(y02*x01 - y01*x02);
  }

  /*! Calculate volume of tetrahedron.
   * @param x0 pointer to 3D position for first point in triangle.
   * @param x1 pointer to 3D position for second point in triangle.
   * @param x2 pointer to 3D position for third point in triangle.
   * @param x3 pointer to 3D position for forth point in triangle.
   */
  real_t volume(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3) const{

    real_t x01 = (x0[0] - x1[0]);
    real_t x02 = (x0[0] - x2[0]);
    real_t x03 = (x0[0] - x3[0]);

    real_t y01 = (x0[1] - x1[1]);
    real_t y02 = (x0[1] - x2[1]);
    real_t y03 = (x0[1] - x3[1]);

    real_t z01 = (x0[2] - x1[2]);
    real_t z02 = (x0[2] - x2[2]);
    real_t z03 = (x0[2] - x3[2]);

    return orientation*inv6*(-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - x01*(z03*y02 - z02*y03));
  }

  /*! Length of an edge as measured in metric space.
   *
   * @param x0 coordinate at start of line segment.
   * @param x1 coordinate at finish of line segment.
   * @param m metric tensor for first point.
   */
  real_t length(const real_t x0[], const real_t x1[], const double m[]) const{
    if(dimension==2){
      return length2d(x0, x1, m);
    }else{
      return length3d(x0, x1, m);
    }
  }

  /*! Length of an edge as measured in metric space.
   *
   * @param x0 coordinate at start of line segment.
   * @param x1 coordinate at finish of line segment.
   * @param m metric tensor for first point.
   */
  static double length2d(const real_t x0[], const real_t x1[], const double m[]){
    double x=x0[0] - x1[0];
    double y=x0[1] - x1[1];
    
    assert((m[1]*x + m[2]*y)*y + (m[0]*x + m[1]*y)*x >= 0.0);

    return sqrt(((m[1]*x + m[2]*y)*y + (m[0]*x + m[1]*y)*x));
  }

  /*! Length of an edge as measured in metric space.
   *
   * @param x0 coordinate at start of line segment.
   * @param x1 coordinate at finish of line segment.
   * @param m metric tensor for first point.
   */
  static double length3d(const real_t x0[], const real_t x1[], const double m[]){
    double x=x0[0] - x1[0];
    double y=x0[1] - x1[1];
    double z=x0[2] - x1[2];

    assert(z*(z*m[5] + y*m[4] + x*m[2]) +
                y*(z*m[4] + y*m[3] + x*m[1]) +
                x*(z*m[2] + y*m[1] + x*m[0]) >= 0.0);

    return sqrt(z*(z*m[5] + y*m[4] + x*m[2]) +
                y*(z*m[4] + y*m[3] + x*m[1]) +
                x*(z*m[2] + y*m[1] + x*m[0]));
  }

  /*! Evaluates the 2D Lipnikov functional. The description for the
   * functional is taken from: Yu. V. Vasileskii and K. N. Lipnikov,
   * An Adaptive Algorithm for Quasioptimal Mesh Generation,
   * Computational Mathematics and Mathematical Physics, Vol. 39,
   * No. 9, 1999, pp. 1468 - 1486.
   *
   * @param x0 pointer to 2D position for first point in triangle.
   * @param x1 pointer to 2D position for second point in triangle.
   * @param x2 pointer to 2D position for third point in triangle.
   * @param m0 2x2 metric tensor for first point.
   * @param m1 2x2 metric tensor for second point.
   * @param m2 2x2 metric tensor for third point.
   */
  double lipnikov(const real_t *x0, const real_t *x1, const real_t *x2,
		  const double *m0, const double *m1, const double *m2){
    // Metric tensor averaged over the element
    double m00 = (m0[0] + m1[0] + m2[0])*inv3;
    double m01 = (m0[1] + m1[1] + m2[1])*inv3;
    double m11 = (m0[2] + m1[2] + m2[2])*inv3;
    
    return lipnikov(x0, x1, x2, m00, m01, m11);
  }

  /*! Evaluates the 2D Lipnikov functional. The description for the
   * functional is taken from: Yu. V. Vasileskii and K. N. Lipnikov,
   * An Adaptive Algorithm for Quasioptimal Mesh Generation,
   * Computational Mathematics and Mathematical Physics, Vol. 39,
   * No. 9, 1999, pp. 1468 - 1486.
   *
   * @param x0 pointer to 2D position for first point in triangle.
   * @param x1 pointer to 2D position for second point in triangle.
   * @param x2 pointer to 2D position for third point in triangle.
   * @param m00 metric index (0,0)
   * @param m01 metric index (0,1)
   * @param m11 metric index (1,1)
   */
  double lipnikov(const real_t *x0, const real_t *x1, const real_t *x2,
                 double m00, double m01, double m11){
    // l is the length of the perimeter, measured in metric space
    double x01 = x0[0] - x1[0];
    double y01 = x0[1] - x1[1];
    double x02 = x0[0] - x2[0];
    double y02 = x0[1] - x2[1];
    double x21 = x2[0] - x1[0];
    double y21 = x2[1] - x1[1];
    
    double l =
      sqrt(y01*(y01*m11 + x01*m01) + 
           x01*(y01*m01 + x01*m00))+
      sqrt(y02*(y02*m11 + x02*m01) + 
           x02*(y02*m01 + x02*m00))+
      sqrt(y21*(y21*m11 + x21*m01) + 
           x21*(y21*m01 + x21*m00));

    double invl = 1.0/l;
    
    // Area in physical space
    double a=orientation*inv2*(y02*x01 - y01*x02);
    
    // Area in metric space
    double a_m = a*sqrt(m00*m11 - m01*m01);

    // Function
    double f = std::min(l*inv3, 3.0*invl);
    double tf = f * (2.0 - f);
    double F = tf*tf*tf;
    double quality = lipnikov_const2d*a_m*F*invl*invl;

    return quality;
  }

  // Gradient of lipnikov functional n0 using a central difference approximation.
  void lipnikov_grad(int moving,
		     const double *x0, const double *x1, const double *x2,
		     const double *m0,
		     double *grad){
    const double sqrt_eps = sqrt(DBL_EPSILON);
    
    // df/dx, df/dy
    for(size_t i=0;i<2;i++){
      double h = std::max(fabs(sqrt_eps*x0[i]), sqrt_eps);
      
      volatile double xnh = x0[i]-h;
      volatile double xph = x0[i]+h;
      
      double Xn[] = {x0[0], x0[1]};
      Xn[i] = xnh;
      double Fxnh = lipnikov(Xn, x1, x2, m0[0], m0[1], m0[2]);
      
      double Xp[] = {x0[0], x0[1]};
      Xp[i] = xph;
      double Fxph = lipnikov(Xp, x1, x2, m0[0], m0[1], m0[2]);
      
      double two_dx = xph - xnh;
      grad[i] = (Fxph - Fxnh)/two_dx;
    }

    return;
  }

  /*! Evaluates the 3D Lipnikov functional. The description for the
   * functional is taken from: A. Agouzal, K Lipnikov,
   * Yu. Vassilevski, Adaptive generation of quasi-optimal tetrahedral
   * meshes, East-West J. Numer. Math., Vol. 7, No. 4, pp. 223-244
   * (1999).
   *
   * @param x0 pointer to 3D position for first point in tetrahedral.
   * @param x1 pointer to 3D position for second point in tetrahedral.
   * @param x2 pointer to 3D position for third point in tetrahedral.
   * @param x3 pointer to 3D position for third point in tetrahedral.
   * @param m0 3x3 metric tensor for first point.
   * @param m1 3x3 metric tensor for second point.
   * @param m2 3x3 metric tensor for third point.
   * @param m3 3x3 metric tensor for forth point.
   */
  double lipnikov(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3,
                  const double *m0, const double *m1, const double *m2, const double *m3){
    // Metric tensor
    double m00 = (m0[0] + m1[0] + m2[0] + m3[0])*inv4;
    double m01 = (m0[1] + m1[1] + m2[1] + m3[1])*inv4;
    double m02 = (m0[2] + m1[2] + m2[2] + m3[2])*inv4;
    double m11 = (m0[3] + m1[3] + m2[3] + m3[3])*inv4;
    double m12 = (m0[4] + m1[4] + m2[4] + m3[4])*inv4;
    double m22 = (m0[5] + m1[5] + m2[5] + m3[5])*inv4;

    // l is the length of the edges of the tet, in metric space
    double z01 = (x0[2] - x1[2]);
    double y01 = (x0[1] - x1[1]);
    double x01 = (x0[0] - x1[0]);

    double z12 = (x1[2] - x2[2]);
    double y12 = (x1[1] - x2[1]);
    double x12 = (x1[0] - x2[0]);

    double z02 = (x0[2] - x2[2]);
    double y02 = (x0[1] - x2[1]);
    double x02 = (x0[0] - x2[0]);

    double z03 = (x0[2] - x3[2]);
    double y03 = (x0[1] - x3[1]);
    double x03 = (x0[0] - x3[0]);

    double z13 = (x1[2] - x3[2]);
    double y13 = (x1[1] - x3[1]);
    double x13 = (x1[0] - x3[0]);

    double z23 = (x2[2] - x3[2]);
    double y23 = (x2[1] - x3[1]);
    double x23 = (x2[0] - x3[0]);

    double dl0 = (z01*(z01*m22 + y01*m12 + x01*m02) + y01*(z01*m12 + y01*m11 + x01*m01) + x01*(z01*m02 + y01*m01 + x01*m00));
    double dl1 = (z12*(z12*m22 + y12*m12 + x12*m02) + y12*(z12*m12 + y12*m11 + x12*m01) + x12*(z12*m02 + y12*m01 + x12*m00));
    double dl2 = (z02*(z02*m22 + y02*m12 + x02*m02) + y02*(z02*m12 + y02*m11 + x02*m01) + x02*(z02*m02 + y02*m01 + x02*m00));
    double dl3 = (z03*(z03*m22 + y03*m12 + x03*m02) + y03*(z03*m12 + y03*m11 + x03*m01) + x03*(z03*m02 + y03*m01 + x03*m00));
    double dl4 = (z13*(z13*m22 + y13*m12 + x13*m02) + y13*(z13*m12 + y13*m11 + x13*m01) + x13*(z13*m02 + y13*m01 + x13*m00));
    double dl5 = (z23*(z23*m22 + y23*m12 + x23*m02) + y23*(z23*m12 + y23*m11 + x23*m01) + x23*(z23*m02 + y23*m01 + x23*m00));
    
    double l = sqrt(dl0)+sqrt(dl1)+sqrt(dl2)+sqrt(dl3)+sqrt(dl4)+sqrt(dl5);
    double invl = 1.0/l;

    // Volume
    double v=orientation*inv6*(-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - (x0[0] - x1[0])*(z03*y02 - z02*y03));

    // Volume in metric space
    double v_m = v*sqrt(((m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02));

    // Function
    double f = std::min(l*inv6, 6*invl);
    double tf = f * (2.0 - f);
    double F = tf*tf*tf;
    double quality = lipnikov_const3d * v_m * F *invl*invl*invl;

    return quality;
  }

  /*! Evaluates the 3D Lipnikov functional. The description for the
   * functional is taken from: A. Agouzal, K Lipnikov,
   * Yu. Vassilevski, Adaptive generation of quasi-optimal tetrahedral
   * meshes, East-West J. Numer. Math., Vol. 7, No. 4, pp. 223-244
   * (1999).
   *
   * @param x0 pointer to 3D position for first point in tetrahedral.
   * @param x1 pointer to 3D position for second point in tetrahedral.
   * @param x2 pointer to 3D position for third point in tetrahedral.
   * @param x3 pointer to 3D position for third point in tetrahedral.
   * @param m0 3x3 metric tensor for first point.
   */
  double lipnikov(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3,
                  const double *m0){
    // Metric tensor
    double m00 = m0[0];
    double m01 = m0[1];
    double m02 = m0[2];
    double m11 = m0[3];
    double m12 = m0[4];
    double m22 = m0[5];

    // l is the length of the edges of the tet, in metric space
    double z01 = (x0[2] - x1[2]);
    double y01 = (x0[1] - x1[1]);
    double x01 = (x0[0] - x1[0]);

    double z12 = (x1[2] - x2[2]);
    double y12 = (x1[1] - x2[1]);
    double x12 = (x1[0] - x2[0]);

    double z02 = (x0[2] - x2[2]);
    double y02 = (x0[1] - x2[1]);
    double x02 = (x0[0] - x2[0]);

    double z03 = (x0[2] - x3[2]);
    double y03 = (x0[1] - x3[1]);
    double x03 = (x0[0] - x3[0]);

    double z13 = (x1[2] - x3[2]);
    double y13 = (x1[1] - x3[1]);
    double x13 = (x1[0] - x3[0]);

    double z23 = (x2[2] - x3[2]);
    double y23 = (x2[1] - x3[1]);
    double x23 = (x2[0] - x3[0]);

    double dl0 = (z01*(z01*m22 + y01*m12 + x01*m02) + y01*(z01*m12 + y01*m11 + x01*m01) + x01*(z01*m02 + y01*m01 + x01*m00));
    double dl1 = (z12*(z12*m22 + y12*m12 + x12*m02) + y12*(z12*m12 + y12*m11 + x12*m01) + x12*(z12*m02 + y12*m01 + x12*m00));
    double dl2 = (z02*(z02*m22 + y02*m12 + x02*m02) + y02*(z02*m12 + y02*m11 + x02*m01) + x02*(z02*m02 + y02*m01 + x02*m00));
    double dl3 = (z03*(z03*m22 + y03*m12 + x03*m02) + y03*(z03*m12 + y03*m11 + x03*m01) + x03*(z03*m02 + y03*m01 + x03*m00));
    double dl4 = (z13*(z13*m22 + y13*m12 + x13*m02) + y13*(z13*m12 + y13*m11 + x13*m01) + x13*(z13*m02 + y13*m01 + x13*m00));
    double dl5 = (z23*(z23*m22 + y23*m12 + x23*m02) + y23*(z23*m12 + y23*m11 + x23*m01) + x23*(z23*m02 + y23*m01 + x23*m00));
    
    double l = sqrt(dl0)+sqrt(dl1)+sqrt(dl2)+sqrt(dl3)+sqrt(dl4)+sqrt(dl5);
    double invl = 1.0/l;

    // Volume
    double v=orientation*inv6*(-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - (x0[0] - x1[0])*(z03*y02 - z02*y03));

    // Volume in metric space
    double v_m = v*sqrt(((m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02));

    // Function
    double f = std::min(l*inv6, 6*invl);
    double tf = f * (2.0 - f);
    double F = tf*tf*tf;
    double quality = lipnikov_const3d * v_m * F *invl*invl*invl;

    return quality;
  }


  // Gradient of lipnikov functional n0 using a central difference approximation.
  void lipnikov_grad(int moving,
		     const double *x0, const double *x1, const double *x2, const double *x3,
		     const double *m0,
		     double *grad){
    const double sqrt_eps = sqrt(DBL_EPSILON);
    
    // df/dx, df/dy, df/dz
    for(size_t i=0;i<3;i++){
      double h = std::max(fabs(sqrt_eps*x0[i]), sqrt_eps);
      
      volatile double xnh = x0[i]-h;
      volatile double xph = x0[i]+h;
      
      double Xn[] = {x0[0], x0[1], x0[2]};
      Xn[i] = xnh;
      double Fxnh = lipnikov(Xn, x1, x2, x3, m0);
      
      double Xp[] = {x0[0], x0[1], x0[2]};
      Xp[i] = xph;
      double Fxph = lipnikov(Xp, x1, x2, x3, m0);
      
      double two_dx = xph - xnh;
      grad[i] = (Fxph - Fxnh)/two_dx;
    }

    return;
  }

  /*! Evaluates the sliver functional. Taken from Computer Methods in
   * Applied Mechanics and Engineering Volume 194, Issues 48-49, 15
   * November 2005, Pages 4915-4950
   *
   * @param x0 pointer to 3D position for first point in tetrahedral.
   * @param x1 pointer to 3D position for second point in tetrahedral.
   * @param x2 pointer to 3D position for third point in tetrahedral.
   * @param x3 pointer to 3D position for third point in tetrahedral.
   * @param m0 3x3 metric tensor for first point.
   * @param m1 3x3 metric tensor for second point.
   * @param m2 3x3 metric tensor for third point.
   * @param m3 3x3 metric tensor for forth point.
   */
  real_t sliver(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3,
                const double *m0, const double *m1, const double *m2, const double *m3){
    // Metric tensor
    double m00 = (m0[0] + m1[0] + m2[0] + m3[0])*inv4;
    double m01 = (m0[1] + m1[1] + m2[1] + m3[1])*inv4;
    double m02 = (m0[2] + m1[2] + m2[2] + m3[2])*inv4;
    double m11 = (m0[4] + m1[4] + m2[4] + m3[4])*inv4;
    double m12 = (m0[5] + m1[5] + m2[5] + m3[5])*inv4;
    double m22 = (m0[8] + m1[8] + m2[8] + m3[8])*inv4;

    double z01 = (x0[2] - x1[2]);
    double y01 = (x0[1] - x1[1]);
    double x01 = (x0[0] - x1[0]);

    double z12 = (x1[2] - x2[2]);
    double y12 = (x1[1] - x2[1]);
    double x12 = (x1[0] - x2[0]);

    double z02 = (x0[2] - x2[2]);
    double y02 = (x0[1] - x2[1]);
    double x02 = (x0[0] - x2[0]);

    double z03 = (x0[2] - x3[2]);
    double y03 = (x0[1] - x3[1]);
    double x03 = (x0[0] - x3[0]);

    double z13 = (x1[2] - x3[2]);
    double y13 = (x1[1] - x3[1]);
    double x13 = (x1[0] - x3[0]);

    double z23 = (x2[2] - x3[2]);
    double y23 = (x2[1] - x3[1]);
    double x23 = (x2[0] - x3[0]);

    // l is the length of the edges of the tet, in metric space
    double dl0 = (z01*(z01*m22 + y01*m12 + x01*m02) + y01*(z01*m12 + y01*m11 + x01*m01) + x01*(z01*m02 + y01*m01 + x01*m00));
    double dl1 = (z12*(z12*m22 + y12*m12 + x12*m02) + y12*(z12*m12 + y12*m11 + x12*m01) + x12*(z12*m02 + y12*m01 + x12*m00));
    double dl2 = (z02*(z02*m22 + y02*m12 + x02*m02) + y02*(z02*m12 + y02*m11 + x02*m01) + x02*(z02*m02 + y02*m01 + x02*m00));
    double dl3 = (z03*(z03*m22 + y03*m12 + x03*m02) + y03*(z03*m12 + y03*m11 + x03*m01) + x03*(z03*m02 + y03*m01 + x03*m00));
    double dl4 = (z13*(z13*m22 + y13*m12 + x13*m02) + y13*(z13*m12 + y13*m11 + x13*m01) + x13*(z13*m02 + y13*m01 + x13*m00));
    double dl5 = (z23*(z23*m22 + y23*m12 + x23*m02) + y23*(z23*m12 + y23*m11 + x23*m01) + x23*(z23*m02 + y23*m01 + x23*m00));
    
    // Volume
    double v=orientation*inv6*(-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - (x0[0] - x1[0])*(z03*y02 - z02*y03));

    // Volume in metric space
    double v_m = v*sqrt(((m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02));

    // Sliver functional
    double ts = dl0+dl1+dl2+dl3+dl4+dl5;
    double sliver = 15552.0*(v_m*v_m)/(ts*ts*ts);
    
    return sliver;
  }

  int getOrientation() {
	return orientation;
  }

 private:
  const double inv2;
  const double inv3;
  const double inv4;
  const double inv6;

  const double lipnikov_const2d; // 12.0*sqrt(3.0);
  const double lipnikov_const3d; // pow(6.0, 4)*sqrt(2.0);

  const int dimension;
  int orientation;
};
#endif
