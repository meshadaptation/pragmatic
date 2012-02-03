/*  Copyright (C) 2010 Imperial College London and others.
    
    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Gerard Gorman
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    g.gorman@imperial.ac.uk
    
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
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

/*! \brief Calculates a number of element properties.
 *
 * The constructers for this class requires a reference element so
 * that the orientation convention can be established. After the
 * orientation has been established a negative area or volume
 * indicated an inverted element. Properties caluclated are:
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
 ElementProperty(const real_t *x0, const real_t *x1, const real_t *x2): dimension(2){
    orientation = 1;
    
    real_t A = area(x0, x1, x2);
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
 ElementProperty(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3) : dimension(3){
    orientation = 1;

    real_t V = volume(x0, x1, x2, x3);
    if(V<0)
      orientation = -1;
    else
      orientation = 1;
  }

  /*! Calculate area of 2D triangle.
   * @param x0 pointer to 2D position for first point in triangle.
   * @param x1 pointer to 2D position for second point in triangle.
   * @param x2 pointer to 2D position for third point in triangle.
   */
  real_t area(const real_t *x0, const real_t *x1, const real_t *x2){
    return orientation*0.5*((x0[1] - x2[1])*(x0[0] - x1[0]) - (x0[1] - x1[1])*(x0[0] - x2[0]));
  }

  /*! Calculate volume of tetrahedron.
   * @param x0 pointer to 3D position for first point in triangle.
   * @param x1 pointer to 3D position for second point in triangle.
   * @param x2 pointer to 3D position for third point in triangle.
   * @param x3 pointer to 3D position for forth point in triangle.
   */
  real_t volume(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3){
    return orientation*(-(x0[0] - x3[0])*((x0[2] - x2[2])*(x0[1] - x1[1]) - (x0[2] - x1[2])*(x0[1] - x2[1])) + (x0[0] - x2[0])*((x0[2] - x3[2])*(x0[1] - x1[1]) - (x0[2] - x1[2])*(x0[1] - x3[1])) - (x0[0] - x1[0])*((x0[2] - x3[2])*(x0[1] - x2[1]) - (x0[2] - x2[2])*(x0[1] - x3[1])))/6;
  }

  /*! Length of an edge as measured in metric space.
   *
   * @param x0 coordinate at start of line segment.
   * @param x1 coordinate at finish of line segment.
   * @param m metric tensor for first point.
   */
  real_t length(const real_t x0[], const real_t x1[], const real_t m[]) const{
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
  static real_t length2d(const real_t x0[], const real_t x1[], const real_t m[]){
    const real_t x=x0[0] - x1[0];
    const real_t y=x0[1] - x1[1];
    
    return sqrt(((m[1]*x + m[3]*y)*y + (m[0]*x + m[1]*y)*x));
  }

  /*! Length of an edge as measured in metric space.
   *
   * @param x0 coordinate at start of line segment.
   * @param x1 coordinate at finish of line segment.
   * @param m metric tensor for first point.
   */
  static real_t length3d(const real_t x0[], const real_t x1[], const real_t m[]){
    const real_t x=x0[0] - x1[0];
    const real_t y=x0[1] - x1[1];
    const real_t z=x0[2] - x1[2];
    
    return sqrt(z*(z*m[8] + y*m[5] + x*m[2]) +
                y*(z*m[5] + y*m[4] + x*m[1]) +
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
  real_t lipnikov(const real_t *x0, const real_t *x1, const real_t *x2,
                  const real_t *m0, const real_t *m1, const real_t *m2){
    // Metric tensor averaged over the element
    real_t m00 = (m0[0] + m1[0] + m2[0])/3;
    real_t m01 = (m0[1] + m1[1] + m2[1])/3;
    real_t m11 = (m0[3] + m1[3] + m2[3])/3;
    
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
  real_t lipnikov(const real_t *x0, const real_t *x1, const real_t *x2,
                  real_t m00, real_t m01, real_t m11){
    // l is the length of the perimeter, measured in metric space
    real_t l =
      sqrt((x0[1] - x1[1])*((x0[1] - x1[1])*m11 + (x0[0] - x1[0])*m01) + 
           (x0[0] - x1[0])*((x0[1] - x1[1])*m01 + (x0[0] - x1[0])*m00))+
      sqrt((x0[1] - x2[1])*((x0[1] - x2[1])*m11 + (x0[0] - x2[0])*m01) + 
           (x0[0] - x2[0])*((x0[1] - x2[1])*m01 + (x0[0] - x2[0])*m00))+
      sqrt((x2[1] - x1[1])*((x2[1] - x1[1])*m11 + (x2[0] - x1[0])*m01) + 
           (x2[0] - x1[0])*((x2[1] - x1[1])*m01 + (x2[0] - x1[0])*m00));

    // Area in physical space
    real_t a=area(x0, x1, x2);

    // Area in metric space
    real_t a_m = a*sqrt(m00*m11 - m01*m01);

    // Function
    real_t f = std::min(l/3.0, 3.0/l);
    real_t F = pow(f * (2.0 - f), 3.0);
    real_t quality = 12.0*sqrt(3.0)*a_m*F/(l*l);

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
   * @param m1 3x3 metric tensor for second point.
   * @param m2 3x3 metric tensor for third point.
   * @param m3 3x3 metric tensor for forth point.
   */
  real_t lipnikov(const real_t *x0, const real_t *x1, const real_t *x2, const real_t *x3,
                  const real_t *m0, const real_t *m1, const real_t *m2, const real_t *m3){
    // Metric tensor
    real_t m00 = (m0[0] + m1[0] + m2[0] + m3[0])/4;
    real_t m01 = (m0[1] + m1[1] + m2[1] + m3[1])/4;
    real_t m02 = (m0[2] + m1[2] + m2[2] + m3[2])/4;
    real_t m11 = (m0[4] + m1[4] + m2[4] + m3[4])/4;
    real_t m12 = (m0[5] + m1[5] + m2[5] + m3[5])/4;
    real_t m22 = (m0[8] + m1[8] + m2[8] + m3[8])/4;

    // l is the length of the edges of the tet, in metric space
    real_t dl0 = ((x0[2] - x1[2])*((x0[2] - x1[2])*m22 + (x0[1] - x1[1])*m12 + (x0[0] - x1[0])*m02) + (x0[1] - x1[1])*((x0[2] - x1[2])*m12 + (x0[1] - x1[1])*m11 + (x0[0] - x1[0])*m01) + (x0[0] - x1[0])*((x0[2] - x1[2])*m02 + (x0[1] - x1[1])*m01 + (x0[0] - x1[0])*m00));
    real_t dl1 = ((x1[2] - x2[2])*((x1[2] - x2[2])*m22 + (x1[1] - x2[1])*m12 + (x1[0] - x2[0])*m02) + (x1[1] - x2[1])*((x1[2] - x2[2])*m12 + (x1[1] - x2[1])*m11 + (x1[0] - x2[0])*m01) + (x1[0] - x2[0])*((x1[2] - x2[2])*m02 + (x1[1] - x2[1])*m01 + (x1[0] - x2[0])*m00));
    real_t dl2 = ((x0[2] - x2[2])*((x0[2] - x2[2])*m22 + (x0[1] - x2[1])*m12 + (x0[0] - x2[0])*m02) + (x0[1] - x2[1])*((x0[2] - x2[2])*m12 + (x0[1] - x2[1])*m11 + (x0[0] - x2[0])*m01) + (x0[0] - x2[0])*((x0[2] - x2[2])*m02 + (x0[1] - x2[1])*m01 + (x0[0] - x2[0])*m00));
    real_t dl3 = ((x0[2] - x3[2])*((x0[2] - x3[2])*m22 + (x0[1] - x3[1])*m12 + (x0[0] - x3[0])*m02) + (x0[1] - x3[1])*((x0[2] - x3[2])*m12 + (x0[1] - x3[1])*m11 + (x0[0] - x3[0])*m01) + (x0[0] - x3[0])*((x0[2] - x3[2])*m02 + (x0[1] - x3[1])*m01 + (x0[0] - x3[0])*m00));
    real_t dl4 = ((x1[2] - x3[2])*((x1[2] - x3[2])*m22 + (x1[1] - x3[1])*m12 + (x1[0] - x3[0])*m02) + (x1[1] - x3[1])*((x1[2] - x3[2])*m12 + (x1[1] - x3[1])*m11 + (x1[0] - x3[0])*m01) + (x1[0] - x3[0])*((x1[2] - x3[2])*m02 + (x1[1] - x3[1])*m01 + (x1[0] - x3[0])*m00));
    real_t dl5 = ((x2[2] - x3[2])*((x2[2] - x3[2])*m22 + (x2[1] - x3[1])*m12 + (x2[0] - x3[0])*m02) + (x2[1] - x3[1])*((x2[2] - x3[2])*m12 + (x2[1] - x3[1])*m11 + (x2[0] - x3[0])*m01) + (x2[0] - x3[0])*((x2[2] - x3[2])*m02 + (x2[1] - x3[1])*m01 + (x2[0] - x3[0])*m00));
    
    real_t l = sqrt(dl0)+sqrt(dl1)+sqrt(dl2)+sqrt(dl3)+sqrt(dl4)+sqrt(dl5);

    // Volume
    real_t v=volume(x0, x1, x2, x3);

    // Volume in metric space
    real_t v_m = v*sqrt(((m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02));

    // Function
    real_t f = std::min(l/6, 6/l);
    real_t F = pow(f * (2.0 - f), 3);
    real_t quality = pow(6.0, 4)*sqrt(2.0) * v_m * F / (l*l*l);

    return quality;
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
                const real_t *m0, const real_t *m1, const real_t *m2, const real_t *m3){
    // Metric tensor
    real_t m00 = (m0[0] + m1[0] + m2[0] + m3[0])/4;
    real_t m01 = (m0[1] + m1[1] + m2[1] + m3[1])/4;
    real_t m02 = (m0[2] + m1[2] + m2[2] + m3[2])/4;
    real_t m11 = (m0[4] + m1[4] + m2[4] + m3[4])/4;
    real_t m12 = (m0[5] + m1[5] + m2[5] + m3[5])/4;
    real_t m22 = (m0[8] + m1[8] + m2[8] + m3[8])/4;

    // l is the length of the edges of the tet, in metric space
    real_t dl0 = ((x0[2] - x1[2])*((x0[2] - x1[2])*m22 + (x0[1] - x1[1])*m12 + (x0[0] - x1[0])*m02) + (x0[1] - x1[1])*((x0[2] - x1[2])*m12 + (x0[1] - x1[1])*m11 + (x0[0] - x1[0])*m01) + (x0[0] - x1[0])*((x0[2] - x1[2])*m02 + (x0[1] - x1[1])*m01 + (x0[0] - x1[0])*m00));
    real_t dl1 = ((x1[2] - x2[2])*((x1[2] - x2[2])*m22 + (x1[1] - x2[1])*m12 + (x1[0] - x2[0])*m02) + (x1[1] - x2[1])*((x1[2] - x2[2])*m12 + (x1[1] - x2[1])*m11 + (x1[0] - x2[0])*m01) + (x1[0] - x2[0])*((x1[2] - x2[2])*m02 + (x1[1] - x2[1])*m01 + (x1[0] - x2[0])*m00));
    real_t dl2 = ((x0[2] - x2[2])*((x0[2] - x2[2])*m22 + (x0[1] - x2[1])*m12 + (x0[0] - x2[0])*m02) + (x0[1] - x2[1])*((x0[2] - x2[2])*m12 + (x0[1] - x2[1])*m11 + (x0[0] - x2[0])*m01) + (x0[0] - x2[0])*((x0[2] - x2[2])*m02 + (x0[1] - x2[1])*m01 + (x0[0] - x2[0])*m00));
    real_t dl3 = ((x0[2] - x3[2])*((x0[2] - x3[2])*m22 + (x0[1] - x3[1])*m12 + (x0[0] - x3[0])*m02) + (x0[1] - x3[1])*((x0[2] - x3[2])*m12 + (x0[1] - x3[1])*m11 + (x0[0] - x3[0])*m01) + (x0[0] - x3[0])*((x0[2] - x3[2])*m02 + (x0[1] - x3[1])*m01 + (x0[0] - x3[0])*m00));
    real_t dl4 = ((x1[2] - x3[2])*((x1[2] - x3[2])*m22 + (x1[1] - x3[1])*m12 + (x1[0] - x3[0])*m02) + (x1[1] - x3[1])*((x1[2] - x3[2])*m12 + (x1[1] - x3[1])*m11 + (x1[0] - x3[0])*m01) + (x1[0] - x3[0])*((x1[2] - x3[2])*m02 + (x1[1] - x3[1])*m01 + (x1[0] - x3[0])*m00));
    real_t dl5 = ((x2[2] - x3[2])*((x2[2] - x3[2])*m22 + (x2[1] - x3[1])*m12 + (x2[0] - x3[0])*m02) + (x2[1] - x3[1])*((x2[2] - x3[2])*m12 + (x2[1] - x3[1])*m11 + (x2[0] - x3[0])*m01) + (x2[0] - x3[0])*((x2[2] - x3[2])*m02 + (x2[1] - x3[1])*m01 + (x2[0] - x3[0])*m00));
    
    // Volume
    real_t v=volume(x0, x1, x2, x3);

    // Volume in metric space
    real_t v_m = v*sqrt(((m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02));

    // Sliver functional
    real_t sliver = 15552.0*pow(v_m, 2)/pow(dl0+dl1+dl2+dl3+dl4+dl5, 3);

    return sliver;
  }

  int getOrientation() {
	return orientation;
  }

 private:
  int orientation;
  const int dimension;
};
#endif
