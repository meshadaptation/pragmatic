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
#ifndef LIPNIKOV_H
#define LIPNIKOV_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <cmath>

/*! \brief Evaluates Lipnikov functional.
 */
template<typename real_t>
class LipnikovFunctional{
 public:
  LipnikovFunctional(const real_t *x0, const real_t *x1, const real_t *x2){
    real_t area = 0.5*((x0[1] - x2[1])*(x0[0] - x1[0]) - (x0[1] - x1[1])*(x0[0] - x2[0]));
    if(area<0)
      orientation = -1;
    else
      orientation = 1;
  }

  // Calculate area.
  real_t area(const real_t *x0, const real_t *x1, const real_t *x2){
    return orientation*0.5*((x0[1] - x2[1])*(x0[0] - x1[0]) - (x0[1] - x1[1])*(x0[0] - x2[0]));
  }

  /// 2D
  real_t calculate(const real_t *x0, const real_t *x1, const real_t *x2,
                          const real_t *m0, const real_t *m1, const real_t *m2){
    // Metric tensor
    real_t m00 = (m0[0] + m1[0] + m2[0])/3;
    real_t m01 = (m0[1] + m1[1] + m2[1])/3;
    real_t m11 = (m0[3] + m1[3] + m2[3])/3;

    // Paremeter
    real_t l =
      sqrt((x0[1] - x1[1])*((x0[1] - x1[1])*m11 + (x0[0] - x1[0])*m01) + 
           (x0[0] - x1[0])*((x0[1] - x1[1])*m01 + (x0[0] - x1[0])*m00))+
      sqrt((x0[1] - x2[1])*((x0[1] - x2[1])*m11 + (x0[0] - x2[0])*m01) + 
           (x0[0] - x2[0])*((x0[1] - x2[1])*m01 + (x0[0] - x2[0])*m00))+
      sqrt((x2[1] - x1[1])*((x2[1] - x1[1])*m11 + (x2[0] - x1[0])*m01) + 
           (x2[0] - x1[0])*((x2[1] - x1[1])*m01 + (x2[0] - x1[0])*m00));

    // Area
    real_t a=orientation*0.5*((x0[1] - x2[1])*(x0[0] - x1[0]) - (x0[1] - x1[1])*(x0[0] - x2[0]));

    // Area in metric space
    real_t a_m = a*(m00*m11 - m01*m01);

    // Function
    real_t f = min(l/3, 3/l);
    real_t F = pow(f * (2.0 - f), 3);
    real_t quality = 12*sqrt(3)*a_m*F/(l*l);
    
    return quality;
  }

 private:
  int orientation;
};
#endif
