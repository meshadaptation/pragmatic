/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Georgios Rokos
 *  Software Performance Optimisation Group
 *  Department of Computing
 *  Imperial College London
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

#include "defs.cuh"

extern "C" {

__constant__ int orientation;

__device__ real_t area(const real_t * x0, const real_t * x1, const real_t * x2)
{
	return orientation * 0.5 *
			( (x0[1] - x2[1]) * (x0[0] - x1[0]) -
			  (x0[1] - x1[1]) * (x0[0] - x2[0]) );
}

__device__ real_t lipnikov2d(const real_t * x0, const real_t * x1, const real_t * x2,
                const real_t * m0, const real_t * m1, const real_t * m2)
{
  // Metric tensor averaged over the element
  real_t m00 = (m0[0] + m1[0] + m2[0])/3;
  real_t m01 = (m0[1] + m1[1] + m2[1])/3;
  real_t m11 = (m0[2] + m1[2] + m2[2])/3;

  // l is the length of the perimeter, measured in metric space
  real_t l =
    sqrt((x0[1] - x1[1])*((x0[1] - x1[1])*m11 + (x0[0] - x1[0])*m01) +
         (x0[0] - x1[0])*((x0[1] - x1[1])*m01 + (x0[0] - x1[0])*m00))+
    sqrt((x0[1] - x2[1])*((x0[1] - x2[1])*m11 + (x0[0] - x2[0])*m01) +
         (x0[0] - x2[0])*((x0[1] - x2[1])*m01 + (x0[0] - x2[0])*m00))+
    sqrt((x2[1] - x1[1])*((x2[1] - x1[1])*m11 + (x2[0] - x1[0])*m01) +
         (x2[0] - x1[0])*((x2[1] - x1[1])*m01 + (x2[0] - x1[0])*m00));

  // Area in physical space
  real_t a = area(x0, x1, x2);

  // Area in metric space
  real_t a_m = a*sqrt(m00*m11 - m01*m01);

  // Function
  real_t f = min(l/3.0, 3.0/l);
  real_t F = pow(f * (2.0 - f), 3.0);
  real_t quality = 12.0 * sqrt(3.0) * a_m * F / (l*l);

  return quality;
}

}

#endif
