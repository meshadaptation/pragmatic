/*
 *    WARNING: Do not edit this file as it is automatically generated
 *             and may be overwritten. Instead, merge your changes into
 *             sage/functional.sage and execute make from there. You will
 *             need sage in your PATH for this to work. If you commit your
 *             changes then commit both sage/functional.sage and the
 *             generated files.
 *
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
  void grad_r(real_t x, real_t y,
              real_t a0, real_t a1, real_t a2, real_t a3, real_t a4, real_t a5,
              real_t b0, real_t b1, real_t b2, real_t b3, real_t b4, real_t b5,
              real_t c0, real_t c1, real_t c2, real_t c3, real_t c4, real_t c5,
              real_t x1, real_t y1, real_t m00_1, real_t m01_1, real_t m11_1,
              real_t x2, real_t y2, real_t m00_2, real_t m01_2, real_t m11_2,
              real_t *grad){
    grad[0] = -1.26424253333333*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2))*((y - y2)*(x - x1) - (y - y1)*(x - x2))*(((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (y - y1)*(b0*y*y + b1*x*x + b2*x*y + (y - y1)*(2*c1*x + c2*y + c4) + (x - x1)*(2*b1*x + b2*y + b4) + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + (y - y1)*(2*b1*x + b2*y + b4) + (x - x1)*(2*a1*x + a2*y + a4) + a3*y + a4*x + a5 + m00_1 + m00_2))/(double)sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + ((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (y - y2)*(b0*y*y + b1*x*x + b2*x*y + (y - y2)*(2*c1*x + c2*y + c4) + (x - x2)*(2*b1*x + b2*y + b4) + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + (y - y2)*(2*b1*x + b2*y + b4) + (x - x2)*(2*a1*x + a2*y + a4) + a3*y + a4*x + a5 + m00_1 + m00_2))/(double)sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + ((y1 - y2)*((y1 - y2)*(2*c1*x + c2*y + c4) + (x1 - x2)*(2*b1*x + b2*y + b4)) + (x1 - x2)*((y1 - y2)*(2*b1*x + b2*y + b4) + (x1 - x2)*(2*a1*x + a2*y + a4)))/(double)sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))))*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)/(double)((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000)*((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000) + 3.79272760000000*(y1 - y2)*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2))/(double)((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000) + 0.210707088888889*((y - y2)*(x - x1) - (y - y1)*(x - x2))*((2*c1*x + c2*y + c4)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 2*(2*b1*x + b2*y + b4)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (2*a1*x + a2*y + a4)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2))/(double)(((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000)*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2)));
    grad[1] = -1.26424253333333*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2))*((y - y2)*(x - x1) - (y - y1)*(x - x2))*(((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (y - y1)*(c0*y*y + c1*x*x + c2*x*y + (y - y1)*(2*c0*y + c2*x + c3) + (x - x1)*(2*b0*y + b2*x + b3) + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + (y - y1)*(2*b0*y + b2*x + b3) + (x - x1)*(2*a0*y + a2*x + a3) + b3*y + b4*x + b5 + m01_1 + m01_2))/(double)sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + ((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (y - y2)*(c0*y*y + c1*x*x + c2*x*y + (y - y2)*(2*c0*y + c2*x + c3) + (x - x2)*(2*b0*y + b2*x + b3) + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + (y - y2)*(2*b0*y + b2*x + b3) + (x - x2)*(2*a0*y + a2*x + a3) + b3*y + b4*x + b5 + m01_1 + m01_2))/(double)sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + ((y1 - y2)*((y1 - y2)*(2*c0*y + c2*x + c3) + (x1 - x2)*(2*b0*y + b2*x + b3)) + (x1 - x2)*((y1 - y2)*(2*b0*y + b2*x + b3) + (x1 - x2)*(2*a0*y + a2*x + a3)))/(double)sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))))*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)/(double)((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000)*((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000) - 3.79272760000000*(x1 - x2)*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2))/(double)((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000) + 0.210707088888889*((y - y2)*(x - x1) - (y - y1)*(x - x2))*((2*c0*y + c2*x + c3)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 2*(2*b0*y + b2*x + b3)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (2*a0*y + a2*x + a3)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2))/(double)(((sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000)*(sqrt(1/(double)3*(y - y1)*((y - y1)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x1)*((y - y1)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x1)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y - y2)*((y - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x - x2)*((y - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) + sqrt(1/(double)3*(y1 - y2)*((y1 - y2)*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2) + (x1 - x2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2)) + 1/(double)3*(x1 - x2)*((y1 - y2)*(b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2) + (x1 - x2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2))) - 1.50000000000000) + 1.89199003000000)*sqrt(1/(double)9*(c0*y*y + c1*x*x + c2*x*y + c3*y + c4*x + c5 + m11_1 + m11_2)*(a0*y*y + a1*x*x + a2*x*y + a3*y + a4*x + a5 + m00_1 + m00_2) - 1/(double)9*pow((b0*y*y + b1*x*x + b2*x*y + b3*y + b4*x + b5 + m01_1 + m01_2), 2)));
    
    return;
  }
