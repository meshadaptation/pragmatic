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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

int main()
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);

    A(0,0) = 0.959;
    A(0,1) = 0.4595;
    A(0,2) = 0.4882;
    A(0,3) = 1.478;
    A(0,4) = 0.8364;
    A(0,5) = 2.405;
    A(1,0) = 0.4595;
    A(1,1) = 0.7646;
    A(1,2) = 0.559;
    A(1,3) = 0.8651;
    A(1,4) = 1.098;
    A(1,5) = 1.708;
    A(2,0) = 0.4882;
    A(2,1) = 0.559;
    A(2,2) = 0.4595;
    A(2,3) = 0.8364;
    A(2,4) = 0.8651;
    A(2,5) = 1.523;
    A(3,0) = 1.478;
    A(3,1) = 0.8651;
    A(3,2) = 0.8364;
    A(3,3) = 2.405;
    A(3,4) = 1.523;
    A(3,5) = 4.2;
    A(4,0) = 0.8364;
    A(4,1) = 1.098;
    A(4,2) = 0.8651;
    A(4,3) = 1.523;
    A(4,4) = 1.708;
    A(4,5) = 2.95;
    A(5,0) = 2.405;
    A(5,1) = 1.708;
    A(5,2) = 1.523;
    A(5,3) = 4.2;
    A(5,4) = 2.95;
    A(5,5) = 8;

    Eigen::Matrix<double, Eigen::Dynamic, 1> b = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);
    b[0] = 1.419;
    b[1] = 1.224;
    b[2] = 1.047;
    b[3] = 2.343;
    b[4] = 1.934;
    b[5] = 4.113;

    Eigen::Matrix<double, Eigen::Dynamic, 1> a = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);

    // Reference solution from MATLAB
    Eigen::Matrix<double, Eigen::Dynamic, 1> ref_a = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);
    ref_a[0] = 1.2767;
    ref_a[1] = 1.0376;
    ref_a[2] = -0.0949;
    ref_a[3] = -0.2713;
    ref_a[4] = 0.0287;
    ref_a[5] = 0.0588;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    bool pass = true;
    a = ldlt.solve(b);
    for(int i=0; i<6; i++)
        if(fabs(a[i] - ref_a[i])>1.0e-4)
            pass = false;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    a = lu.solve(b);
    for(int i=0; i<6; i++)
        if(fabs(a[i] - ref_a[i])>1.0e-4)
            pass = false;

    if(pass)
        std::cout<<"pass\n";
    else
        std::cout<<"fail\n";

    return 0;
}
