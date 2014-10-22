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

#include <iostream>

#include "ElementProperty.h"

int main(){
  // Check triangles
  {
    double fx0[] = {0.0, 0.0};
    double fx1[] = {1.0, 0.0};
    double fx2[] = {0.0, 1.0};
    double fm[] = {1.0, 0.0, 1.0};

    ElementProperty<double> ftriangle(fx0, fx1, fx2);

    std::cout<<"Test ElementProperty<double>::area:"<<std::endl;
    if(fabs((double)0.5-ftriangle.area(fx0, fx1, fx2))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";
    
    std::cout<<"Test ElementProperty<double>::length:"<<std::endl;
    if((sqrt(2.0)-ftriangle.length<2>(fx1, fx2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length2d:"<<std::endl;
    if((sqrt(2.0)-ftriangle.length2d(fx1, fx2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    double x0[] = {0.0, 0.0};
    double x1[] = {1.0, 0.0};
    double x2[] = {0.0, 1.0};
    
    ElementProperty<double> triangle(x0, x1, x2);

    std::cout<<"Test ElementProperty<double> double precision area:"<<std::endl;
    if(fabs(0.5-triangle.area(x0, x1, x2))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length:"<<std::endl;
    if((sqrt(2.0)-triangle.length<2>(x1, x2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length2d:"<<std::endl;
    if((sqrt(2.0)-triangle.length2d(x1, x2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::lipnikov/ElementProperty<double>::lipnikov 2D:"<<std::endl;
    if(ftriangle.lipnikov(fx0, fx1, fx2, fm[0], fm[1], fm[2])==
       triangle.lipnikov(x0, x1, x2, fm[0], fm[1], fm[2]))
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";
  }

  // Check tetrahedra
  {
    double fx0[] = {0.0, 0.0, 0.0};
    double fx1[] = {1.0, 0.0, 0.0};
    double fx2[] = {0.0, 1.0, 0.0};
    double fx3[] = {0.0, 1.0, 1.0};
    double fm[] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    
    ElementProperty<double> ftetrahedron(fx0, fx1, fx2, fx3);

    std::cout<<"Test ElementProperty<double> single precision volume:"<<std::endl;
    if(fabs((double)(1.0/6.0)-ftetrahedron.volume(fx0, fx1, fx2, fx3))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length:"<<std::endl;
    if((sqrt(2.0)-ftetrahedron.length<3>(fx1, fx2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length3d:"<<std::endl;
    if((sqrt(2.0)-ftetrahedron.length3d(fx1, fx2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";
    
    double x0[] = {0.0, 0.0, 0.0};
    double x1[] = {1.0, 0.0, 0.0};
    double x2[] = {0.0, 1.0, 0.0};
    double x3[] = {0.0, 1.0, 1.0};
    
    ElementProperty<double> tetrahedron(x0, x1, x2, x3);

    std::cout<<"Test ElementProperty<double> double precision volume:"<<std::endl;
    if(fabs(1.0/6.0-tetrahedron.volume(x0, x1, x2, x3))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double> single precision volume:"<<std::endl;
    if(fabs((double)(1.0/6.0)-tetrahedron.volume(x0, x1, x2, x3))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length:"<<std::endl;
    if((sqrt(2.0)-tetrahedron.length<3>(x1, x2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::length3d:"<<std::endl;
    if((sqrt(2.0)-tetrahedron.length3d(x1, x2, fm))==0)
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";

    std::cout<<"Test ElementProperty<double>::lipnikov/ElementProperty<double>::lipnikov 3D:"<<std::endl;
    if(ftetrahedron.lipnikov(fx0, fx1, fx2, fx3, fm, fm, fm, fm)==
       tetrahedron.lipnikov(x0, x1, x2, x3, fm, fm, fm, fm))
      std::cout<<"pass\n";
    else
      std::cout<<"fail\n";
  }

  return 0;
}
