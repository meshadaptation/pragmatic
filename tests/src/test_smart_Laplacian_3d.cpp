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
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"

#include "Smooth.h"
#include "ticker.h"

int main(int argc, char **argv){
  Mesh<double> *mesh=VTKTools<double>::import_vtu("../data/box20x20x20.vtu");

  Surface3D<double> surface(*mesh);
  surface.find_surface();

  MetricField3D<double> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  std::vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(mesh->get_coords(i)[0], 3) +
      pow(mesh->get_coords(i)[1], 3) + pow(mesh->get_coords(i)[2], 3);

  metric_field.add_field(&(psi[0]), 0.6);

  size_t NElements = mesh->get_number_elements();

  metric_field.apply_nelements(NElements);
  metric_field.update_mesh();

  Smooth3D<double> smooth(*mesh, surface);

  double tic = get_wtime();
  smooth.smooth("smart Laplacian");
  double toc = get_wtime();

  double lrms = mesh->get_lrms();
  double qrms = mesh->get_qrms();

  std::cout<<"Smooth loop time:     "<<toc-tic<<std::endl
           <<"Edge length RMS:      "<<lrms<<std::endl
           <<"Quality RMS:          "<<qrms<<std::endl;

  VTKTools<double>::export_vtu("../data/test_smart_Laplacian_3d", mesh);
  delete mesh;

  if((lrms<0.45)&&(qrms<2.0))
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
