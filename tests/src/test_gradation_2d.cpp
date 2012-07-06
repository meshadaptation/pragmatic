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
#include "MetricTensor.h"
#include "ticker.h"

using namespace std;

int main(int argc, char **argv){
  Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu("../data/box20x20.vtu");

  Surface<double, int> surface(*mesh);
  surface.find_surface();

  MetricField<double, int> metric_field(*mesh, surface);

  size_t NNodes = mesh->get_number_nodes();

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
  {
    if ( mesh->get_coords(i)[0] < 1e-4 && mesh->get_coords(i)[1] < 1e-4 )
    {
      psi[i] = 2.42;
    }
    else
    {
      psi[i] = pow(mesh->get_coords(i)[0]+0.1, 2) + pow(mesh->get_coords(i)[1]+0.1, 2);
    }
  }

  double start_tic = get_wtime();
  metric_field.add_field(&(psi[0]), 1.0);

  double gradation = 1.3;
  metric_field.apply_gradation(gradation);
  metric_field.update_mesh();

  std::cout<<"Hessian loop time = "<<get_wtime()-start_tic<<std::endl;

  vector<double> metric(NNodes*4);
  metric_field.get_metric(&(metric[0]));

  bool eigenvalues_ok = true;

  // loop over nodes and measure distance to each neighbour in turn
  for( int nid0=0; nid0 < (int) NNodes; nid0++ )
  {
    std::set<int> adjacent_nodes = mesh->get_node_patch(nid0);

    MetricTensor<double> metric_tensor_0(2,&(metric[4*nid0]));
    std::vector<double> D0(2), V0(4);
    metric_tensor_0.eigen_decomp(D0, V0);

    for (std::set<int>::iterator it = adjacent_nodes.begin(); it != adjacent_nodes.end(); it++)
    {
      int nid1 = *it;
      
      MetricTensor<double> metric_tensor_1(2,&(metric[4*nid1]));
      std::vector<double> D1(2), V1(4);
      metric_tensor_1.eigen_decomp(D1, V1);

      // Pair the eigenvectors by minimising the angle between them.
      std::vector<int> pairs(2, -1);
      std::vector<bool> paired(2, false);
      for(int d=0; d<2; d++)
      {
        std::vector<double> angle(2);
        for(int k=0; k<2; k++)
        {
          if(paired[k])
            continue;
          angle[k] = V0[d*2]*V1[k*2];
          for(int l=1; l<2; l++)
            angle[k] += V0[d*2+l]*V1[k*2+l];
          angle[k] = acos(fabs(angle[k]));
        }

        int r=0;
        for(;r<2;r++)
        {
          if(!paired[r])
          {
            pairs[d] = r;
            break;
          }
        }
        r++;

        for(;r<2;r++)
        {
          if(angle[pairs[d]]<angle[r])
          {
            pairs[d] = r;
          }
        }

        paired[pairs[d]] = true;

        assert(pairs[d]!=-1);
      }

      // Check that no eigenvalues need resizing
      double L01=mesh->calc_edge_length(nid0, nid1);
      for(int k=0; k<2; k++)
      {
        double h0 = 1.0/sqrt(D0[k]);
        double h1 = 1.0/sqrt(D1[pairs[k]]);
        double gamma = exp(fabs(h0 - h1)/L01);
        
        if (gamma > 1.05*gradation)
        {
          eigenvalues_ok = false;
        }
      }
    }
  }

  VTKTools<double, int>::export_vtu("../data/test_gradation_2d", mesh, &(psi[0]));

  delete mesh;

  if(eigenvalues_ok)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  return 0;
}

