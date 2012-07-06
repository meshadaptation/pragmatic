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

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCell.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#include <iostream>
#include <vector>

#include "MetricField.h"
#include "Smooth.h"
#include "Coarsen.h"
#include "Refine.h"
#include "Surface.h"
#include "VTKTools.h"

using namespace std;

int main(int argc, char **argv){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName("../data/coarse_slab0003.vtu");
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();

  int NNodes = ug->GetNumberOfPoints();
  int NElements = ug->GetNumberOfCells();

  vector<double> x(NNodes),  y(NNodes), z(NNodes);
  for(int i=0;i<NNodes;i++){
    double r[3];
    ug->GetPoints()->GetPoint(i, r);
    x[i] = r[0];
    y[i] = r[1];
    z[i] = r[2];
  }

  vector<int> ENList;
  for(int i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    for(size_t j=0;j<4;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }

  Mesh<double, int> mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));

  Surface<double, int> surface(mesh);
  surface.find_surface();

  MetricField<double, int> metric_field(mesh, surface);
  
  vector<double> psi(NNodes);
  
  vtkPointData *p_point_data = ug->GetPointData();
  vtkDataArray *p_scalars = p_point_data->GetArray("Vm");
  for(int i=0;i<NNodes;i++){
    psi[i] = p_scalars->GetTuple(i)[0];
  }
  
  reader->Delete();
  
  metric_field.add_field(&(psi[0]), 0.5);
  
  // metric_field.apply_gradation(1.3);
  metric_field.update_mesh();

  VTKTools<double, int>::export_vtu("../data/test_chaste_metric", &mesh, &(psi[0]));
  
  // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
  double L_up = 1.0; // sqrt(2);
  double L_low = L_up/2;
    
  Coarsen<double, int> coarsen(mesh, surface);
  coarsen.coarsen(L_low, L_up);
  
  Smooth<double, int> smooth(mesh, surface);
  double L_max = mesh.maximal_edge_length();
  
  int adapt_iter=0;
  double alpha = 0.95; //sqrt(2)/2;
  Refine<double, int> refine(mesh, surface);
  do{
    double L_ref = std::max(alpha*L_max, L_up);
      
    refine.refine(L_ref);    
    coarsen.coarsen(L_low, L_ref);

    L_max = mesh.maximal_edge_length();
  }while((L_max>L_up)&&(adapt_iter++<20));
    
  double lrms = mesh.get_lrms();
  double qrms = mesh.get_qrms();
  
  std::map<int, int> active_vertex_map;
  mesh.defragment(&active_vertex_map);
  surface.defragment(&active_vertex_map);

  smooth.smooth("smart Laplacian");
  
  int nelements = mesh.get_number_elements();
  
  std::cout<<"Number elements:      "<<nelements<<std::endl
           <<"Edge length RMS:      "<<lrms<<std::endl
           <<"Quality RMS:          "<<qrms<<std::endl;
  
  VTKTools<double, int>::export_vtu("../data/test_chaste_mesh", &mesh);
    
  if((lrms<0.8)&&(qrms<2.2))
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
