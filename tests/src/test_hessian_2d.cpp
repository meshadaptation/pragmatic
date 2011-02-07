/* 
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
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>

#include <iostream>
#include <vector>

#include <omp.h>

#include "MetricField.h"
#include "Surface.h"

using namespace std;

int main(int argc, char **argv){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName("../data/box20x20.vtu");
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();

  size_t NNodes = ug->GetNumberOfPoints();
  size_t NElements = ug->GetNumberOfCells();

  vector<double> x(NNodes),  y(NNodes), z(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double r[3];
    ug->GetPoints()->GetPoint(i, r);
    x[i] = r[0];
    y[i] = r[1];
    z[i] = r[2];
  }

  vector<int> ENList;
  for(size_t i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    for(size_t j=0;j<3;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }
  reader->Delete();

  Mesh<double, int> mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));

  ENList.clear();
  x.clear();
  y.clear();

  Surface<double, int> surface(mesh);

  MetricField<double, int> metric_field(mesh, surface);

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++)
    psi[i] = pow(x[i], 2) + pow(y[i], 2);
  
  double start_tic = omp_get_wtime();
  metric_field.add_field(&(psi[0]), 1.0);
  
  std::cerr<<"Hessian loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  vector<double> metric(NNodes*4);
  metric_field.get_metric(&(metric[0]));
  
  double rms[] = {0., 0., 0., 0.};
  for(size_t i=0;i<NNodes;i++){
    rms[0] += pow(2.0-metric[i*4  ], 2); rms[1] += pow(    metric[i*4+1], 2);
    rms[2] += pow(    metric[i*4+2], 2); rms[3] += pow(2.0-metric[i*4+3], 2);
  }
  
  double max_rms = 0;
  for(size_t i=0;i<4;i++){
    rms[i] = sqrt(rms[i]/NNodes);
    max_rms = std::max(max_rms, rms[i]);
  }

  if(max_rms>0.01)
    std::cout<<"fail\n";
  else
    std::cout<<"pass\n";
  
  // Create VTU object to write out.
  ug = vtkUnstructuredGrid::New();
  
  vtkPoints *vtk_points = vtkPoints::New();
  vtk_points->SetNumberOfPoints(NNodes);
  
  vtkDoubleArray *vtk_psi = vtkDoubleArray::New();
  vtk_psi->SetNumberOfComponents(1);
  vtk_psi->SetNumberOfTuples(NNodes);
  vtk_psi->SetName("psi");

  vtkIntArray *vtk_numbering = vtkIntArray::New();
  vtk_numbering->SetNumberOfComponents(1);
  vtk_numbering->SetNumberOfTuples(NNodes);
  vtk_numbering->SetName("nid");
  
  vtkDoubleArray *vtk_metric = vtkDoubleArray::New();
  vtk_metric->SetNumberOfComponents(4);
  vtk_metric->SetNumberOfTuples(NNodes);
  vtk_metric->SetName("Metric");
  
  for(size_t i=0;i<NNodes;i++){
    const double *r = mesh.get_coords()+i*2;
    vtk_psi->SetTuple1(i, pow(r[0], 2)+pow(r[1], 2));
    vtk_numbering->SetTuple1(i, i);
    vtk_points->SetPoint(i, r[0], r[1], 0.0);
    vtk_metric->SetTuple4(i,
                          metric[i*4  ], metric[i*4+1],
                          metric[i*4+2], metric[i*4+3]);
  }
  
  ug->SetPoints(vtk_points);
  vtk_points->Delete();
  
  ug->GetPointData()->AddArray(vtk_psi);
  vtk_psi->Delete();

  ug->GetPointData()->AddArray(vtk_numbering);
  vtk_numbering->Delete();
  
  ug->GetPointData()->AddArray(vtk_metric);
  vtk_metric->Delete();

  assert(NElements == (size_t)mesh.get_number_elements());
  for(size_t i=0;i<NElements;i++){
    vtkIdType pts[] = {mesh.get_enlist()[i*3],
                       mesh.get_enlist()[i*3+1], 
                       mesh.get_enlist()[i*3+2]};
    ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
  }

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_hessian_2d.vtu");
  writer->SetInput(ug);
  writer->Write();

  writer->Delete();
  ug->Delete();

  return 0;
}
