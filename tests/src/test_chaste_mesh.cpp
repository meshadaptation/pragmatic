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
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#include <iostream>
#include <vector>

#include "MetricField.h"
#include "Smooth.h"
#include "Surface.h"

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

  MetricField<double, int> metric_field(mesh, surface);

  vector<double> psi(NNodes);
  vector<double> metric(NNodes*9);

  vtkPointData *p_point_data = ug->GetPointData();
  vtkDataArray *p_scalars = p_point_data->GetArray("Vm");
  for(int i=0;i<NNodes;i++){
    psi[i] = p_scalars->GetTuple(mesh.new2old(i))[0];
  }

  reader->Delete();

  metric_field.add_field(&(psi[0]), 1.0);

  metric_field.apply_gradation(1.3);
  metric_field.apply_nelements(NElements);

  metric_field.get_metric(&(metric[0]));

  Smooth<double, int> smooth(mesh, surface, &(metric[0]));

  double start_tic = omp_get_wtime();
  double prev_mean_quality = smooth.smooth();
  int iter=1;
  for(;iter<500;iter++){
    double mean_quality = smooth.smooth();
    double res = abs(mean_quality-prev_mean_quality)/prev_mean_quality;
    prev_mean_quality = mean_quality;
    if(res<1.0e-4)
      break;
    std::cout<<"quality "<<iter<<" "<<mean_quality<<std::endl;
  }
  std::cout<<"Simple smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  if(iter<500)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  start_tic = omp_get_wtime();
  prev_mean_quality = smooth.smooth(true);
  iter=1;
  for(;iter<500;iter++){
    double mean_quality = smooth.smooth(true);
    double res = abs(mean_quality-prev_mean_quality)/prev_mean_quality;
    prev_mean_quality = mean_quality;
    if(res<1.0e-5)
      break;
  }
  std::cout<<"Constrained smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  if(iter<500)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  // Export
  ug = vtkUnstructuredGrid::New();

  vtkPoints *vtk_points = vtkPoints::New();
  assert(NNodes==(int)mesh.get_number_nodes());
  vtk_points->SetNumberOfPoints(NNodes);

  vtkIntArray *vtk_numbering = vtkIntArray::New();
  vtk_numbering->SetNumberOfComponents(1);
  vtk_numbering->SetNumberOfTuples(NNodes);
  vtk_numbering->SetName("nid");

  size_t ndims = mesh.get_number_dimensions();

  vtkDoubleArray *vtk_metric = vtkDoubleArray::New();
  vtk_metric->SetNumberOfComponents(ndims*ndims);
  vtk_metric->SetNumberOfTuples(NNodes);
  vtk_metric->SetName("Metric");

  for(int i=0;i<NNodes;i++){
    const double *r = mesh.get_coords(i);
    vtk_numbering->SetTuple1(i, i);

    vtk_points->SetPoint(i, r[0], r[1], r[2]);
    vtk_metric->SetTuple9(i,
                          metric[i*9],   metric[i*9+1], metric[i*9+2],
                          metric[i*9+3], metric[i*9+4], metric[i*9+5],
                          metric[i*9+6], metric[i*9+7], metric[i*9+8]);
  }

  ug->SetPoints(vtk_points);
  vtk_points->Delete();

  ug->GetPointData()->AddArray(vtk_numbering);
  vtk_numbering->Delete();

  ug->GetPointData()->AddArray(vtk_metric);
  vtk_metric->Delete();

  assert(NElements==(int)mesh.get_number_elements());
  for(int i=0;i<NElements;i++){
    vtkIdType pts[] = {mesh.get_enlist(i)[0],
                       mesh.get_enlist(i)[1],
                       mesh.get_enlist(i)[2],
                       mesh.get_enlist(i)[3]};
    ug->InsertNextCell(VTK_TETRA, 4, pts);
  }

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_chaste_mesh.vtu");
  writer->SetInput(ug);
  writer->Write();

  writer->Delete();
  ug->Delete();

  return 0;
}
