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
#include <vtkPointData.h>

#include <iostream>
#include <vector>

#include "MetricField.h"
#include "Smooth.h"

using namespace std;

int main(int argc, char **argv){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();

  int NNodes = ug->GetNumberOfPoints();
  int NElements = ug->GetNumberOfCells();

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

  MetricField<double, int> metric_field;
  metric_field.set_mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));

  vector<double> psi(NNodes);
  for(size_t i=0;i<NNodes;i++){
    double X = x[i]*2 - 1;
    double Y = y[i]*2 - 1;
    psi[i] = Y*X*X+Y*Y*Y+tanh(10*(sin(5*Y)-2*X));
  }
  metric_field.add_field(&(psi[0]), 0.5, false);

  vector<double> metric(NNodes*4);
  metric_field.get_metric(&(metric[0]));

  Smooth<double, int> smooth;
  smooth.set_mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(metric[0]));
  
  for(int i=0;i<100;i++){
    int moves = smooth.smooth();
    std::cout<<"iteration "<<i<<" moved "<<moves<<" nodes\n";
  }
  
  vtkUnstructuredGrid *ug_out = vtkUnstructuredGrid::New();
  ug_out->DeepCopy(ug);
  
  for(size_t i=0;i<NNodes;i++){
    ug_out->GetPoints()->SetPoint(i, x[i], y[i], z[i]);
  }

  vtkDoubleArray *mfield = vtkDoubleArray::New();
  mfield->SetNumberOfComponents(4);
  mfield->SetNumberOfTuples(NNodes);
  mfield->SetName("Metric");
  for(size_t i=0;i<NNodes;i++)
    mfield->SetTuple4(i, metric[i*4], metric[i*4+1], metric[i*4+2], metric[i*4+3]);
  ug_out->GetPointData()->AddArray(mfield);

  vtkDoubleArray *scalar = vtkDoubleArray::New();
  scalar->SetNumberOfComponents(1);
  scalar->SetNumberOfTuples(NNodes);
  scalar->SetName("psi");
  for(size_t i=0;i<NNodes;i++)
    scalar->SetTuple1(i, psi[i]);
  ug_out->GetPointData()->AddArray(scalar);

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("test_smooth_2d.vtu");
  writer->SetInput(ug_out);
  writer->Write();

  return 0;
}
