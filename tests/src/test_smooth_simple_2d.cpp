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

#include <omp.h>

#include "MetricField.h"
#include "Smooth.h"
#include "Surface.h"

using namespace std;

int main(int argc, char **argv){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName("../data/box20x20.vtu");
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
    for(size_t j=0;j<3;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }

  Mesh<double, int> mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));

  Surface<double, int> surface(mesh);

  MetricField<double, int> metric_field(mesh, surface);

  vector<double> psi(NNodes);  
  for(int i=0;i<NNodes;i++)
    psi[i] = x[i]*x[i]*x[i]+y[i]*y[i]*y[i];
  
  metric_field.add_field(&(psi[0]), 0.6);

  metric_field.apply_nelements(NElements);

  vector<double> metric(NNodes*4);
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
  }
  std::cerr<<"Smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  // recalculate
  for(int i=0;i<NNodes;i++)
    psi[i] = x[i]*x[i]*x[i]+y[i]*y[i]*y[i];

  vtkUnstructuredGrid *ug_out = vtkUnstructuredGrid::New();
  ug_out->DeepCopy(ug);
  
  for(int i=0;i<NNodes;i++){
    double *r = mesh.get_coords()+i*2;
    ug_out->GetPoints()->SetPoint(i, r[0], r[1], 0.0);
  }

  vtkDoubleArray *mfield = vtkDoubleArray::New();
  mfield->SetNumberOfComponents(4);
  mfield->SetNumberOfTuples(NNodes);
  mfield->SetName("Metric");
  for(int i=0;i<NNodes;i++)
    mfield->SetTuple4(i, metric[i*4], metric[i*4+1], metric[i*4+2], metric[i*4+3]);
  ug_out->GetPointData()->AddArray(mfield);

  vtkDoubleArray *scalar = vtkDoubleArray::New();
  scalar->SetNumberOfComponents(1);
  scalar->SetNumberOfTuples(NNodes);
  scalar->SetName("psi");
  for(int i=0;i<NNodes;i++)
    scalar->SetTuple1(i, psi[i]);
  ug_out->GetPointData()->AddArray(scalar);

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_smooth_simple_2d.vtu");
  writer->SetInput(ug_out);
  writer->Write();

  std::cout<<"iter "<<iter<<std::endl;

  if(iter<80)
    std::cout<<"pass"<<std::endl;
  else
    std::cout<<"fail"<<std::endl;

  return 0;
}
