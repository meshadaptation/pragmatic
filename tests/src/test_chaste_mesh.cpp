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
  vtkDataArray *p_scalars = p_point_data->GetArray( "Vm" );
  for (int i = 0; i < NNodes; i++)
  {
    psi[i] = p_scalars->GetTuple(i)[0];
  }
	      
  metric_field.add_field(&(psi[0]), 0.6);

  metric_field.apply_nelements(NElements);

  metric_field.get_metric(&(metric[0]));

  Smooth<double, int> smooth(mesh, surface, &(metric[0]));
  
  double start_tic = omp_get_wtime();
  double initial_rms = smooth.smooth();
  for(int iter=1;iter<500;iter++){    
    double rms = smooth.smooth();
    
    if(rms<0.05*initial_rms){
      std::cout<<"Terminating at iteration "<<iter<<", rms = "<<rms<<std::endl;
      break;
    }
  }
  std::cerr<<"Simple smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;
  
  start_tic = omp_get_wtime();
  initial_rms = smooth.smooth(true);
  for(int iter=1;iter<500;iter++){    
    double rms = smooth.smooth(true);
        
    if(rms<0.05*initial_rms){
      std::cout<<"Terminating at iteration "<<iter<<", rms = "<<rms<<std::endl;
      break;
    }
  }
  std::cerr<<"Quality constrained smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;

  /*
  // recalculate
  for(int i=0;i<NNodes;i++)
    psi[i] = x[i]*x[i]*x[i]+y[i]*y[i]*y[i]+z[i]*z[i]*z[i];
  */
    
  vtkUnstructuredGrid *ug_out = vtkUnstructuredGrid::New();
  ug_out->DeepCopy(ug);
  
  for(int i=0;i<NNodes;i++){
    ug_out->GetPoints()->SetPoint(i, x[i], y[i], z[i]);
  }

  vtkDoubleArray *mfield = vtkDoubleArray::New();
  mfield->SetNumberOfComponents(9);
  mfield->SetNumberOfTuples(NNodes);
  mfield->SetName("Metric");
  for(int i=0;i<NNodes;i++)
    mfield->SetTuple9(i,
                      metric[i*9  ], metric[i*9+1], metric[i*9+2], 
                      metric[i*9+3], metric[i*9+4], metric[i*9+5],
                      metric[i*9+6], metric[i*9+7], metric[i*9+8]);
  ug_out->GetPointData()->AddArray(mfield);
  mfield->Delete();

  vtkDoubleArray *scalar = vtkDoubleArray::New();
  scalar->SetNumberOfComponents(1);
  scalar->SetNumberOfTuples(NNodes);
  scalar->SetName("psi");
  for(int i=0;i<NNodes;i++)
    scalar->SetTuple1(i, psi[i]);
  ug_out->GetPointData()->AddArray(scalar);
  scalar->Delete();

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_chaste_mesh.vtu");
  writer->SetInput(ug_out);
  writer->Write();

  reader->Delete();
  ug_out->Delete();
  writer->Delete();

  return 0;
}
