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
#include <vtkCellData.h>

#include <iostream>
#include <vector>

#include "MetricField.h"
#include "Surface.h"

using namespace std;

/* Tests
   1. Assert number of coplanar id's is 6.
 */

int main(int argc, char **argv){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName("../data/box20x20x20.vtu");
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
    for(int j=0;j<4;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }

  Surface<double, int> surface;
  surface.set_mesh(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));

  vtkUnstructuredGrid *ug_out = vtkUnstructuredGrid::New();
  ug_out->SetPoints(ug->GetPoints());

  // Need to get out the facets
  int NSElements = surface.get_number_facets();
  const int *facets = surface.get_facets();
  for(int i=0;i<NSElements;i++){
    vtkIdType pts[] = {facets[i*3], facets[i*3+1], facets[i*3+2]};
    ug_out->InsertNextCell(VTK_TRIANGLE, 3, pts);
  }

  // Need the facet ID's
  const int *coplanar_ids = surface.get_coplanar_ids();

  vtkIntArray *scalar = vtkIntArray::New();
  scalar->SetNumberOfComponents(1);
  scalar->SetNumberOfTuples(NSElements);
  scalar->SetName("coplanar_ids");
  std::set<int> unique_ids;
  for(int i=0;i<NSElements;i++){
    unique_ids.insert(coplanar_ids[i]);
    scalar->SetTuple1(i, coplanar_ids[i]);
  }
  ug_out->GetCellData()->AddArray(scalar);

  vtkDoubleArray *normal = vtkDoubleArray::New();
  normal->SetNumberOfComponents(3);
  normal->SetNumberOfTuples(NSElements);
  normal->SetName("normals");
  for(int i=0;i<NSElements;i++){
    const double *n = surface.get_normal(i);
    normal->SetTuple3(i, n[0], n[1], n[2]);
  }
  ug_out->GetCellData()->AddArray(normal);

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_surface_3d.vtu");
  writer->SetInput(ug_out);
  writer->Write();

  if(unique_ids.size()==6)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  return 0;
}
