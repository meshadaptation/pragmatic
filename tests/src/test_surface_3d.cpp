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

#include "Surface.h"
#include "Mesh.h"
#include "vtk_tools.h"

using namespace std;

/* Tests
   1. Assert number of coplanar id's is 6.
 */

int main(int argc, char **argv){
  Mesh<double, int> *mesh=NULL;
  import_vtu("../data/box20x20x20.vtu", mesh);
  
  Surface<double, int> surface(*mesh);

  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();

  vtkPoints *vtk_points = vtkPoints::New();
  size_t NNodes = mesh->get_number_nodes();
  vtk_points->SetNumberOfPoints(NNodes);
  for(size_t i=0;i<NNodes;i++){
    const double *r = mesh->get_coords(i);
    vtk_points->SetPoint(i, r[0], r[1], 0.0);
  }
  ug->SetPoints(ug->GetPoints());
  vtk_points->Delete();

  // Need to get out the facets
  int NSElements = surface.get_number_facets();
  const int *facets = surface.get_facets();
  for(int i=0;i<NSElements;i++){
    vtkIdType pts[] = {facets[i*3], facets[i*3+1], facets[i*3+2]};
    ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
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
  ug->GetCellData()->AddArray(scalar);
  scalar->Delete();

  vtkDoubleArray *normal = vtkDoubleArray::New();
  normal->SetNumberOfComponents(3);
  normal->SetNumberOfTuples(NSElements);
  normal->SetName("normals");
  for(int i=0;i<NSElements;i++){
    const double *n = surface.get_normal(i);
    normal->SetTuple3(i, n[0], n[1], n[2]);
  }
  ug->GetCellData()->AddArray(normal);
  normal->Delete();

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName("../data/test_surface_3d.vtu");
  writer->SetInput(ug);
  writer->Write();

  writer->Delete();
  ug->Delete();

  delete mesh;

  if(unique_ids.size()==6)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";

  return 0;
}
