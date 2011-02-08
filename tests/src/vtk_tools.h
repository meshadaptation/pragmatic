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
#ifndef VTK_TOOLS_H
#define VTK_TOOLS_H

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>

#include <vector>
#include <string>

#include "Mesh.h"

template<typename real_t, typename index_t>
void import_vtu(const char *filename, Mesh<real_t, index_t>* &mesh){
  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName(filename);
  reader->Update();
  
  vtkUnstructuredGrid *ug = reader->GetOutput();

  size_t NNodes = ug->GetNumberOfPoints();
  size_t NElements = ug->GetNumberOfCells();

  std::vector<real_t> x(NNodes),  y(NNodes), z(NNodes);
  for(size_t i=0;i<NNodes;i++){
    real_t r[3];
    ug->GetPoints()->GetPoint(i, r);
    x[i] = r[0];
    y[i] = r[1];
    z[i] = r[2];
  }

  int cell_type = ug->GetCell(0)->GetCellType();
  
  int nloc = -1;
  int ndims = -1;
  if(cell_type==VTK_TRIANGLE){
    nloc = 3;
    ndims = 2;
  }else if(cell_type==VTK_TETRA){
    nloc = 4;
    ndims = 3;
  }else{
    std::cerr<<"ERROR: unsupported element type\n";
    exit(-1);
  }

  std::vector<int> ENList;
  for(size_t i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    assert(cell->GetCellType()==cell_type);
    for(int j=0;j<nloc;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }
  reader->Delete();

  if(ndims==2)
    mesh = new Mesh<real_t, int>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
  else
    mesh = new Mesh<real_t, int>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));
  
  return;
}

template<typename real_t, typename index_t>
void export_vtu(const char *filename, const Mesh<real_t, index_t> *mesh, const real_t *psi, const real_t *metric){
  // Create VTU object to write out.
  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();
  
  vtkPoints *vtk_points = vtkPoints::New();
  size_t NNodes = mesh->get_number_nodes();
  vtk_points->SetNumberOfPoints(NNodes);

  vtkDoubleArray *vtk_psi = vtkDoubleArray::New();
  vtk_psi->SetNumberOfComponents(1);
  vtk_psi->SetNumberOfTuples(NNodes);
  vtk_psi->SetName("psi");

  vtkIntArray *vtk_numbering = vtkIntArray::New();
  vtk_numbering->SetNumberOfComponents(1);
  vtk_numbering->SetNumberOfTuples(NNodes);
  vtk_numbering->SetName("nid");
  
  size_t ndims = mesh->get_number_dimensions();

  vtkDoubleArray *vtk_metric = NULL;
  if(metric!=NULL){
    vtk_metric = vtkDoubleArray::New();
    vtk_metric->SetNumberOfComponents(ndims*ndims);
    vtk_metric->SetNumberOfTuples(NNodes);
    vtk_metric->SetName("Metric");
  }

  for(size_t i=0;i<NNodes;i++){
    const real_t *r = mesh->get_coords(i);
    vtk_psi->SetTuple1(i, psi[i]);
    vtk_numbering->SetTuple1(i, i);
    if(ndims==2){
      vtk_points->SetPoint(i, r[0], r[1], 0.0);
      if(vtk_metric!=NULL)
        vtk_metric->SetTuple4(i,
                              metric[i*4  ], metric[i*4+1],
                              metric[i*4+2], metric[i*4+3]);
    }else{
      vtk_points->SetPoint(i, r[0], r[1], r[2]);
      if(vtk_metric!=NULL)
        vtk_metric->SetTuple9(i,
                              metric[i*9],   metric[i*9+1], metric[i*9+2],
                              metric[i*9+3], metric[i*9+4], metric[i*9+5],
                              metric[i*9+6], metric[i*9+7], metric[i*9+8]); 
    }
  }
  
  ug->SetPoints(vtk_points);
  vtk_points->Delete();

  ug->GetPointData()->AddArray(vtk_psi);
  vtk_psi->Delete();

  ug->GetPointData()->AddArray(vtk_numbering);
  vtk_numbering->Delete();
  
  if(vtk_metric!=NULL){
    ug->GetPointData()->AddArray(vtk_metric);
    vtk_metric->Delete();
  }

  size_t NElements = mesh->get_number_elements();
  for(size_t i=0;i<NElements;i++){
    if(ndims==2){
      vtkIdType pts[] = {mesh->get_enlist(i)[0],
                         mesh->get_enlist(i)[1], 
                         mesh->get_enlist(i)[2]};
      ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
    }else{
      vtkIdType pts[] = {mesh->get_enlist(i)[0],
                         mesh->get_enlist(i)[1], 
                         mesh->get_enlist(i)[2],
                         mesh->get_enlist(i)[3]};
      ug->InsertNextCell(VTK_TETRA, 4, pts);
    }
  }
  
  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName(filename);
  writer->SetInput(ug);
  writer->Write();
  
  writer->Delete();
  ug->Delete();
  
  return;
}

#endif
