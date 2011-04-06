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
#include <vtkCellData.h>

#include <vector>
#include <string>

#include "Mesh.h"
#include "Surface.h"

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
void export_vtu(const char *filename, const Mesh<real_t, index_t> *mesh, const real_t *psi=NULL){
  // Create VTU object to write out.
  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();
  
  vtkPoints *vtk_points = vtkPoints::New();
  size_t NNodes = mesh->get_number_nodes();
  vtk_points->SetNumberOfPoints(NNodes);

  vtkDoubleArray *vtk_psi = NULL;

  if(psi!=NULL){
    vtk_psi = vtkDoubleArray::New();
    vtk_psi->SetNumberOfComponents(1);
    vtk_psi->SetNumberOfTuples(NNodes);
    vtk_psi->SetName("psi");
  }
  
  vtkIntArray *vtk_node_numbering = vtkIntArray::New();
  vtk_node_numbering->SetNumberOfComponents(1);
  vtk_node_numbering->SetNumberOfTuples(NNodes);
  vtk_node_numbering->SetName("nid");

  vtkIntArray *vtk_node_tpartition = vtkIntArray::New();
  vtk_node_tpartition->SetNumberOfComponents(1);
  vtk_node_tpartition->SetNumberOfTuples(NNodes);
  vtk_node_tpartition->SetName("node_tpartition");

  size_t ndims = mesh->get_number_dimensions();

  vtkDoubleArray *vtk_metric = NULL;
  vtkDoubleArray *vtk_edge_length = NULL;
  const real_t *metric=NULL;
  if(mesh->metric.size()>0)
    metric = &(mesh->metric[0]);
  if(metric!=NULL){
    vtk_metric = vtkDoubleArray::New();
    vtk_metric->SetNumberOfComponents(ndims*ndims);
    vtk_metric->SetNumberOfTuples(NNodes);
    vtk_metric->SetName("Metric");

    vtk_edge_length = vtkDoubleArray::New();
    vtk_edge_length->SetNumberOfComponents(1);
    vtk_edge_length->SetNumberOfTuples(NNodes);
    vtk_edge_length->SetName("mean_edge_length");
  }

  for(size_t i=0;i<NNodes;i++){
    const real_t *r = mesh->get_coords(i);
    if(vtk_psi!=NULL)
      vtk_psi->SetTuple1(i, psi[i]);
    vtk_node_numbering->SetTuple1(i, i);
    vtk_node_tpartition->SetTuple1(i, mesh->get_node_towner(i));
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
    if(vtk_edge_length!=NULL){
      int nedges=mesh->NNList[i].size();
      real_t mean_edge_length=0;
      for(typename std::deque<index_t>::const_iterator it=mesh->NNList[i].begin();it!=mesh->NNList[i].end();++it){
        Edge<real_t, index_t> edge(i, *it);
        mean_edge_length+=mesh->Edges.find(edge)->get_length();
      }
      mean_edge_length/=nedges;
      vtk_edge_length->SetTuple1(i, mean_edge_length);
    }
  }
  
  ug->SetPoints(vtk_points);
  vtk_points->Delete();

  if(vtk_psi!=NULL){
    ug->GetPointData()->AddArray(vtk_psi);
    vtk_psi->Delete();
  }

  ug->GetPointData()->AddArray(vtk_node_numbering);
  vtk_node_numbering->Delete();
  
  ug->GetPointData()->AddArray(vtk_node_tpartition);
  vtk_node_tpartition->Delete();

  if(vtk_metric!=NULL){
    ug->GetPointData()->AddArray(vtk_metric);
    vtk_metric->Delete();
  }

  if(vtk_edge_length!=NULL){
    ug->GetPointData()->AddArray(vtk_edge_length);
    vtk_edge_length->Delete();
  }

  size_t NElements = mesh->get_number_elements();

  vtkIntArray *vtk_cell_numbering = vtkIntArray::New();
  vtk_cell_numbering->SetNumberOfComponents(1);
  vtk_cell_numbering->SetNumberOfTuples(NElements);
  vtk_cell_numbering->SetName("eid");
  
  vtkIntArray *vtk_cell_tpartition = vtkIntArray::New();
  vtk_cell_tpartition->SetNumberOfComponents(1);
  vtk_cell_tpartition->SetNumberOfTuples(NElements);
  vtk_cell_tpartition->SetName("cell_partition");

  for(size_t i=0;i<NElements;i++){
    vtk_cell_numbering->SetTuple1(i, i);
    vtk_cell_tpartition->SetTuple1(i, mesh->get_element_towner(i));
    const index_t *n = mesh->get_element(i);
    assert(n[0]>=0);
    if(ndims==2){
      vtkIdType pts[] = {n[0], n[1], n[2]};
      ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
    }else{
      vtkIdType pts[] = {n[0], n[1], n[2], n[3]};
      ug->InsertNextCell(VTK_TETRA, 4, pts);
    }
  }

  ug->GetCellData()->AddArray(vtk_cell_numbering);
  vtk_cell_numbering->Delete();
  
  ug->GetCellData()->AddArray(vtk_cell_tpartition);
  vtk_cell_tpartition->Delete();
    
  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName(filename);
  writer->SetInput(ug);
  writer->Write();
  
  writer->Delete();
  ug->Delete();
  
  return;
}

template<typename real_t, typename index_t>
  void export_vtu(const char *filename, const Surface<real_t, index_t> *surface){
  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();
  
  vtkPoints *vtk_points = vtkPoints::New();
  size_t NNodes = surface->get_number_nodes();
  vtk_points->SetNumberOfPoints(NNodes);
  int ndims = surface->get_number_dimensions();
  for(size_t i=0;i<NNodes;i++){
    if(ndims==2)
      vtk_points->SetPoint(i, surface->get_x(i), surface->get_y(i), 0.0);
    else
      vtk_points->SetPoint(i, surface->get_x(i), surface->get_y(i), surface->get_z(i));
  }
  ug->SetPoints(vtk_points);
  vtk_points->Delete();

  // Need to get out the facets
  int NSElements = surface->get_number_facets();
  for(int i=0;i<NSElements;i++){
    const int *facet = surface->get_facet(i);
    if(ndims==2){
      vtkIdType pts[] = {facet[0], facet[1]};
      ug->InsertNextCell(VTK_LINE, 2, pts);
    }else{
      vtkIdType pts[] = {facet[0], facet[1], facet[2]};
      ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
    }
  }

  // Need the facet ID's
  vtkIntArray *scalar = vtkIntArray::New();
  scalar->SetNumberOfComponents(1);
  scalar->SetNumberOfTuples(NSElements);
  scalar->SetName("coplanar_ids");
  for(int i=0;i<NSElements;i++){
    scalar->SetTuple1(i, surface->get_coplanar_id(i));
  }
  ug->GetCellData()->AddArray(scalar);
  scalar->Delete();
  
  vtkDoubleArray *normal = vtkDoubleArray::New();
  normal->SetNumberOfComponents(3);
  normal->SetNumberOfTuples(NSElements);
  normal->SetName("normals");
  for(int i=0;i<NSElements;i++){
    const double *n = surface->get_normal(i);
    if(ndims==2)
      normal->SetTuple3(i, n[0], n[1], 0.0);
    else
      normal->SetTuple3(i, n[0], n[1], n[2]);
  }
  ug->GetCellData()->AddArray(normal);
  normal->Delete();

  vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
  writer->SetFileName(filename);
  writer->SetInput(ug);
  writer->Write();

  writer->Delete();
  ug->Delete();
}

#endif
