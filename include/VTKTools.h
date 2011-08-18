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

#include <vtkCellType.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#ifndef vtkFloatingPointType
#define vtkFloatingPointType vtkFloatingPointType
typedef float vtkFloatingPointType;
#endif

#include <vector>
#include <string>
#include <cfloat>

#include "Mesh.h"
#include "MetricTensor.h"
#include "Surface.h"
#include "Metis.h"
#include "ElementProperty.h"


#ifdef HAVE_MPI
#include <mpi.h>
#endif

template<typename real_t, typename index_t> class VTKTools{
 public:
  static Mesh<real_t, index_t>* import_vtu(const char *filename){
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
    
    int nparts=1;
    Mesh<real_t, index_t> *mesh=NULL;
    std::vector<index_t> owner_range;
    std::vector<index_t> lnn2gnn;
    {
#ifdef HAVE_MPI
      // Handle mpi parallel run.
      if(MPI::Is_initialized()){
        nparts = MPI::COMM_WORLD.Get_size();
      }

      if(nparts>1){
        std::vector<idxtype> epart(NElements, 0), npart(NNodes, 0);
        int rank = MPI::COMM_WORLD.Get_rank();
        if(rank==0){
          int numflag = 0, edgecut;
          int etype = 1; // triangles
          if(ndims==3)
            etype = 2; // tetrahedra

          std::vector<idxtype> metis_ENList(NElements*nloc);
          for(size_t i=0;i<NElements*nloc;i++)
            metis_ENList[i] = ENList[i];
          int intNElements = NElements;
          int intNNodes = NNodes;
          METIS_PartMeshNodal(&intNElements, &intNNodes, &(metis_ENList[0]), &etype,
                              &numflag, &nparts, &edgecut, &(epart[0]), &(npart[0]));
        }
        
        // This is a bug right here if idxtype is not of size int.
        MPI::COMM_WORLD.Bcast(&(epart[0]), NElements, MPI_INT, 0);
        MPI::COMM_WORLD.Bcast(&(npart[0]), NNodes, MPI_INT, 0);
        
        // Seperate out owned nodes.
        std::vector< std::deque<index_t> > node_partition(nparts);
        for(size_t i=0;i<NNodes;i++)
          node_partition[npart[i]].push_back(i);

        std::map<index_t, index_t> renumber;
        {
          index_t pos=0;
          owner_range.push_back(0);
          for(int i=0;i<nparts;i++){
            int pNNodes = node_partition[i].size();
            owner_range.push_back(owner_range[i]+pNNodes);
            for(int j=0;j<pNNodes;j++)
              renumber[node_partition[i][j]] = pos++;
          }
        }
        std::deque<index_t> element_partition;
        std::set<index_t> halo_nodes;
        for(size_t i=0;i<NElements;i++){
          std::set<index_t> residency;
          for(int j=0;j<nloc;j++)
            residency.insert(npart[ENList[i*nloc+j]]);
          
          if(residency.count(rank)){
            element_partition.push_back(i);
            
            for(int j=0;j<nloc;j++){
              index_t nid = ENList[i*nloc+j];
              if(npart[nid]!=rank)
                halo_nodes.insert(nid);
            }
          }
        }
        
        // Append halo nodes to local node partition.
        for(typename std::set<index_t>::const_iterator it=halo_nodes.begin();it!=halo_nodes.end();++it){
          node_partition[rank].push_back(*it);
        }
        
        // Global numbering to partition numbering look up table.
        NNodes = node_partition[rank].size();
        std::map<index_t, index_t> gnn2lnn;
        lnn2gnn.resize(NNodes);
        for(size_t i=0;i<NNodes;i++){
          index_t gnn = renumber[node_partition[rank][i]];
          gnn2lnn[gnn] = i;
          lnn2gnn[i] = gnn;
        }
        
        // Construct local mesh.
        std::vector<real_t> lx(NNodes), ly(NNodes), lz(NNodes);
        for(size_t i=0;i<NNodes;i++){
          lx[i] = x[node_partition[rank][i]];
          ly[i] = y[node_partition[rank][i]];
          if(ndims==3)
            lz[i] = z[node_partition[rank][i]];
        }
        
        NElements = element_partition.size();
        std::vector<index_t> lENList(NElements*nloc);
        for(size_t i=0;i<NElements;i++){
          for(int j=0;j<nloc;j++){
            index_t nid = renumber[ENList[element_partition[i]*nloc+j]];
            lENList[i*nloc+j] = nid;
          }
        }
        
        // Swap
        x.swap(lx);
        y.swap(ly);
        if(ndims==3)
          z.swap(lz);
        ENList.swap(lENList);
      }
#endif
    }

#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm comm = MPI_COMM_WORLD;
      if(ndims==2)
        mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]),
                                         &(lnn2gnn[0]), &(owner_range[0]), comm);
      else
        mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]),
                                         &(lnn2gnn[0]), &(owner_range[0]), comm);
    }else{
      if(ndims==2)
        mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
      else
        mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));
    }
#else
    if(ndims==2)
      mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
    else
      mesh = new Mesh<real_t, index_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));
#endif

    return mesh;
  }
  
  static void export_vtu(const char *basename, const Mesh<real_t, index_t> *mesh, const real_t *psi=NULL){
    size_t NElements = mesh->get_number_elements();
    size_t ndims = mesh->get_number_dimensions();
    
    // Set the orientation of elements.
    ElementProperty<real_t> *property = NULL;
    for(size_t i=0;i<NElements;i++){
      const int *n=mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      if(ndims==2)
        property = new ElementProperty<real_t>(mesh->get_coords(n[0]),
                                               mesh->get_coords(n[1]),
                                               mesh->get_coords(n[2]));
      else
        property = new ElementProperty<real_t>(mesh->get_coords(n[0]),
                                               mesh->get_coords(n[1]),
                                               mesh->get_coords(n[2]),
                                               mesh->get_coords(n[3]));
      break;
    }

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

    vtkDoubleArray *vtk_metric = vtkDoubleArray::New();
    vtk_metric->SetNumberOfComponents(ndims*ndims);
    vtk_metric->SetNumberOfTuples(NNodes);
    vtk_metric->SetName("Metric");

    vtkDoubleArray *vtk_edge_length = vtkDoubleArray::New();
    vtk_edge_length->SetNumberOfComponents(1);
    vtk_edge_length->SetNumberOfTuples(NNodes);
    vtk_edge_length->SetName("mean_edge_length");

    vtkDoubleArray *vtk_max_desired_length = vtkDoubleArray::New();
    vtk_max_desired_length->SetNumberOfComponents(1);
    vtk_max_desired_length->SetNumberOfTuples(NNodes);
    vtk_max_desired_length->SetName("max_desired_edge_length");

    vtkDoubleArray *vtk_min_desired_length = vtkDoubleArray::New();
    vtk_min_desired_length->SetNumberOfComponents(1);
    vtk_min_desired_length->SetNumberOfTuples(NNodes);
    vtk_min_desired_length->SetName("min_desired_edge_length");

    for(size_t i=0;i<NNodes;i++){
      const real_t *r = mesh->get_coords(i);
      const real_t *m = mesh->get_metric(i);

      if(vtk_psi!=NULL)
        vtk_psi->SetTuple1(i, psi[i]);
      vtk_node_numbering->SetTuple1(i, i);
      vtk_node_tpartition->SetTuple1(i, mesh->get_node_towner(i));
      if(ndims==2){
        vtk_points->SetPoint(i, r[0], r[1], 0.0);
        vtk_metric->SetTuple4(i,
                              m[0], m[1],
                              m[2], m[3]);
      }else{
        vtk_points->SetPoint(i, r[0], r[1], r[2]);
        vtk_metric->SetTuple9(i,
                              m[0], m[1], m[2],
                              m[3], m[4], m[5],
                              m[6], m[7], m[8]); 
      }
      int nedges=mesh->NNList[i].size();
      real_t mean_edge_length=0;
      real_t max_desired_edge_length=0;
      real_t min_desired_edge_length=DBL_MAX;
      for(typename std::deque<index_t>::const_iterator it=mesh->NNList[i].begin();it!=mesh->NNList[i].end();++it){
        Edge<real_t, index_t> edge(i, *it);
        mean_edge_length+=mesh->Edges.find(edge)->get_length();
        
        MetricTensor<real_t> M(ndims, m);
        max_desired_edge_length = std::max(max_desired_edge_length, M.max_length());
        min_desired_edge_length = std::min(min_desired_edge_length, M.min_length());
      }
      mean_edge_length/=nedges;
      vtk_edge_length->SetTuple1(i, mean_edge_length);
      vtk_max_desired_length->SetTuple1(i, max_desired_edge_length);
      vtk_min_desired_length->SetTuple1(i, min_desired_edge_length);
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

    ug->GetPointData()->AddArray(vtk_metric);
    vtk_metric->Delete();
    
    ug->GetPointData()->AddArray(vtk_edge_length);
    vtk_edge_length->Delete();

    ug->GetPointData()->AddArray(vtk_max_desired_length);
    vtk_max_desired_length->Delete();

    ug->GetPointData()->AddArray(vtk_min_desired_length);
    vtk_min_desired_length->Delete();

    vtkIntArray *vtk_cell_numbering = vtkIntArray::New();
    vtk_cell_numbering->SetNumberOfComponents(1);
    vtk_cell_numbering->SetNumberOfTuples(NElements);
    vtk_cell_numbering->SetName("eid");
  
    vtkIntArray *vtk_cell_tpartition = vtkIntArray::New();
    vtk_cell_tpartition->SetNumberOfComponents(1);
    vtk_cell_tpartition->SetNumberOfTuples(NElements);
    vtk_cell_tpartition->SetName("cell_partition");

    vtkDoubleArray *vtk_quality = vtkDoubleArray::New();
    vtk_quality->SetNumberOfComponents(1);
    vtk_quality->SetNumberOfTuples(NElements);
    vtk_quality->SetName("quality");

    for(size_t i=0;i<NElements;i++){
      vtk_cell_numbering->SetTuple1(i, i);
      vtk_cell_tpartition->SetTuple1(i, mesh->get_element_towner(i));
      const index_t *n = mesh->get_element(i);
      assert(n[0]>=0);
      if(ndims==2){
        vtkIdType pts[] = {n[0], n[1], n[2]};
        ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
        
        vtk_quality->SetTuple1(i, property->lipnikov(mesh->get_coords(n[0]), mesh->get_coords(n[1]), mesh->get_coords(n[2]), 
                                                     mesh->get_metric(n[0]), mesh->get_metric(n[1]), mesh->get_metric(n[2])));
      }else{
        vtkIdType pts[] = {n[0], n[1], n[2], n[3]};
        ug->InsertNextCell(VTK_TETRA, 4, pts);
        
        vtk_quality->SetTuple1(i, property->lipnikov(mesh->get_coords(n[0]), mesh->get_coords(n[1]), mesh->get_coords(n[2]), mesh->get_coords(n[3]), 
                                                     mesh->get_metric(n[0]), mesh->get_metric(n[1]), mesh->get_metric(n[2]), mesh->get_metric(n[3])));
      }
    }

    ug->GetCellData()->AddArray(vtk_cell_numbering);
    vtk_cell_numbering->Delete();
  
    ug->GetCellData()->AddArray(vtk_cell_tpartition);
    vtk_cell_tpartition->Delete();

    ug->GetCellData()->AddArray(vtk_quality);
    vtk_quality->Delete();
    
    int nparts=1;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      nparts = MPI::COMM_WORLD.Get_size();
    }
#endif
    if(nparts==1){
      vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
      std::string filename = std::string(basename)+std::string(".vtu");
      writer->SetFileName(filename.c_str());
      writer->SetInput(ug);
      writer->Write();
      
      writer->Delete();
    }else{
#ifdef HAVE_MPI
      int rank = MPI::COMM_WORLD.Get_rank();
      int nparts = MPI::COMM_WORLD.Get_size();
      
      vtkXMLPUnstructuredGridWriter *writer = vtkXMLPUnstructuredGridWriter::New();
      std::string filename = std::string(basename)+std::string(".pvtu");
      writer->SetFileName(filename.c_str());
      writer->SetNumberOfPieces(nparts);
      writer->SetGhostLevel(1);
      writer->SetStartPiece(rank);
      writer->SetEndPiece(rank);
      writer->SetInput(ug);
      writer->Write();
      writer->Delete();
#endif
    }
    ug->Delete();
    delete property;

    return;
  }

  static void export_vtu(const char *basename, const Surface<real_t, index_t> *surface){
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

    int nparts=1;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      nparts = MPI::COMM_WORLD.Get_size();
    }
#endif
    if(nparts==1){
      vtkXMLUnstructuredGridWriter *writer = vtkXMLUnstructuredGridWriter::New();
      std::string filename = std::string(basename)+std::string(".vtu");
      writer->SetFileName(filename.c_str());
      writer->SetInput(ug);
      writer->Write();
      
      writer->Delete();
    }else{
#ifdef HAVE_MPI
      int rank = MPI::COMM_WORLD.Get_rank();
      int nparts = MPI::COMM_WORLD.Get_size();
      
      vtkXMLPUnstructuredGridWriter *writer = vtkXMLPUnstructuredGridWriter::New();
      std::string filename = std::string(basename)+std::string(".pvtu");
      writer->SetFileName(filename.c_str());
      writer->SetNumberOfPieces(nparts);
      writer->SetGhostLevel(1);
      writer->SetStartPiece(rank);
      writer->SetEndPiece(rank);
      writer->SetInput(ug);
      writer->Write();
      writer->Delete();
#endif
    }
    
    ug->Delete();
  }
};
#endif
