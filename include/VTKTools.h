/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Gerard Gorman
 *  Applied Modelling and Computation Group
 *  Department of Earth Science and Engineering
 *  Imperial College London
 *
 *  g.gorman@imperial.ac.uk
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials provided
 *  with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 *  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#ifndef VTK_TOOLS_H
#define VTK_TOOLS_H

#include <vtkCellType.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkIntArray.h>
#include <vtkIdTypeArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>

#include <vtkJPEGReader.h>
#include <vtkImageData.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkImageAnisotropicDiffusion2D.h>

#ifndef vtkFloatingPointType
#define vtkFloatingPointType vtkFloatingPointType
typedef float vtkFloatingPointType;
#endif

#include <vector>
#include <string>
#include <cfloat>
#include <typeinfo>

#include "Mesh.h"
#include "MetricTensor.h"
#include "ElementProperty.h"

extern "C" {
#ifdef HAVE_METIS
#include "metis.h"
#endif
}

#ifdef HAVE_MPI
#include "mpi_tools.h"
#endif

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

/*! \brief Toolkit for importing and exporting VTK files. This is
 * mostly used as part of the internal unit testing framework and
 * should not intended to be part of the public API.
 */
template<typename real_t> class VTKTools
{
public:
    static Mesh<real_t>* import_vtu(std::string filename)
    {
        vtkSmartPointer<vtkUnstructuredGrid> ug = vtkSmartPointer<vtkUnstructuredGrid>::New();

        if(filename.substr(filename.find_last_of('.'))==".pvtu") {
            vtkSmartPointer<vtkXMLPUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLPUnstructuredGridReader>::New();
            reader->SetFileName(filename.c_str());
            reader->Update();

            ug->DeepCopy(reader->GetOutput());
        } else {
            vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
            reader->SetFileName(filename.c_str());
            reader->Update();

            ug->DeepCopy(reader->GetOutput());
        }

        return import_vtu(ug);
    }

    static Mesh<real_t>* import_vtu(vtkUnstructuredGrid *ug)
    {
        int rank=0;
	int nparts=1;
#ifdef HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nparts);

        mpi_type_wrapper<index_t> mpi_index_t_wrapper;
        MPI_Datatype MPI_INDEX_T = mpi_index_t_wrapper.mpi_type;
#endif

        std::vector<real_t> x, y, z;
        std::vector<int> ENList;

        index_t NNodes, NElements;
        int nloc, ndims;
        int cell_type;
        if (rank==0) {
            NNodes = ug->GetNumberOfPoints();
            NElements = ug->GetNumberOfCells();

            cell_type = ug->GetCell(0)->GetCellType();
            if(cell_type==VTK_TRIANGLE) {
                nloc = 3;
                ndims = 2;
            } else if(cell_type==VTK_TETRA) {
                nloc = 4;
                ndims = 3;
            } else {
                std::cerr<<"ERROR("<<__FILE__<<"): unsupported element type\n";
                exit(-1);
            }

            x.reserve(NNodes);
            y.reserve(NNodes);
            z.reserve(NNodes);
            ENList.reserve(nloc*NElements);

            std::map<int, int> renumber;
            std::map<int, int> gnns;

            int ncnt=0;
            bool have_global_ids = ug->GetPointData()->GetArray("GlobalId")!=NULL;
            if(have_global_ids) {
                for(index_t i=0; i<NNodes; i++) {
                    int gnn = ug->GetPointData()->GetArray("GlobalId")->GetTuple1(i);
                    if(gnns.find(gnn)==gnns.end()) {
                        gnns[gnn] = ncnt;;
                        renumber[i] = ncnt;
                        ncnt++;

                        real_t r[3];
                        ug->GetPoints()->GetPoint(i, r);
                        x.push_back(r[0]);
                        y.push_back(r[1]);
                        z.push_back(r[2]);
                    } else {
                        renumber[i] = gnns[gnn];
                    }
                }
            } else {
                for(index_t i=0; i<NNodes; i++) {
                    real_t r[3];
                    ug->GetPoints()->GetPoint(i, r);
                    x.push_back(r[0]);
                    y.push_back(r[1]);
                    z.push_back(r[2]);
                }
            }

            NNodes = x.size();

            for(index_t i=0; i<NElements; i++) {
                int ghost=0;
                if(ug->GetCellData()->GetArray("vtkGhostLevels")!=NULL)
                    ghost = ug->GetCellData()->GetArray("vtkGhostLevels")->GetTuple1(i);

                if(ghost>0)
                    continue;

                vtkCell *cell = ug->GetCell(i);
                assert(cell->GetCellType()==cell_type);
                for(int j=0; j<nloc; j++) {
                    if(have_global_ids)
                        ENList.push_back(renumber[cell->GetPointId(j)]);
                    else
                        ENList.push_back(cell->GetPointId(j));
                }
            }
            NElements = ENList.size()/nloc;
        }

#ifdef HAVE_MPI
        MPI_Bcast(&NNodes, 1, MPI_INDEX_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&NElements, 1, MPI_INDEX_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nloc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ndims, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        if (rank!=0) {
            ENList.resize(nloc*NElements);
        }

        Mesh<real_t> *mesh=NULL;

#ifdef HAVE_MPI
        MPI_Bcast(ENList.data(), nloc*NElements, MPI_INDEX_T, 0, MPI_COMM_WORLD);

        if(nparts>1) {
            std::vector<index_t> owner_range;
            std::vector<index_t> lnn2gnn;
            std::vector<int> node_owner;

            std::vector<int> epart(NElements, 0), npart(NNodes, 0);

            if(rank==0) {
                int edgecut;

                std::vector<int> eind(NElements*nloc);
                eind[0] = 0;
                for(index_t i=0; i<NElements*nloc; i++)
                    eind[i] = ENList[i];

                int intNElements = NElements;
                int intNNodes = NNodes;

#ifdef METIS_VER_MAJOR
                int vsize = nloc - 1;
                std::vector<int> eptr(NElements+1);
                for(index_t i=0; i<NElements; i++)
                    eptr[i+1] = eptr[i]+nloc;
                METIS_PartMeshNodal(&intNElements,
                                    &intNNodes,
                                    eptr.data(),
                                    eind.data(),
                                    NULL,
                                    &vsize,
                                    &nparts,
                                    NULL,
                                    NULL,
                                    &edgecut,
                                    epart.data(),
                                    npart.data());
#else
                std::vector<int> etype(NElements, ndims-1);
                int numflag = 0;
                METIS_PartMeshNodal(&intNElements,
                                    &intNNodes,
                                    eind.data(),
                                    etype.data(),
                                    &numflag,
                                    &nparts,
                                    &edgecut,
                                    epart.data(),
                                    npart.data());
#endif
            }

            MPI_Bcast(epart.data(), NElements, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(npart.data(), NNodes, MPI_INT, 0, MPI_COMM_WORLD);

            // Separate out owned nodes.
            std::vector< std::vector<index_t> > node_partition(nparts);
            for(index_t i=0; i<NNodes; i++)
                node_partition[npart[i]].push_back(i);

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
            boost::unordered_map<index_t, index_t> renumber;
#else
            std::map<index_t, index_t> renumber;
#endif
            {
                index_t pos=0;
                owner_range.push_back(0);
                for(int i=0; i<nparts; i++) {
                    int pNNodes = node_partition[i].size();
                    owner_range.push_back(owner_range[i]+pNNodes);
                    for(int j=0; j<pNNodes; j++)
                        renumber[node_partition[i][j]] = pos++;
                }
            }
            std::vector<index_t> element_partition;
            std::set<index_t> halo_nodes;
            for(index_t i=0; i<NElements; i++) {
                std::set<index_t> residency;
                for(int j=0; j<nloc; j++)
                    residency.insert(npart[ENList[i*nloc+j]]);

                if(residency.count(rank)) {
                    element_partition.push_back(i);

                    for(int j=0; j<nloc; j++) {
                        index_t nid = ENList[i*nloc+j];
                        if(npart[nid]!=rank)
                            halo_nodes.insert(nid);
                    }
                }
            }

            // Append halo nodes to local node partition.
            for(typename std::set<index_t>::const_iterator it=halo_nodes.begin(); it!=halo_nodes.end(); ++it) {
                node_partition[rank].push_back(*it);
            }

            // Global numbering to partition numbering look up table.
            int lNNodes = node_partition[rank].size();

            lnn2gnn.resize(lNNodes);
            node_owner.resize(lNNodes);
            for(index_t i=0; i<lNNodes; i++) {
                index_t nid = node_partition[rank][i];
                index_t gnn = renumber[nid];
                lnn2gnn[i] = gnn;
                node_owner[i] = npart[nid];
            }

            // Construct local mesh.
            NElements = element_partition.size();
            std::vector<index_t> lENList(NElements*nloc);
            for(index_t i=0; i<NElements; i++) {
                for(int j=0; j<nloc; j++) {
                    index_t nid = renumber[ENList[element_partition[i]*nloc+j]];
                    lENList[i*nloc+j] = nid;
                }
            }
            ENList.clear();
            ENList.swap(lENList);

            // Construct local x
            x.resize(NNodes);
            MPI_Bcast(x.data(), NNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            std::vector<real_t> lx(lNNodes);
            for(index_t i=0; i<lNNodes; i++)
                lx[i] = x[node_partition[rank][i]];
            x.clear();
            x.swap(lx);

            // Construct local y.
            y.resize(NNodes);
            MPI_Bcast(y.data(), NNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            std::vector<real_t> ly(lNNodes);
            for(index_t i=0; i<lNNodes; i++)
                ly[i] = y[node_partition[rank][i]];
            y.clear();
            y.swap(ly);

            if(ndims==3) {
                z.resize(NNodes);
                MPI_Bcast(z.data(), NNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                std::vector<real_t> lz(lNNodes);
                for(index_t i=0; i<lNNodes; i++)
                    lz[i] = z[node_partition[rank][i]];
                z.clear();
                z.swap(lz);
            }

            NNodes = lNNodes;

            MPI_Comm comm = MPI_COMM_WORLD;

            if(ndims==2)
                mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(lnn2gnn[0]), &(owner_range[0]), comm);
            else
                mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]), &(lnn2gnn[0]), &(owner_range[0]), comm);
        }
#endif

        if(nparts==1) { // If nparts!=1, then the mesh has been created already by the code a few lines above.
            if(ndims==2)
                mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
            else
                mesh = new Mesh<real_t>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(z[0]));
        }

        return mesh;
    }

    static void export_vtu(const char *basename, const Mesh<real_t> *mesh, const real_t *psi=NULL)
    {
        index_t NElements = mesh->get_number_elements();
        index_t ndims = mesh->get_number_dimensions();

        // Set the orientation of elements.
        ElementProperty<real_t> *property = NULL;
        for(index_t i=0; i<NElements; i++) {
            const int *n=mesh->get_element(i);
            assert(n[0]>=0);

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
        vtkSmartPointer<vtkUnstructuredGrid> ug = vtkSmartPointer<vtkUnstructuredGrid>::New();

        vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
        index_t NNodes = mesh->get_number_nodes();
        vtk_points->SetNumberOfPoints(NNodes);

        vtkSmartPointer<vtkDoubleArray> vtk_psi = NULL;

        if(psi!=NULL) {
            vtk_psi = vtkSmartPointer<vtkDoubleArray>::New();
            vtk_psi->SetNumberOfComponents(1);
            vtk_psi->SetNumberOfTuples(NNodes);
            vtk_psi->SetName("psi");
        }

        vtkSmartPointer<vtkIntArray> vtk_node_numbering = vtkSmartPointer<vtkIntArray>::New();
        vtk_node_numbering->SetNumberOfComponents(1);
        vtk_node_numbering->SetNumberOfTuples(NNodes);
        vtk_node_numbering->SetName("nid");

        vtkSmartPointer<vtkDoubleArray> vtk_metric = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_metric->SetNumberOfComponents(ndims*ndims);
        vtk_metric->SetNumberOfTuples(NNodes);
        vtk_metric->SetName("Metric");

        vtkSmartPointer<vtkDoubleArray> vtk_edge_length = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_edge_length->SetNumberOfComponents(1);
        vtk_edge_length->SetNumberOfTuples(NNodes);
        vtk_edge_length->SetName("mean_edge_length");

        vtkSmartPointer<vtkDoubleArray> vtk_max_desired_length = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_max_desired_length->SetNumberOfComponents(1);
        vtk_max_desired_length->SetNumberOfTuples(NNodes);
        vtk_max_desired_length->SetName("max_desired_edge_length");

        vtkSmartPointer<vtkDoubleArray> vtk_min_desired_length = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_min_desired_length->SetNumberOfComponents(1);
        vtk_min_desired_length->SetNumberOfTuples(NNodes);
        vtk_min_desired_length->SetName("min_desired_edge_length");

        #pragma omp parallel for
        for(index_t i=0; i<NNodes; i++) {
            const real_t *r = mesh->get_coords(i);
            const double *m = mesh->get_metric(i);

            if(vtk_psi!=NULL)
                vtk_psi->SetTuple1(i, psi[i]);
            vtk_node_numbering->SetTuple1(i, i);
            if(ndims==2) {
                vtk_points->SetPoint(i, r[0], r[1], 0.0);
                vtk_metric->SetTuple4(i,
                                      m[0], m[1],
                                      m[1], m[2]);
            } else {
                vtk_points->SetPoint(i, r[0], r[1], r[2]);
                vtk_metric->SetTuple9(i,
                                      m[0], m[1], m[2],
                                      m[1], m[3], m[4],
                                      m[2], m[4], m[5]);
            }
            int nedges=mesh->NNList[i].size();
            real_t mean_edge_length=0;
            real_t max_desired_edge_length=0;
            real_t min_desired_edge_length=DBL_MAX;

            if(ndims==2) {
                MetricTensor<double,2> M(m, false);
                double maxL = M.max_length();
                double minL = M.min_length();
                for(typename std::vector<index_t>::const_iterator it=mesh->NNList[i].begin(); it!=mesh->NNList[i].end(); ++it) {
                    double length = mesh->calc_edge_length(i, *it);
                    mean_edge_length += length;

                    max_desired_edge_length = std::max(max_desired_edge_length, maxL);
                    min_desired_edge_length = std::min(min_desired_edge_length, minL);
                }
            }
            else if(ndims==3) {
                MetricTensor<double,3> M(m, false);
                double maxL = M.max_length();
                double minL = M.min_length();

                for(typename std::vector<index_t>::const_iterator it=mesh->NNList[i].begin(); it!=mesh->NNList[i].end(); ++it) {
                    double length = mesh->calc_edge_length(i, *it);
                    mean_edge_length += length;

                    max_desired_edge_length = std::max(max_desired_edge_length, maxL);
                    min_desired_edge_length = std::min(min_desired_edge_length, minL);
                }
            }

            mean_edge_length/=nedges;
            vtk_edge_length->SetTuple1(i, mean_edge_length);
            vtk_max_desired_length->SetTuple1(i, max_desired_edge_length);
            vtk_min_desired_length->SetTuple1(i, min_desired_edge_length);
        }

        ug->SetPoints(vtk_points);
        if(vtk_psi!=NULL) {
            ug->GetPointData()->AddArray(vtk_psi);
        }
        ug->GetPointData()->AddArray(vtk_node_numbering);
        ug->GetPointData()->AddArray(vtk_metric);
        ug->GetPointData()->AddArray(vtk_edge_length);
        ug->GetPointData()->AddArray(vtk_max_desired_length);
        ug->GetPointData()->AddArray(vtk_min_desired_length);

        // Create a point data array to illustrate the boundary nodes.
        vtkSmartPointer<vtkIntArray> vtk_boundary_nodes = vtkSmartPointer<vtkIntArray>::New();
        vtk_boundary_nodes->SetNumberOfComponents(1);
        vtk_boundary_nodes->SetNumberOfTuples(NNodes);
        vtk_boundary_nodes->SetName("BoundaryNodes");

        std::vector<int> boundary_nodes(NNodes, 0);
        for(index_t i=0; i<NElements; i++) {
            const int *n=mesh->get_element(i);
            if(n[0]==-1)
                continue;

            if(ndims==2) {
                for(int j=0; j<3; j++) {
                    boundary_nodes[n[(j+1)%3]] = std::max(boundary_nodes[n[(j+1)%3]], mesh->boundary[i*3+j]);
                    boundary_nodes[n[(j+2)%3]] = std::max(boundary_nodes[n[(j+2)%3]], mesh->boundary[i*3+j]);
                }
            } else {
                for(int j=0; j<4; j++) {
                    boundary_nodes[n[(j+1)%4]] = std::max(boundary_nodes[n[(j+1)%4]], mesh->boundary[i*4+j]);
                    boundary_nodes[n[(j+2)%4]] = std::max(boundary_nodes[n[(j+2)%4]], mesh->boundary[i*4+j]);
                    boundary_nodes[n[(j+3)%4]] = std::max(boundary_nodes[n[(j+3)%4]], mesh->boundary[i*4+j]);
                }
            }
        }
        for(index_t i=0; i<NNodes; i++)
            vtk_boundary_nodes->SetTuple1(i, boundary_nodes[i]);
        ug->GetPointData()->AddArray(vtk_boundary_nodes);

        vtkSmartPointer<vtkIntArray> vtk_cell_numbering = vtkSmartPointer<vtkIntArray>::New();
        vtk_cell_numbering->SetNumberOfComponents(1);
        vtk_cell_numbering->SetNumberOfTuples(NElements);
        vtk_cell_numbering->SetName("eid");

        vtkSmartPointer<vtkIntArray> vtk_boundary = vtkSmartPointer<vtkIntArray>::New();
        if(ndims==2)
            vtk_boundary->SetNumberOfComponents(3);
        else
            vtk_boundary->SetNumberOfComponents(4);
        vtk_boundary->SetNumberOfTuples(NElements);
        vtk_boundary->SetName("Boundary");

        vtkSmartPointer<vtkDoubleArray> vtk_quality = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_quality->SetNumberOfComponents(1);
        vtk_quality->SetNumberOfTuples(NElements);
        vtk_quality->SetName("quality");

        for(index_t i=0, k=0; i<NElements; i++) {
            const index_t *n = mesh->get_element(i);
            assert(n[0]>=0);

            if(ndims==2) {
                vtkIdType pts[] = {n[0], n[1], n[2]};
                ug->InsertNextCell(VTK_TRIANGLE, 3, pts);
                vtk_boundary->SetTuple3(i, mesh->boundary[i*3], mesh->boundary[i*3+1], mesh->boundary[i*3+2]);

                vtk_quality->SetTuple1(k, property->lipnikov(mesh->get_coords(n[0]), mesh->get_coords(n[1]), mesh->get_coords(n[2]),
                                       mesh->get_metric(n[0]), mesh->get_metric(n[1]), mesh->get_metric(n[2])));
            } else {
                vtkIdType pts[] = {n[0], n[1], n[2], n[3]};
                ug->InsertNextCell(VTK_TETRA, 4, pts);
                vtk_boundary->SetTuple4(i, mesh->boundary[i*4], mesh->boundary[i*4+1], mesh->boundary[i*4+2], mesh->boundary[i*4+3]);

                vtk_quality->SetTuple1(k, property->lipnikov(mesh->get_coords(n[0]), mesh->get_coords(n[1]), mesh->get_coords(n[2]), mesh->get_coords(n[3]),
                                       mesh->get_metric(n[0]), mesh->get_metric(n[1]), mesh->get_metric(n[2]), mesh->get_metric(n[3])));
            }

            vtk_cell_numbering->SetTuple1(k, i);

            k++;
        }

        ug->GetCellData()->AddArray(vtk_cell_numbering);
        ug->GetCellData()->AddArray(vtk_quality);
        ug->GetCellData()->AddArray(vtk_boundary);

        int rank=0, nparts=1;
#ifdef HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nparts);
#endif

        if(nparts==1) {
            vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
            std::string filename = std::string(basename)+std::string(".vtu");
            writer->SetFileName(filename.c_str());
#if VTK_MAJOR_VERSION < 6
            writer->SetInput(ug);
#else
            writer->SetInputData(ug);
#endif
            writer->Write();
        }
#ifdef HAVE_MPI
        else {
            // Set ghost levels
            vtkSmartPointer<vtkUnsignedCharArray> vtk_ghost = vtkSmartPointer<vtkUnsignedCharArray>::New();
            vtk_ghost->SetNumberOfComponents(1);
            vtk_ghost->SetNumberOfTuples(NElements);
            vtk_ghost->SetName("vtkGhostLevels");

            for(index_t i=0; i<NElements; i++) {
                const index_t *n = mesh->get_element(i);
                int owner;
                if(ndims==2)
                    owner=std::min(mesh->node_owner[n[0]], std::min(mesh->node_owner[n[1]], mesh->node_owner[n[2]]));
                else
                    owner=std::min(std::min(mesh->node_owner[n[0]], mesh->node_owner[n[1]]), std::min(mesh->node_owner[n[2]], mesh->node_owner[n[3]]));

                if(owner==rank) {
                    vtk_ghost->SetTuple1(i, 0);
                } else {
                    vtk_ghost->SetTuple1(i, 1);
                }
            }
            ug->GetCellData()->AddArray(vtk_ghost);

            // Set GlobalIds
            vtkSmartPointer<vtkIdTypeArray> vtk_gnn = vtkSmartPointer<vtkIdTypeArray>::New();
            vtk_gnn->SetNumberOfComponents(1);
            vtk_gnn->SetNumberOfTuples(NNodes);
            vtk_gnn->SetName("GlobalId");

            for(index_t i=0; i<NNodes; i++) {
                vtk_gnn->SetTuple1(i, mesh->lnn2gnn[i]);
            }
            // ug->GetPointData()->AddArray(vtk_gnn);
            // ug->GetPointData()->SetActiveGlobalIds("GlobalId");
            ug->GetPointData()->SetGlobalIds(vtk_gnn);

            vtkSmartPointer<vtkXMLPUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
            std::string filename = std::string(basename)+std::string(".pvtu");
            writer->SetFileName(filename.c_str());
            writer->SetNumberOfPieces(nparts);
            writer->SetGhostLevel(1);
            writer->SetStartPiece(rank);
            writer->SetEndPiece(rank);
#if VTK_MAJOR_VERSION < 6
            writer->SetInput(ug);
#else
            writer->SetInputData(ug);
#endif
            writer->Write();
        }
#endif

        delete property;

        return;
    }

};
#endif
