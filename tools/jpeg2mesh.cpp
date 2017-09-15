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

#include <iostream>

#include <getopt.h>

#include "Mesh.h"
#ifdef HAVE_VTK
#include "VTKTools.h"
#endif
#include "MetricField.h"

#include "Coarsen.h"
#include "Smooth.h"
#include "Swapping.h"
#include "ticker.h"

#include <mpi.h>

void usage(char *cmd)
{
    std::cout<<"Usage: "<<cmd<<" [options] infile\n"
             <<"\nOptions:\n"
             <<" -h, --help\n\tHelp! Prints this message.\n"
             <<" -v, --verbose\n\tVerbose output.\n"
             <<" -c factor, --compression factor\n\tCurrently an art...\n";
    return;
}

int parse_arguments(int argc, char **argv, std::string &infilename, bool &verbose, double &factor)
{

    // Set defaults
    verbose = false;
    factor = 1.0;

    if(argc==1) {
        usage(argv[0]);
        exit(0);
    }

    struct option longOptions[] = {
        {"help",    0,                 0, 'h'},
        {"verbose", 0,                 0, 'v'},
        {"compression", optional_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    int optionIndex = 0;
    int c;
    const char *shortopts = "hvc:";

    // Set opterr to nonzero to make getopt print error messages
    opterr=1;
    while (true) {
        c = getopt_long(argc, argv, shortopts, longOptions, &optionIndex);

        if (c == -1) break;

        switch (c) {
        case 'h':
            usage(argv[0]);
            break;
        case 'v':
            verbose = true;
            break;
        case 'c':
            factor = atof(optarg);
            break;
        case '?':
            // missing argument only returns ':' if the option string starts with ':'
            // but this seems to stop the printing of error messages by getopt?
            std::cerr<<"ERROR: unknown option or missing argument\n";
            usage(argv[0]);
            exit(-1);
        case ':':
            std::cerr<<"ERROR: missing argument\n";
            usage(argv[0]);
            exit(-1);
        default:
            // unexpected:
            std::cerr<<"ERROR: getopt returned unrecognized character code\n";
            exit(-1);
        }
    }

    infilename = std::string(argv[argc-1]);

    return 0;
}

void cout_quality(const Mesh<double> *mesh, std::string operation)
{
    double qmean = mesh->get_qmean();
    double qmin = mesh->get_qmin();

    int rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0)
        std::cout<<operation<<": step in quality (mean, min): ("<<qmean<<", "<<qmin<<")"<<std::endl;
}

int main(int argc, char **argv)
{
    int rank = 0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc==1) {
        usage(argv[0]);
        exit(-1);
    }

    std::string infilename, outfilename;
    bool verbose=false;
    double factor=1.0;

    parse_arguments(argc, argv, infilename, verbose, factor);

    // Read in image
#ifdef HAVE_VTK
    vtkSmartPointer<vtkJPEGReader> reader = vtkSmartPointer<vtkJPEGReader>::New();
    reader->SetFileName(infilename.c_str());
    vtkSmartPointer<vtkImageGaussianSmooth> gsmooth = vtkSmartPointer<vtkImageGaussianSmooth>::New();
    gsmooth->SetStandardDeviation(9);
    gsmooth->SetDimensionality(2);
#if VTK_MAJOR_VERSION < 6
    gsmooth->SetInput(reader->GetOutput());
#else
    gsmooth->SetInputConnection(reader->GetOutputPort());
#endif
    gsmooth->Update();

    // Convert image to triangulated unstructured grid
    vtkSmartPointer<vtkDataSetTriangleFilter> image2ug = vtkSmartPointer<vtkDataSetTriangleFilter>::New();
#if VTK_MAJOR_VERSION < 6
    image2ug->SetInput(gsmooth->GetOutput());
#else
    image2ug->SetInputConnection(gsmooth->GetOutputPort());
#endif

    vtkUnstructuredGrid *ug = image2ug->GetOutput();
#if VTK_MAJOR_VERSION < 6
    ug->Update();
#endif

    size_t NNodes = ug->GetNumberOfPoints();
    std::vector<double> x(NNodes),y(NNodes), imageR(NNodes), imageG(NNodes), imageB(NNodes);
    for(size_t i=0; i<NNodes; i++) {
        double r[3];
        ug->GetPoints()->GetPoint(i, r);
        x[i] = r[0];
        y[i] = r[1];

        const double *tuple = ug->GetPointData()->GetArray(0)->GetTuple(i);

        imageR[i] = tuple[0];
        imageG[i] = tuple[1];
        imageB[i] = tuple[2];
    }

    int cell_type = ug->GetCell(0)->GetCellType();
    assert(cell_type==VTK_TRIANGLE);

    int nloc = 3;
    int ndims = 2;

    size_t NElements = ug->GetNumberOfCells();

    std::vector<int> ENList(NElements*nloc);
    for(size_t i=0; i<NElements; i++) {
        vtkCell *cell = ug->GetCell(i);
        assert(cell->GetCellType()==cell_type);

        for(int j=0; j<nloc; j++) {
            ENList[i*nloc+j] = cell->GetPointId(j);
        }
    }

    int nparts=1;
    Mesh<double> *mesh=NULL;

    // Handle mpi parallel run.
    MPI_Comm_size(MPI_COMM_WORLD, &nparts);

    if(nparts>1) {
        std::vector<index_t> owner_range;
        std::vector<index_t> lnn2gnn;
        std::vector<int> node_owner;

        std::vector<int> epart(NElements, 0), npart(NNodes, 0);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(rank==0) {
            int edgecut;

            std::vector<int> eind(NElements*nloc);
            eind[0] = 0;
            for(size_t i=0; i<NElements*nloc; i++)
                eind[i] = ENList[i];

            int intNElements = NElements;
            int intNNodes = NNodes;

#ifdef METIS_VER_MAJOR
            int vsize = nloc - 1;
            std::vector<int> eptr(NElements+1);
            for(size_t i=0; i<NElements; i++)
                eptr[i+1] = eptr[i]+nloc;
            METIS_PartMeshNodal(&intNElements,
                                &intNNodes,
                                &(eptr[0]),
                                &(eind[0]),
                                NULL,
                                &vsize,
                                &nparts,
                                NULL,
                                NULL,
                                &edgecut,
                                &(epart[0]),
                                &(npart[0]));
#else
            std::vector<int> etype(NElements);
            for(size_t i=0; i<NElements; i++)
                etype[i] = ndims-1;
            int numflag = 0;
            METIS_PartMeshNodal(&intNElements,
                                &intNNodes,
                                &(eind[0]),
                                &(etype[0]),
                                &numflag,
                                &nparts,
                                &edgecut,
                                &(epart[0]),
                                &(npart[0]));
#endif
        }

        mpi_type_wrapper<index_t> mpi_index_t_wrapper;
        MPI_Datatype MPI_INDEX_T = mpi_index_t_wrapper.mpi_type;

        MPI_Bcast(&(epart[0]), NElements, MPI_INDEX_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(npart[0]), NNodes, MPI_INDEX_T, 0, MPI_COMM_WORLD);

        // Separate out owned nodes.
        std::vector< std::vector<index_t> > node_partition(nparts);
        for(size_t i=0; i<NNodes; i++)
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
        for(size_t i=0; i<NElements; i++) {
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
        NNodes = node_partition[rank].size();
        lnn2gnn.resize(NNodes);
        node_owner.resize(NNodes);
        for(size_t i=0; i<NNodes; i++) {
            index_t nid = node_partition[rank][i];
            index_t gnn = renumber[nid];
            lnn2gnn[i] = gnn;
            node_owner[i] = npart[nid];
        }

        // Construct local mesh.
        std::vector<double> lx(NNodes), ly(NNodes), lz(NNodes), limageR(NNodes), limageG(NNodes), limageB(NNodes);
        for(size_t i=0; i<NNodes; i++) {
            lx[i] = x[node_partition[rank][i]];
            ly[i] = y[node_partition[rank][i]];

            limageR[i] = imageR[node_partition[rank][i]];
            limageG[i] = imageG[node_partition[rank][i]];
            limageB[i] = imageB[node_partition[rank][i]];
        }

        NElements = element_partition.size();
        std::vector<index_t> lENList(NElements*nloc);
        for(size_t i=0; i<NElements; i++) {
            for(int j=0; j<nloc; j++) {
                index_t nid = renumber[ENList[element_partition[i]*nloc+j]];
                lENList[i*nloc+j] = nid;
            }
        }

        // Swap
        x.swap(lx);
        y.swap(ly);
        imageR.swap(limageR);
        imageG.swap(limageG);
        imageB.swap(limageB);
        ENList.swap(lENList);

        MPI_Comm comm = MPI_COMM_WORLD;

        int pNNodes = node_partition[rank].size();
        mesh = new Mesh<double>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(lnn2gnn[0]), pNNodes, comm);
    } else
    {
        mesh = new Mesh<double>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
    }

    mesh->create_boundary();

    MetricField<double, 2> metric_field(*mesh);

    double time_metric = get_wtime();
    metric_field.add_field(imageR.data(), 5.0, 1);
    std::cout<<"Predicted number of elementsR: "<<metric_field.predict_nelements()<<std::endl;
    // metric_field.add_field(imageG.data(), 5.0, 3);
    // std::cout<<"Predicted number of elementsG: "<<metric_field.predict_nelements()<<std::endl;
    // metric_field.add_field(imageB.data(), 1.0, 3);
    // std::cout<<"Predicted number of elementsB: "<<metric_field.predict_nelements()<<std::endl;

    metric_field.apply_min_edge_length(0.5);
    std::cout<<"Predicted number of elements1: "<<metric_field.predict_nelements()<<std::endl;

    // metric_field.apply_max_aspect_ratio(10.0);
    // std::cout<<"Predicted number of elements1: "<<metric_field.predict_nelements()<<std::endl;

    // metric_field.apply_max_nelements(std::max(2, (int)NElements/2));
    // std::cout<<"Predicted number of elements2: "<<metric_field.predict_nelements()<<std::endl;
    // metric_field.apply_max_edge_length(10.0);
    // std::cout<<"Predicted number of elements3: "<<metric_field.predict_nelements()<<std::endl;

    time_metric = (get_wtime() - time_metric);

    metric_field.update_mesh();

    if(verbose) {
        cout_quality(mesh, "Initial quality");
        VTKTools<double>::export_vtu("initial_mesh_3d", mesh);
    }

    double L_up = sqrt(2.0);

    Coarsen<double, 2> coarsen(*mesh);
    Smooth<double, 2> smooth(*mesh);
    Swapping<double, 2> swapping(*mesh);

    double time_coarsen=0, time_swapping=0;
    for(size_t i=0; i<5; i++) {
        if(verbose)
            std::cout<<"INFO: Sweep "<<i<<std::endl;

        double tic = get_wtime();
        coarsen.coarsen(L_up, L_up, true);
        time_coarsen += (get_wtime()-tic);
        if(verbose)
            cout_quality(mesh, "Quality after coarsening");

        tic = get_wtime();
        swapping.swap(0.1);
        time_swapping += (get_wtime()-tic);
        if(verbose)
            cout_quality(mesh, "Quality after swapping");
    }

    mesh->defragment();

    double time_smooth=get_wtime();
    smooth.smart_laplacian(20);
    smooth.optimisation_linf(20);
    time_smooth = (get_wtime() - time_smooth);
    if(verbose)
        cout_quality(mesh, "Quality after final smoothening");

    if(verbose)
        std::cout<<"Times for metric, coarsen, swapping, smoothing = "<<time_metric<<", "<<time_coarsen<<", "<<time_swapping<<", "<<time_smooth<<std::endl;

    if(outfilename.size()==0)
        VTKTools<double>::export_vtu("scaled_mesh_3d", mesh);
    else
        VTKTools<double>::export_vtu(outfilename.c_str(), mesh);

    delete mesh;
#else
    std::cerr<<"Pragmatic was configured without VTK"<<std::endl;
#endif

    MPI_Finalize();

    return 0;
}
