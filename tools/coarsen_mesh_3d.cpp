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
#include "VTKTools.h"
#include "MetricField.h"

#include "Coarsen.h"
#include "Smooth.h"
#include "Swapping.h"

#include <mpi.h>

void usage(char *cmd){
  std::cout<<"Usage: "<<cmd<<" [options] infile\n"
           <<"\nOptions:\n"
           <<" -h, --help\n\tHelp! Prints this message.\n"
           <<" -v, --verbose\n\tVerbose output.\n"
           <<" -c factor, --coarsen factor\n\tCoarsening factor is some number greater than 1, e.g. -c 2 with half the mesh resolution.\n"
           <<" -o filename, --output filename\n\tName of outfile -- without the extension.\n";
  return;
}

int parse_arguments(int argc, char **argv, std::string &infilename, std::string &outfilename, bool &verbose, double &factor){

  // Set defaults
  verbose = false;
  factor = 1.0;

  if(argc==1){
    usage(argv[0]);
    exit(0);
  }

  struct option longOptions[] = {
    {"help",    0,                 0, 'h'},
    {"verbose", 0,                 0, 'v'},
    {"coarsen", optional_argument, 0, 'c'},
    {"output",  optional_argument, 0, 'o'},
    {0, 0, 0, 0}
  };

  int optionIndex = 0;
  int c;
  const char *shortopts = "hvc:o:";

  // Set opterr to nonzero to make getopt print error messages
  opterr=1;
  while (true){
    c = getopt_long(argc, argv, shortopts, longOptions, &optionIndex);
    
    if (c == -1) break;
    
    switch (c){
    case 'h':
      usage(argv[0]);
      break;
    case 'v':
      verbose = true;
      break;
    case 'c':
      factor = atof(optarg);
      break;
    case 'o':
      outfilename = std::string(optarg);
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

void cout_quality(const Mesh<double> *mesh, std::string operation){
  double qmean = mesh->get_qmean();
  double qmin = mesh->get_qmin();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank==0)
    std::cout<<operation<<": step in quality (mean, min): ("<<qmean<<", "<<qmin<<")"<<std::endl;
}

int main(int argc, char **argv){
  int required_thread_support=MPI_THREAD_SINGLE;
  int provided_thread_support;
  MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
  assert(required_thread_support==provided_thread_support);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc==1){
    usage(argv[0]);
    exit(-1);
  }
  
  std::string infilename, outfilename;
  bool verbose=false;
  double factor=1.0;

  parse_arguments(argc, argv, infilename, outfilename, verbose, factor);

  Mesh<double> *mesh=VTKTools<double>::import_vtu(infilename.c_str());
  mesh->create_boundary();

  MetricField<double,3> metric_field(*mesh);

  metric_field.generate_mesh_metric(factor);
  metric_field.update_mesh();

  if(verbose){
    cout_quality(mesh, "Initial quality");
    VTKTools<double>::export_vtu("initial_mesh_3d", mesh);
  }
  
  double L_up = sqrt(2.0);

  Coarsen<double, 3> coarsen(*mesh);
  Smooth<double, 3> smooth(*mesh);
  Swapping<double, 3> swapping(*mesh);

  for(size_t i=0;i<5;i++){
    if(verbose)
      std::cout<<"INFO: Sweep "<<i<<std::endl;
    
    coarsen.coarsen(L_up, L_up, true);
    if(verbose)
      cout_quality(mesh, "Quality after coarsening");
    
    swapping.swap(0.3);
    if(verbose)
      cout_quality(mesh, "Quality after swapping");
    
    smooth.smooth(1);
    if(verbose)
      cout_quality(mesh, "Quality after smoothening");
  }

  mesh->defragment();
  
  smooth.smooth(20);
  if(verbose)
    cout_quality(mesh, "Quality after final smoothening");

  if(outfilename.size()==0)
    VTKTools<double>::export_vtu("scaled_mesh_3d", mesh);
  else
    VTKTools<double>::export_vtu(outfilename.c_str(), mesh);

  delete mesh;

  MPI_Finalize();

  return 0;
}
