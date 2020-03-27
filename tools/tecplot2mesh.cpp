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

#include <cmath>
#include <iostream>
#include <vector>

#include <stdio.h> 
#include <string.h> 

#include "Mesh.h"
#ifdef HAVE_LIBMESHB
#include "GMFTools.h"
#endif
#include "cpragmatic.h"

#include <mpi.h>


int main(int argc, char **argv)
{
	if (argc < 1) exit(1);

    int rank=0;
    int required_thread_support=MPI_THREAD_SINGLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);
    assert(required_thread_support==provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char * filename_in = argv[1];
    std::string name(filename_in);
    printf("DEBUG begin mesh import\n");
    
    FILE * file_in = fopen(filename_in, "r");

    char  buf[512];
    char* token;
    char* rest;
    int nvar, nVer, nTri;

    // 4 lines of headers
    if (fgets(buf, sizeof buf, file_in) != NULL) {
        token = strtok(buf, " ");
        if (strcmp(token, "TITLE")) {
            exit(2);
        }
    }
    else {exit(2);}
    if (fgets(buf, sizeof buf, file_in) != NULL) {
        token = strtok_r(buf, " = ", &rest);
        if (strcmp(token, "VARIABLES")) {
            exit(2);
        }
        // first pass to remove the =
        token = strtok_r(rest, " ", &rest);
        nvar = 0;
        while ((token = strtok_r(rest, " ", &rest))) {
            nvar++;
        }
        nvar -= 2; // remove x and y
        if (nvar>8) {
            exit(5);
        }
    }
    else {exit(2);}
    if (fgets(buf, sizeof buf, file_in) != NULL) {
        if (sscanf(buf, "ZONE T=\"sampletext\", DATAPACKING=POINT, NODES=  %d, ELEMENTS=   %d, ZONETYPE=FETRIANGLE", &nVer, &nTri) != 2) {
            exit(2);
        }
    }
    else {exit(2);}
    if (fgets(buf, sizeof buf, file_in) != NULL) {
        // skip this line, assyme doubles for now
        token = strtok(buf, "=");
        if (strcmp(token, "DT")) {
            exit(2);
        }
    }
    else {exit(2);}

    std::vector<double> x(nVer), y(nVer);
    std::vector<double> data(nVer*nvar);
    // point data
    double dbuf1, dbuf2, dbuf[8];
    for (int line = 0; line < nVer; ++line) {
        if (fgets(buf, sizeof buf, file_in) == NULL) {
            exit(3);
        }
        if (sscanf(buf, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &dbuf1, &dbuf2, &dbuf[0], &dbuf[1], &dbuf[2], &dbuf[3], &dbuf[4], &dbuf[5], &dbuf[6], &dbuf[7]) != nvar+2){
            int result = sscanf(buf, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &dbuf1, &dbuf2, &dbuf[0], &dbuf[1], &dbuf[2], &dbuf[3], &dbuf[4], &dbuf[5], &dbuf[6], &dbuf[7]);
            printf("ERROR in parser:  buf: \"%s\"  and result = %d\n", buf, result);
            exit(32);
        }
        x[line] = dbuf1;
        y[line] = dbuf2;
        for (int i=0; i<nvar; ++i) {
            data[line*nvar+i] = dbuf[i];
        }
    }

    std::vector<index_t> ENList(nTri*3);
    // elements
    int ibuf[3];
    for (int line = 0; line < nTri; ++line) {
        if (fgets(buf, sizeof buf, file_in) == NULL) {
            exit(4);
        }
        if (sscanf(buf, "%d %d %d", &ibuf[0], &ibuf[1], &ibuf[2]) != 3){
            exit(42);
        }
        for (int i=0; i<3; ++i) {
            ENList[line*3+i] = ibuf[i]-1;
        }
    }

    printf("DEBUG mesh imported\n");

    Mesh<double> *mesh;
    mesh = new Mesh<double>(nVer, nTri, &ENList[0], &x[0], &y[0]);

    printf("DEBUG mesh created\n");

    pragmatic_init_light((void*)mesh);
    mesh->create_boundary();
    mesh->set_regions(NULL);

    printf("DEBUG begin mesh export\n");

#ifdef HAVE_LIBMESHB
    std::vector<int> solTypes(nvar);
    for (int i=0; i<nvar; ++i) {solTypes[i] = 1;}
    GMFTools<double>::export_gmf_mesh(filename_in, mesh, true);
    GMFTools<double>::export_gmf_solutions(filename_in, nvar, &solTypes[0], data, mesh, true);
#else
    std::cerr<<"Warning: Pragmatic was configured without LIBMESHB support"<<std::endl;
#endif

    return 0;
}