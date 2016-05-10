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

/*! \mainpage Parallel anisotRopic Adaptive Mesh ToolkIt
 *
 * PRAgMaTIc provides 2D/3D anisotropic mesh adaptivity for meshes of
 * simplexes. The target applications are finite element and finite
 * volume methods although the it can also be used as a lossy
 * compression algorithm for 2 and 3D data (e.g. image
 * compression). It takes as its input the mesh and a metric tensor
 * field which encodes desired mesh element size anisotropically. The
 * toolkit is written in C++ and has OpenMP and MPI parallel support.
 *
 * \section links Useful links:
 * \li Production releases are available from <a href="http://www.openpetascale.org/">Open Petascale Libraries</a>.
 * \li Bleeding edge developer site on <a href="https://launchpad.net/pragmatic">Launchpad</a>.
 */

extern "C" {
    void pragmatic_2d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y);
    void pragmatic_3d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z);
    void pragmatic_vtk_begin(const char *filename);
    void pragmatic_add_field(const double *psi, const double *error);
    void pragmatic_set_metric(const double *metric, const double *min_length, const double *max_length);
    void pragmatic_adapt();
    void pragmatic_get_info(int *NNodes, int *NElements, int *NSElements);
    void pragmatic_get_coords_2d(double *x, double *y);
    void pragmatic_get_coords_3d(double *x, double *y, double *z);
    void pragmatic_get_elements(int *elements);
    void pragmatic_get_facets(int *facets);
    void pragmatic_get_lnn2gnn(int *nodes_per_partition, int *lnn2gnn);
    void pragmatic_get_metric(double *metric);
    void pragmatic_dump(const char *filename);
    void pragmatic_end();
}
