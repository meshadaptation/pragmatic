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
