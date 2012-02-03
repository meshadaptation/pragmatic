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

#ifndef ZOLTAN_COLOUR_H
#define ZOLTAN_COLOUR_H

#include <stddef.h>
#include "mpi.h"

#include "pragmatic_config.h"

#ifndef HAVE_ZOLTAN
#error No Zoltan support.
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /*! \brief This is used internally to package the data required by
   * the Zoltan C interface.
   */
  typedef struct {
    /* Rank of local process.
     */
    int rank;

    /* Number of nodes in the graph assigned to the local process.
     */
    size_t npnodes;
    
    /* Total number of nodes on local process.
     */
    size_t nnodes;
    
    /* Array storing the number of edges connected to each node.
     */
    size_t *nedges;
    
    /* Array storing the edges in compressed row storage format.
     */
    size_t *csr_edges;
    
    /* Mapping from local node numbers to global node numbers.
     */
    int *gid;
    
    /* Process owner of each node.
     */
    size_t *owner;
    
    /* Graph colouring.
     */
    int *colour;
  } zoltan_colour_graph_t;
  
  void zoltan_colour(zoltan_colour_graph_t *graph, int distance, MPI_Comm mpi_comm);
  
#ifdef __cplusplus
}
#endif
#endif
