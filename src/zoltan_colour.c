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

#include <assert.h>
#include <mpi.h>

#include "zoltan_colour.h"

#include "zoltan.h"

#include <stdio.h>

/* A ZOLTAN_NUM_OBJ_FN query function returns the number of objects
   that are currently assigned to the processor. */
int num_obj_fn(void* data, int* ierr){
  *ierr = ZOLTAN_OK;
  
  zoltan_colour_graph_t *graph = (zoltan_colour_graph_t *)data;
  return graph->npnodes;
}

/* A ZOLTAN_OBJ_LIST_FN query function fills two (three if weights are
   used) arrays with information about the objects currently assigned
   to the processor. Both arrays are allocated (and subsequently
   freed) by Zoltan; their size is determined by a call to a
   ZOLTAN_NUM_OBJ_FN query function to get the array size. For many
   algorithms, either a ZOLTAN_OBJ_LIST_FN query function or a
   ZOLTAN_FIRST_OBJ_FN/ZOLTAN_NEXT_OBJ_FN query-function pair must be
   registered; however, both query options need not be provided. The
   ZOLTAN_OBJ_LIST_FN is preferred for efficiency. */
void obj_list_fn(void* data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids,
                 ZOLTAN_ID_PTR local_ids, int wgt_dim, float* obj_wgts, int* ierr){
  int i;
  int loc=0;

  *ierr = ZOLTAN_OK; 
  
  zoltan_colour_graph_t *graph = (zoltan_colour_graph_t *)data;

  for(i=0;i<graph->nnodes;i++){
    if(graph->owner[i]==graph->rank){
      global_ids[loc] = graph->gid[i];
      local_ids[loc] = loc;
      loc++;
    }
  }
}

/* A ZOLTAN_NUM_EDGES_MULTI_FN query function returns the number of
   edges in the communication graph of the application for each object
   in a list of objects. That is, for each object in the
   global_ids/local_ids arrays, the number of objects with which the
   given object must share information is returned. */
void num_edges_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                        ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, int* ierr){
  int i;
  int loc=0;

  *ierr = ZOLTAN_OK; 
  
  zoltan_colour_graph_t *graph = (zoltan_colour_graph_t *)data;
  
  for(i=0;i<graph->nnodes;i++){
    if(graph->owner[i]==graph->rank)
      num_edges[loc++] = graph->nedges[i];
  }
}

void edge_list_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                        ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, ZOLTAN_ID_PTR nbor_global_ids,
                        int* nbor_procs, int wgt_dim, float *ewgts, int* ierr){
  int i, j, lid, loc=0, sum=0, sum2=0;

  *ierr = ZOLTAN_OK;
  
  zoltan_colour_graph_t *graph = (zoltan_colour_graph_t *)data;
  
  for(i=0;i<graph->nnodes;i++){
    if(graph->owner[i]==graph->rank){
      assert(num_edges[loc]==graph->nedges[i]);
      for(j=0;j<num_edges[loc];j++){
        lid = graph->csr_edges[sum2+j];
        nbor_global_ids[sum] = graph->gid[lid];
        nbor_procs[sum] = graph->owner[lid];
        sum++;
      }
      loc++;
    }
    sum2+=graph->nedges[i];
  }
}

void zoltan_colour(zoltan_colour_graph_t *graph, int distance, MPI_Comm mpi_comm){
  int ierr, i, j, loc=0;
  float ver;
  struct Zoltan_Struct *zz;
  int num_gid_entries;
  int num_obj;
  ZOLTAN_ID_PTR global_ids;
  int *color_exp;

  ierr = Zoltan_Initialize(-1, NULL, &ver);
  zz = Zoltan_Create(mpi_comm);
  
  /* The number of array entries used to describe a single global ID.
   */
  num_gid_entries = 1;

  /* Number of objects for which we want to know the color on this
     processor. Objects may be non-local or duplicated. */
  num_obj = graph->nnodes;

  /* An array of global IDs of objects for which we want to know the
     color on this processor. Size of this array must be num_obj. */
  global_ids = (ZOLTAN_ID_PTR) ZOLTAN_MALLOC(num_obj*sizeof(ZOLTAN_ID_TYPE));
  for(i=0;i<graph->nnodes;i++)
    global_ids[loc++] = graph->gid[i];

#ifndef NDEBUG 
  ierr = Zoltan_Set_Param(zz, "CHECK_GRAPH", "2");
  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
#else
  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
#endif


  ierr = Zoltan_Set_Param(zz, "VERTEX_VISIT_ORDER", "S");
  if(distance==1){
    ierr = Zoltan_Set_Param(zz, "COLORING_PROBLEM", "DISTANCE-1");
  }else if(distance==2){
    ierr = Zoltan_Set_Param(zz, "COLORING_PROBLEM", "DISTANCE-2");
  }else{
    fprintf(stderr, "WARNING unexpected distance for coloring graph.\n");
  }

  /* Register the callbacks.
   */
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_OBJ_FN_TYPE, (void *)&num_obj_fn, (void *)graph);
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_OBJ_LIST_FN_TYPE, (void *)&obj_list_fn, (void *)graph);
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_EDGES_MULTI_FN_TYPE, (void *)&num_edges_multi_fn, (void *)graph);
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_EDGE_LIST_MULTI_FN_TYPE, (void *)&edge_list_multi_fn, (void *)graph);
  
  ierr = Zoltan_Color(zz, num_gid_entries, num_obj, global_ids, graph->colour);

  ZOLTAN_FREE(&global_ids);
  Zoltan_Destroy(&zz);
}
