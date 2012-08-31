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

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "zoltan_tools.h"

void pragmatic_zoltan_verify(int ierr, const char *str){
  if(ierr==ZOLTAN_WARN){
    fprintf(stderr, "ZOLTAN_WARN: %s\n", str);
    return;
  }

  if(ierr==ZOLTAN_FATAL){
    fprintf(stderr, "ZOLTAN_FATAL: %s\n", str);
    exit(-1);
  }

  if(ierr==ZOLTAN_MEMERR){
    fprintf(stderr, "ZOLTAN_MEMERR: %s\n", str);
    exit(-1);
  }

  return;
}

/* A ZOLTAN_NUM_OBJ_FN query function returns the number of objects
   that are currently assigned to the processor. */
int num_obj_fn(void* data, int* ierr){
  *ierr = ZOLTAN_OK;
  
  zoltan_graph_t *graph = (zoltan_graph_t *)data;
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
  size_t i;
  int loc=0, verbose=0;
  
  if(verbose){
    printf("VERBOSE: num_gid_entries=%d, num_lid_entries=%d, wgt_dim=%d, obj_wgts=%p\n",
           num_gid_entries, num_lid_entries, wgt_dim, obj_wgts);
  }

  *ierr = ZOLTAN_OK; 
  
  zoltan_graph_t *graph = (zoltan_graph_t *)data;

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
  size_t i;
  int loc=0, verbose=0;

  if(verbose){
    printf("VERBOSE: num_gid_entries=%d, num_lid_entries=%d, num_obj=%d, local_ids=%p, global_ids=%p\n",
           num_gid_entries, num_lid_entries, num_obj, local_ids, global_ids);
  }

  *ierr = ZOLTAN_OK; 
  
  zoltan_graph_t *graph = (zoltan_graph_t *)data;
  
  for(i=0;i<graph->nnodes;i++){
    if(graph->owner[i]==graph->rank)
      num_edges[loc++] = graph->nedges[i];
  }
}

void edge_list_multi_fn(void* data, int num_gid_entries, int num_lid_entries, int num_obj, 
                        ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int* num_edges, ZOLTAN_ID_PTR nbor_global_ids,
                        int* nbor_procs, int wgt_dim, float *ewgts, int* ierr){
  size_t i;
  int j, lid, loc=0, sum=0, sum2=0;
  int verbose=0;
  
  if(verbose){
    printf("VERBOSE: num_gid_entries=%d, num_lid_entries=%d, num_obj=%d, int wgt_dim=%d, local_ids=%p, global_ids=%p, ewgts=%p\n",
           num_gid_entries, num_lid_entries, num_obj, wgt_dim, local_ids, global_ids, ewgts);
  }
  
  *ierr = ZOLTAN_OK;
  
  zoltan_graph_t *graph = (zoltan_graph_t *)data;
  
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

void zoltan_colour(zoltan_graph_t *graph, int distance, MPI_Comm mpi_comm){
  size_t i;
  int ierr, loc=0;
  float ver;
  struct Zoltan_Struct *zz;
  int num_gid_entries;
  int num_obj;
  ZOLTAN_ID_PTR global_ids;

  ierr = Zoltan_Initialize(-1, NULL, &ver); 
  pragmatic_zoltan_verify(ierr, "Zoltan_Initialize\0");

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
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");

  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
#else
  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
#endif

  if(distance==1){
    ierr = Zoltan_Set_Param(zz, "COLORING_PROBLEM", "DISTANCE-1");
    pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  }else if(distance==2){
    ierr = Zoltan_Set_Param(zz, "COLORING_PROBLEM", "DISTANCE-2");
    pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  }else{
    fprintf(stderr, "WARNING unexpected distance for coloring graph.\n");
  }

  ierr = Zoltan_Set_Param(zz, "SUPERSTEP_SIZE", "100");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");

  ierr = Zoltan_Set_Param(zz, "COMM_PATTERN", "S");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");

  ierr = Zoltan_Set_Param(zz, "VERTEX_VISIT_ORDER", "S");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  
  ierr = Zoltan_Set_Param(zz, "RECOLORING_NUM_OF_ITERATIONS", "0");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  
  ierr = Zoltan_Set_Param(zz, "RECOLORING_TYPE", "ASYNCHRONOUS");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  
  ierr = Zoltan_Set_Param(zz, "RECOLORING_PERMUTATION", "NONDECREASING");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
  
  /* Register the callbacks.
   */
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_OBJ_FN_TYPE, (ZOLTAN_VOID_FN *)&num_obj_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_OBJ_LIST_FN_TYPE, (ZOLTAN_VOID_FN *)&obj_list_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_EDGES_MULTI_FN_TYPE, (ZOLTAN_VOID_FN *)&num_edges_multi_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_EDGE_LIST_MULTI_FN_TYPE, (ZOLTAN_VOID_FN *)&edge_list_multi_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");
  
  ierr = Zoltan_Color(zz, num_gid_entries, num_obj, global_ids, graph->colour);
  pragmatic_zoltan_verify(ierr, "Zoltan_Color\0");

  ZOLTAN_FREE(&global_ids);
  Zoltan_Destroy(&zz);
}

void zoltan_reorder(zoltan_graph_t *graph){
  size_t i;
  int ierr, loc=0;
  float ver;
  struct Zoltan_Struct *zz;
  int num_gid_entries;
  int num_obj;
  ZOLTAN_ID_PTR global_ids;

  ierr = Zoltan_Initialize(-1, NULL, &ver); 
  pragmatic_zoltan_verify(ierr, "Zoltan_Initialize\0");

  zz = Zoltan_Create(MPI_COMM_SELF);
  
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
    global_ids[loc++] = i;

#ifndef NDEBUG 
  ierr = Zoltan_Set_Param(zz, "CHECK_GRAPH", "2");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");

  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
#else
  ierr = Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");
#endif

  ierr = Zoltan_Set_Param(zz, "ORDER_METHOD", "METIS");
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Param\0");

  /* Register the callbacks.
   */
  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_OBJ_FN_TYPE, (ZOLTAN_VOID_FN *)&num_obj_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_OBJ_LIST_FN_TYPE, (ZOLTAN_VOID_FN *)&obj_list_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_NUM_EDGES_MULTI_FN_TYPE, (ZOLTAN_VOID_FN *)&num_edges_multi_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");

  ierr = Zoltan_Set_Fn(zz, ZOLTAN_EDGE_LIST_MULTI_FN_TYPE, (ZOLTAN_VOID_FN *)&edge_list_multi_fn, (void *)graph);
  pragmatic_zoltan_verify(ierr, "Zoltan_Set_Fn\0");
  
  ierr = Zoltan_Order(zz, num_gid_entries, num_obj, global_ids, graph->order);

  pragmatic_zoltan_verify(ierr, "Zoltan_Color\0");

  ZOLTAN_FREE(&global_ids);
  Zoltan_Destroy(&zz);
}
