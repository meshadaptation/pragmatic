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

#include <cassert>

#include "Mesh.h"
#include "Surface.h"
#include "MetricField.h"
#include "Coarsen.h"
#include "Refine.h"
#include "Swapping.h"
#include "Smooth.h"

#include "VTKTools.h"

static void *_pragmatic_mesh=NULL;
static void *_pragmatic_surface=NULL;
static void *_pragmatic_metric_field=NULL;

extern "C" {
  void pragmatic_2d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);

    int *fenlist = new int [(*NElements)*3];
    for(int i=0;i<(*NElements)*3;i++)
      fenlist[i] = enlist[i]-1;
    Mesh<double, int> *mesh = new Mesh<double, int>(*NNodes, *NElements, fenlist, x, y);
    delete [] fenlist;

    _pragmatic_mesh = mesh;
  }

  void pragmatic_3d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);
    
    int *fenlist = new int [(*NElements)*4];
    for(int i=0;i<(*NElements)*4;i++)
      fenlist[i] = enlist[i]-1;
    Mesh<double, int> *mesh = new Mesh<double, int>(*NNodes, *NElements, fenlist, x, y, z);
    delete [] fenlist;
    
    _pragmatic_mesh = mesh;
  }

  void pragmatic_vtk_begin(const char *filename){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);
    
    Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu(filename);
    _pragmatic_mesh = mesh;

    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      Surface2D<double, int> *surface = new Surface2D<double, int>(*mesh);
      _pragmatic_surface = surface;
      
      surface->find_surface(true);
    }else{
      Surface3D<double, int> *surface = new Surface3D<double, int>(*mesh);
      _pragmatic_surface = surface;
      
      surface->find_surface(true);
    }
  }
  
  void pragmatic_add_field(const double *psi, const double *error, int *pnorm){
    assert(_pragmatic_mesh!=NULL);
    assert(_pragmatic_surface!=NULL);

    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;
    
    if(_pragmatic_metric_field==NULL){

      if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
        Surface2D<double, int> *surface = (Surface2D<double, int> *)_pragmatic_surface;

        MetricField2D<double, int> *metric_field = new MetricField2D<double, int>(*mesh, *surface);
        _pragmatic_metric_field = metric_field;
      }else{
        Surface3D<double, int> *surface = (Surface3D<double, int> *)_pragmatic_surface;

        MetricField3D<double, int> *metric_field = new MetricField3D<double, int>(*mesh, *surface);
        _pragmatic_metric_field = metric_field;
      }
    }

    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      ((MetricField2D<double, int> *)_pragmatic_metric_field)->add_field(psi, *error, *pnorm);
      ((MetricField2D<double, int> *)_pragmatic_metric_field)->update_mesh();
    }else{
      ((MetricField3D<double, int> *)_pragmatic_metric_field)->add_field(psi, *error, *pnorm);
      ((MetricField3D<double, int> *)_pragmatic_metric_field)->update_mesh();
    }
  }

  void pragmatic_set_surface(const int *nfacets, const int *facets, const int *boundary_ids, const int *coplanar_ids){
    assert(_pragmatic_mesh!=NULL);
    assert(_pragmatic_surface==NULL);
    
    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;

    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      Surface2D<double, int> *surface = new Surface2D<double, int>(*mesh);
      _pragmatic_surface = surface;
      
      size_t NSElements = *nfacets;
      
      const size_t snloc = 2;
      
      int *fenlist = new int [NSElements*snloc];
      for(size_t i=0;i<NSElements*snloc;i++)
        fenlist[i] = facets[i]-1;
      surface->set_surface(NSElements, fenlist, boundary_ids, coplanar_ids);
      delete [] fenlist;
    }else{
      Surface3D<double, int> *surface = new Surface3D<double, int>(*mesh);
      _pragmatic_surface = surface;
      
      size_t NSElements = *nfacets;
      
      const size_t snloc = 3;
      
      int *fenlist = new int [NSElements*snloc];
      for(size_t i=0;i<NSElements*snloc;i++)
        fenlist[i] = facets[i]-1;
      surface->set_surface(NSElements, fenlist, boundary_ids, coplanar_ids);
      delete [] fenlist;
    }
  }

  void pragmatic_set_metric(const double *metric){
    assert(_pragmatic_mesh!=NULL);
    assert(_pragmatic_surface!=NULL);
    assert(_pragmatic_metric_field==NULL);

    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;

    if(_pragmatic_metric_field==NULL){
      if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
        Surface2D<double, int> *surface = (Surface2D<double, int> *)_pragmatic_surface;

        MetricField2D<double, int> *metric_field = new MetricField2D<double, int>(*mesh, *surface);
        _pragmatic_metric_field = metric_field;
      }else{
        Surface3D<double, int> *surface = (Surface3D<double, int> *)_pragmatic_surface;

        MetricField3D<double, int> *metric_field = new MetricField3D<double, int>(*mesh, *surface);
        _pragmatic_metric_field = metric_field;
      }
    }

    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      ((MetricField2D<double, int> *)_pragmatic_metric_field)->set_metric(metric);
      ((MetricField2D<double, int> *)_pragmatic_metric_field)->update_mesh();
    }else{
      ((MetricField3D<double, int> *)_pragmatic_metric_field)->set_metric(metric);
      ((MetricField3D<double, int> *)_pragmatic_metric_field)->update_mesh();
    }
  }

  void pragmatic_adapt(){
    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;

    const size_t ndims = mesh->get_number_dimensions();

    // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
    double L_up = sqrt(2.0);
    double L_low = L_up*0.5;

    if(ndims==2){
      Surface2D<double, int> *surface = (Surface2D<double, int> *)_pragmatic_surface;

      Coarsen2D<double, int> coarsen(*mesh, *surface);
      Smooth2D<double, int> smooth(*mesh, *surface);
      Refine2D<double, int> refine(*mesh, *surface);
      Swapping2D<double, int> swapping(*mesh, *surface);
      
      coarsen.coarsen(L_low, L_up);
      
      double L_max = mesh->maximal_edge_length();
      
      double alpha = sqrt(2.0)/2.0;
      for(size_t i=0;i<10;i++){
        double L_ref = std::max(alpha*L_max, L_up);
        
        refine.refine(L_ref);
        coarsen.coarsen(L_low, L_ref);
        swapping.swap(0.95);
        
        L_max = mesh->maximal_edge_length();
        
        if((L_max-L_up)<0.01)
          break;
      }
      
      std::vector<int> active_vertex_map;
      mesh->defragment(&active_vertex_map);
      surface->defragment(&active_vertex_map);
      
      smooth.smooth("optimisation Linf", 10);
    }else{
      Surface3D<double, int> *surface = (Surface3D<double, int> *)_pragmatic_surface;

      Coarsen3D<double, int> coarsen(*mesh, *surface);
      Smooth3D<double, int> smooth(*mesh, *surface);
      Refine3D<double, int> refine(*mesh, *surface);
      Swapping3D<double, int> swapping(*mesh, *surface);
      
      coarsen.coarsen(L_low, L_up);
      
      double L_max = mesh->maximal_edge_length();
      
      double alpha = sqrt(2.0)/2.0;
      for(size_t i=0;i<10;i++){
        double L_ref = std::max(alpha*L_max, L_up);
        
        refine.refine(L_ref);
        coarsen.coarsen(L_low, L_ref);
        swapping.swap(0.95);
        
        L_max = mesh->maximal_edge_length();
        
        if((L_max-L_up)<0.01)
          break;
      }
      
      std::vector<int> active_vertex_map;
      mesh->defragment(&active_vertex_map);
      surface->defragment(&active_vertex_map);
      
      smooth.smooth("optimisation Linf", 10);
    }
  }

  void pragmatic_get_info(int *NNodes, int *NElements, int *NSElements){
    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;
    
    *NNodes = mesh->get_number_nodes();
    *NElements = mesh->get_number_elements();
    
    const size_t ndims = mesh->get_number_dimensions();
    
    if(ndims==2){
      Surface2D<double, int> *surface = (Surface2D<double, int> *)_pragmatic_surface;
      *NSElements = surface->get_number_facets();
    }else{
      Surface3D<double, int> *surface = (Surface3D<double, int> *)_pragmatic_surface;
      *NSElements = surface->get_number_facets();
    }
  }

  void pragmatic_get_coords_2d(double *x, double *y){
    size_t NNodes = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_nodes();
    for(size_t i=0;i<NNodes;i++){
      x[i] = ((Mesh<double, int> *)_pragmatic_mesh)->get_coords(i)[0];
      y[i] = ((Mesh<double, int> *)_pragmatic_mesh)->get_coords(i)[1];
    }
  }

  void pragmatic_get_coords_3d(double *x, double *y, double *z){
    size_t NNodes = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_nodes();
    for(size_t i=0;i<NNodes;i++){
      x[i] = ((Mesh<double, int> *)_pragmatic_mesh)->get_coords(i)[0];
      y[i] = ((Mesh<double, int> *)_pragmatic_mesh)->get_coords(i)[1];
      z[i] = ((Mesh<double, int> *)_pragmatic_mesh)->get_coords(i)[2];
    }
  }

  void pragmatic_get_elements(int *elements){
    const size_t ndims = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions();
    const size_t NElements = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_elements();
    const size_t nloc = (ndims==2)?3:4;

    for(size_t i=0;i<NElements;i++){
      const int *n=((Mesh<double, int> *)_pragmatic_mesh)->get_element(i);

      for(size_t j=0;j<nloc;j++){
        assert(n[j]>=0);
        elements[i*nloc+j] = n[j]+1;
      }
    }
  }

  void pragmatic_get_surface(int *facets, int *boundary_ids, int *coplanar_ids){
    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;
    const size_t ndims = mesh->get_number_dimensions();
    
    if(ndims==2){
      Surface2D<double, int> *surface = (Surface2D<double, int> *)_pragmatic_surface;
      
      size_t NSElements = surface->get_number_facets();
      const size_t snloc = 2;
      
      for(size_t i=0;i<NSElements;i++){
        const int *n=surface->get_facet(i);
        
        for(size_t j=0;j<snloc;j++){
          assert(n[j]>=0);
          facets[i*snloc+j] = n[j]+1;
        }
        
        boundary_ids[i] = surface->get_boundary_id(i);
        coplanar_ids[i] = surface->get_coplanar_id(i);
      }
    }else{
      Surface3D<double, int> *surface = (Surface3D<double, int> *)_pragmatic_surface;
      
      size_t NSElements = surface->get_number_facets();
      const size_t snloc = 3;
      
      for(size_t i=0;i<NSElements;i++){
        const int *n=surface->get_facet(i);
        
        for(size_t j=0;j<snloc;j++){
          assert(n[j]>=0);
          facets[i*snloc+j] = n[j]+1;
        }
        
        boundary_ids[i] = surface->get_boundary_id(i);
        coplanar_ids[i] = surface->get_coplanar_id(i);
      }
    }
  }

  void pragmatic_get_lnn2gnn(int *nodes_per_partition, int *lnn2gnn){
    std::vector<int> _NPNodes, _lnn2gnn;
    ((Mesh<double, int> *)_pragmatic_mesh)->get_global_node_numbering(_NPNodes, _lnn2gnn);
    size_t len0 = _NPNodes.size();
    for(size_t i=0;i<len0;i++)
      nodes_per_partition[i] = _NPNodes[i];
    
    size_t len1 = _lnn2gnn.size();
    for(size_t i=0;i<len1;i++)
      lnn2gnn[i] = _lnn2gnn[i];
  }

  void pragmatic_get_metric(double *metric){
    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      ((MetricField2D<double, int> *)_pragmatic_metric_field)->get_metric(metric);
    }else{
      ((MetricField3D<double, int> *)_pragmatic_metric_field)->get_metric(metric);
    }
  }

  void pragmatic_dump(const char *filename){
    VTKTools<double, int>::export_vtu(filename, (Mesh<double, int>*)_pragmatic_mesh);
  }

  void pragmatic_end(){
    if(((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions()==2){
      delete (MetricField2D<double, int> *)_pragmatic_metric_field; 
      delete (Surface2D<double, int> *)_pragmatic_surface;
    }else{
      delete (MetricField3D<double, int> *)_pragmatic_metric_field; 
      delete (Surface3D<double, int> *)_pragmatic_surface;
    }
    _pragmatic_metric_field=NULL;
    _pragmatic_surface=NULL;

    delete (Mesh<double, int> *)_pragmatic_mesh;
    _pragmatic_mesh=NULL;
  }
}
