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

#include <cassert>

#include "Mesh.h"
#include "Surface.h"
#include "MetricField.h"
#include "Coarsen.h"
#include "Refine.h"
#include "Swapping.h"
#include "Smooth.h"

#include "VTKTools.h"

using namespace std;

static void *_pragmatic_mesh=NULL;
static void *_pragmatic_surface=NULL;
static void *_pragmatic_metric_field=NULL;

extern "C" {
  void pragmatic_2d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);
    
    Mesh<double, int> *mesh = new Mesh<double, int>(*NNodes, *NElements, enlist, x, y);
    _pragmatic_mesh = mesh;
    
    Surface<double, int> *surface = new Surface<double, int>(*mesh);
    _pragmatic_surface = surface;

    MetricField<double, int> *metric_field = new MetricField<double, int>(*mesh, *surface);
    metric_field->set_hessian_method("qls");
    _pragmatic_metric_field = metric_field;
  }

  void pragmatic_3d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);
    
    Mesh<double, int> *mesh = new Mesh<double, int>(*NNodes, *NElements, enlist, x, y, z);
    _pragmatic_mesh = mesh;
    
    Surface<double, int> *surface = new Surface<double, int>(*mesh);
    _pragmatic_surface = surface;
    
    MetricField<double, int> *metric_field = new MetricField<double, int>(*mesh, *surface);
    metric_field->set_hessian_method("qls");
    _pragmatic_metric_field = metric_field;
  }

  void pragmatic_vtk_begin(const char *filename){
    assert(_pragmatic_mesh==NULL);
    assert(_pragmatic_surface==NULL);
    assert(_pragmatic_metric_field==NULL);
    
    Mesh<double, int> *mesh=VTKTools<double, int>::import_vtu(filename);
    _pragmatic_mesh = mesh;
    
    Surface<double, int> *surface = new Surface<double, int>(*mesh, true);
    _pragmatic_surface = surface;
    
    MetricField<double, int> *metric_field = new MetricField<double, int>(*mesh, *surface);
    metric_field->set_hessian_method("qls");
    _pragmatic_metric_field = metric_field;
  }
  
  void pragmatic_add_field(const double *psi, const double *error){
    assert(_pragmatic_metric_field!=NULL);
    
    ((MetricField<double, int> *)_pragmatic_metric_field)->add_field(psi, *error);
    ((MetricField<double, int> *)_pragmatic_metric_field)->update_mesh();
  }
  
  void pragmatic_set_metric(const double *metric, const double *min_length, const double *max_length){
    ((MetricField<double, int> *)_pragmatic_metric_field)->set_metric(metric);
    ((MetricField<double, int> *)_pragmatic_metric_field)->apply_min_edge_length(*min_length);
    ((MetricField<double, int> *)_pragmatic_metric_field)->apply_max_edge_length(*max_length);
    ((MetricField<double, int> *)_pragmatic_metric_field)->update_mesh();
  }
  
  void pragmatic_adapt(){
    Mesh<double, int> *mesh = (Mesh<double, int> *)_pragmatic_mesh;
    Surface<double, int> *surface = (Surface<double, int> *)_pragmatic_surface;
    
    // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
    double L_up = sqrt(2.0);
    double L_low = L_up*0.5;
    
    Coarsen<double, int> coarsen(*mesh, *surface);
    Smooth<double, int> smooth(*mesh, *surface);
    Refine<double, int> refine(*mesh, *surface);
    Swapping<double, int> swapping(*mesh, *surface);
    
    coarsen.coarsen(L_low, L_up);
    
    double L_max = mesh->maximal_edge_length();
    
    double alpha = sqrt(2)/2;  
    for(size_t i=0;i<10;i++){
      double L_ref = std::max(alpha*L_max, L_up);
      
      refine.refine(L_ref);
      coarsen.coarsen(L_low, L_ref);
      swapping.swap(0.95);
      
      L_max = mesh->maximal_edge_length();
      
      if((L_max-L_up)<0.01)
        break;
    }
    
    std::map<int, int> active_vertex_map;
    mesh->defragment(&active_vertex_map);
    surface->defragment(&active_vertex_map);
  }

  void pragmatic_get_info(int *NNodes, int *NElements, int *NSElements){
    *NNodes = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_nodes();
    *NElements = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_elements();
    *NSElements = ((Surface<double, int> *)_pragmatic_surface)->get_number_facets();
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
      
      for(size_t j=0;j<nloc;j++)
        elements[i*nloc+j] = n[j];
    }
  }

  void pragmatic_get_facets(int *facets){
    const size_t ndims = ((Mesh<double, int> *)_pragmatic_mesh)->get_number_dimensions();
    size_t NSElements = ((Surface<double, int> *)_pragmatic_surface)->get_number_facets();
    const size_t snloc = (ndims==2)?2:3;

    for(size_t i=0;i<NSElements;i++){
      const int *n=((Surface<double, int> *)_pragmatic_surface)->get_facet(i);
      
      for(size_t j=0;j<snloc;j++)
        facets[i*snloc+j] = n[j];
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
    ((MetricField<double, int> *)_pragmatic_metric_field)->get_metric(metric);
  }

  void pragmatic_dump(const char *filename){
    VTKTools<double, int>::export_vtu(filename, (Mesh<double, int>*)_pragmatic_mesh);
  }

  void pragmatic_end(){
    delete (Mesh<double, int> *)_pragmatic_mesh;
    delete (Surface<double, int> *)_pragmatic_surface;
    delete (MetricField<double, int> *)_pragmatic_metric_field;
  }
}
