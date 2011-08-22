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
#include "MetricField.h"
#include "Surface.h"

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
  
  void pragmatic_addfield(const double *psi, const double *error){
    assert(_pragmatic_metric_field!=NULL);
    
    ((MetricField<double, int> *)_pragmatic_metric_field)->add_field(psi, *error);
    ((MetricField<double, int> *)_pragmatic_metric_field)->update_mesh();
  }
  
  void pragmatic_get_metric(double *metric){
    ((MetricField<double, int> *)_pragmatic_metric_field)->get_metric(metric);
  }

  void pragmatic_set_metric(const double *metric){
    ((MetricField<double, int> *)_pragmatic_metric_field)->set_metric(metric);
  }

  void pragmatic_end(){
    delete (Mesh<double, int> *)_pragmatic_mesh;
    delete (Surface<double, int> *)_pragmatic_surface;
    delete (MetricField<double, int> *)_pragmatic_metric_field;
  }
}
