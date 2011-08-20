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

static Mesh<double, int> *mesh=NULL;
static Surface<double, int> *surface=NULL;
static MetricField<double, int> *metric_field=NULL;

extern "C" {
  void pragmatic_metric_2d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y){
    assert(mesh==NULL);
    mesh= new Mesh<double, int>(*NNodes, *NElements, enlist, x, y);
    
    assert(surface==NULL);
    surface = new Surface<double, int>(*mesh);
    
    assert(metric_field==NULL);
    metric_field = new MetricField<double, int>(*mesh, *surface);
  }

  void pragmatic_metric_3d_begin(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z){
    assert(mesh==NULL);
    mesh= new Mesh<double, int>(*NNodes, *NElements, enlist, x, y, z);
    
    assert(surface==NULL);
    surface = new Surface<double, int>(*mesh);
    
    assert(metric_field==NULL);
    metric_field = new MetricField<double, int>(*mesh, *surface);
  }
  
  void pragmatic_metric_addfield(const double *psi, const double *error){
    metric_field->set_hessian_method("qls");
    metric_field->add_field(psi, *error);
  }
  
  void pragmatic_metric_end(double *metric){
    metric_field->update_mesh();
    metric_field->get_metric(metric);

    delete mesh;
    delete surface;
    delete metric_field;
  }
}
