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
#ifndef HAVE_LIBCGAL
#warning No CGAL support. Cannot run test suite.
int main(){
    return 0;
}
#else

#include <iostream>
#include <vector>

#include <omp.h>

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"
#include "ticker.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_triangulation_plus_2.h> 
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_conformer_2.h>

struct K : public CGAL::Exact_predicates_inexact_constructions_kernel {};
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds, CGAL::Exact_predicates_tag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;

typedef CDT::Point Point;
typedef CDT::Vertex_handle Vertex_handle;

using namespace std;

int main(int argc, char **argv){
  CDT cdt;
  double xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0;
  
  // Define points - corners of bounding box.
  deque<Vertex_handle> cdt_pts(4);
  cdt_pts[0] = cdt.insert(Point(xmin, ymin));
  cdt_pts[1] = cdt.insert(Point(xmax, ymin));
  cdt_pts[2] = cdt.insert(Point(xmax, ymax));
  cdt_pts[3] = cdt.insert(Point(xmin, ymax));

  // Define edges
  int edges[] = {0,1, 1,2, 2,3, 3,0};
  for(size_t i=0;i<4;i++)
    cdt.insert_constraint(cdt_pts[edges[i*2]], cdt_pts[edges[i*2+1]]);

  // Refine mesh
  double shape_bound = 0.125; // Shape bound. It corresponds to about 20.6 degree.
  double max_edge_length=(xmax-xmin)/100;
  CGAL::refine_Delaunay_mesh_2(cdt, Criteria(shape_bound, max_edge_length));

  // Export points
  vector<double> x, y;
  map<Point, size_t> coordToId;
  size_t index = 0;
  for(CDT::Point_iterator it=cdt.points_begin();it!=cdt.points_end();it++, index++){
    x.push_back(CGAL::to_double(it->x()));
    y.push_back(CGAL::to_double(it->y()));
    coordToId[*it] = index;
  }
  int NNodes = x.size();

  // Export edges
  vector<int> enlist;
  for(CDT::Finite_faces_iterator it=cdt.faces_begin();it!=cdt.faces_end();it++){
    enlist.push_back(coordToId[it->vertex(0)->point()]);
    enlist.push_back(coordToId[it->vertex(1)->point()]);
    enlist.push_back(coordToId[it->vertex(2)->point()]);
  }
  size_t NElements = enlist.size()/3;
  Mesh<double, int> *mesh= new Mesh<double, int>(NNodes, NElements, &(enlist[0]), &(x[0]), &(y[0]));

  Surface<double, int> surface(*mesh);

  // Set up field - use first touch policy
  vector<double> psi(NNodes);
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for(int i=0;i<NNodes;i++)
      psi[i] = pow(mesh->get_coords(i)[0]+0.1, 2) + pow(mesh->get_coords(i)[1]+0.1, 2);
  }
  
  const char *methods[] = {"qls", "qls2"};
  for(int m=0;m<2;m++){
    const char *method = methods[m];
    
    MetricField<double, int> metric_field(*mesh, surface);
    metric_field.set_hessian_method(method); // default
    
    double tic = get_wtime();
    metric_field.add_field(&(psi[0]), 1.0);
    double toc = get_wtime();

    metric_field.update_mesh();
    
    vector<double> metric(NNodes*4);
    metric_field.get_metric(&(metric[0]));
    
    double rms[] = {0., 0., 0., 0.};
    for(int i=0;i<NNodes;i++){
      rms[0] += pow(2.0-metric[i*4  ], 2); rms[1] += pow(    metric[i*4+1], 2);
      rms[2] += pow(    metric[i*4+2], 2); rms[3] += pow(2.0-metric[i*4+3], 2);
    }
    
    double max_rms = 0;
    for(size_t i=0;i<4;i++){
      rms[i] = sqrt(rms[i]/NNodes);
      max_rms = std::max(max_rms, rms[i]);
    }

    string vtu_filename = string("../data/test_hessian_2d_")+string(method);
    VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh);
    
    std::cout<<"Hessian ("<<method<<") loop time = "<<toc-tic<<std::endl
             <<"Max RMS = "<<max_rms<<std::endl;
    if(max_rms>0.01)
      std::cout<<"fail\n";
    else
      std::cout<<"pass\n";
  }

  delete mesh;

  
  return 0;
}
#endif

