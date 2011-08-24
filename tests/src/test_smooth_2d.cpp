/* 
 *    Copyright (C) 2010 Imperial College London and others.
 *    
 *    Please see the AUTHORS file in the main source directory for a
 *    full list of copyright holders.
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
 *    License as published by the Free Software Foundation, version
 *    2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free
 *    Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *    MA 02111-1307 USA
 */
#include "pragmatic_config.h"

#ifndef HAVE_LIBCGAL
#warning No CGAL support. Cannot run test (__FILE__).
int main(){
      return 0;
}
#else

#include <iostream>
#include <vector>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Mesh.h"
#include "Surface.h"
#include "VTKTools.h"
#include "MetricField.h"
#include "Smooth.h"
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
  double max_edge_length=(xmax-xmin)/20;
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

  const char *methods[] = {"Laplacian", "smart Laplacian", "optimisation L2", "optimisation Linf", "combined"};
  const double target_quality_mean[] = {0.06, 0.2,  0.2, 0.1,    0.2};
  const double target_quality_min[]  = {0.0,  0.008, 0.0, 0.0004, 0.008};
  for(int m=0;m<5;m++){
    const char *method = methods[m];
    
    Mesh<double, int> *mesh= new Mesh<double, int>(NNodes, NElements, &(enlist[0]), &(x[0]), &(y[0]));
    Surface<double, int> surface(*mesh);
    
    MetricField<double, int> metric_field(*mesh, surface);
        
    for(int i=0;i<NNodes;i++){
      double hx = std::max(0.01, fabs(mesh->get_coords(i)[0]));
      double hy = std::max(0.01, fabs(mesh->get_coords(i)[1]));
      
      double m[] =
        {1/(hx*hx), 0.0,
         0.0,       1/(hy*hy)};
      metric_field.set_metric(m, i);
    }
    
    metric_field.apply_nelements(NElements);
    metric_field.update_mesh();

    if(m==0){
      VTKTools<double, int>::export_vtu("../data/test_smooth_2d_init", mesh);
      double qmean = mesh->get_qmean();
      double qrms = mesh->get_qrms();
      double qmin = mesh->get_qmin();
      
      std::cout<<"Initial quality:"<<std::endl
               <<"Quality mean:    "<<qmean<<std::endl
               <<"Quality min:     "<<qmin<<std::endl
               <<"Quality RMS:     "<<qrms<<std::endl;
    }

    Smooth<double, int> smooth(*mesh, surface);
    
    double tic = get_wtime();
    if(m<4){
      smooth.smooth(method);
    }else{
      smooth.smooth("smart Laplacian");
      smooth.smooth("optimisation Linf");
    }
    double toc = get_wtime();
    
    double lmean = mesh->get_lmean();
    double lrms = mesh->get_lrms();
    
    double qmean = mesh->get_qmean();
    double qrms = mesh->get_qrms();
    double qmin = mesh->get_qmin();
    
    std::cout<<"Smooth loop time ("<<method<<"):     "<<toc-tic<<std::endl
             <<"Edge length mean:      "<<lmean<<std::endl
             <<"Edge length RMS:      "<<lrms<<std::endl
             <<"Quality mean:          "<<qmean<<std::endl
             <<"Quality min:          "<<qmin<<std::endl
             <<"Quality RMS:          "<<qrms<<std::endl;
    
    string vtu_filename = string("../data/test_smooth_2d_")+string(method);
    VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh);
    delete mesh;
    
    std::cout<<"Smooth - "<<methods[m]<<": ";
    if((qmean>target_quality_mean[m])&&(qmin>target_quality_min[m]))
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail"<<std::endl;
  }

  return 0;
}
#endif
