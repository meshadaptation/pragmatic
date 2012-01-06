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

#if !defined HAVE_LIBCGAL || !defined HAVE_MPI
#warning CGAL and MPI are both required for this test (__FILE__). Skipping.
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

#include <mpi.h>

using namespace std;

int main(int argc, char **argv){
  MPI::Init(argc,argv);

  int rank = MPI::COMM_WORLD.Get_rank();
  int nparts = MPI::COMM_WORLD.Get_size();

  vector<double> x, y;
  vector<int> ENList;
  int NNodes=0, NElements=0;
  if(rank==0){
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
    map<Point, size_t> coordToId;
    size_t index = 0;
    for(CDT::Point_iterator it=cdt.points_begin();it!=cdt.points_end();it++, index++){
      x.push_back(CGAL::to_double(it->x()));
      y.push_back(CGAL::to_double(it->y()));
      coordToId[*it] = index;
    }
    NNodes = x.size();
    
    // Export edges
    for(CDT::Finite_faces_iterator it=cdt.faces_begin();it!=cdt.faces_end();it++){
      ENList.push_back(coordToId[it->vertex(0)->point()]);
      ENList.push_back(coordToId[it->vertex(1)->point()]);
      ENList.push_back(coordToId[it->vertex(2)->point()]);
    }
    NElements = ENList.size()/3; 
  }

  std::vector<int> owner_range;
  std::vector<int> lnn2gnn;
  if(nparts>1){
    // Distribute the mesh to everyone.
    MPI::COMM_WORLD.Bcast(&NNodes, 1, MPI_INT, 0);
    MPI::COMM_WORLD.Bcast(&NElements, 1, MPI_INT, 0);
    
    if(rank!=0){
      x.resize(NNodes);
      y.resize(NNodes);
      ENList.resize(NElements*3);
    }
    
    MPI::COMM_WORLD.Bcast(&(x[0]), NNodes, MPI_DOUBLE, 0);
    MPI::COMM_WORLD.Bcast(&(y[0]), NNodes, MPI_DOUBLE, 0);
    MPI::COMM_WORLD.Bcast(&(ENList[0]), NElements*3, MPI_INT, 0);
    
    // Perform partitioning only on the root process.
    std::vector<idxtype> epart(NElements, 0), npart(NNodes, 0);
    if(rank==0){
      int numflag = 0, edgecut;
      int etype = 1; // triangles
      
      std::vector<idxtype> metis_ENList(NElements*3);
      for(int i=0;i<NElements*3;i++)
        metis_ENList[i] = ENList[i];
      METIS_PartMeshNodal(&NElements, &NNodes, &(metis_ENList[0]), &etype,
                          &numflag, &nparts, &edgecut, &(epart[0]), &(npart[0]));
    }
    
    // Broadcast this partitioning to other processes.
    MPI::COMM_WORLD.Bcast(&(epart[0]), NElements, MPI_INT, 0);
    MPI::COMM_WORLD.Bcast(&(npart[0]), NNodes, MPI_INT, 0);
        
    // Seperate out owned nodes.
    std::vector< std::deque<int> > node_partition(nparts);
    for(int i=0;i<NNodes;i++)
      node_partition[npart[i]].push_back(i);
    
    std::map<int, int> renumber;
    {
      int pos=0;
      owner_range.push_back(0);
      for(int i=0;i<nparts;i++){
        int pNNodes = node_partition[i].size();
        owner_range.push_back(owner_range[i]+pNNodes);
        for(int j=0;j<pNNodes;j++)
          renumber[node_partition[i][j]] = pos++;
      }
    }
    
    // Find elements that have at least one vertex in the local
    // partition.
    std::deque<int> element_partition;
    std::set<int> halo_nodes;
    for(int i=0;i<NElements;i++){
      std::set<int> residency;
      for(int j=0;j<3;j++)
        residency.insert(npart[ENList[i*3+j]]);
      
      if(residency.count(rank)){
        element_partition.push_back(i);
        
        for(int j=0;j<3;j++){
          int nid = ENList[i*3+j];
          if(npart[nid]!=rank)
            halo_nodes.insert(nid);
        }
      }
    }
    
    // Append halo nodes to local node partition.
    for(std::set<int>::const_iterator it=halo_nodes.begin();it!=halo_nodes.end();++it){
      node_partition[rank].push_back(*it);
    }
    
    // Global numbering to partition numbering look up table.
    NNodes = node_partition[rank].size();
    std::map<int, int> gnn2lnn;
    lnn2gnn.resize(NNodes);
    for(int i=0;i<NNodes;i++){
      int gnn = renumber[node_partition[rank][i]];
      gnn2lnn[gnn] = i;
      lnn2gnn[i] = gnn;
    }
    
    // Construct local mesh.
    std::vector<double> lx(NNodes), ly(NNodes), lz(NNodes);
    for(int i=0;i<NNodes;i++){
      lx[i] = x[node_partition[rank][i]];
      ly[i] = y[node_partition[rank][i]];
    }
    
    NElements = element_partition.size();
    std::vector<int> lENList(NElements*3);
    for(int i=0;i<NElements;i++){
      for(int j=0;j<3;j++){
        int nid = renumber[ENList[element_partition[i]*3+j]];
        lENList[i*3+j] = nid;
      }
    }
    
    // Swap
    x.swap(lx);
    y.swap(ly);
    ENList.swap(lENList);
  }

  const char *methods[] = {"Laplacian", "smart Laplacian", "smart Laplacian search", "optimisation L2", "optimisation Linf", "combined"};
  const double target_quality_mean[] = {0.06, 0.08,  0.5,  0.3,   0.7, 0.7};
  const double target_quality_min[]  = {0.0,  0.001, 0.02, 0.001, 0.4, 0.4};
  for(int m=0;m<6;m++){
    const char *method = methods[m];

    Mesh<double, int> *mesh;
    if(nparts>1){
      MPI_Comm comm = MPI_COMM_WORLD;
      mesh= new Mesh<double, int>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]), &(lnn2gnn[0]), &(owner_range[0]), comm);
    }else{
      mesh= new Mesh<double, int>(NNodes, NElements, &(ENList[0]), &(x[0]), &(y[0]));
    }
    
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
      
      if(rank==0)
        std::cout<<"Initial quality:"<<std::endl
                 <<"Quality mean:    "<<qmean<<std::endl
                 <<"Quality min:     "<<qmin<<std::endl
                 <<"Quality RMS:     "<<qrms<<std::endl;
    }
    
    Smooth<double, int> smooth(*mesh, surface);
    
    double tic = get_wtime();
    int max_smooth_iter=200;
    if(m<5){
      smooth.smooth(method, max_smooth_iter);
    }else{
      smooth.smooth("smart Laplacian search", max_smooth_iter);
      smooth.smooth("optimisation Linf", max_smooth_iter);
    }
    double toc = get_wtime();
    
    double lmean = mesh->get_lmean();
    double lrms = mesh->get_lrms();
    
    double qmean = mesh->get_qmean();
    double qrms = mesh->get_qrms();
    double qmin = mesh->get_qmin();
    
    if(rank==0)
      std::cout<<"Smooth loop time ("<<method<<"):     "<<toc-tic<<std::endl
               <<"Edge length mean:      "<<lmean<<std::endl
               <<"Edge length RMS:      "<<lrms<<std::endl
               <<"Quality mean:          "<<qmean<<std::endl
               <<"Quality min:          "<<qmin<<std::endl
               <<"Quality RMS:          "<<qrms<<std::endl;
    
    string vtu_filename = string("../data/test_smooth_2d_")+string(method);
    VTKTools<double, int>::export_vtu(vtu_filename.c_str(), mesh);
    delete mesh;
    
    if(rank==0){
      std::cout<<"Smooth - "<<methods[m]<<": ";
      if((qmean>target_quality_mean[m])&&(qmin>target_quality_min[m]))
        std::cout<<"pass"<<std::endl;
      else
        std::cout<<"fail"<<std::endl;
    }
  }
  
  MPI::Finalize();

  return 0;
}

#endif
