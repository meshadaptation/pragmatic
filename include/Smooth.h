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

#ifndef SMOOTH_H
#define SMOOTH_H

#include <omp.h>
#include <set>
#include <map>
#include <vector>
#include <deque>

#include "errno.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <errno.h>

#include "ElementProperty.h"
#include "Surface.h"
#include "Mesh.h"
#include "Colour.h"
#include "MetricTensor.h"

/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t, typename index_t>
  class Smooth{
 public:
  /// Default constructor.
  Smooth(Mesh<real_t, index_t> &mesh, Surface<real_t, index_t> &surface){
    _mesh = &mesh;
    _surface = &surface;

    ndims = _mesh->get_number_dimensions();
    nloc = (ndims==2)?3:4;

    // Set the orientation of elements.
    property = NULL;
    int NElements = _mesh->get_number_elements();
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      if(ndims==2)
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]));
      else
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]),
                                               _mesh->get_coords(n[3]));
      break;
    }    
  }

  /// Default destructor.
  ~Smooth(){
    delete property;
  }
  
  // Smooth the mesh using a given method. Valid methods are: "Laplacian", "smart Laplacian", "optimisation"
  void smooth(std::string method, int max_iterations=100){
    init_cache(method);
    
    bool (Smooth<real_t, index_t>::*smooth_kernel)(index_t) = NULL;
    if(method=="Laplacian"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::laplacian_2d_kernel;
      else
        smooth_kernel = &Smooth<real_t, index_t>::laplacian_3d_kernel;
    }else if(method=="smart Laplacian"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_2d_kernel;
      else
        smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_3d_kernel;
    }else if(method=="optimisation"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::optimisation_2d_kernel;
      //else
      //  smooth_kernel = &Smooth<real_t, index_t>::optimisation_3d_kernel;
    }else{
      std::cerr<<"WARNING: Unknown smoothing method \""<<method<<"\"\nUsing \"smart Laplacian\"\n";
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_2d_kernel;
      else
        smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_3d_kernel;
    }

    // Use this to keep track of vertices that are still to be visited.
#ifdef _OPENMP
    std::vector< std::set<index_t> > partial_active_vertices(omp_get_max_threads());
#endif
    std::set<index_t> active_vertices;

    // First sweep through all vertices. Add vertices adjancent to any
    // vertex moved into the active_vertex list.
    for(size_t colour=0; colour<colour_sets.size(); colour++){
#pragma omp parallel
      {
        int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
        for(int cn=0;cn<node_set_size;cn++){
          index_t node = colour_sets[colour][cn];
          
          if((this->*smooth_kernel)(node)){
            for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
#ifdef _OPENMP
              partial_active_vertices[omp_get_thread_num()].insert(*it);
#else
              active_vertices.insert(*it);
#endif
            }
          }
        }
      }
    }

#ifdef _OPENMP
    for(int t=0;t<omp_get_max_threads();t++){
      active_vertices.insert(partial_active_vertices[t].begin(), partial_active_vertices[t].end());
      partial_active_vertices[t].clear();
    }
#endif

    for(int iter=1;iter<max_iterations;iter++){
      for(size_t colour=0;colour<colour_sets.size(); colour++){
#pragma omp parallel
        {
          int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[colour][cn];
            // Jump to next one if this has not changed.
            if(active_vertices.count(node)==0)
              continue;

            if((this->*smooth_kernel)(node)){
              for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
#ifdef _OPENMP
                partial_active_vertices[omp_get_thread_num()].insert(*it);
#else
                active_vertices.insert(*it);
#endif
              }
            }
          }
        }
      }
      
#ifdef _OPENMP
      active_vertices.clear();
      for(int t=0;t<omp_get_max_threads();t++){
        active_vertices.insert(partial_active_vertices[t].begin(), partial_active_vertices[t].end());
        partial_active_vertices[t].clear();
      }
#endif
      
      if(active_vertices.empty())
        break;
    }
    
    return;
  }

  bool laplacian_2d_kernel(index_t node){
    real_t p[2], mp[4];
    
    if(laplacian_2d_kernel(node, p, mp)){
      // Looks good so lets copy it back;
      for(size_t j=0;j<2;j++)
        _mesh->_coords[node*2+j] = p[j];
      
      for(size_t j=0;j<4;j++)
        _mesh->metric[node*4+j] = mp[j];
      
      return true;
    }
    return false;
  }
  
  bool laplacian_2d_kernel(index_t node, real_t *p, real_t *mp){
    const real_t *normal=NULL;
    std::deque<index_t> adj_nodes;
    if(_surface->contains_node(node)){
      // Check how many different planes intersect at this node.
      std::set<int> coids;
      std::set<index_t> patch = _surface->get_surface_patch(node);
      for(typename std::set<index_t>::const_iterator e=patch.begin();e!=patch.end();++e)
        coids.insert(_surface->get_coplanar_id(*e));

      if(coids.size()==1){
        /* We will need the normal later when making sure that point
           is on the surface to within roundoff.*/
        normal = _surface->get_normal(*patch.begin());

        // Find the adjacent nodes that are on this surface.
        std::set<index_t> adj_nodes_set;
        for(typename std::set<index_t>::const_iterator e=patch.begin();e!=patch.end();++e){
          const index_t *facet = _surface->get_facet(*e);
          adj_nodes_set.insert(facet[0]);
          adj_nodes_set.insert(facet[1]);
        }
        for(typename std::set<index_t>::const_iterator il=adj_nodes_set.begin();il!=adj_nodes_set.end();++il){
          if((*il)!=node)
            adj_nodes.push_back(*il);
        }
        assert(adj_nodes.size()==2);
      }else{
        // Corner node, in which case it cannot be moved.
        return false;
      }
    }else{
      adj_nodes.insert(adj_nodes.end(), _mesh->NNList[node].begin(), _mesh->NNList[node].end());
    }
    
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);
    for(typename std::deque<index_t>::const_iterator il=adj_nodes.begin();il!=adj_nodes.end();++il){
      const real_t *m0 = _mesh->get_metric(node);
      const real_t *m1 = _mesh->get_metric(*il);
      
      real_t ml00 = 0.5*(m0[0]+m1[0]);
      real_t ml01 = 0.5*(m0[1]+m1[1]);
      real_t ml11 = 0.5*(m0[3]+m1[3]);
      
      q[0] += (ml00*get_x(*il) + ml01*get_y(*il));
      q[1] += (ml01*get_x(*il) + ml11*get_y(*il));
      
      A[0] += ml00;
      A[1] += ml01;
      A[3] += ml11;
    }
    A[2]=A[1];
    
    // Want to solve the system Ap=q to find the new position, p.
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);
    A.ldlt().solve(q, &b);
    //A.llt().solve(q, &b);
    //A.lu().solve(q, &b);
    
    p[0] = b[0];
    p[1] = b[1];

    // If this is on the surface, then make a roundoff correction.
    if(normal!=NULL){
      p[0] -= (p[0]-get_x(node))*fabs(normal[0]);
      p[1] -= (p[1]-get_y(node))*fabs(normal[1]);
    }

    // Interpolate metric at this new position.
    real_t l[3], L;
    int best_e;
    bool inverted;
    // 5 bisections along the search line
    for(size_t bisections=0;bisections<5;bisections++){
      // Reset values
      for(int i=0;i<3;i++)
        l[i]=-1.0;
      L=-1;
      best_e=-1;
      inverted=false;
      real_t tol=-1;
      
      for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        
        /* Check for inversion by looking at the area of element who
           node is being moved.*/
        real_t area;
        if(n[0]==node){
          area = property->area(p, x1, x2);
        }else if(n[1]==node){
          area = property->area(x0, p, x2);
        }else{
          area = property->area(x0, x1, p);
        }
        if(area<=0){
          inverted = true;
          break;
        }
        
        real_t ll[3];
        ll[0] = property->area(p,  x1, x2);
        ll[1] = property->area(x0, p,  x2);
        ll[2] = property->area(x0, x1, p);
        
        real_t min_l = min(ll[0], min(ll[1], ll[2]));
        if(best_e<0){
          tol = min_l;
          best_e = *ie;
          for(int i=0;i<3;i++)
            l[i] = ll[i];
          L = property->area(x0, x1, x2);
        }else{
          if(min_l>tol){
            tol = min_l;
            best_e = *ie;
            for(int i=0;i<3;i++)
              l[i] = ll[i];
            L = property->area(x0, x1, x2);
          }
        }
      }
      if(inverted){
        p[0] = (get_x(node)+p[0])/2;
        p[1] = (get_y(node)+p[1])/2;
      }else{
        break;
      }
    }
    
    if(inverted){
      return false;
    }
    
    assert(best_e>=0);
    {
      const index_t *n=_mesh->get_element(best_e);
      assert(n[0]>=0);
      
      for(size_t i=0;i<4;i++)
        mp[i] = (l[0]*_mesh->metric[n[0]*4+i]+
                 l[1]*_mesh->metric[n[1]*4+i]+
                 l[2]*_mesh->metric[n[2]*4+i])/L;
    }
    
    MetricTensor<real_t>::positive_definiteness(2, mp);

    return true;
  }

  bool smart_laplacian_2d_kernel(index_t node){
    real_t dq_tol=0.01;
    
    real_t p[2], mp[4];
    if(!laplacian_2d_kernel(node, p, mp))
      return false;
    
    // Check if this positions improves the local mesh quality.  
    std::map<int, real_t> new_quality;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      if(n[0]<0)
        continue;
      
      int iloc = 0;
      while(n[iloc]!=(int)node){
        iloc++;
      }
      int loc1 = (iloc+1)%3;
      int loc2 = (iloc+2)%3;
      
      const real_t *x1 = _mesh->get_coords(n[loc1]);
      const real_t *x2 = _mesh->get_coords(n[loc2]);
      
      new_quality[*ie] = property->lipnikov(p, x1, x2, 
                                            mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
    }
    
    real_t min_q=0;
    {
      typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
      min_q = quality[*ie];
      for(;ie!=_mesh->NEList[node].end();++ie)
        min_q = std::min(min_q, quality[*ie]);
    }
    real_t new_min_q=0;
    {
      typename std::map<int, real_t>::iterator ie=new_quality.begin();
      new_min_q = ie->second;
      for(;ie!=new_quality.end();++ie)
        new_min_q = std::min(new_min_q, ie->second);
    }
    
    // So - is this an improvement?
    if((new_min_q-min_q)<dq_tol)
      return false;
    
    // Looks good so lets copy it back;
    for(typename std::map<int, real_t>::const_iterator it=new_quality.begin();it!=new_quality.end();++it)
      quality[it->first] = it->second;
    
    for(size_t j=0;j<2;j++){
      _mesh->_coords[node*2+j] = p[j];
    }

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }

  bool laplacian_3d_kernel(index_t node){
    real_t p[3], mp[9];
    if(laplacian_3d_kernel(node, p, mp)){
      // Looks good so lets copy it back;
      for(size_t j=0;j<3;j++)
        _mesh->_coords[node*3+j] = p[j];
      
      for(size_t j=0;j<9;j++)
        _mesh->metric[node*9+j] = mp[j];
      
      return true;
    }
    return false;
  }

  bool laplacian_3d_kernel(index_t node, real_t *p, real_t *mp){
    const real_t *normal[]={NULL, NULL};
    std::deque<index_t> adj_nodes;
    if(_surface->contains_node(node)){
      // Check how many different planes intersect at this node.
      std::set<index_t> patch = _surface->get_surface_patch(node);
      std::map<int, std::set<int> > coids;
      for(typename std::set<index_t>::const_iterator e=patch.begin();e!=patch.end();++e)
        coids[_surface->get_coplanar_id(*e)].insert(*e);
      
      int loc=0;
      if(coids.size()<3){
        /* We will need the normals later when making sure that point
           is on the surface to within roundoff.*/
        for(std::map<int, std::set<int> >::const_iterator ic=coids.begin();ic!=coids.end();++ic){
          normal[loc++] = _surface->get_normal(*(ic->second.begin()));
        }
        
        // Find the adjacent nodes that are on this surface.
        std::set<index_t> adj_nodes_set;
        for(typename std::set<index_t>::const_iterator e=patch.begin();e!=patch.end();++e){
          const index_t *facet = _surface->get_facet(*e);
          if(facet[0]<0)
            continue;

          adj_nodes_set.insert(facet[0]); assert(_mesh->NNList[facet[0]].size()>0);
          adj_nodes_set.insert(facet[1]); assert(_mesh->NNList[facet[1]].size()>0);
          adj_nodes_set.insert(facet[2]); assert(_mesh->NNList[facet[2]].size()>0);
        }
        for(typename std::set<index_t>::const_iterator il=adj_nodes_set.begin();il!=adj_nodes_set.end();++il){
          if((*il)!=node)
            adj_nodes.push_back(*il);
        }
      }else{
        // Corner node, in which case it cannot be moved.
        return false;
      }
    }else{
      adj_nodes.insert(adj_nodes.end(), _mesh->NNList[node].begin(), _mesh->NNList[node].end());
    }
      
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 3);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
    for(typename std::deque<index_t>::const_iterator il=adj_nodes.begin();il!=adj_nodes.end();++il){
      const real_t *m0 = _mesh->get_metric(node);
      const real_t *m1 = _mesh->get_metric(*il);
      
      real_t ml00 = 0.5*(m0[0] + m1[0]);
      real_t ml01 = 0.5*(m0[1] + m1[1]);
      real_t ml02 = 0.5*(m0[2] + m1[2]);
      real_t ml11 = 0.5*(m0[4] + m1[4]);
      real_t ml12 = 0.5*(m0[5] + m1[5]);
      real_t ml22 = 0.5*(m0[8] + m1[8]);
      
      q[0] += ml00*get_x(*il) + ml01*get_y(*il) + ml02*get_z(*il);
      q[1] += ml01*get_x(*il) + ml11*get_y(*il) + ml12*get_z(*il);
      q[2] += ml02*get_x(*il) + ml12*get_y(*il) + ml22*get_z(*il);
      
      A[0] += ml00;
      A[1] += ml01;
      A[2] += ml02;
      A[4] += ml11;
      A[5] += ml12;
      A[8] += ml22;
    }
    A[3] = A[1];
    A[6] = A[2];
    A[7] = A[5];
    
    // Want to solve the system Ap=q to find the new position, p.
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
    A.ldlt().solve(q, &b);
    
    for(int i=0;i<3;i++)
      p[i] = b[i];

    if(isnan(p[0])||isnan(p[1])||isnan(p[2])){
      errno = 0;
      return false;
    }

    // If this is on the surface or edge, then make a roundoff correction.
    for(int i=0;i<2;i++)
      if(normal[i]!=NULL){
        p[0] -= (p[0]-get_x(node))*fabs(normal[i][0]);
        p[1] -= (p[1]-get_y(node))*fabs(normal[i][1]);
        p[2] -= (p[2]-get_z(node))*fabs(normal[i][2]);
      }
    
    // Interpolate metric at this new position.
    real_t l[4], L;
    int best_e;
    bool inverted=false;
    // 5 bisections along the search line
    for(size_t bisections=0;bisections<5;bisections++){ 
      // Reset values
      for(int i=0;i<4;i++)
        l[i]=-1.0;
      L=-1;
      best_e=-1;
      inverted=false;
      real_t tol=-1;

      for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        
        real_t vectors[] = {get_x(n[0]), get_y(n[0]), get_z(n[0]),
                            get_x(n[1]), get_y(n[1]), get_z(n[1]),
                            get_x(n[2]), get_y(n[2]), get_z(n[2]),
                            get_x(n[3]), get_y(n[3]), get_z(n[3])};
        real_t *x0 = vectors;
        real_t *x1 = vectors+3;
        real_t *x2 = vectors+6;
        real_t *x3 = vectors+9;
        
        real_t *r[4];
        for(int iloc=0;iloc<4;iloc++)
          if(n[iloc]==node){
            r[iloc] = p;
          }else{
            r[iloc] = vectors+3*iloc;
          }
        /* Check for inversion by looking at the volume of element who
           node is being moved.*/
        real_t volume = property->volume(r[0], r[1], r[2], r[3]);
        if(volume<=0){
          inverted = true;
          break;
        }
        
        real_t ll[4];
        ll[0] = property->volume(p,  x1, x2, x3);
        ll[1] = property->volume(x0, p,  x2, x3);
        ll[2] = property->volume(x0, x1, p,  x3);
        ll[3] = property->volume(x0, x1, x2, p);
        
        real_t min_l = min(min(ll[0], ll[1]), min(ll[2], ll[3]));
        if(best_e<0){
          tol = min_l;
          best_e = *ie;
          for(int i=0;i<4;i++)
            l[i] = ll[i];
          L = property->volume(x0, x1, x2, x3);
        }else{
          if(min_l>tol){
            tol = min_l;
            best_e = *ie;
            for(int i=0;i<4;i++)
              l[i] = ll[i];
            L = property->volume(x0, x1, x2, x3);
          }
        }
      }
      if(inverted){
        p[0] = (get_x(node)+p[0])/2;
        p[1] = (get_y(node)+p[1])/2;
        p[2] = (get_z(node)+p[2])/2;
      }else{
        break;
      }
    }
    
    if(inverted)
      return false;
    
    {
      const index_t *n=_mesh->get_element(best_e);
      assert(n[0]>=0);

      for(size_t i=0;i<9;i++)
        mp[i] =
          (l[0]*_mesh->metric[n[0]*9+i]+
           l[1]*_mesh->metric[n[1]*9+i]+
           l[2]*_mesh->metric[n[2]*9+i]+
           l[3]*_mesh->metric[n[3]*9+i])/L;
    }
    
    MetricTensor<real_t>::positive_definiteness(3, mp);

    return true;
  }
  
  bool smart_laplacian_3d_kernel(index_t node){
    real_t dq_tol=0.01;

    real_t p[3], mp[9];
    if(!laplacian_3d_kernel(node, p, mp))
      return false;
    
    // Check if this positions improves the local mesh quality.  
    std::map<int, real_t> new_quality;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      if(n[0]<0)
        continue;
      
      int iloc = 0;
      while(n[iloc]!=(int)node){
        iloc++;
      }
      int loc1 = (iloc+1)%4;
      int loc2 = (iloc+2)%4;
      int loc3 = (iloc+3)%4;
      
      const real_t *x1 = _mesh->get_coords(n[loc1]);
      const real_t *x2 = _mesh->get_coords(n[loc2]);
      const real_t *x3 = _mesh->get_coords(n[loc3]);
      
      new_quality[*ie] = property->lipnikov(p, x1, x2, x3, 
                                            mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]), _mesh->get_metric(n[loc3]));
    }
    
    real_t min_q=0;
    {
      typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
      min_q = quality[*ie];
      for(;ie!=_mesh->NEList[node].end();++ie)
        min_q = std::min(min_q, quality[*ie]);
    }
    real_t new_min_q=0;
    {
      typename std::map<int, real_t>::iterator ie=new_quality.begin();
      new_min_q = ie->second;
      for(;ie!=new_quality.end();++ie)
        new_min_q = std::min(new_min_q, ie->second);
    }
    
    // So - is this an improvement?
    if((new_min_q-min_q)<dq_tol)
      return false;
    
    // Looks good so lets copy it back;
    for(typename std::map<int, real_t>::const_iterator it=new_quality.begin();it!=new_quality.end();++it)
      quality[it->first] = it->second;
    
    for(size_t j=0;j<3;j++)
      _mesh->_coords[node*3+j] = p[j];
    
    for(size_t j=0;j<9;j++)
      _mesh->metric[node*9+j] = mp[j];
    
    return true;
  }

  bool optimisation_2d_kernel(index_t node){
    // If this is on the surface then just so a smart Laplacian smooth
    // and return.
    if(_surface->contains_node(node)){
      return smart_laplacian_2d_kernel(node);
    }

    // Use a smart Laplacian smooth to get a good starting point.
    bool node_updated = smart_laplacian_2d_kernel(node);

    double patch_quality;
    {
      typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();
      patch_quality = quality[*ie]; ++ie;
      for(;ie!=_mesh->NEList[node].end();++ie)
        patch_quality = std::min(patch_quality, quality[*ie]);
    }
    if(patch_quality>0.8)
      return node_updated;

    real_t dq_tol=0.0;
    const size_t min_patch_size = 6;
    
    // First job - create quadric fit for the local metric field.
    size_t lmin_patch_size = _surface->contains_node(node)?min_patch_size*2:min_patch_size;
    std::set<index_t> patch = _mesh->get_node_patch(node, lmin_patch_size);
    patch.erase(node);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = y^2+x^2+xy+y+x+1
    // A = P^TP
    double x=_mesh->get_coords(node)[0], y=_mesh->get_coords(node)[1];
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);
    
    A[0]+=pow(y,4); A[1]+=pow(x,2)*pow(y,2); A[2]+=x*pow(y,3); A[3]+=pow(y,3); A[4]+=x*pow(y,2); A[5]+=pow(y,2);
    A[6]+=pow(x,2)*pow(y,2); A[7]+=pow(x,4); A[8]+=pow(x,3)*y; A[9]+=pow(x,2)*y; A[10]+=pow(x,3); A[11]+=pow(x,2);
    A[12]+=x*pow(y,3); A[13]+=pow(x,3)*y; A[14]+=pow(x,2)*pow(y,2); A[15]+=x*pow(y,2); A[16]+=pow(x,2)*y; A[17]+=x*y;
    A[18]+=pow(y,3); A[19]+=pow(x,2)*y; A[20]+=x*pow(y,2); A[21]+=pow(y,2); A[22]+=x*y; A[23]+=y;
    A[24]+=x*pow(y,2); A[25]+=pow(x,3); A[26]+=pow(x,2)*y; A[27]+=x*y; A[28]+=pow(x,2); A[29]+=x;
    A[30]+=pow(y,2); A[31]+=pow(x,2); A[32]+=x*y; A[33]+=y; A[34]+=x; A[35]+=1;
    
    real_t m00 = _mesh->get_metric(node)[0];
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    b00[0]+=m00*pow(y,2); b00[1]+=m00*pow(x,2); b00[2]+=m00*x*y; b00[3]+=m00*y; b00[4]+=m00*x; b00[5]+=m00;

    real_t m01 = _mesh->get_metric(node)[1];
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    b01[0]+=m01*pow(y,2); b01[1]+=m01*pow(x,2); b01[2]+=m01*x*y; b01[3]+=m01*y; b01[4]+=m01*x; b01[5]+=m01;

    real_t m11 = _mesh->get_metric(node)[3];
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    b11[0]+=m11*pow(y,2); b11[1]+=m11*pow(x,2); b11[2]+=m11*x*y; b11[3]+=m11*y; b11[4]+=m11*x; b11[5]+=m11;
    
    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      x=_mesh->get_coords(*n)[0]; y=_mesh->get_coords(*n)[1];
      A[0]+=pow(y,4); A[1]+=pow(x,2)*pow(y,2); A[2]+=x*pow(y,3); A[3]+=pow(y,3); A[4]+=x*pow(y,2); A[5]+=pow(y,2);
      A[6]+=pow(x,2)*pow(y,2); A[7]+=pow(x,4); A[8]+=pow(x,3)*y; A[9]+=pow(x,2)*y; A[10]+=pow(x,3); A[11]+=pow(x,2);
      A[12]+=x*pow(y,3); A[13]+=pow(x,3)*y; A[14]+=pow(x,2)*pow(y,2); A[15]+=x*pow(y,2); A[16]+=pow(x,2)*y; A[17]+=x*y;
      A[18]+=pow(y,3); A[19]+=pow(x,2)*y; A[20]+=x*pow(y,2); A[21]+=pow(y,2); A[22]+=x*y; A[23]+=y;
      A[24]+=x*pow(y,2); A[25]+=pow(x,3); A[26]+=pow(x,2)*y; A[27]+=x*y; A[28]+=pow(x,2); A[29]+=x;
      A[30]+=pow(y,2); A[31]+=pow(x,2); A[32]+=x*y; A[33]+=y; A[34]+=x; A[35]+=1;
      
      m00 = _mesh->get_metric(*n)[0];
      b00[0]+=m00*pow(y,2); b00[1]+=m00*pow(x,2); b00[2]+=m00*x*y; b00[3]+=m00*y; b00[4]+=m00*x; b00[5]+=m00;
      m01 = _mesh->get_metric(*n)[1];
      b01[0]+=m01*pow(y,2); b01[1]+=m01*pow(x,2); b01[2]+=m01*x*y; b01[3]+=m01*y; b01[4]+=m01*x; b01[5]+=m01;
      m11 = _mesh->get_metric(*n)[3];
      b11[0]+=m11*pow(y,2); b11[1]+=m11*pow(x,2); b11[2]+=m11*x*y; b11[3]+=m11*y; b11[4]+=m11*x; b11[5]+=m11;
    }

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.ldlt().solve(b00, &a00);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.ldlt().solve(b01, &a01);
    
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.ldlt().solve(b11, &a11);

    real_t a0 = a00[0]; real_t a1 = a00[1]; real_t a2 = a00[2]; real_t a3 = a00[3]; real_t a4 = a00[4]; real_t a5 = a00[5];
    real_t b0 = a01[0]; real_t b1 = a01[1]; real_t b2 = a01[2]; real_t b3 = a01[3]; real_t b4 = a01[4]; real_t b5 = a01[5];
    real_t c0 = a11[0]; real_t c1 = a11[1]; real_t c2 = a11[2]; real_t c3 = a11[3]; real_t c4 = a11[4]; real_t c5 = a11[5];

    x=_mesh->get_coords(node)[0];
    y=_mesh->get_coords(node)[1];
    
    std::map<index_t, std::vector<real_t> > local_gradients;
    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[node].begin();it!=_mesh->NEList[node].end();++it){
      const index_t *ele=_mesh->get_element(*it);
      size_t loc=0;
      for(;loc<3;loc++)
        if(ele[loc]==node)
          break;
      const real_t *r1=_mesh->get_coords(ele[(loc+1)%3]);
      const real_t *m1=_mesh->get_metric(ele[(loc+1)%3]);
      const real_t *r2=_mesh->get_coords(ele[(loc+2)%3]);
      const real_t *m2=_mesh->get_metric(ele[(loc+2)%3]);

      std::vector<real_t> grad_functional(2);
      grad_r(x, y,
             a0, a1, a2, a3, a4, a5,
             b0, b1, b2, b3, b4, b5,
             c0, c1, c2, c3, c4, c5,
             r1[0], r1[1], m1[0], m1[1], m1[3],
             r2[0], r2[1], m2[0], m2[1], m2[3],
             &(grad_functional[0]));
      local_gradients[*it] = grad_functional;
    }

    /*
    // Find the element corresponding to the steepest gradient.
    index_t target_element;
    {
      typename std::map<index_t, std::vector<real_t> >::iterator lg=local_gradients.begin();
      target_element = lg->first;
      real_t steepest_grad_2 = lg->second[0]*lg->second[0]+lg->second[1]*lg->second[1];
      lg++;
      for(;lg!=local_gradients.end();++lg){
        real_t gradient_2=lg->second[0]*lg->second[0]+lg->second[1]*lg->second[1];
        if(steepest_grad_2<gradient_2){
          steepest_grad_2 = gradient_2;
          target_element = lg->first;
        }
      }
      }*/
    
    // Focus on improving the worst element.
    index_t target_element;
    {
      typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
      target_element = *ie; ++ie;
      real_t min_q=quality[target_element];
      for(;ie!=_mesh->NEList[node].end();++ie){
        if(quality[*ie]<min_q){
          target_element = *ie;
          min_q = quality[target_element];
        }
      }
    }
    
    // Find the distance we have to step to reach the local quality maximum.
    real_t alpha=-1.0;
    {
      // Initialise alpha.
      typename std::map<index_t, std::vector<real_t> >::iterator lg=local_gradients.begin();
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat = lg->second;
        real_t mag = sqrt((double)hat[0]*hat[0]+hat[1]*hat[1]);
        
        hat[0]/=mag;
        hat[1]/=mag;
        alpha = (quality[lg->first]-quality[target_element])/
          (hat[0]*(local_gradients[target_element][0]-lg->second[0])+
           hat[1]*(local_gradients[target_element][1]-lg->second[1]));
        
        if(isnan(alpha))
          continue;

        break;
      }

      // Adjust alpha to the nearest point where the patch functional intersects with another.
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat = lg->second;
        real_t mag = sqrt((double)hat[0]*hat[0]+hat[1]*hat[1]);
        
        hat[0]/=mag;
        hat[1]/=mag;
        real_t new_alpha = (quality[lg->first]-quality[target_element])/
          (hat[0]*(local_gradients[target_element][0]-lg->second[0])+
           hat[1]*(local_gradients[target_element][1]-lg->second[1]));
        
        if(isnan(new_alpha))
          continue;
        
        if(fabs(new_alpha)<fabs(alpha))
          alpha = new_alpha;
      }
    }
    
    // Check that there is a valid search direction.
    if((isnan(alpha))||(alpha<0.0))
      return false;
      
    real_t p[2], mp[4];
    {
      std::vector<real_t> hat = local_gradients[target_element];
      real_t mag = sqrt((double)hat[0]*hat[0]+hat[1]*hat[1]);
      hat[0]/=mag;
      hat[1]/=mag;
      p[0] = x + alpha*hat[0];
      p[1] = y + alpha*hat[1];
    }
    // This loop is used to try to recover if we appear to overshoot.
    bool improvement=false;
    std::map<int, real_t> new_quality;
    for(int i=0;i<4;i++){
      mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0]+a5;
      mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0]+b5;
      mp[2] = mp[1];
      mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0]+c5;
      MetricTensor<real_t>::positive_definiteness(2, mp);

      // Check if this positions improves the local mesh quality.  
      for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        if(n[0]<0)
          continue;
        
        int iloc = 0;
        while(n[iloc]!=(int)node){
          iloc++;
        }
        int loc1 = (iloc+1)%3;
        int loc2 = (iloc+2)%3;
        
        const real_t *x1 = _mesh->get_coords(n[loc1]);
        const real_t *x2 = _mesh->get_coords(n[loc2]);
        
        new_quality[*ie] = property->lipnikov(p, x1, x2, 
                                              mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
      }
      
      real_t min_q=0;
      {
        typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
        min_q = quality[*ie];
        for(;ie!=_mesh->NEList[node].end();++ie)
          min_q = std::min(min_q, quality[*ie]);
      }
      real_t new_min_q=0;
      {
        typename std::map<int, real_t>::iterator ie=new_quality.begin();
        new_min_q = ie->second;
        for(;ie!=new_quality.end();++ie)
          new_min_q = std::min(new_min_q, ie->second);
      }
      
      // So - is this an improvement?
      if((new_min_q-min_q)<dq_tol){
        p[0] = 0.5*(x+p[0]);
        p[1] = 0.5*(y+p[1]);
      }else{
        improvement=true;
        break;
      }
    }
    
    assert((!isnan(p[0]))&&(!isnan(p[1])));
    assert((!isnan(mp[0]))&&(!isnan(mp[1]))&&(!isnan(mp[3])));
    
    if(!improvement)
      return node_updated;
    
    /* I want to come back to this for plotting
    std::cout<<"What an improvement looks like = "<<alpha<<" :: "<<target_element<<std::endl;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      std::cout<<"testing element "<<*ie<<std::endl;
      for(int i=-5;i<30;i++){
        real_t p[2], mp[4];
        real_t beta=i*alpha/10;
        {
          std::vector<real_t> hat = local_gradients[target_element];
          real_t mag = sqrt((double)hat[0]*hat[0]+hat[1]*hat[1]);
          hat[0]/=mag;
          hat[1]/=mag;
          p[0] = x + beta*hat[0];
          p[1] = y + beta*hat[1];
        }
        mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0]+a5;
        mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0]+b5;
        mp[2] = mp[1];
        mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0]+c5;
      
        const index_t *n=_mesh->get_element(*ie);
        if(n[0]<0)
          continue;
      
        int iloc = 0;
        while(n[iloc]!=(int)node){
          iloc++;
        }
        int loc1 = (iloc+1)%3;
        int loc2 = (iloc+2)%3;
        
        const real_t *x1 = _mesh->get_coords(n[loc1]);
        const real_t *x2 = _mesh->get_coords(n[loc2]);
        
        std::cout<<beta<<" "<<property->lipnikov(p, x1, x2, 
                                                 mp,
                                                 _mesh->get_metric(n[loc1]), 
                                                 _mesh->get_metric(n[loc2]))<<"\n";
      }
      std::cout<<std::endl;
    } 
    
    */

    // Looks good so lets copy it back;
    for(typename std::map<int, real_t>::const_iterator it=new_quality.begin();it!=new_quality.end();++it)
      quality[it->first] = it->second;
    
    for(size_t j=0;j<2;j++){
      _mesh->_coords[node*2+j] = p[j];
    }

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }

#include "dqdx_2d.h"

 private:
  void init_cache(std::string method){
    colour_sets.clear();

    int NNodes = _mesh->get_number_nodes();
    std::vector<index_t> colour(NNodes, -1);
    Colour<index_t>::greedy(_mesh->NNList, &(colour[0]));
    
    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(_mesh->is_halo_node(i)))
        continue;
      colour_sets[colour[i]].push_back(i);
    }

    if((method=="smart Laplacian")||(method=="optimisation")){
      int NElements = _mesh->get_number_elements();
      quality.resize(NElements);
#pragma omp parallel
      {
#pragma omp for schedule(static)
        for(int i=0;i<NElements;i++){
          const int *n=_mesh->get_element(i);
          if(n[0]<0){
            quality[i] = 1.0;
            continue;
          }
          
          if(ndims==2)
            quality[i] = property->lipnikov(_mesh->get_coords(n[0]),
                                            _mesh->get_coords(n[1]),
                                            _mesh->get_coords(n[2]),
                                            _mesh->get_metric(n[0]),
                                            _mesh->get_metric(n[1]),
                                            _mesh->get_metric(n[2]));
          else
            quality[i] = property->lipnikov(_mesh->get_coords(n[0]),
                                            _mesh->get_coords(n[1]),
                                            _mesh->get_coords(n[2]),
                                            _mesh->get_coords(n[3]),
                                            _mesh->get_metric(n[0]),
                                            _mesh->get_metric(n[1]),
                                            _mesh->get_metric(n[2]),
                                            _mesh->get_metric(n[3]));
        }
      }
    }

    return;
  }

  inline real_t get_x(index_t nid){
    return _mesh->_coords[nid*ndims];
  }

  inline real_t get_y(index_t nid){
    return _mesh->_coords[nid*ndims+1];
  }

  inline real_t get_z(index_t nid){
    return _mesh->_coords[nid*ndims+2];
  }

  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  size_t ndims, nloc;
  std::vector<real_t> quality;
  std::map<int, std::deque<index_t> > colour_sets;
};
#endif
