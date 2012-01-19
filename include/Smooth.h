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

#include <algorithm>
#include <omp.h>
#include <set>
#include <map>
#include <vector>
#include <deque>
#include <limits>

#include "errno.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <errno.h>

#include "ElementProperty.h"
#include "Surface.h"
#include "Mesh.h"
#include "MetricTensor.h"

#include "zoltan_colour.h"

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
    
    mpi_nparts = 1;
    rank=0;
#ifdef HAVE_MPI
    if(MPI::Is_initialized()){
      MPI_Comm_size(_mesh->get_mpi_comm(), &mpi_nparts);
      MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
    }
#endif

    sigma_q = 0.0001;
    good_q = 0.7;

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
  
  // Smooth the mesh using a given method. Valid methods are:
  // "Laplacian", "smart Laplacian", "smart Laplacian search", "optimisation Linf"
  void smooth(std::string method, int max_iterations=10){
    init_cache(method);

    std::deque<int> halo_elements;
    int NElements = _mesh->get_number_elements();
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      for(size_t j=0;j<nloc;j++){
        if(!_mesh->is_owned_node(n[j])){
          halo_elements.push_back(i);
          break;
        }
      }
    } 

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
    }else if(method=="smart Laplacian search"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_search_2d_kernel;
      //else
      //  smooth_kernel = &Smooth<real_t, index_t>::smart_laplacian_search_3d_kernel;
    }else if(method=="optimisation Linf"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::optimisation_linf_2d_kernel;
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
    int NNodes = _mesh->get_number_nodes();
    std::vector<int> active_vertices(NNodes, 0);

    // First sweep through all vertices. Add vertices adjancent to any
    // vertex moved into the active_vertex list.
    int max_colour = colour_sets.rbegin()->first;
#ifdef HAVE_MPI
    if(mpi_nparts>1)
      MPI_Allreduce(MPI_IN_PLACE, &max_colour, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
#endif

    for(int ic=1;ic<=max_colour;ic++){
      if(colour_sets.count(ic)){
#pragma omp parallel
        {
          int node_set_size = colour_sets[ic].size();
#pragma omp for schedule(static)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[ic][cn];
            
            if((this->*smooth_kernel)(node)){
              for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                active_vertices[*it] = 1;
              }
            }
          }
        }
      }
      _mesh->halo_update(&(_mesh->_coords[0]), ndims);
      _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
      for(std::deque<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
        quality[*ie] = -1;
    }

    for(int iter=1;iter<max_iterations;iter++){
      for(int ic=1;ic<=max_colour;ic++){
        if(colour_sets.count(ic)){
#pragma omp parallel
          {
            int node_set_size = colour_sets[ic].size();
#pragma omp for schedule(dynamic)
            for(int cn=0;cn<node_set_size;cn++){
              index_t node = colour_sets[ic][cn];
              
              // Only process if it is active.
              if(!active_vertices[node])
                continue;

              // Reset mask
              active_vertices[node] = 0;

              if((this->*smooth_kernel)(node)){
                for(typename std::deque<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                  active_vertices[*it] = 1;
                }
              }
            }
          }
        }
        _mesh->halo_update(&(_mesh->_coords[0]), ndims);
        _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
        for(std::deque<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
          quality[*ie] = -1;
      }
      
      // Count number of active vertices.
      int nav = 0;      
#pragma omp parallel reduction(+:nav)
      {
#pragma omp for schedule(static)
        for(int i=0;i<NNodes;i++){
          if(_mesh->is_owned_node(i))
            nav += active_vertices[i];
        }
      }
#ifdef HAVE_MPI
      if(mpi_nparts>1)
        MPI_Allreduce(MPI_IN_PLACE, &nav, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
#endif
      
      if(nav==0)
        break;
    }

    return;
  }

  bool laplacian_2d_kernel(index_t node){
    real_t p[2];
    bool valid = laplacian_2d_kernel(node, p);
    if(!valid)
      return false;
    
    real_t mp[4];
    valid = generate_location_2d(node, p, mp);
    if(!valid)
      return false;

    for(size_t j=0;j<2;j++)
      _mesh->_coords[node*2+j] = p[j];
    
    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }
  
  bool smart_laplacian_2d_kernel(index_t node){
    real_t p[2];
    if(!laplacian_2d_kernel(node, p))
      return false;

    real_t mp[4];
    bool valid = generate_location_2d(node, p, mp);
    if(!valid)
      return false;
    
    real_t functional = functional_Linf(node, p, mp);
    real_t functional_orig = functional_Linf(node);

    if(functional-functional_orig<sigma_q)
      return false;

    // Reset quality cache.
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie)
      quality[*ie] = -1;
    
    _mesh->_coords[node*2  ] = p[0];
    _mesh->_coords[node*2+1] = p[1];

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }

  bool smart_laplacian_search_2d_kernel(index_t node){
    real_t x0 = get_x(node);
    real_t y0 = get_y(node);

    real_t p[2];
    if(!laplacian_2d_kernel(node, p))
      return false;
    
    p[0] -= x0;
    p[1] -= y0;
    
    real_t mag = sqrt(p[0]*p[0]+p[1]*p[1]);
    real_t hat[] = {p[0]/mag, p[1]/mag};

    // This can happen if there is zero mag.
    if(!isnormal(hat[0]+hat[1]))
      return false;

    real_t mp[4];

    // Initialise alpha.
    bool valid=false;
    real_t alpha = mag, functional;
    for(int rb=0;rb<5;rb++){
      p[0] = x0 + alpha*hat[0];
      p[1] = y0 + alpha*hat[1];
      
      valid = generate_location_2d(node, p, mp);

      if(valid){
        functional = functional_Linf(node, p, mp);
        break;
      }else{
        alpha*=0.5;
        continue;
      }
    }
    if(!valid)
      return false;

    // Recursive bisection search along line.
    const real_t functional_orig = functional_Linf(node);
    std::pair<real_t, real_t> alpha_lower(0, functional_orig), alpha_upper(alpha, functional);
    
    for(int rb=0;rb<10;rb++){
      alpha = (alpha_lower.first+alpha_upper.first)*0.5;
      p[0] = x0 + alpha*hat[0];
      p[1] = y0 + alpha*hat[1];

      valid = generate_location_2d(node, p, mp);
      assert(valid);

      // Check if this positions improves the L-infinity norm.
      functional = functional_Linf(node, p, mp);
      
      if(alpha_lower.second<functional){
        alpha_lower.first = alpha;
        alpha_lower.second = functional;
      }else{
        if(alpha_upper.second<functional){
          alpha_upper.first = alpha;
          alpha_upper.second = functional;
        }else{
          alpha = alpha_upper.first;
          functional = alpha_upper.second;
          p[0] = x0 + alpha*hat[0];
          p[1] = y0 + alpha*hat[1];
          break;
        }
      }
    }
    assert(valid);

    if(functional-functional_orig<sigma_q)
      return false;

    // Reset quality cache.
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie)
      quality[*ie] = -1;
    
    _mesh->_coords[node*2  ] = p[0];
    _mesh->_coords[node*2+1] = p[1];

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }
  
  bool laplacian_2d_kernel(index_t node, real_t *p){
    const real_t *normal=NULL;
    std::set<index_t> patch;
    if(_surface->contains_node(node)){
      // Check how many different planes intersect at this node.
      std::set<int> coids;
      std::set<index_t> epatch = _surface->get_surface_patch(node);
      for(typename std::set<index_t>::const_iterator e=epatch.begin();e!=epatch.end();++e)
        coids.insert(_surface->get_coplanar_id(*e));

      if(coids.size()==1){
        /* We will need the normal later when making sure that point
           is on the surface to within roundoff.*/
        normal = _surface->get_normal(*epatch.begin());
        
        // Find the adjacent nodes that are on this surface.
        for(typename std::set<index_t>::const_iterator e=epatch.begin();e!=epatch.end();++e){
          const index_t *facet = _surface->get_facet(*e);
          patch.insert(facet[0]);
          patch.insert(facet[1]);
        }
        patch.erase(node);
        assert(patch.size()==2);
      }else{
        // Corner node, in which case it cannot be moved.
        return false;
      }
    }else{
      patch = _mesh->get_node_patch(node);
    }
    
    real_t x0 = get_x(node);
    real_t y0 = get_y(node);

    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);

    for(typename std::set<index_t>::const_iterator il=patch.begin();il!=patch.end();++il){
      const real_t *m0 = _mesh->get_metric(node);
      const real_t *m1 = _mesh->get_metric(*il);
      
      real_t ml00 = 0.5*(m0[0]+m1[0]);
      real_t ml01 = 0.5*(m0[1]+m1[1]);
      real_t ml11 = 0.5*(m0[3]+m1[3]);
      
      real_t x = get_x(*il)-x0;
      real_t y = get_y(*il)-y0;
      
      q[0] += (ml00*x + ml01*y);
      q[1] += (ml01*x + ml11*y);
      
      A[0] += ml00; A[1] += ml01;
                    A[3] += ml11;
    }
    A[2]=A[1];
    
    // Want to solve the system Ap=q to find the new position, p.
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);
    A.svd().solve(q, &b);
    
    for(size_t i=0;i<2;i++)
      p[i] = b[i];

    // If this is on the surface, then make a roundoff correction.
    if(normal!=NULL){
      p[0] -= p[0]*fabs(normal[0]);
      p[1] -= p[1]*fabs(normal[1]);
    }

    p[0] += x0;
    p[1] += y0;

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
    A.svd().solve(q, &b);
    
    for(int i=0;i<3;i++)
      p[i] = b[i];

    if(!isnormal(p[0]+p[1]+p[2])){
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
        
        real_t min_l = std::min(std::min(ll[0], ll[1]), std::min(ll[2], ll[3]));
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

    return true;
  }
  
  bool smart_laplacian_3d_kernel(index_t node){
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
      ++ie;
      for(;ie!=_mesh->NEList[node].end();++ie)
        min_q = std::min(min_q, quality[*ie]);
    }
    real_t new_min_q=0;
    {
      typename std::map<int, real_t>::iterator ie=new_quality.begin();
      new_min_q = ie->second;
      ++ie;
      for(;ie!=new_quality.end();++ie)
        new_min_q = std::min(new_min_q, ie->second);
    }
    
    // Check if this is an improvement.
    if(new_min_q-min_q<sigma_q)
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

  bool optimisation_linf_2d_kernel(index_t node){
    bool update = smart_laplacian_search_2d_kernel(node);

    if(_surface->contains_node(node))
      return update;
        
    for(int hill_climb_iteration=0;hill_climb_iteration<5;hill_climb_iteration++){
      // As soon as the tolerence quality is reached, break.
      const real_t functional_0 = functional_Linf(node);  
      if(functional_0>good_q)
        break;

      // Differentiate quality functional for elements with respect to x,y
      std::map<index_t, std::vector<real_t> > local_gradients;
      std::multimap<real_t, index_t> priority_elements;
      for(typename std::set<index_t>::const_iterator it=_mesh->NEList[node].begin();it!=_mesh->NEList[node].end();++it){
        const index_t *ele=_mesh->get_element(*it);
        size_t loc=0;
        for(;loc<3;loc++)
          if(ele[loc]==node)
            break;
        
        const real_t *r1=_mesh->get_coords(ele[(loc+1)%3]);
        const real_t *r2=_mesh->get_coords(ele[(loc+2)%3]);
        
        const real_t *m1=_mesh->get_metric(ele[(loc+1)%3]);
        const real_t *m2=_mesh->get_metric(ele[(loc+2)%3]);
        
        std::vector<real_t> grad_functional(2);
        grad_r(node,
               r1, m1,
               r2, m2,
               grad_functional);
        local_gradients[*it] = grad_functional;
        
        // Focusing on improving the worst element
        real_t key = quality[*it];
        priority_elements.insert(std::pair<real_t, index_t>(key, *it));
      }
      
      // Find the distance we have to step to reach the local quality maximum.
      real_t alpha=-1.0;
      index_t target_element = priority_elements.begin()->second;
      
      std::vector<real_t> hat0 = local_gradients[target_element];
      real_t mag0 = sqrt(hat0[0]*hat0[0]+hat0[1]*hat0[1]);
      hat0[0]/=mag0;
      hat0[1]/=mag0;
      
      typename std::map<index_t, std::vector<real_t> >::iterator lg=local_gradients.begin();
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat1 = lg->second;
        real_t mag1 = sqrt(hat1[0]*hat1[0]+hat1[1]*hat1[1]);
        hat1[0]/=mag1;
        hat1[1]/=mag1;
        
        alpha = (quality[lg->first] - quality[target_element])/
          (mag0-(hat0[0]*hat1[0]+hat0[1]*hat1[1])*mag1);
        
        if((!isnormal(alpha)) || (alpha<0)){
          alpha = -1;
          continue;
        }
        
        break;
      }

      // Adjust alpha to the nearest point where the patch functional intersects with another.
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat1 = lg->second;
        real_t mag1 = sqrt(hat1[0]*hat1[0]+hat1[1]*hat1[1]);
        hat1[0]/=mag1;
        hat1[1]/=mag1;
        
        real_t new_alpha = (quality[lg->first] - quality[target_element])/
          (mag0-(hat0[0]*hat1[0]+hat0[1]*hat1[1])*mag1);
        
        if((!isnormal(new_alpha)) || (new_alpha<0))
          continue;
        
        if(new_alpha<alpha)
          alpha = new_alpha;
      }
      
      // If there is no viable direction, break.
      if((!isnormal(alpha))||(alpha<=0))
        break;
    
      // -
      real_t p[2], gp[2], mp[4];
      std::map<int, real_t> new_quality;
      bool valid_move = false;
      for(int i=0;i<10;i++){
        // If the predicted improvement is less than sigma, break;
        if(mag0*alpha<sigma_q)
          break;

        new_quality.clear();
        
        p[0] = alpha*hat0[0];
        p[1] = alpha*hat0[1];
        
        if(!isnormal(p[0]+p[1])){ // This can happen if there is zero gradient.
          std::cerr<<"WARNING: apparently no gradients for mesh smoothing!!\n";
          break;
        }
        
        const real_t *r0=_mesh->get_coords(node);
        gp[0] = r0[0]+p[0]; gp[1] = r0[1]+p[1];      
        valid_move = generate_location_2d(node, gp, mp);
        if(!valid_move){
          alpha/=2;
          continue;
        }
        
        assert(isnormal(p[0]+p[0]));
        assert(isnormal(mp[0]+mp[1]+mp[3]));
        
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
          
          real_t functional = property->lipnikov(gp, x1, x2, 
                                                 mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
          assert(isnormal(functional));
          if(functional-functional_0<sigma_q){
            alpha/=2;
            valid_move = false;
            break;
          }
          
          new_quality[*ie] = functional;
        }
        if(valid_move)
          break;
      }
      
      if(valid_move)
        update = true;
      else
        break;
      
      // Looks good so lets copy it back;
      for(typename std::map<int, real_t>::const_iterator it=new_quality.begin();it!=new_quality.end();++it)
        quality[it->first] = it->second;
      
      _mesh->_coords[node*2  ] = gp[0];
      _mesh->_coords[node*2+1] = gp[1];
      
      for(size_t j=0;j<4;j++)
        _mesh->metric[node*4+j] = mp[j];
    }

    return update;
  }

 private:
  void init_cache(std::string method){
    colour_sets.clear();

    zoltan_colour_graph_t graph;
    graph.rank = rank; 
    
    int NNodes = _mesh->get_number_nodes();
    assert(NNodes==(int)_mesh->NNList.size());
    graph.nnodes = NNodes;
    
    int NPNodes;
    std::vector<index_t> lnn2gnn;
    std::vector<size_t> owner;
    _mesh->create_global_node_numbering(NPNodes, lnn2gnn, owner);
    graph.npnodes = NPNodes;
    
    std::vector<size_t> nedges(NNodes);
    size_t sum = 0;
    for(int i=0;i<NNodes;i++){
      size_t cnt = _mesh->NNList[i].size();
      nedges[i] = cnt;
      sum+=cnt;
    }
    graph.nedges = &(nedges[0]);

    std::vector<size_t> csr_edges(sum);
    sum=0;
    for(int i=0;i<NNodes;i++){
      for(typename std::deque<index_t>::iterator it=_mesh->NNList[i].begin();it!=_mesh->NNList[i].end();++it){
        csr_edges[sum++] = *it;
      }
    }
    graph.csr_edges = &(csr_edges[0]);

    graph.gid = &(lnn2gnn[0]);
    graph.owner = &(owner[0]);

    std::vector<int> colour(NNodes);
    graph.colour = &(colour[0]);
    zoltan_colour(&graph, 1, _mesh->get_mpi_comm());
    
    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(!_mesh->is_owned_node(i)))
        continue;
      colour_sets[colour[i]].push_back(i);
    }

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

    return;
  }

  void grad_r(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4, real_t a5,
              real_t b0, real_t b1, real_t b2, real_t b3, real_t b4, real_t b5,
              real_t c0, real_t c1, real_t c2, real_t c3, real_t c4, real_t c5,
              const real_t *r1, const real_t *m1,
              const real_t *r2, const real_t *m2,
              real_t *grad){
    real_t linf = std::max(std::max(fabs(r1[0]), fabs(r2[0])), std::max(fabs(r1[1]), fabs(r2[1])));    
    real_t delta=linf*1.0e-2;

    real_t p[2];
    real_t mp[4];
    
    p[0] = -delta/2; p[1] = 0;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = mp[1];
    mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor<real_t>::positive_definiteness(2, mp);
    real_t functional_minus_dx = property->lipnikov(p, r1, r2, 
                                                    mp, m1, m2);

    p[0] = delta/2; p[1] = 0;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = mp[1];
    mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor<real_t>::positive_definiteness(2, mp);
    real_t functional_plus_dx = property->lipnikov(p, r1, r2, 
                                                    mp, m1, m2);
    
    grad[0] = (functional_plus_dx-functional_minus_dx)/delta;
    
    p[0] = 0; p[1] = -delta/2;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = mp[1];
    mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor<real_t>::positive_definiteness(2, mp);
    real_t functional_minus_dy = property->lipnikov(p, r1, r2, 
                                                    mp, m1, m2);
    
    p[0] = 0; p[1] = delta/2;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = mp[1];
    mp[3] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor<real_t>::positive_definiteness(2, mp);
    real_t functional_plus_dy = property->lipnikov(p, r1, r2, 
                                                   mp, m1, m2);
    
    grad[1] = (functional_plus_dy-functional_minus_dy)/delta;
  }

  void grad_r(index_t node,
              const real_t *r1, const real_t *m1,
              const real_t *r2, const real_t *m2,
              std::vector<real_t> &grad){

    grad[0] = 0;
    grad[1] = 0;
    
    const real_t *r0=_mesh->get_coords(node);

    real_t linf_x = std::max(fabs(r1[0]-r0[0]), fabs(r2[0]-r0[0]));
    real_t delta_x=linf_x*1.0e-2;

    real_t linf_y = std::max(fabs(r1[1]-r0[1]), fabs(r2[1]-r0[1]));
    real_t delta_y=linf_y*1.0e-1;

    real_t p[2];
    real_t mp[4];

    bool valid_move_minus_x=false, valid_move_plus_x=false;
    real_t functional_minus_dx=0, functional_plus_dx=0;
    for(int i=0;(i<5)&&(!valid_move_minus_x)&&(!valid_move_plus_x);i++){
      p[0] = r0[0]-delta_x/2; p[1] = r0[1];
      valid_move_minus_x = generate_location_2d(node, p, mp);
      if(valid_move_minus_x)
        functional_minus_dx = property->lipnikov(p,  r1, r2, 
                                                 mp, m1, m2);
      
      p[0] = r0[0]+delta_x/2; p[1] = r0[1];
      valid_move_plus_x = generate_location_2d(node, p, mp);
      if(valid_move_plus_x)
        functional_plus_dx = property->lipnikov(p,  r1, r2, 
                                                mp, m1, m2);
      
      if((!valid_move_minus_x)&&(!valid_move_plus_x))
        delta_x/=2;
    }

    bool valid_move_minus_y=false, valid_move_plus_y=false;
    real_t functional_minus_dy=0, functional_plus_dy=0;
    for(int i=0;(i<5)&&(!valid_move_minus_y)&&(!valid_move_plus_y);i++){
      p[0] = r0[0]; p[1] = r0[1]-delta_y/2;
      valid_move_minus_y = generate_location_2d(node, p, mp);
      if(valid_move_minus_y)
        functional_minus_dy = property->lipnikov(p,  r1, r2, 
                                                 mp, m1, m2);
    
      p[0] = r0[0]; p[1] = r0[1]+delta_y/2;
      valid_move_plus_y = generate_location_2d(node, p, mp);
      if(valid_move_plus_y)
        functional_plus_dy = property->lipnikov(p,  r1, r2, 
                                                mp, m1, m2);

      if((!valid_move_minus_y)&&(!valid_move_plus_y))
        delta_y/=2;
    }

    if(valid_move_minus_x && valid_move_plus_x){
      grad[0] = (functional_plus_dx-functional_minus_dx)/delta_x;    
    }else if(valid_move_minus_x){
      grad[0] = (quality[node]-functional_minus_dx)/(delta_x*0.5);    
    }else if(valid_move_plus_x){
      grad[0] = (functional_plus_dx-quality[node])/(delta_x*0.5);    
    }

    if(valid_move_minus_y && valid_move_plus_y){
      grad[1] = (functional_plus_dy-functional_minus_dy)/delta_y;    
    }else if(valid_move_minus_y){
      grad[1] = (quality[node]-functional_minus_dy)/(delta_y*0.5);    
    }else if(valid_move_plus_y){
      grad[1] = (functional_plus_dy-quality[node])/(delta_y*0.5);    
    }
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

  real_t functional_Linf(index_t node){
    double patch_quality = std::numeric_limits<double>::max();
    
    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      // Check cache - if it's stale then recalculate. 
      if(quality[*ie]<0){
        const int *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        std::vector<const real_t *> x(nloc), m(nloc);
        for(size_t i=0;i<nloc;i++){
          x[i] = _mesh->get_coords(n[i]);
          m[i] = _mesh->get_metric(n[i]);
        }
        if(ndims==2)
          quality[*ie] = property->lipnikov(x[0], x[1], x[2], 
                                            m[0], m[1], m[2]);
        else
          quality[*ie] = property->lipnikov(x[0], x[1], x[2], x[3],
                                            m[0], m[1], m[2], m[3]);
      }
      
      patch_quality = std::min(patch_quality, quality[*ie]);
    }

    return patch_quality;
  }
  
  real_t functional_Linf(index_t node, const real_t *p, const real_t *mp) const{
    real_t functional = DBL_MAX;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      assert(n[0]>=0);
      int iloc = 0;
      
      while(n[iloc]!=(int)node){
        iloc++;
      }
      int loc1 = (iloc+1)%3;
      int loc2 = (iloc+2)%3;
      
      const real_t *x1 = _mesh->get_coords(n[loc1]);
      const real_t *x2 = _mesh->get_coords(n[loc2]);
      
      real_t fnl = property->lipnikov(p, x1, x2, 
                                      mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
      functional = std::min(functional, fnl);
    }
    return functional;
  }

  bool generate_location_2d(index_t node, const real_t *p, real_t *mp){
    // Interpolate metric at this new position.
    real_t l[]={-1, -1, -1};
    int best_e=-1;
    real_t tol=-1;
    
    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
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
      if(area<0)
        return false;
      
      real_t L = property->area(x0, x1, x2);

      real_t ll[3];
      ll[0] = property->area(p,  x1, x2)/L;
      ll[1] = property->area(x0, p,  x2)/L;
      ll[2] = property->area(x0, x1, p)/L;

      real_t min_l = std::min(ll[0], std::min(ll[1], ll[2]));
      if(best_e==-1){
        tol = min_l;
        best_e = *ie;
        for(int i=0;i<3;i++)
          l[i] = ll[i];
      }else{
        if(min_l>tol){
          tol = min_l;
          best_e = *ie;
          for(int i=0;i<3;i++)
            l[i] = ll[i];
        }
      }
    }

    const index_t *n=_mesh->get_element(best_e);
    assert(n[0]>=0);
    
    for(size_t i=0;i<4;i++)
      mp[i] = 
        l[0]*_mesh->metric[n[0]*4+i]+
        l[1]*_mesh->metric[n[1]*4+i]+
        l[2]*_mesh->metric[n[2]*4+i];
    
    return true;
  }
  
  Mesh<real_t, index_t> *_mesh;
  Surface<real_t, index_t> *_surface;
  ElementProperty<real_t> *property;
  size_t ndims, nloc;
  int mpi_nparts, rank;
  real_t good_q, sigma_q;
  std::vector<real_t> quality;
  std::map<int, std::deque<index_t> > colour_sets;
};
#endif
