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
    
    rtol = 0.001;
    relax=1.0;
    // relax=0.001;

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
  // "Laplacian", "smart Laplacian", "optimisation L2", "optimisation Linf"
  void smooth(std::string method, int max_iterations=10){
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
    }else if(method=="optimisation Linf"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::optimisation_linf_2d_kernel;
      //else
      //  smooth_kernel = &Smooth<real_t, index_t>::optimisation_3d_kernel;
    }else if(method=="optimisation L2"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::optimisation_l2_2d_kernel;
      //else
      //  smooth_kernel = &Smooth<real_t, index_t>::optimisation_3d_kernel;
    }else if(method=="combined"){
      if(ndims==2)
        smooth_kernel = &Smooth<real_t, index_t>::combined_2d_kernel;
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
    int max_colour = colour_sets.rbegin()->first;
    if(MPI::Is_initialized()){
      MPI_Allreduce(MPI_IN_PLACE, &max_colour, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
    }

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
                // Don't add node to the active set if it is not owned.
                if(_mesh->is_not_owned_node(*it))
                  continue;
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
      _mesh->halo_update(&(_mesh->_coords[0]), ndims);
      _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
    }

#ifdef _OPENMP
    for(int t=0;t<omp_get_max_threads();t++){
      active_vertices.insert(partial_active_vertices[t].begin(), partial_active_vertices[t].end());
      partial_active_vertices[t].clear();
    }
#endif

    for(int iter=1;iter<max_iterations;iter++){
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
                  // Don't add node to the active set if it is not owned.
                  if(_mesh->is_not_owned_node(*it))
                    continue;
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
        _mesh->halo_update(&(_mesh->_coords[0]), ndims);
        _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);
      }
      
#ifdef _OPENMP
      active_vertices.clear();
      for(int t=0;t<omp_get_max_threads();t++){
        active_vertices.insert(partial_active_vertices[t].begin(), partial_active_vertices[t].end());
        partial_active_vertices[t].clear();
      }
#endif
      int nav = active_vertices.size();
#ifdef HAVE_MPI
      if(MPI::Is_initialized()){
        MPI_Allreduce(MPI_IN_PLACE, &nav, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
      }
#endif
      if(nav==0)
        break;
    }
    _mesh->halo_update(&(_mesh->_coords[0]), ndims);
    _mesh->halo_update(&(_mesh->metric[0]), ndims*ndims);

    return;
  }

  bool laplacian_2d_kernel(index_t node){
    real_t p[2];
    bool valid = laplacian_2d_kernel(node, p);
    if(!valid)
      return false;
    
    real_t mp[4];
    valid = generate_location_2d(node, p, mp);
    if(!valid){
      /* Some verticies cannot be moved without causing inverted
         elements. To try to free up this element we inform the outter
         loop that the vertex has indeed moved so that the local
         verticies are flagged for further smoothing. This gives the
         chance of arriving at a new configuration where a valid
         smooth can be performed.*/
      return true;
    }

    for(size_t j=0;j<2;j++)
      _mesh->_coords[node*2+j] = p[j];
    
    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }
  
  bool smart_laplacian_2d_kernel(index_t node){
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
    for(int rb=0;rb<10;rb++){
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
    const real_t orig_functional = functional_Linf(node);
    std::pair<real_t, real_t> alpha_lower(0, orig_functional), alpha_upper(alpha, functional);
    
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

    //if((functional-orig_functional)/orig_functional<rtol)
    if(functional<relax*orig_functional)
      return false;

    // Recalculate qualities.
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
      quality[*ie] = property->lipnikov(p, x1, x2, 
                                        mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
    }
    
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

  bool combined_2d_kernel(index_t node){
    // If this is on the surface then we just apply regular Laplacian smooth.
    if(_surface->contains_node(node))
      return laplacian_2d_kernel(node);
    
    bool move0 = smart_laplacian_2d_kernel(node);
    bool move1 = true; // optimisation_l2_2d_kernel(node);
    bool move2 = optimisation_linf_2d_kernel(node);

    return move0||move1||move2;
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
    if((new_min_q-min_q)/min_q<rtol)
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
    real_t functional_orig = functional_Linf(node);
    if(functional_orig>0.5)
      return false;

    if(_surface->contains_node(node))
      return false;

    const size_t min_patch_size = 6;
    
    // Create quadric fit for the local metric field.
    std::set<index_t> patch = _mesh->get_node_patch(node, min_patch_size);
    patch.insert(node);

    // Form quadratic system to be solved. The quadratic fit is:
    // P = y^2+x^2+xy+y+x+1
    // A = P^TP
    real_t x0=_mesh->get_coords(node)[0], y0=_mesh->get_coords(node)[1];

    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    
    for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
      double x=_mesh->get_coords(*n)[0]-x0, y=_mesh->get_coords(*n)[1]-y0;

      A[0]+=y*y*y*y;
      A[6]+=x*x*y*y;  A[7]+=x*x*x*x;  
      A[12]+=x*y*y*y; A[13]+=x*x*x*y; A[14]+=x*x*y*y;
      A[18]+=y*y*y;   A[19]+=x*x*y;   A[20]+=x*y*y;   A[21]+=y*y;
      A[24]+=x*y*y;   A[25]+=x*x*x;   A[26]+=x*x*y;   A[27]+=x*y; A[28]+=x*x;
      A[30]+=y*y;     A[31]+=x*x;     A[32]+=x*y;     A[33]+=y;   A[34]+=x;   A[35]+=1;
      
      double m00 = _mesh->get_metric(*n)[0];
      b00[0]+=m00*pow(y,2); b00[1]+=m00*pow(x,2); b00[2]+=m00*x*y; b00[3]+=m00*y; b00[4]+=m00*x; b00[5]+=m00;

      double m01 = _mesh->get_metric(*n)[1];
      b01[0]+=m01*pow(y,2); b01[1]+=m01*pow(x,2); b01[2]+=m01*x*y; b01[3]+=m01*y; b01[4]+=m01*x; b01[5]+=m01;

      double m11 = _mesh->get_metric(*n)[3];
      b11[0]+=m11*pow(y,2); b11[1]+=m11*pow(x,2); b11[2]+=m11*x*y; b11[3]+=m11*y; b11[4]+=m11*x; b11[5]+=m11;
    }
    A[1] = A[6]; A[2] = A[12]; A[3] = A[18]; A[4] = A[24]; A[5] = A[30];
                 A[8] = A[13]; A[9] = A[19]; A[10]= A[25]; A[11]= A[31];
                               A[15]= A[20]; A[16]= A[26]; A[17]= A[32];
                                             A[22]= A[27]; A[23]= A[33];
                                                           A[29]= A[34];

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.svd().solve(b00, &a00);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.svd().solve(b01, &a01);
    
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> a11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
    A.svd().solve(b11, &a11);

    real_t a0 = a00[0]; real_t a1 = a00[1]; real_t a2 = a00[2]; real_t a3 = a00[3]; real_t a4 = a00[4]; real_t a5 = a00[5];
    real_t b0 = a01[0]; real_t b1 = a01[1]; real_t b2 = a01[2]; real_t b3 = a01[3]; real_t b4 = a01[4]; real_t b5 = a01[5];
    real_t c0 = a11[0]; real_t c1 = a11[1]; real_t c2 = a11[2]; real_t c3 = a11[3]; real_t c4 = a11[4]; real_t c5 = a11[5];

    // Differentiate quality functional for elements with respect to x,y
    std::map<index_t, std::vector<real_t> > local_gradients;
    std::multimap<real_t, index_t> priority_elements;
    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[node].begin();it!=_mesh->NEList[node].end();++it){
      const index_t *ele=_mesh->get_element(*it);
      size_t loc=0;
      for(;loc<3;loc++)
        if(ele[loc]==node)
          break;

      const real_t *_r1=_mesh->get_coords(ele[(loc+1)%3]);
      real_t r1[] = {_r1[0]-x0, _r1[1]-y0};

      const real_t *_r2=_mesh->get_coords(ele[(loc+2)%3]);
      real_t r2[] = {_r2[0]-x0, _r2[1]-y0};

      const real_t *m1=_mesh->get_metric(ele[(loc+1)%3]);
      const real_t *m2=_mesh->get_metric(ele[(loc+2)%3]);

      std::vector<real_t> grad_functional(2);
      grad_r(a0, a1, a2, a3, a4, a5,
             b0, b1, b2, b3, b4, b5,
             c0, c1, c2, c3, c4, c5,
             r1, m1,
             r2, m2,
             &(grad_functional[0]));
      local_gradients[*it] = grad_functional;
      
      //real_t key = grad_functional[0]*grad_functional[0]+grad_functional[1]*grad_functional[1];
      real_t key = quality[*it];
      priority_elements.insert(std::pair<real_t, index_t>(key, *it));
    }

    // Find the distance we have to step to reach the local quality maximum.
    real_t alpha=-1;
    index_t target_element;
    for(typename std::multimap<real_t, index_t>::const_iterator pe=priority_elements.begin();pe!=priority_elements.end();++pe){
      // Initialise alpha.
      alpha = -1.0;
      target_element = pe->second;
      typename std::map<index_t, std::vector<real_t> >::iterator lg=local_gradients.begin();
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat = lg->second;
        real_t mag = sqrt(hat[0]*hat[0]+hat[1]*hat[1]);
        
        hat[0]/=mag;
        hat[1]/=mag;

        alpha = (quality[lg->first]-quality[target_element])/
          (hat[0]*(local_gradients[target_element][0]-lg->second[0])+
           hat[1]*(local_gradients[target_element][1]-lg->second[1]));
        
        if(!isnormal(alpha) || alpha<=0){
          alpha = -1;
          continue;
        }

        break;
      }
      
      // Adjust alpha to the nearest point where the patch functional intersects with another.
      for(;lg!=local_gradients.end();++lg){
        if(lg->first == target_element)
          continue;
        
        std::vector<real_t> hat = lg->second;
        real_t mag = sqrt(hat[0]*hat[0]+hat[1]*hat[1]);
        
        hat[0]/=mag;
        hat[1]/=mag;

        real_t new_alpha = (quality[lg->first]-quality[target_element])/
          (hat[0]*(local_gradients[target_element][0]-lg->second[0])+
           hat[1]*(local_gradients[target_element][1]-lg->second[1]));
        
        if(!isnormal(new_alpha) || new_alpha<=0)
          continue;
        
        if(new_alpha<alpha)
          alpha = new_alpha;
      }

      // If this looks like a viable step then break from loop.
      if(isnormal(alpha)&&(alpha>0.0))
        break;
    }
    
    // If there is no viable direction to step in then return.
    if((!isnormal(alpha))||(alpha<=0))
      return false;

    // -
    real_t p[2], mp[4];
    {
      std::vector<real_t> hat = local_gradients[target_element];
      real_t mag = sqrt(hat[0]*hat[0]+hat[1]*hat[1]);
      hat[0]/=mag;
      hat[1]/=mag;
      p[0] = alpha*hat[0];
      p[1] = alpha*hat[1];
    }
    
    if(!isnormal(p[0]+p[1])){ // This can happen if there is zero gradient.
      return false;
    }

    real_t gp[] = {x0+p[0], y0+p[1]};      
    bool valid_move = generate_location_2d(node, gp, mp);
    if(!valid_move){
      return false;
    }

    assert(isnormal(p[0]+p[0]));
    assert(isnormal(mp[0]+mp[1]+mp[3]));

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
      
      const real_t *_x1 = _mesh->get_coords(n[loc1]);
      const real_t *_x2 = _mesh->get_coords(n[loc2]);
      real_t x1[] = {_x1[0]-x0, _x1[1]-y0};
      real_t x2[] = {_x2[0]-x0, _x2[1]-y0};
      real_t functional = property->lipnikov(p, x1, x2, 
                                             mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
      assert(isnormal(functional));
      if(functional<functional_orig){
        return false;
      }

      new_quality[*ie] = functional;
    }

    /*
        // I want to come back to this for plotting
        std::cout<<"What an improvement looks like = "<<alpha<<std::endl;
        for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
          std::cout<<"testing element "<<*ie<<std::endl;
          for(int i=-10;i<=10;i++){
            real_t t_p[2], t_mp[4];
            real_t t_alpha=alpha + i*alpha*0.2;
            {
              std::vector<real_t> t_hat = local_gradients[target_element];
              real_t t_mag = sqrt(t_hat[0]*t_hat[0]+t_hat[1]*t_hat[1]);
              t_hat[0]/=t_mag;
              t_hat[1]/=t_mag;
              t_p[0] = t_alpha*t_hat[0];
              t_p[1] = t_alpha*t_hat[1];
            }
            t_mp[0] = a0*t_p[1]*t_p[1]+a1*t_p[0]*t_p[0]+a2*t_p[0]*t_p[1]+a3*t_p[1]+a4*t_p[0]+a5;
            t_mp[1] = b0*t_p[1]*t_p[1]+b1*t_p[0]*t_p[0]+b2*t_p[0]*t_p[1]+b3*t_p[1]+b4*t_p[0]+b5;
            t_mp[2] = t_mp[1];
            t_mp[3] = c0*t_p[1]*t_p[1]+c1*t_p[0]*t_p[0]+c2*t_p[0]*t_p[1]+c3*t_p[1]+c4*t_p[0]+c5;
            
            const index_t *n=_mesh->get_element(*ie);
            if(n[0]<0)
              continue;
            
            int iloc = 0;
            while(n[iloc]!=(int)node){
              iloc++;
            }
            int loc1 = (iloc+1)%3;
            int loc2 = (iloc+2)%3;
            
            const real_t *_x1 = _mesh->get_coords(n[loc1]);
            const real_t *_x2 = _mesh->get_coords(n[loc2]);
            real_t x1[] = {_x1[0]-x0, _x1[1]-y0};
            real_t x2[] = {_x2[0]-x0, _x2[1]-y0};
            std::cout<<t_alpha<<" "<<property->lipnikov(t_p, x1, x2, 
                                                        t_mp,
                                                        _mesh->get_metric(n[loc1]), 
                                                        _mesh->get_metric(n[loc2]))-quality[*ie]<<"\n";
          }
          std::cout<<std::endl;
        } 
    */
        
    // Looks good so lets copy it back;
    for(typename std::map<int, real_t>::const_iterator it=new_quality.begin();it!=new_quality.end();++it)
      quality[it->first] = it->second;
    
    _mesh->_coords[node*2  ] = p[0] + x0;
    _mesh->_coords[node*2+1] = p[1] + y0;

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }

  bool optimisation_l2_2d_kernel(index_t node){
    const size_t min_patch_size = 6;
    real_t x0=get_x(node), y0=get_y(node);
    real_t p[2], hat[2], alpha;
    
    if(_surface->contains_node(node)){
      if(!laplacian_2d_kernel(node, p))
        return false;
      
      p[0] -= x0;
      p[1] -= y0;
    
      alpha = sqrt(p[0]*p[0]+p[1]*p[1]);

      hat[0] = p[0]/alpha;
      hat[1] = p[1]/alpha;
    }else{
      // Create quadric fit for the local metric field.
      std::set<index_t> patch = _mesh->get_node_patch(node, min_patch_size);
      patch.insert(node);
      
      // Form quadratic system to be solved. The quadratic fit is:
      // P = y^2+x^2+xy+y+x+1
      // A = P^TP
      
      Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);
      
      Eigen::Matrix<real_t, Eigen::Dynamic, 1> b00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      Eigen::Matrix<real_t, Eigen::Dynamic, 1> b01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      Eigen::Matrix<real_t, Eigen::Dynamic, 1> b11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      
      real_t l_inf=0;
      for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++){
        double x=_mesh->get_coords(*n)[0]-x0, y=_mesh->get_coords(*n)[1]-y0;
        l_inf = std::max(l_inf, std::max(fabs(x), fabs(y)));
        
        A[0]+=y*y*y*y;
        A[6]+=x*x*y*y;  A[7]+=x*x*x*x;  
        A[12]+=x*y*y*y; A[13]+=x*x*x*y; A[14]+=x*x*y*y;
        A[18]+=y*y*y;   A[19]+=x*x*y;   A[20]+=x*y*y;   A[21]+=y*y;
        A[24]+=x*y*y;   A[25]+=x*x*x;   A[26]+=x*x*y;   A[27]+=x*y; A[28]+=x*x;
        A[30]+=y*y;     A[31]+=x*x;     A[32]+=x*y;     A[33]+=y;   A[34]+=x;   A[35]+=1;
        
        double m00 = _mesh->get_metric(*n)[0];
        b00[0]+=m00*pow(y,2); b00[1]+=m00*pow(x,2); b00[2]+=m00*x*y; b00[3]+=m00*y; b00[4]+=m00*x; b00[5]+=m00;
        
        double m01 = _mesh->get_metric(*n)[1];
        b01[0]+=m01*pow(y,2); b01[1]+=m01*pow(x,2); b01[2]+=m01*x*y; b01[3]+=m01*y; b01[4]+=m01*x; b01[5]+=m01;
        
        double m11 = _mesh->get_metric(*n)[3];
        b11[0]+=m11*pow(y,2); b11[1]+=m11*pow(x,2); b11[2]+=m11*x*y; b11[3]+=m11*y; b11[4]+=m11*x; b11[5]+=m11;
      }
      A[1] = A[6]; A[2] = A[12]; A[3] = A[18]; A[4] = A[24]; A[5] = A[30];
                   A[8] = A[13]; A[9] = A[19]; A[10]= A[25]; A[11]= A[31];
                                 A[15]= A[20]; A[16]= A[26]; A[17]= A[32];
                                               A[22]= A[27]; A[23]= A[33];
                                                             A[29]= A[34];

      Eigen::Matrix<real_t, Eigen::Dynamic, 1> a00 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      A.svd().solve(b00, &a00);
      
      Eigen::Matrix<real_t, Eigen::Dynamic, 1> a01 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      A.svd().solve(b01, &a01);
      
      Eigen::Matrix<real_t, Eigen::Dynamic, 1> a11 = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(6);
      A.svd().solve(b11, &a11);
      
      real_t a0 = a00[0]; real_t a1 = a00[1]; real_t a2 = a00[2]; real_t a3 = a00[3]; real_t a4 = a00[4]; real_t a5 = a00[5];
      real_t b0 = a01[0]; real_t b1 = a01[1]; real_t b2 = a01[2]; real_t b3 = a01[3]; real_t b4 = a01[4]; real_t b5 = a01[5];
      real_t c0 = a11[0]; real_t c1 = a11[1]; real_t c2 = a11[2]; real_t c3 = a11[3]; real_t c4 = a11[4]; real_t c5 = a11[5];
      
      // Differentiate quality functional for elements with respect to x,y
      real_t gradient[]={0,0};
      for(typename std::set<index_t>::const_iterator it=_mesh->NEList[node].begin();it!=_mesh->NEList[node].end();++it){
        const index_t *ele=_mesh->get_element(*it);
        size_t loc=0;
        for(;loc<3;loc++)
          if(ele[loc]==node)
            break;
        
        const real_t *_r1=_mesh->get_coords(ele[(loc+1)%3]);
        real_t r1[] = {_r1[0]-x0, _r1[1]-y0};
        
        const real_t *_r2=_mesh->get_coords(ele[(loc+2)%3]);
        real_t r2[] = {_r2[0]-x0, _r2[1]-y0};
        
        const real_t *m1=_mesh->get_metric(ele[(loc+1)%3]);
        const real_t *m2=_mesh->get_metric(ele[(loc+2)%3]);
        
        real_t grad_functional[2];
        grad_r(a0, a1, a2, a3, a4, a5,
               b0, b1, b2, b3, b4, b5,
               c0, c1, c2, c3, c4, c5,
               r1, m1,
               r2, m2,
               grad_functional);
        gradient[0] += grad_functional[0];
        gradient[1] += grad_functional[1];
      }
      real_t mag = sqrt(gradient[0]*gradient[0]+gradient[1]*gradient[1]);
      hat[0] = gradient[0]/mag;
      hat[1] = gradient[1]/mag;

      alpha = l_inf;
    }

    // This can happen if there is zero mag.
    if(!isnormal(hat[0]+hat[1]))
      return false;

    real_t mp[4];

    // Initialise alpha.
    bool valid=false;
    real_t functional;
    for(int rb=0;rb<10;rb++){
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
    const real_t orig_functional = functional_Linf(node);
    std::pair<real_t, real_t> alpha_lower(0, orig_functional), alpha_upper(alpha, functional);
    
    for(int rb=0;rb<10;rb++){
      alpha = (alpha_lower.first+alpha_upper.first)*0.5;
      p[0] = x0 + alpha*hat[0];
      p[1] = y0 + alpha*hat[1];

      // Check if this positions improves the L2 norm.
      valid = generate_location_2d(node, p, mp);
      if(valid){
        functional = functional_Linf(node, p, mp);
      }else{
        functional = -1;
        alpha_upper.first*=0.5;
      }

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
    if(!valid)
      return false;

    //if((functional-orig_functional)/orig_functional<rtol)
    if(functional<relax*orig_functional)
      return false;

    // Recalculate qualities.
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
      quality[*ie] = property->lipnikov(p, x1, x2, 
                                        mp, _mesh->get_metric(n[loc1]), _mesh->get_metric(n[loc2]));
    }
    
    _mesh->_coords[node*2  ] = p[0];
    _mesh->_coords[node*2+1] = p[1];

    for(size_t j=0;j<4;j++)
      _mesh->metric[node*4+j] = mp[j];
    
    return true;
  }
 private:
  void init_cache(std::string method){
    colour_sets.clear();

    zoltan_colour_graph_t graph;
    if(MPI::Is_initialized()){
      MPI_Comm_rank(_mesh->get_mpi_comm(), &graph.rank);
    }else{
      graph.rank = 0; 
    }

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
    zoltan_colour(&graph);
    
    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(_mesh->is_not_owned_node(i)))
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
    real_t delta=linf*1.0e-4;

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

  inline real_t get_x(index_t nid){
    return _mesh->_coords[nid*ndims];
  }

  inline real_t get_y(index_t nid){
    return _mesh->_coords[nid*ndims+1];
  }

  inline real_t get_z(index_t nid){
    return _mesh->_coords[nid*ndims+2];
  }

  real_t functional_Linf(index_t node) const{
    typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();
    double patch_quality = quality[*ie];

    ++ie;

    for(;ie!=_mesh->NEList[node].end();++ie)
      patch_quality = std::min(patch_quality, quality[*ie]);
  
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

  real_t functional_L2(index_t node) const{
    typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();
    double patch_quality = quality[*ie]*quality[*ie];

    ++ie;

    for(;ie!=_mesh->NEList[node].end();++ie)
      patch_quality += quality[*ie]*quality[*ie];
    
    return sqrt(patch_quality);
  }

  real_t functional_L2(index_t node, const real_t *p, const real_t *mp) const{
    real_t functional = 0;
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
      functional += fnl*fnl;
    }
    return sqrt(functional);
  }

  bool generate_location_2d(index_t node, const real_t *p, real_t *mp){
    // Interpolate metric at this new position.
    real_t l[]={-1, -1, -1};
    int best_e=-1;
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
  real_t rtol, relax;
  std::vector<real_t> quality;
  std::map<int, std::deque<index_t> > colour_sets;
};
#endif
