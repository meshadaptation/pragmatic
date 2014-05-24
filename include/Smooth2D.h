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

#ifndef SMOOTH2D_H
#define SMOOTH2D_H

#include "Colour.h"

/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t>
  class Smooth2D{
 public:
  /// Default constructor.
  Smooth2D(Mesh<real_t> &mesh){
    _mesh = &mesh;

    mpi_nparts = 1;
    rank=0;
#ifdef HAVE_MPI
    MPI_Comm_size(_mesh->get_mpi_comm(), &mpi_nparts);
    MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
#endif

    sigma_q = 0.0001;

    // Set the orientation of elements.
    property = NULL;
    int NElements = _mesh->get_number_elements();
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                             _mesh->get_coords(n[1]),
                                             _mesh->get_coords(n[2]));
      break;
    }

    kernels["Laplacian"]              = &Smooth2D<real_t>::laplacian_2d_kernel;
    kernels["smart Laplacian"]        = &Smooth2D<real_t>::smart_laplacian_2d_kernel;
    kernels["optimisation Linf"]      = &Smooth2D<real_t>::optimisation_linf_2d_kernel;
  }

  /// Default destructor.
  ~Smooth2D(){
    delete property;
  }

  // Smooth the mesh using a given method. Valid methods are:
  // "Laplacian", "smart Laplacian", "optimisation Linf"
  void smooth(std::string method, int max_iterations=10, double quality_tol=0.5){
    good_q = quality_tol;

    init_cache(method);

    std::vector<int> halo_elements;
    int NElements = _mesh->get_number_elements();
    if(mpi_nparts>1){
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
    }

    bool (Smooth2D<real_t>::*smooth_kernel)(index_t) = NULL;

    if(kernels.count(method)){
      smooth_kernel = kernels[method];
    }else{
      std::cerr<<"WARNING: Unknown smoothing method \""<<method<<"\"\nUsing \"optimisation Linf\"\n";
      smooth_kernel = kernels["optimisation Linf"];
    }

    // Use this to keep track of vertices that are still to be visited.
    int NNodes = _mesh->get_number_nodes();
    std::vector<int> active_vertices(NNodes, 0);

    // First sweep through all vertices. Add vertices adjacent to any
    // vertex moved into the active_vertex list.
    int max_colour = colour_sets.rbegin()->first;
#ifdef HAVE_MPI
    if(mpi_nparts>1){
      MPI_Allreduce(MPI_IN_PLACE, &max_colour, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
    }
#endif

#pragma omp parallel
    {    
      for(int ic=1;ic<=max_colour;ic++){
        if(colour_sets.count(ic)){
          int node_set_size = colour_sets[ic].size();
#pragma omp for schedule(guided)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[ic][cn];
            
            if((this->*smooth_kernel)(node)){
              for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                active_vertices[*it] = 1;
              }
            }
          }
        }

#pragma omp single
        if(mpi_nparts>1){
          _mesh->halo_update(&(_mesh->_coords[0]), ndims);
          _mesh->halo_update(&(_mesh->metric[0]), msize);

          for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
            quality[*ie] = -1;
        }
      }

      for(int iter=1;iter<max_iterations;iter++){
        for(int ic=1;ic<=max_colour;ic++){
          if(colour_sets.count(ic)){
            int node_set_size = colour_sets[ic].size();
#pragma omp for schedule(guided)
            for(int cn=0;cn<node_set_size;cn++){
              index_t node = colour_sets[ic][cn];

              // Only process if it is active.
              if(active_vertices[node]){
                // Reset mask
                active_vertices[node] = 0;

                if((this->*smooth_kernel)(node)){
                  for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                    active_vertices[*it] = 1;
                  }
                }
              }
            }
          }
          if(mpi_nparts>1){
#pragma omp single
            {
              _mesh->halo_update(&(_mesh->_coords[0]), ndims);
              _mesh->halo_update(&(_mesh->metric[0]), msize);

              for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
                quality[*ie] = -1;
            }
          }
        }
      }
    }

    return;
  }

  bool laplacian_2d_kernel(index_t node){
    real_t p[2];
    bool valid = laplacian_2d_kernel(node, p);
    if(!valid)
      return false;
    
    real_t mp[3];
    valid = generate_location_2d(node, p, mp);
    if(!valid)
      return false;

    for(size_t j=0;j<2;j++)
      _mesh->_coords[node*2+j] = p[j];

    for(size_t j=0;j<3;j++)
      _mesh->metric[node*3+j] = mp[j];

    return true;
  }

  bool smart_laplacian_2d_kernel(index_t node){
    real_t p[2];
    if(!laplacian_2d_kernel(node, p))
      return false;

    double mp[3];
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

    for(size_t j=0;j<3;j++)
      _mesh->metric[node*3+j] = mp[j];

    return true;
  }

  bool laplacian_2d_kernel(index_t node, real_t *p){
    std::set<index_t> patch(_mesh->get_node_patch(node));

    real_t x0 = get_x(node);
    real_t y0 = get_y(node);

    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);

    const real_t *m = _mesh->get_metric(node);
    for(typename std::set<index_t>::const_iterator il=patch.begin();il!=patch.end();++il){
      real_t x = get_x(*il)-x0;
      real_t y = get_y(*il)-y0;

      q[0] += (m[0]*x + m[1]*y);
      q[1] += (m[1]*x + m[2]*y);

      A[0] += m[0]; A[1] += m[1];
                    A[3] += m[2];
    }
    A[2]=A[1];

    // Want to solve the system Ap=q to find the new position, p.
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);
    A.svd().solve(q, &b);

    for(size_t i=0;i<2;i++)
      p[i] = b[i];

    p[0] += x0;
    p[1] += y0;

    return true;
  }

  bool optimisation_linf_2d_kernel(index_t node){
    bool update = smart_laplacian_2d_kernel(node);

    for(int hill_climb_iteration=0;hill_climb_iteration<5;hill_climb_iteration++){
      // As soon as the tolerance quality is reached, break.
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

        if((!pragmatic_isnormal(alpha)) || (alpha<0)){
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

        if((!pragmatic_isnormal(new_alpha)) || (new_alpha<0))
          continue;

        if(new_alpha<alpha)
          alpha = new_alpha;
      }

      // If there is no viable direction, break.
      if((!pragmatic_isnormal(alpha))||(alpha<=0))
        break;

      // -
      real_t p[2], gp[2];
      double mp[3];
      std::map<int, real_t> new_quality;
      bool valid_move = false;
      for(int i=0;i<10;i++){
        // If the predicted improvement is less than sigma, break;
        if(mag0*alpha<sigma_q)
          break;

        new_quality.clear();

        p[0] = alpha*hat0[0];
        p[1] = alpha*hat0[1];

        if(!pragmatic_isnormal(p[0]+p[1])){ // This can happen if there is zero gradient.
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

        assert(pragmatic_isnormal(p[0]+p[1]));
        assert(pragmatic_isnormal(mp[0]+mp[1]+mp[2]));

        // Check if this position improves the local mesh quality.
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
          assert(pragmatic_isnormal(functional));
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

      for(size_t j=0;j<3;j++)
        _mesh->metric[node*3+j] = mp[j];
    }

    return update;
  }

 private:
  void init_cache(std::string method){
    colour_sets.clear();

    int NNodes = _mesh->get_number_nodes();
    std::vector<char> colour(NNodes);
    Colour::GebremedhinManne(NNodes, _mesh->NNList, colour);

    int NElements = _mesh->get_number_elements();
    std::vector<bool> is_boundary(NNodes, false);
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]==-1)
        continue;
  
      for(int j=0;j<3;j++){
        if(_mesh->boundary[i*3+j]>0){
          is_boundary[n[(j+1)%3]] = true;
          is_boundary[n[(j+2)%3]] = true;
        }
      }
    }

    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(!_mesh->is_owned_node(i))||(_mesh->NNList[i].empty())||is_boundary[i])
        continue;

      colour_sets[colour[i]].push_back(i);
    }

    quality.resize(NElements);
#pragma omp parallel
    {
#pragma omp for schedule(guided)
      for(int i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0){
          quality[i] = 1.0;
          continue;
        }

        quality[i] = property->lipnikov(_mesh->get_coords(n[0]),
                                        _mesh->get_coords(n[1]),
                                        _mesh->get_coords(n[2]),
                                        _mesh->get_metric(n[0]),
                                        _mesh->get_metric(n[1]),
                                        _mesh->get_metric(n[2]));
      }
    }

    return;
  }

  void grad_r(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4, real_t a5,
              real_t b0, real_t b1, real_t b2, real_t b3, real_t b4, real_t b5,
              real_t c0, real_t c1, real_t c2, real_t c3, real_t c4, real_t c5,
              const real_t *r1, const double *m1,
              const real_t *r2, const double *m2,
              real_t *grad){
    real_t linf = std::max(std::max(fabs(r1[0]), fabs(r2[0])), std::max(fabs(r1[1]), fabs(r2[1])));    
    real_t delta=linf*1.0e-2;

    real_t p[2];
    real_t mp[3];

    p[0] = -delta/2; p[1] = 0;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor2D<real_t>::positive_definiteness(mp);
    double functional_minus_dx = property->lipnikov(p, r1, r2, 
                                                   mp, m1, m2);

    p[0] = delta/2; p[1] = 0;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor2D<real_t>::positive_definiteness(mp);
    double functional_plus_dx = property->lipnikov(p, r1, r2, 
                                                  mp, m1, m2);

    grad[0] = (functional_plus_dx-functional_minus_dx)/delta;

    p[0] = 0; p[1] = -delta/2;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor2D<real_t>::positive_definiteness(mp);
    double functional_minus_dy = property->lipnikov(p, r1, r2, 
                                                   mp, m1, m2);

    p[0] = 0; p[1] = delta/2;
    mp[0] = a0*p[1]*p[1]+a1*p[0]*p[0]+a2*p[0]*p[1]+a3*p[1]+a4*p[0];
    mp[1] = b0*p[1]*p[1]+b1*p[0]*p[0]+b2*p[0]*p[1]+b3*p[1]+b4*p[0];
    mp[2] = c0*p[1]*p[1]+c1*p[0]*p[0]+c2*p[0]*p[1]+c3*p[1]+c4*p[0];
    MetricTensor2D<real_t>::positive_definiteness(mp);
    double functional_plus_dy = property->lipnikov(p, r1, r2, 
                                                  mp, m1, m2);

    grad[1] = (functional_plus_dy-functional_minus_dy)/delta;
  }

  void grad_r(index_t node,
              const real_t *r1, const double *m1,
              const real_t *r2, const double *m2,
              std::vector<real_t> &grad){

    grad[0] = 0;
    grad[1] = 0;

    const real_t *r0=_mesh->get_coords(node);

    real_t linf_x = std::max(fabs(r1[0]-r0[0]), fabs(r2[0]-r0[0]));
    real_t delta_x=linf_x*1.0e-2;

    real_t linf_y = std::max(fabs(r1[1]-r0[1]), fabs(r2[1]-r0[1]));
    real_t delta_y=linf_y*1.0e-1;

    real_t p[2];
    double mp[3];

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

  real_t functional_Linf(index_t node){
    double patch_quality = std::numeric_limits<double>::max();

    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      // Check cache - if it's stale then recalculate. 
      if(quality[*ie]<0){
        const int *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        std::vector<const real_t *> x(nloc);
        std::vector<const double *> m(nloc);
        for(size_t i=0;i<nloc;i++){
          x[i] = _mesh->get_coords(n[i]);
          m[i] = _mesh->get_metric(n[i]);
        }

        quality[*ie] = property->lipnikov(x[0], x[1], x[2], 
                                          m[0], m[1], m[2]);
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

  bool generate_location_2d(index_t node, const real_t *p, double *mp){
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

    for(size_t i=0;i<3;i++)
      mp[i] = 
        l[0]*_mesh->metric[n[0]*3+i]+
        l[1]*_mesh->metric[n[1]*3+i]+
        l[2]*_mesh->metric[n[2]*3+i];

    return true;
  }

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  const static size_t ndims=2;
  const static size_t nloc=3;
  const static size_t msize=3;

  int mpi_nparts, rank;
  real_t good_q, sigma_q;
  std::vector<real_t> quality;
  std::map<int, std::vector<index_t> > colour_sets;

  std::map<std::string, bool (Smooth2D<real_t>::*)(index_t)> kernels;
};

#endif
