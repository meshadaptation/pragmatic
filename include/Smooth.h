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

#ifndef SMOOTH_H
#define SMOOTH_H

#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <vector>
#include <limits>
#include <random>

#include "Colour.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ElementProperty.h"
#include "Mesh.h"
#include "MetricTensor.h"


/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t, int dim>
  class Smooth{
 public:
  /// Default constructor.
 Smooth(Mesh<real_t> &mesh):nloc(dim+1), msize(dim==2?3:6){
    _mesh = &mesh;

    mpi_nparts = 1;
    rank=0;
#ifdef HAVE_MPI
    MPI_Comm_size(_mesh->get_mpi_comm(), &mpi_nparts);
    MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
#endif

    epsilon_q = DBL_EPSILON;

    // Set the orientation of elements.
    property = NULL;
    int NElements = _mesh->get_number_elements();
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;
      
      if(dim==2){
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]));
      }else{
        property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                               _mesh->get_coords(n[1]),
                                               _mesh->get_coords(n[2]),
                                               _mesh->get_coords(n[3]));
      }
      
      break;
    }
  }
  
  /// Default destructor.
  ~Smooth(){
    delete property;
  }

  // Smooth the mesh.
  void smooth(int max_iterations=10, double quality_tol=-1.0){
    if(quality_tol>0)
      good_q = quality_tol;

    init_cache();

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

    // Use this to keep track of vertices that are still to be visited.
    int NNodes = _mesh->get_number_nodes();
    std::vector<int> active_vertices(NNodes, 0);

    // First sweep through all vertices. Add vertices adjacent to any
    // vertex moved into the active_vertex list.
    int max_colour = 0;
    if(!colour_sets.empty())
      max_colour = colour_sets.rbegin()->first;
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

            if(optimisation_linf_kernel(node)){
              for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                active_vertices[*it] = 1;
              }
            }
          }
        }

        if(mpi_nparts>1){
#pragma omp single
          {
            halo_update<real_t, dim, dim==2?3:6>(_mesh->get_mpi_comm(), _mesh->send, _mesh->recv, _mesh->_coords, _mesh->metric);

            for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
              update_quality(*ie);
          }
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

                if(optimisation_linf_kernel(node)){
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
              halo_update<real_t, dim, dim==2?3:6>(_mesh->get_mpi_comm(), _mesh->send, _mesh->recv, _mesh->_coords, _mesh->metric);

              for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
                update_quality(*ie);
            }
          }
        }
      }
    }

    return;
  }

 private:

  // Laplacian smooth kernels
  inline bool laplacian_kernel(index_t node){
    if(dim==2)
      return laplacian_2d_kernel(node);
    else
      return laplacian_3d_kernel(node);
  }
  
  inline bool laplacian_2d_kernel(index_t node){
    real_t p[2];
    bool valid = laplacian_2d_kernel(node, p);
    if(!valid)
      return false;
    
    double mp[3];
    valid = generate_location_2d(node, p, mp);
    if(!valid)
      return false;
    
    for(size_t j=0;j<2;j++)
      _mesh->_coords[node*2+j] = p[j];
    
    for(size_t j=0;j<3;j++)
      _mesh->metric[node*3+j] = mp[j];
    
    return true;
  }
  
  inline bool laplacian_3d_kernel(index_t node){
    real_t p[3];
    bool valid =laplacian_3d_kernel(node, p);
    if(!valid)
      return false;
    
    double mp[6];
    valid = generate_location_3d(node, p, mp);
    if(!valid)
      return false;
    
    for(size_t j=0;j<3;j++)
      _mesh->_coords[node*3+j] = p[j];
    
    for(size_t j=0;j<6;j++)
      _mesh->metric[node*6+j] = mp[j];
    
    return true;
  }
  
  inline bool laplacian_2d_kernel(index_t node, real_t *p){
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
  
  bool laplacian_3d_kernel(index_t node, real_t *p){
    std::set<index_t> patch(_mesh->get_node_patch(node));
    
    real_t x0 = get_x(node);
    real_t y0 = get_y(node);
    real_t z0 = get_z(node);
    
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 3);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
    
    const real_t *m = _mesh->get_metric(node);
    for(typename std::set<index_t>::const_iterator il=patch.begin();il!=patch.end();++il){
      real_t x = get_x(*il)-x0;
      real_t y = get_y(*il)-y0;
      real_t z = get_z(*il)-z0;
      
      q[0] += m[0]*x + m[1]*y + m[2]*z;
      q[1] += m[1]*x + m[3]*y + m[4]*z;
      q[2] += m[2]*x + m[4]*y + m[5]*z;

      A[0] += m[0]; A[1] += m[1]; A[2] += m[2];
      A[4] += m[3]; A[5] += m[4];
      A[8] += m[5];
    }
    A[3] = A[1];
    A[6] = A[2];
    A[7] = A[5];

    // Want to solve the system Ap=q to find the new position, p.
    Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
    A.svd().solve(q, &b);

    for(int i=0;i<3;i++)
      p[i] = b[i];

    p[0] += x0;
    p[1] += y0;
    p[2] += z0;

    return true;
  }

  // Smart Laplacian kernels
  inline bool smart_laplacian_kernel(index_t node){
    if(dim==2)
      return smart_laplacian_2d_kernel(node);
    else
      return smart_laplacian_3d_kernel(node);
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

    if(functional-functional_orig<epsilon_q)
      return false;

    for(size_t j=0;j<2;j++)
      _mesh->_coords[node*2+j] = p[j];
    
    for(size_t j=0;j<3;j++)
      _mesh->metric[node*3+j] = mp[j];
    
    // Reset quality cache.
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      update_quality(*ie);
    }

    return true;
  }

  inline bool smart_laplacian_3d_kernel(index_t node){
    real_t p[3];
    if(!laplacian_3d_kernel(node, p))
      return false;
    
    double mp[6];
    bool valid = generate_location_3d(node, p, mp);
    if(!valid)
      return false;
    
    real_t functional = functional_Linf(node, p, mp);
    real_t functional_orig = functional_Linf(node);
    
    if(functional-functional_orig<epsilon_q)
      return false;

    for(size_t j=0;j<3;j++)
      _mesh->_coords[node*3+j] = p[j];
    
    for(size_t j=0;j<6;j++)
      _mesh->metric[node*6+j] = mp[j];
    
    // Reset quality cache.
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      update_quality(*ie);
    }
    
    return true;
  }

  inline bool stocastic_3d_kernel(index_t node){
    real_t p[3];
    real_t r=0;
    for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
      r += sqrt(pow(_mesh->_coords[node*3+0]-_mesh->_coords[(*it)*3+0],2)+
		pow(_mesh->_coords[node*3+1]-_mesh->_coords[(*it)*3+1],2)+
		pow(_mesh->_coords[node*3+2]-_mesh->_coords[(*it)*3+2],2));
    }
    r/=_mesh->NNList[node].size();
    
    std::default_random_engine generator;
    std::normal_distribution<real_t> distribution(0.0, r*0.5);

    bool valid = false;
    real_t functional_orig = functional_Linf(node);
    for(int hail_marys=0;hail_marys<1000;hail_marys++){
      // Randomally jump to a new location in the neighbourhood.
      for(int i=0;i<3;i++)
        p[i] = _mesh->_coords[node*3+i]+distribution(generator);
      
      // Interpolate metric and check for inversion.
      double mp[6];
      bool lvalid = generate_location_3d(node, p, mp);
      if(!lvalid)
        continue;
      
      // Check to see if the position is an improvement.
      real_t functional = functional_Linf(node, p, mp);
      if(functional-functional_orig<epsilon_q)
        continue;

      // Otherwise accept.
      valid = true;
      for(size_t j=0;j<3;j++)
        _mesh->_coords[node*3+j] = p[j];
    
      for(size_t j=0;j<6;j++)
        _mesh->metric[node*6+j] = mp[j];
    
      // Reset quality cache.
      for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie)
        update_quality(*ie);

      break;
    }

    return valid;
  }

  // l-infinity optimisation kernels
  inline bool optimisation_linf_kernel(index_t n0){
    if(dim==2)
      return optimisation_linf_2d_kernel(n0);
    else
      return optimisation_linf_3d_kernel(n0);
  }

  inline bool optimisation_linf_2d_kernel(index_t n0){
    const double *m0 = _mesh->get_metric(n0);
    const double *x0 = _mesh->get_coords(n0);
    
    // Find the worst element.
    std::pair<double, index_t> worst_element(DBL_MAX, -1);
    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
      if(quality[*it]<worst_element.first)
        worst_element = std::pair<double, index_t>(quality[*it], *it);
    }
    assert(worst_element.second!=-1);

    // Return if it is already "good enough".
    if(worst_element.first>good_q)
      return false;
    
    // Find direction of steepest ascent for quality of worst element.
    double search[2], grad_w[2];
    {
      const index_t *n=_mesh->get_element(worst_element.second);
      size_t loc=0;
      for(;loc<3;loc++)
        if(n[loc]==n0)
          break;
      
      int n1 = n[(loc+1)%3];
      int n2 = n[(loc+2)%3];
      
      const double *x1 = _mesh->get_coords(n1);
      const double *x2 = _mesh->get_coords(n2);
      
      property->lipnikov_grad(loc, x0, x1, x2, m0, grad_w);
      
      double mag = sqrt(grad_w[0]*grad_w[0]+grad_w[1]*grad_w[1]);
      assert(mag!=0);
      
      for(int i=0;i<2;i++)
        search[i] = grad_w[i]/mag;
    }

    // Estimate how far we move along this search path until we make
    // another element of a similar quality to the current worst. This
    // is effectively a simplex method for linear programming.
    double alpha;
    {
      double bbox[] = {DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX};
      for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[n0].begin();it!=_mesh->NNList[n0].end();++it){
        const double *x1 = _mesh->get_coords(*it);
        
        bbox[0] = std::min(bbox[0], x1[0]);
        bbox[1] = std::max(bbox[1], x1[0]);

        bbox[2] = std::min(bbox[2], x1[1]);
        bbox[3] = std::max(bbox[3], x1[1]);
      }
      alpha = (bbox[1]-bbox[0] + bbox[3]-bbox[2])/2.0;
    }

    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
      if(*it==worst_element.second)
        continue;

      const index_t *n=_mesh->get_element(*it);
      size_t loc=0;
      for(;loc<3;loc++)
        if(n[loc]==n0)
          break;
	
      int n1 = n[(loc+1)%3];
      int n2 = n[(loc+2)%3];
	
      const double *x1 = _mesh->get_coords(n1);
      const double *x2 = _mesh->get_coords(n2);
	
      double grad[2];
      property->lipnikov_grad(loc, x0, x1, x2, m0, grad);
	
      double new_alpha =
          (quality[*it]-worst_element.first)/
          ((search[0]*grad_w[0]+search[1]*grad_w[1])-
          (search[0]*grad[0]+search[1]*grad[1]));

      if(new_alpha>0)
        alpha = std::min(alpha, new_alpha);
    }
    
    bool linf_update;
    for(int isearch=0;isearch<10;isearch++){
      linf_update = false;
      
      // Only want to step half that distance so we do not degrade the other elements too much.
      alpha*=0.5;
      
      double new_x0[2];
      for(int i=0;i<2;i++){
        new_x0[i] = x0[i] + alpha*search[i];
        if(!std::isnormal(new_x0[i]))
          return false;
      }

      double new_m0[3];
      bool valid = generate_location_2d(n0, new_x0, new_m0);
      
      if(!valid)
        continue;
      
      // Need to check that we have not decreased the Linf norm. Start by assuming the best.
      linf_update = true;
      std::vector<double> new_quality;
      for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
        const index_t *n=_mesh->get_element(*it);
        size_t loc=0;
        for(;loc<3;loc++)
          if(n[loc]==n0)
            break;
	
        int n1 = n[(loc+1)%3];
        int n2 = n[(loc+2)%3];

        const double *x1 = _mesh->get_coords(n1);
        const double *x2 = _mesh->get_coords(n2);

        const double *m1 = _mesh->get_metric(n1);
        const double *m2 = _mesh->get_metric(n2);

        double new_q = property->lipnikov(new_x0, x1, x2, new_m0, m1, m2);
        new_quality.push_back(new_q);

        if(!std::isnormal(new_q) || new_quality.back()<worst_element.first){
          linf_update = false;
          break;
        }
      }
      
      if(!linf_update)
        continue;
      
      // Update information
      // go backwards and pop quality
      assert(_mesh->NEList[n0].size()==new_quality.size());
      for(typename std::set<index_t>::const_reverse_iterator it=_mesh->NEList[n0].rbegin();it!=_mesh->NEList[n0].rend();++it){
        quality[*it] = new_quality.back();
        new_quality.pop_back();
      }
      assert(new_quality.empty());
      
      for(size_t i=0;i<dim;i++)
        _mesh->_coords[n0*dim+i] = new_x0[i];
      
      for(size_t i=0;i<msize;i++)
        _mesh->metric[n0*msize+i] = new_m0[i];

      break;
    }
  
    return linf_update;
  }

  inline bool optimisation_linf_3d_kernel(index_t n0){
    const double *m0 = _mesh->get_metric(n0);
    const double *x0 = _mesh->get_coords(n0);
    
    // Find the worst element.
    std::pair<double, index_t> worst_element(DBL_MAX, -1);
    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
      if(quality[*it]<worst_element.first)
        worst_element = std::pair<double, index_t>(quality[*it], *it);
    }
    assert(worst_element.second!=-1);
    
    // Jump out if already good enough.
    if(worst_element.first>good_q)
      return false;

    // Find direction of steepest ascent for quality of worst element.
    double grad_w[3], search[3];
    {
      const index_t *n=_mesh->get_element(worst_element.second);
      size_t loc=0;
      for(;loc<4;loc++)
        if(n[loc]==n0)
          break;
      
      int n1, n2, n3;
      switch(loc){
      case 0:
        n1 = n[1];
        n2 = n[2];
        n3 = n[3];
        break;
      case 1:
        n1 = n[2];
        n2 = n[0];
        n3 = n[3];
        break;
      case 2:
        n1 = n[0];
        n2 = n[1];
        n3 = n[3];
        break;
      case 3:
        n1 = n[0];
        n2 = n[2];
        n3 = n[1];
        break;
      }
      
      const double *x1 = _mesh->get_coords(n1);
      const double *x2 = _mesh->get_coords(n2);
      const double *x3 = _mesh->get_coords(n3);
      
      property->lipnikov_grad(loc, x0, x1, x2, x3, m0, grad_w);
      
      double mag = sqrt(grad_w[0]*grad_w[0] + grad_w[1]*grad_w[1] + grad_w[2]*grad_w[2]);
      if(!std::isnormal(mag)){
        std::cout<<"mag issues "<<mag<<", "<<grad_w[0]<<", "<<grad_w[1]<<", "<<grad_w[2]<<std::endl;
        std::cout<<"This usually means that the metric field is rubbish\n";
      }
      assert(std::isnormal(mag));
      
      for(int i=0;i<3;i++)
        search[i] = grad_w[i]/mag;
    }

    // Estimate how far we move along this search path until we make
    // another element of a similar quality to the current worst. This
    // is effectively a simplex method for linear programming.
    double alpha;
    {
      double bbox[] = {DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX};
      for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[n0].begin();it!=_mesh->NNList[n0].end();++it){
        const double *x1 = _mesh->get_coords(*it);
	
        bbox[0] = std::min(bbox[0], x1[0]);
        bbox[1] = std::max(bbox[1], x1[0]);

        bbox[2] = std::min(bbox[2], x1[1]);
        bbox[3] = std::max(bbox[3], x1[1]);

        bbox[4] = std::min(bbox[4], x1[2]);
        bbox[5] = std::max(bbox[5], x1[2]);
      }
      alpha = (bbox[1]-bbox[0] + bbox[3]-bbox[2] + bbox[5]-bbox[4])/6.0;
    }
    for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
      if(*it==worst_element.second)
        continue;

      const index_t *n=_mesh->get_element(*it);
      size_t loc=0;
      for(;loc<4;loc++)
        if(n[loc]==n0)
          break;

      int n1, n2, n3;
      switch(loc){
      case 0:
        n1 = n[1];
        n2 = n[2];
        n3 = n[3];
        break;
      case 1:
        n1 = n[2];
        n2 = n[0];
        n3 = n[3];
        break;
      case 2:
        n1 = n[0];
        n2 = n[1];
        n3 = n[3];
        break;
      case 3:
        n1 = n[0];
        n2 = n[2];
        n3 = n[1];
        break;
      }
      
      const double *x1 = _mesh->get_coords(n1);
      const double *x2 = _mesh->get_coords(n2);
      const double *x3 = _mesh->get_coords(n3);
	
      double grad[3];
      property->lipnikov_grad(loc, x0, x1, x2, x3, m0, grad);
	
      double new_alpha =
          (quality[*it]-worst_element.first)/
          ((search[0]*grad_w[0]+search[1]*grad_w[1]+search[2]*grad_w[2])-
          (search[0]*grad[0]+search[1]*grad[1]+search[2]*grad[2]));

      if(new_alpha>0)
        alpha = std::min(alpha, new_alpha);
    }
    
    bool linf_update;
    for(int isearch=0;isearch<10;isearch++){
      linf_update = false;
      
      // Only want to step half that distance so we do not degrade the other elements too much.
      alpha*=0.5;
      
      double new_x0[3];
      for(int i=0;i<3;i++){
        new_x0[i] = x0[i] + alpha*search[i];
      }

      double new_m0[6];
      bool valid = generate_location_3d(n0, new_x0, new_m0);
      
      if(!valid)
        continue;

      // Need to check that we have not decreased the Linf norm. Start by assuming the best.
      linf_update = true;
      std::vector<double> new_quality;
      for(typename std::set<index_t>::const_iterator it=_mesh->NEList[n0].begin();it!=_mesh->NEList[n0].end();++it){
        const index_t *n=_mesh->get_element(*it);
        size_t loc=0;
        for(;loc<4;loc++)
          if(n[loc]==n0)
            break;

        int n1, n2, n3;
        switch(loc){
        case 0:
          n1 = n[1];
          n2 = n[2];
          n3 = n[3];
          break;
        case 1:
          n1 = n[2];
          n2 = n[0];
          n3 = n[3];
          break;
        case 2:
          n1 = n[0];
          n2 = n[1];
          n3 = n[3];
          break;
        case 3:
          n1 = n[0];
          n2 = n[2];
          n3 = n[1];
          break;
        }
	
        const double *x1 = _mesh->get_coords(n1);
        const double *x2 = _mesh->get_coords(n2);
        const double *x3 = _mesh->get_coords(n3);


        const double *m1 = _mesh->get_metric(n1);
        const double *m2 = _mesh->get_metric(n2);
        const double *m3 = _mesh->get_metric(n3);

        double new_q = property->lipnikov(new_x0, x1, x2, x3, new_m0, m1, m2, m3);

        if(new_q>worst_element.first){
          new_quality.push_back(new_q);
        }else{
          // This means that the linear approximation was not sufficient.
          linf_update = false;
          break;
        }
      }
      
      if(!linf_update)
        continue;

      // Update information
      // go backwards and pop quality
      assert(_mesh->NEList[n0].size()==new_quality.size());
      for(typename std::set<index_t>::const_reverse_iterator it=_mesh->NEList[n0].rbegin();it!=_mesh->NEList[n0].rend();++it){
        quality[*it] = new_quality.back();
        new_quality.pop_back();
      }
      assert(new_quality.empty());
      
      for(size_t i=0;i<dim;i++)
        _mesh->_coords[n0*dim+i] = new_x0[i];
      
      for(size_t i=0;i<msize;i++)
        _mesh->metric[n0*msize+i] = new_m0[i];

      break;
    }
  
    return linf_update;
  }

  void init_cache(){
    colour_sets.clear();

    int NNodes = _mesh->get_number_nodes();
    std::vector<char> colour(NNodes);

    Colour::GebremedhinManne(MPI_COMM_WORLD, NNodes, _mesh->NNList, _mesh->send, _mesh->recv, _mesh->node_owner, colour);

    int NElements = _mesh->get_number_elements();
    std::vector<bool> is_boundary(NNodes, false);
    for(int i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]==-1)
        continue;
  
      for(size_t j=0;j<nloc;j++){
        if(_mesh->boundary[i*nloc+j]>0){
          for(size_t k=1;k<nloc;k++){
            is_boundary[n[(j+k)%nloc]] = true;
          }
        }
      }
    }

    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(!_mesh->is_owned_node(i))||(_mesh->NNList[i].empty())||is_boundary[i])
        continue;

      colour_sets[colour[i]].push_back(i);
    }

    quality.resize(NElements);

    double qsum=0;
#pragma omp parallel
    {
#pragma omp for schedule(guided) reduction(+:qsum)
      for(int i=0;i<NElements;i++){
        const int *n=_mesh->get_element(i);
        if(n[0]<0){
          quality[i] = 1.0;
          continue;
        }

        update_quality(i);
        qsum+=quality[i];
      }
    }
    good_q = qsum/NElements;

    return;
  }

  inline real_t get_x(index_t nid){
    return _mesh->_coords[nid*dim];
  }

  inline real_t get_y(index_t nid){
    return _mesh->_coords[nid*dim+1];
  }
    
  inline real_t get_z(index_t nid){
    if(dim==3){
      return _mesh->_coords[nid*dim+2];
    }else{
      std::cerr<<"ERROR: inline real_t get_z(index_t nid) should not get called for 2D";
      return -1;
    }
  }

  inline real_t functional_Linf(index_t node){
    double patch_quality = std::numeric_limits<double>::max();

    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      patch_quality = std::min(patch_quality, quality[*ie]);
    }

    return patch_quality;
  }
  
  inline real_t functional_Linf(index_t n0, const real_t *p, const real_t *mp) const{
    real_t f;
    if(dim==2){
      f = functional_Linf_2d(n0, p, mp);
    }else{
      f = functional_Linf_3d(n0, p, mp);
    }
    return f;
  }

  inline real_t functional_Linf_2d(index_t n0, const real_t *p, const real_t *mp) const{
    real_t functional = DBL_MAX;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[n0].begin();ie!=_mesh->NEList[n0].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      assert(n[0]>=0);
      int iloc = 0;

      while(n[iloc]!=(int)n0){
        iloc++;
      }
      int loc1 = (iloc+1)%3;
      int loc2 = (iloc+2)%3;

      const real_t *x1 = _mesh->get_coords(n[loc1]);
      const real_t *x2 = _mesh->get_coords(n[loc2]);

      const double *m1 = _mesh->get_metric(n[loc1]);
      const double *m2 = _mesh->get_metric(n[loc2]);

      real_t fnl = property->lipnikov(p,  x1, x2, 
				      mp, m1, m2);
      functional = std::min(functional, fnl);
    }

    return functional;
  }

  inline real_t functional_Linf_3d(index_t n0, const real_t *p, const real_t *mp) const{
    real_t functional = DBL_MAX;
    for(typename std::set<index_t>::iterator ie=_mesh->NEList[n0].begin();ie!=_mesh->NEList[n0].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      size_t loc=0;
      for(;loc<4;loc++)
        if(n[loc]==n0)
          break;
      
      int n1, n2, n3;
      switch(loc){
      case 0:
        n1 = n[1];
        n2 = n[2];
        n3 = n[3];
        break;
      case 1:
        n1 = n[2];
        n2 = n[0];
        n3 = n[3];
        break;
      case 2:
        n1 = n[0];
        n2 = n[1];
        n3 = n[3];
        break;
      case 3:
        n1 = n[0];
        n2 = n[2];
        n3 = n[1];
        break;
      }
      
      const double *x1 = _mesh->get_coords(n1);
      const double *x2 = _mesh->get_coords(n2);
      const double *x3 = _mesh->get_coords(n3);
      
      const double *m1 = _mesh->get_metric(n1);
      const double *m2 = _mesh->get_metric(n2);
      const double *m3 = _mesh->get_metric(n3);
      
      real_t fnl = property->lipnikov(p, x1, x2, x3,
				      mp,m1, m2, m3);

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

      /* Check for inversion by looking at the area
       * of the element whose node is being moved.*/
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
      ll[2] = property->area(x0, x1, p )/L;

      real_t min_l = std::min(ll[0], std::min(ll[1], ll[2]));
      if(best_e==-1){
        tol = min_l;
        best_e = *ie;
        for(size_t i=0;i<nloc;i++)
          l[i] = ll[i];
      }else{
        if(min_l>tol){
          tol = min_l;
          best_e = *ie;
          for(size_t i=0;i<nloc;i++)
            l[i] = ll[i];
        }
      }
    }
    assert(best_e!=-1);
    assert(tol>-DBL_EPSILON);

    const index_t *n=_mesh->get_element(best_e);
    assert(n[0]>=0);

    for(size_t i=0;i<msize;i++)
      mp[i] = 
	l[0]*_mesh->metric[n[0]*msize+i]+
	l[1]*_mesh->metric[n[1]*msize+i]+
	l[2]*_mesh->metric[n[2]*msize+i];

    return true;
  }

  bool generate_location_3d(index_t node, const real_t *p, double *mp){
    // Interpolate metric at this new position.
    real_t l[]={-1, -1, -1, -1};
    int best_e=-1;
    real_t tol=-1;

    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
      const index_t *n=_mesh->get_element(*ie);
      assert(n[0]>=0);

      const real_t *x0 = _mesh->get_coords(n[0]);
      const real_t *x1 = _mesh->get_coords(n[1]);
      const real_t *x2 = _mesh->get_coords(n[2]);
      const real_t *x3 = _mesh->get_coords(n[3]);

      /* Check for inversion by looking at the volume
       * of element whose node is being moved.*/
      real_t volume;
      if(n[0]==node){
        volume = property->volume(p, x1, x2, x3);
      }else if(n[1]==node){
        volume = property->volume(x0, p, x2, x3);
      }else if(n[2]==node){
        volume = property->volume(x0, x1, p, x3);
      }else{
        volume = property->volume(x0, x1, x2, p);
      }
      if(volume<0)
        return false;

      real_t L = property->volume(x0, x1, x2, x3);

      real_t ll[4];
      ll[0] = property->volume(p,  x1, x2, x3)/L;
      ll[1] = property->volume(x0, p,  x2, x3)/L;
      ll[2] = property->volume(x0, x1, p,  x3)/L;
      ll[3] = property->volume(x0, x1, x2, p )/L;

      real_t min_l = std::min(std::min(ll[0], ll[1]), std::min(ll[2], ll[3]));
      if(best_e==-1){
        tol = min_l;
        best_e = *ie;
        for(size_t i=0;i<nloc;i++)
          l[i] = ll[i];
      }else{
        if(min_l>tol){
          tol = min_l;
          best_e = *ie;
          for(size_t i=0;i<nloc;i++)
            l[i] = ll[i];
        }
      }
    }
    assert(best_e!=-1);
    assert(tol>-10*DBL_EPSILON);

    const index_t *n=_mesh->get_element(best_e);
    assert(n[0]>=0);

    for(size_t i=0;i<msize;i++)
      mp[i] =
	l[0]*_mesh->metric[n[0]*msize+i]+
	l[1]*_mesh->metric[n[1]*msize+i]+
	l[2]*_mesh->metric[n[2]*msize+i]+
	l[3]*_mesh->metric[n[3]*msize+i];

    return true;
  }

  inline void update_quality(index_t element){
    if(dim==2)
      update_quality_2d(element);
    else
      update_quality_3d(element);
  }

  inline void update_quality_2d(index_t element){
    const index_t *n=_mesh->get_element(element);

    const double *x0 = _mesh->get_coords(n[0]);
    const double *x1 = _mesh->get_coords(n[1]);
    const double *x2 = _mesh->get_coords(n[2]);

    const double *m0 = _mesh->get_metric(n[0]);
    const double *m1 = _mesh->get_metric(n[1]);
    const double *m2 = _mesh->get_metric(n[2]);

    quality[element] = property->lipnikov(x0, x1, x2,
					  m0, m1, m2);
    return;
  }

  inline void update_quality_3d(index_t element){
    const index_t *n=_mesh->get_element(element);

    const double *x0 = _mesh->get_coords(n[0]);
    const double *x1 = _mesh->get_coords(n[1]);
    const double *x2 = _mesh->get_coords(n[2]);
    const double *x3 = _mesh->get_coords(n[3]);

    const double *m0 = _mesh->get_metric(n[0]);
    const double *m1 = _mesh->get_metric(n[1]);
    const double *m2 = _mesh->get_metric(n[2]);
    const double *m3 = _mesh->get_metric(n[3]);

    quality[element] = property->lipnikov(x0, x1, x2, x3,
					  m0, m1, m2, m3);
    return;
  }

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  const size_t nloc, msize;

  int mpi_nparts, rank;
  real_t good_q, epsilon_q;
  std::vector<real_t> quality;
  std::map<int, std::vector<index_t> > colour_sets;
};

#endif

