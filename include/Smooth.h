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

#include <Eigen/Core>
#include <Eigen/Dense>

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
  
  int smooth(real_t tolerance, int max_iterations, bool qconstrain=false){
    init_cache();
    
    double prev_mean_quality = -1;
    int NElements = _mesh->get_number_elements();
    std::vector<real_t> qvec(NElements);
    int iter=0;
    for(;iter<max_iterations;iter++){    
      for(int i=0;i<NElements;i++)
        qvec[i] = 0;
      
      real_t qlinfinity = std::numeric_limits<real_t>::max();
      real_t qmean = 0.0, qrms=0.0;
      
      int ncolours = colour_sets.size();
 
      if(ndims==2){
        // Smoothing loop.
        for(int colour=0; colour<ncolours; colour++){
#pragma omp parallel
          {
            int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
            for(int cn=0;cn<node_set_size;cn++){
              index_t node = colour_sets[colour][cn];
              smooth_2d_kernel(node, qconstrain);
            }
          }
        }
        
#pragma omp parallel
        {
          real_t lqlinfinity = std::numeric_limits<real_t>::max();
#pragma omp for schedule(static) reduction(+:qmean)
          for(int i=0;i<NElements;i++){
            const int *n=_mesh->get_element(i);
            if(n[0]<0)
              continue;

            const real_t *x0 = _mesh->get_coords(n[0]);
            const real_t *x1 = _mesh->get_coords(n[1]);
            const real_t *x2 = _mesh->get_coords(n[2]);
            
            qvec[i] = property->lipnikov(x0, x1, x2,
                                         &(_mesh->metric[n[0]*4]),
                                         &(_mesh->metric[n[1]*4]),
                                         &(_mesh->metric[n[2]*4]));
            lqlinfinity = std::min(lqlinfinity, qvec[i]);
            qmean += qvec[i]/NElements;
          }
#pragma omp for schedule(static) reduction(+:qrms)
          for(int i=0;i<NElements;i++){
            qrms += pow(qvec[i]-qmean, 2);
          }
#pragma omp critical 
          {
            qlinfinity = std::min(qlinfinity, lqlinfinity);
          }
        }
      }else{
        // Smoothing loop.
        for(int colour=0; colour<ncolours; colour++){
#pragma omp parallel
          {
            int node_set_size = colour_sets[colour].size();
#pragma omp for schedule(static)
            for(int cn=0;cn<node_set_size;cn++){
              index_t node = colour_sets[colour][cn];
              smooth_3d_kernel(node, qconstrain);          
            }
          }
        }
        
#pragma omp parallel
        {
          real_t lqlinfinity = std::numeric_limits<real_t>::max();
#pragma omp for schedule(static) reduction(+:qmean)
          for(int i=0;i<NElements;i++){
            const int *n=_mesh->get_element(i);
            if(n[0]<0)
              continue;
            
            const real_t *x0 = _mesh->get_coords(n[0]);
            const real_t *x1 = _mesh->get_coords(n[1]);
            const real_t *x2 = _mesh->get_coords(n[2]);
            const real_t *x3 = _mesh->get_coords(n[3]);
            
            qvec[i] = property->lipnikov(x0, x1, x2, x3,
                                         &(_mesh->metric[n[0]*9]),
                                         &(_mesh->metric[n[1]*9]),
                                         &(_mesh->metric[n[2]*9]),
                                         &(_mesh->metric[n[3]*9]));
            
            lqlinfinity = std::min(lqlinfinity, qvec[i]);
            qmean += qvec[i]/NElements;
          }
#pragma omp for schedule(static) reduction(+:qrms)
          for(int i=0;i<NElements;i++){
            
            qrms += pow(qvec[i]-qmean, 2);
          }
#pragma omp critical 
          {
            qlinfinity = std::min(qlinfinity, lqlinfinity);
          }
        }
      }
      
      //qrms=sqrt(qrms/NElements);
      //std::cout<<NElements<<" "<<qmean<<" "<<qrms<<" "<<qlinfinity<<std::endl;

      if(prev_mean_quality<0){
        prev_mean_quality = qmean;
        continue;
      }else{
        double res = fabs(qmean-prev_mean_quality)/prev_mean_quality;
        prev_mean_quality = qmean;
        if(res<tolerance){
          iter++; // Ensure number of iterations returned is correct.
          break;
        }
      }
    }
    
    return iter;
  }

  void smooth_2d_kernel(index_t node, bool qconstrain=false){
    real_t min_q=0, mean_q=0;
    if(qconstrain){
      typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
      {
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        min_q = property->lipnikov(x0, x1, x2, 
                                   &(_mesh->metric[n[0]*4]),
                                   &(_mesh->metric[n[1]*4]),
                                   &(_mesh->metric[n[2]*4]));
        mean_q = min_q;
      }
      for(;ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);
        
        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        real_t q = property->lipnikov(x0, x1, x2,
                                      &(_mesh->metric[n[0]*4]),
                                      &(_mesh->metric[n[1]*4]),
                                      &(_mesh->metric[n[2]*4]));
        min_q = min(q, min_q);
        mean_q += q;
      }
      mean_q/=_mesh->NEList[node].size();
    }

    real_t p[2], mp[4];
    const real_t *normal=NULL;
    std::deque<index_t> adj_nodes;
    if(_surface->contains_node(node)){
      // Check how many different planes intersect at this node.
      std::set<int> coids;
      std::set<index_t> patch = _surface->get_surface_patch(node);
      for(typename std::set<index_t>::const_iterator e=patch.begin();e!=patch.end();++e)
        coids.insert(_surface->get_coplanar_id(*e));

      if(coids.size()==1){
        // We will need this later when making sure that point is on the surface to within roundoff.
        normal = _surface->get_normal(*patch.begin());

        std::set<index_t> adj_nodes_set;;
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
        return;
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
      return;
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
  
    bool improvement = true;
    if(qconstrain){
      // Check if this positions improves the local mesh quality.
      real_t min_q_new = std::numeric_limits<real_t>::max(), mean_q_new=0;
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
        
        real_t q = property->lipnikov(p, x1, x2, 
                                      mp,
                                      &(_mesh->metric[n[loc1]*4]),
                                      &(_mesh->metric[n[loc2]*4]));
        mean_q_new += q;
        min_q_new = min(min_q_new, q);
      }
      mean_q_new /= _mesh->NEList[node].size();
      
      improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
    }  
    if(improvement){
      for(size_t j=0;j<2;j++)
        _mesh->_coords[node*2+j] = p[j];
      
      _mesh->metric[node*4  ] = mp[0];
      _mesh->metric[node*4+1] = mp[1];
      _mesh->metric[node*4+2] = mp[2];
      _mesh->metric[node*4+3] = mp[3];
    }
  }

  void smooth_3d_kernel(index_t node, bool qconstrain=false){
    real_t min_q=0, mean_q=0;
    if(qconstrain){
      typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();
      {
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);

        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        const real_t *x3 = _mesh->get_coords(n[3]);
        min_q = property->lipnikov(x0, x1, x2, x3,
                                   &(_mesh->metric[n[0]*9]),
                                   &(_mesh->metric[n[1]*9]),
                                   &(_mesh->metric[n[2]*9]),
                                   &(_mesh->metric[n[3]*9]));
        mean_q = min_q;
      }
      for(;ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);

        const real_t *x0 = _mesh->get_coords(n[0]);
        const real_t *x1 = _mesh->get_coords(n[1]);
        const real_t *x2 = _mesh->get_coords(n[2]);
        const real_t *x3 = _mesh->get_coords(n[3]);
        real_t q = property->lipnikov(x0, x1, x2, x3,
                                      &(_mesh->metric[n[0]*9]),
                                      &(_mesh->metric[n[1]*9]),
                                      &(_mesh->metric[n[2]*9]),
                                      &(_mesh->metric[n[3]*9]));
        min_q = min(q, min_q);
        mean_q += q;
      }
      mean_q/=_mesh->NEList[node].size();
    }
    real_t A00=0, A01=0, A02=0, A11=0, A12=0, A22=0, q0=0, q1=0, q2=0;
    for(typename std::deque<index_t>::const_iterator il=_mesh->NNList[node].begin();il!=_mesh->NNList[node].end();++il){
      real_t ml00 = 0.5*(_mesh->metric[node*9  ] + _mesh->metric[*il*9  ]);
      real_t ml01 = 0.5*(_mesh->metric[node*9+1] + _mesh->metric[*il*9+1]);
      real_t ml02 = 0.5*(_mesh->metric[node*9+2] + _mesh->metric[*il*9+2]);
      real_t ml11 = 0.5*(_mesh->metric[node*9+4] + _mesh->metric[*il*9+4]);
      real_t ml12 = 0.5*(_mesh->metric[node*9+5] + _mesh->metric[*il*9+5]);
      real_t ml22 = 0.5*(_mesh->metric[node*9+8] + _mesh->metric[*il*9+8]);
      
      q0 += ml00*get_x(*il) + ml01*get_y(*il) + ml02*get_z(*il);
      q1 += ml01*get_x(*il) + ml11*get_y(*il) + ml12*get_z(*il);
      q2 += ml02*get_x(*il) + ml12*get_y(*il) + ml22*get_z(*il);
      
      A00 += ml00;
      A01 += ml01;
      A02 += ml02;
      A11 += ml11;
      A12 += ml12;
      A22 += ml22;
    }
    // Want to solve the system Ap=q to find the new position, p.
    real_t p[] = {-(((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*(A01*A02/A00 - A12)/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - A01/((A01*A01/A00 - A11)*A00))*q1 + (pow((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00, 2)/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22) - A01*A01/((A01*A01/A00 - A11)*A00*A00) + 1/A00)*q0 + ((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*q2/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22),
                  (pow(A01*A02/A00 - A12, 2)/(pow(A01*A01/A00 - A11, 2)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - 1/(A01*A01/A00 - A11))*q1 - (((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*(A01*A02/A00 - A12)/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) - A01/((A01*A01/A00 - A11)*A00))*q0 - (A01*A02/A00 - A12)*q2/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)),
                  ((A01*A02/A00 - A12)*A01/((A01*A01/A00 - A11)*A00) - A02/A00)*q0/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22) - (A01*A02/A00 - A12)*q1/((A01*A01/A00 - A11)*(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)) + q2/(pow(A01*A02/A00 - A12, 2)/(A01*A01/A00 - A11) - A02*A02/A00 + A22)};
    
    if(_surface->contains_node(node)){
      // If this node is on the surface then we have to project
      // this position back onto the surface.
      std::set<index_t> *patch;
      patch = new std::set<index_t>;
      *patch = _surface->get_surface_patch(node);
      
      std::map<int, std::set<int> > *coids;
      coids = new std::map<int, std::set<int> >;
      
      for(typename std::set<index_t>::const_iterator e=patch->begin();e!=patch->end();++e)
        (*coids)[_surface->get_coplanar_id(*e)].insert(*e);
      
      if(coids->size()<3)
        for(std::map<int, std::set<int> >::const_iterator ic=coids->begin();ic!=coids->end();++ic){
          const real_t *normal = _surface->get_normal(*(ic->second.begin()));
          p[0] -= (p[0]-get_x(node))*fabs(normal[0]);
          p[1] -= (p[1]-get_y(node))*fabs(normal[1]);
          p[2] -= (p[2]-get_z(node))*fabs(normal[2]);
        }
      
      size_t coids_size = coids->size();
      
      delete patch;
      delete coids;
      
      // Test if this is a corner node, or edge node in which case it cannot be moved.
      if(coids_size>2)
        return;
    }
    
    // Interpolate metric at this new position.
    real_t mp[9], l[4], L;
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
      return;
    
    {
      const index_t *n=_mesh->get_element(best_e);
      assert(n[0]>=0);

      for(size_t i=0;i<9;i++)
        mp[i] =
          (l[0]*_mesh->metric[n[0]*9+i]+
           l[1]*_mesh->metric[n[1]*9+i]+
           l[2]*_mesh->metric[n[2]*9+i]+
           l[3]*_mesh->metric[n[3]*9+i])/L;
      
      MetricTensor<real_t>::positive_definiteness(3, mp);
    }
    
    bool improvement=true;
    if(qconstrain){
      // Check if this positions improves the local mesh quality.
      real_t min_q_new = std::numeric_limits<real_t>::max(), mean_q_new=0;
      for(typename std::set<index_t>::iterator ie=_mesh->NEList[node].begin();ie!=_mesh->NEList[node].end();++ie){
        const index_t *n=_mesh->get_element(*ie);
        assert(n[0]>=0);

        real_t vectors[] = {get_x(n[0]), get_y(n[0]), get_z(n[0]),
                            get_x(n[1]), get_y(n[1]), get_z(n[1]),
                            get_x(n[2]), get_y(n[2]), get_z(n[2]),
                            get_x(n[3]), get_y(n[3]), get_z(n[3])};
        
        real_t *r[4], *m[4];
        for(int iloc=0;iloc<4;iloc++)
          if(n[iloc]==(node)){
            r[iloc] = p;
            m[iloc] = mp;
          }else{
            r[iloc] = vectors+3*iloc;
            m[iloc] = &(_mesh->metric[n[iloc]*9]);
          }
        real_t q = property->lipnikov(r[0], r[1], r[2], r[3],
                                      m[0], m[1], m[2], m[3]);
        mean_q_new += q;
        min_q_new = min(min_q_new, q);
      }
      
      mean_q_new /= _mesh->NEList[node].size();
      
      improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
    }
    if(improvement){
      for(size_t j=0;j<ndims;j++)
        _mesh->_coords[node*ndims+j] = p[j];
      
      _mesh->metric[node*9  ] = mp[0]; _mesh->metric[node*9+1] = mp[1]; _mesh->metric[node*9+2] = mp[2];
      _mesh->metric[node*9+3] = mp[1]; _mesh->metric[node*9+4] = mp[4]; _mesh->metric[node*9+5] = mp[5];
      _mesh->metric[node*9+6] = mp[2]; _mesh->metric[node*9+7] = mp[5]; _mesh->metric[node*9+8] = mp[8];
    }
  }
  
 private:
  void init_cache(){
    colour_sets.clear();

    int NNodes = _mesh->get_number_elements();
    std::vector<index_t> colour(NNodes, -1);
    Colour<index_t>::greedy(_mesh->NNList, &(colour[0]));
    
    for(int i=0;i<NNodes;i++){
      if((colour[i]<0)||(_mesh->is_halo_node(i)))
        continue;
      colour_sets[colour[i]].push_back(i);
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

  std::map<int, std::deque<index_t> > colour_sets;
};
#endif
