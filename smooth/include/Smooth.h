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
#include "LipnikovFunctional.h"
#include "Surface.h"

/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t, typename index_t>
  class Smooth{
 public:
  /// Default constructor.
  Smooth(){
    _NNodes = 0;
    _NElements = 0;
    _ndims = 0;
    nloc = 0;
    _ENList = NULL;
    _node_distribution = NULL;
    _x = NULL;
    _y = NULL;
    _z = NULL;
#ifdef HAVE_MPI
    _mesh_comm = NULL;
#endif
    _metric = NULL;
  }

  /// Default destructor.
  ~Smooth(){
  }
  
  /*! Set the source mesh (2D triangular meshes).
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param metric is the metric tensor field.
   * @param node_distribution gives the number of nodes owned
   *        by each process when this is using MPI, where it is
   *        assumed that each processes ownes n_i consecutive
   *        vertices of the mesh. Its contents are identical for every process.
   */
  void set_mesh(int NNodes, int NElements, const index_t *ENList, Surface<real_t, index_t> *surface,
                real_t *x, real_t *y, real_t *metric, const index_t *node_distribution=NULL){
    nloc = 3;
    _ndims = 2;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _surface = surface;
    _x = x;
    _y = y;
    _z = NULL;
    _metric = metric;
    _node_distribution = node_distribution;
  }
  
  /*! Set the source mesh (3D tetrahedral meshes).
   * @param NNodes number of nodes in the local mesh.
   * @param NElements number of nodes in the local mesh.
   * @param ENList array storing the global node number for each element.
   * @param x is the X coordinate.
   * @param y is the Y coordinate.
   * @param z is the Z coordinate.
   * @param metric is the metric tensor field.
   * @param node_distribution gives the number of nodes owned
   *        by each process when this is using MPI, where it is
   *        assumed that each processes ownes n_i consecutive
   *        vertices of the mesh. Its contents are identical for every process.
   */
  void set_mesh(int NNodes, int NElements, const index_t *ENList, Surface<real_t, index_t> *surface,
                real_t *x, real_t *y, real_t *z, real_t *metric, const index_t *node_distribution=NULL){
    nloc = 4;
    _ndims = 3;
    _NNodes = NNodes;
    _NElements = NElements;
    _ENList = ENList;
    _surface = surface;
    _x = x;
    _y = y;
    _z = z;
    _metric = metric;
    _node_distribution = node_distribution;
  }
  
  real_t smooth(bool qconstrain=false){
    real_t rms=0;

    // Create node-node list
    if(NNList.empty()){
      NNList.resize(_NNodes);
      NEList.resize(_NNodes);
      for(int i=0; i<_NElements; i++){
        for(int j=0;j<nloc;j++){
          NEList[_ENList[i*nloc+j]].insert(i);
          for(int k=j+1;k<nloc;k++){
            NNList[_ENList[i*nloc+j]].insert(_ENList[i*nloc+k]);
            NNList[_ENList[i*nloc+k]].insert(_ENList[i*nloc+j]);
          }
        }
      }
      Metis<index_t>::reorder(NNList, norder);
    }

    double start_tic = omp_get_wtime();

    if(_ndims==2){
      // Smoothing loop.
      real_t refx0[] = {_x[_ENList[0]], _y[_ENList[0]]};
      real_t refx1[] = {_x[_ENList[1]], _y[_ENList[1]]};
      real_t refx2[] = {_x[_ENList[2]], _y[_ENList[2]]};
      LipnikovFunctional<real_t> functional(refx0, refx1, refx2);

      for(std::vector<int>::const_iterator it=norder.begin();it!=norder.end();++it){
        real_t min_q=0, mean_q=0;
        if(qconstrain){
          typename std::set<index_t>::iterator ie=NEList[*it].begin();
          {
            const index_t *n=_ENList+(*ie)*3;
            real_t x0[] = {_x[n[0]], _y[n[0]]};
            real_t x1[] = {_x[n[1]], _y[n[1]]};
            real_t x2[] = {_x[n[2]], _y[n[2]]};
            min_q = functional.calculate(x0, x1, x2, _metric+n[0]*4, _metric+n[1]*4, _metric+n[2]*4);
            mean_q = min_q;
          }
          for(;ie!=NEList[*it].end();++ie){
            const index_t *n=_ENList+(*ie)*3;
            real_t x0[] = {_x[n[0]], _y[n[0]]};
            real_t x1[] = {_x[n[1]], _y[n[1]]};
            real_t x2[] = {_x[n[2]], _y[n[2]]};
            real_t q = functional.calculate(x0, x1, x2, _metric+n[0]*4, _metric+n[1]*4, _metric+n[2]*4);
            min_q = min(q, min_q);
            mean_q += q;
          }
          mean_q/=NEList[*it].size();
        }
        real_t A00=0, A01=0, A11=0, q0=0, q1=0;
        for(std::set<int>::const_iterator il=NNList[*it].begin();il!=NNList[*it].end();++il){
          real_t ml00 = 0.5*(_metric[*it*4  ] + _metric[*il*4  ]);
          real_t ml01 = 0.5*(_metric[*it*4+1] + _metric[*il*4+1]);
          real_t ml11 = 0.5*(_metric[*it*4+3] + _metric[*il*4+3]);
          
          q0 += ml00*_x[*il] + ml01*_y[*il];
          q1 += ml01*_x[*il] + ml11*_y[*il];
          
          A00 += ml00;
          A01 += ml01;
          A11 += ml11;
        }
        // Want to solve the system Ap=q to find the new position, p.
        real_t p[] = {-(pow(A01, 2)/((pow(A01, 2)/A00 - A11)*pow(A00, 2)) - 1/A00)*q0 + 
                      A01*q1/((pow(A01, 2)/A00 - A11)*A00),
                      A01*q0/((pow(A01, 2)/A00 - A11)*A00) - q1/(pow(A01, 2)/A00 - A11)};
        
        if(_surface->contains_node(*it)){
          // If this node is on the surface then we have to project
          // this position back onto the surface.
          std::set<size_t> patch = _surface->get_surface_patch(*it);
          std::set<int> coids;
          for(std::set<size_t>::const_iterator e=patch.begin();e!=patch.end();++e)
            coids.insert(_surface->get_coplanar_id(*e));
          
          // Test if this is a corner node, in which case it cannot be moved.
          if(coids.size()>1)
            continue;
          
          const real_t *normal = _surface->get_normal(*patch.begin());
          p[0] -= (p[0]-_x[*it])*fabs(normal[0]);
          p[1] -= (p[1]-_y[*it])*fabs(normal[1]);
        }
        
        // Interpolate metric at this new position.
        real_t mp[4], l[3];
        int best_e=*NEList[*it].begin();
        for(size_t b=0;b<5;b++){
          real_t tol=-1;
          for(typename std::set<index_t>::iterator ie=NEList[*it].begin();ie!=NEList[*it].end();++ie){
            const index_t *n=_ENList+(*ie)*3;
            
            real_t x0[] = {_x[n[0]], _y[n[0]]};
            real_t x1[] = {_x[n[1]], _y[n[1]]};
            real_t x2[] = {_x[n[2]], _y[n[2]]};
            
            l[0] = functional.area(p,  x1, x2);
            l[1] = functional.area(x0, p,  x2);
            l[2] = functional.area(x0, x1, p);
            
            real_t min_l = min(l[0], min(l[1], l[2]));
            if(min_l>tol){
              tol = min_l;
              best_e = *ie;
            }
            if(tol>=0){
              break;
            }
          }
          if(tol>=0){
            break;
          }else{
            p[0] = (_x[*it]+p[0])/2;
            p[1] = (_y[*it]+p[1])/2;
          }
        }
        if((l[0]<0)||(l[1]<0)||(l[2]<0))
          continue;
        
        {
          const index_t *n=_ENList+best_e*3;
          real_t x0[] = {_x[n[0]], _y[n[0]]};
          real_t x1[] = {_x[n[1]], _y[n[1]]};
          real_t x2[] = {_x[n[2]], _y[n[2]]};
          real_t L = functional.area(x0, x1, x2);
          if(L<0){
            std::cerr<<"negative area :: "<<*it<<", "<<L<<std::endl;
          }
          for(size_t i=0;i<4;i++){
            mp[i] = (l[0]*_metric[n[0]*4+i]+l[1]*_metric[n[1]*4+i]+l[2]*_metric[n[2]*4+i])/L;
          }
        }
        
        bool improvement = true;
        if(qconstrain){
          // Check if this positions improves the local mesh quality.
          real_t min_q_new=0, mean_q_new=0;
          {
            typename std::set<index_t>::iterator ie=NEList[*it].begin();
            {
              const index_t *n=_ENList+(*ie)*nloc;
              int iloc = 0;
              while(n[iloc]!=(*it)){
                iloc++;
              }
              int loc1 = (iloc+1)%3;
              int loc2 = (iloc+2)%3;
              
              real_t x1[] = {_x[n[loc1]], _y[n[loc1]]};
              real_t x2[] = {_x[n[loc2]], _y[n[loc2]]};
              
              min_q_new = functional.calculate(p, x1, x2, mp, _metric+n[loc1]*4, _metric+n[loc2]*4);
              mean_q_new = min_q_new;
            }
            for(;ie!=NEList[*it].end();++ie){
              const index_t *n=_ENList+(*ie)*nloc;
              int iloc = 0;
              while(n[iloc]!=(*it)){
                iloc++;
              }
              int loc1 = (iloc+1)%3;
              int loc2 = (iloc+2)%3;
              
              real_t x1[] = {_x[n[loc1]], _y[n[loc1]]};
              real_t x2[] = {_x[n[loc2]], _y[n[loc2]]};
              
              real_t q = functional.calculate(p, x1, x2, mp, _metric+n[loc1]*4, _metric+n[loc2]*4);
              mean_q_new += q;
              min_q_new = min(min_q_new, q);
            }
          }
          mean_q_new /= NEList[*it].size();
          
          improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
          if(improvement)
            rms += pow(min_q_new-min_q, 2);
        }

        if(improvement){
          _x[*it] = p[0];
          _y[*it] = p[1];
          _metric[(*it)*4  ] = mp[0];
          _metric[(*it)*4+1] = mp[1];
          _metric[(*it)*4+2] = mp[1];
          _metric[(*it)*4+3] = mp[3];
        }
      }
    }else{
      // Smoothing loop.
      real_t refx0[] = {_x[_ENList[0]], _y[_ENList[0]], _z[_ENList[0]]};
      real_t refx1[] = {_x[_ENList[1]], _y[_ENList[1]], _z[_ENList[1]]};
      real_t refx2[] = {_x[_ENList[2]], _y[_ENList[2]], _z[_ENList[2]]};
      real_t refx3[] = {_x[_ENList[3]], _y[_ENList[3]], _z[_ENList[3]]};
      LipnikovFunctional<real_t> functional(refx0, refx1, refx2, refx3);

      for(std::vector<int>::const_iterator it=norder.begin();it!=norder.end();++it){
        real_t min_q=0, mean_q=0;
        if(qconstrain){
          typename std::set<index_t>::iterator ie=NEList[*it].begin();
          {
            const index_t *n=_ENList+(*ie)*4;
            real_t x0[] = {_x[n[0]], _y[n[0]], _z[n[0]]};
            real_t x1[] = {_x[n[1]], _y[n[1]], _z[n[1]]};
            real_t x2[] = {_x[n[2]], _y[n[2]], _z[n[2]]};
            real_t x3[] = {_x[n[3]], _y[n[3]], _z[n[3]]};
            min_q = functional.calculate(x0, x1, x2, x3,
                                         _metric+n[0]*9, _metric+n[1]*9, _metric+n[2]*9, _metric+n[3]*9);
            mean_q = min_q;
          }
          for(;ie!=NEList[*it].end();++ie){
            const index_t *n=_ENList+(*ie)*4;
            real_t x0[] = {_x[n[0]], _y[n[0]], _z[n[0]]};
            real_t x1[] = {_x[n[1]], _y[n[1]], _z[n[1]]};
            real_t x2[] = {_x[n[2]], _y[n[2]], _z[n[2]]};
            real_t x3[] = {_x[n[3]], _y[n[3]], _z[n[3]]};
            real_t q = functional.calculate(x0, x1, x2, x3,
                                            _metric+n[0]*9, _metric+n[1]*9, _metric+n[2]*9, _metric+n[3]*9);
            min_q = min(q, min_q);
            mean_q += q;
          }
          mean_q/=NEList[*it].size();
        }
        real_t A00=0, A01=0, A02=0, A11=0, A12=0, A22=0, q0=0, q1=0, q2=0;
        for(std::set<int>::const_iterator il=NNList[*it].begin();il!=NNList[*it].end();++il){
          real_t ml00 = 0.5*(_metric[*it*9  ] + _metric[*il*9  ]);
          real_t ml01 = 0.5*(_metric[*it*9+1] + _metric[*il*9+1]);
          real_t ml02 = 0.5*(_metric[*it*9+2] + _metric[*il*9+2]);
          real_t ml11 = 0.5*(_metric[*it*9+4] + _metric[*il*9+4]);
          real_t ml12 = 0.5*(_metric[*it*9+5] + _metric[*il*9+5]);
          real_t ml22 = 0.5*(_metric[*it*9+8] + _metric[*il*9+8]);

          q0 += ml00*_x[*il] + ml01*_y[*il] + ml02*_z[*il];
          q1 += ml01*_x[*il] + ml11*_y[*il] + ml12*_z[*il];
          q2 += ml02*_x[*il] + ml12*_y[*il] + ml22*_z[*il];
          
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
        
        if(_surface->contains_node(*it)){
          // If this node is on the surface then we have to project
          // this position back onto the surface.
          std::set<size_t> patch = _surface->get_surface_patch(*it);
          std::map<int, std::set<int> > coids;
          for(std::set<size_t>::const_iterator e=patch.begin();e!=patch.end();++e)
            coids[_surface->get_coplanar_id(*e)].insert(*e);
          
          // Test if this is a corner node, or edge node in which case it cannot be moved.
          if(coids.size()>2)
            continue;
          
          for(std::map<int, std::set<int> >::const_iterator ic=coids.begin();ic!=coids.end();++ic){
            const real_t *normal = _surface->get_normal(*(ic->second.begin()));
            p[0] -= (p[0]-_x[*it])*fabs(normal[0]);
            p[1] -= (p[1]-_y[*it])*fabs(normal[1]);
            p[2] -= (p[2]-_z[*it])*fabs(normal[2]);
          }
        }
        
        // Interpolate metric at this new position.
        real_t mp[9], l[4];
        int best_e=*NEList[*it].begin();
        bool inverted=false;
        for(size_t bisections=0;bisections<5;bisections++){ // 5 bisections along the search line
          real_t tol=-1;
          for(typename std::set<index_t>::iterator ie=NEList[*it].begin();ie!=NEList[*it].end();++ie){
            const index_t *n=_ENList+(*ie)*4;
            real_t vectors[] = {_x[n[0]], _y[n[0]], _z[n[0]],
                                _x[n[1]], _y[n[1]], _z[n[1]],
                                _x[n[2]], _y[n[2]], _z[n[2]],
                                _x[n[3]], _y[n[3]], _z[n[3]]};
            real_t *x0 = vectors;
            real_t *x1 = vectors+3;
            real_t *x2 = vectors+6;
            real_t *x3 = vectors+9;
            
            real_t *r[4];
            for(int iloc=0;iloc<4;iloc++)
              if(n[iloc]==(*it)){
                r[iloc] = p;
              }else{
                r[iloc] = vectors+3*iloc;
              }
            real_t volume = functional.volume(r[0], r[1], r[2], r[3]);
            if(volume<0){
              inverted = true;
              break;
            }
            
            if(tol<0){
              real_t L = functional.volume(x0, x1, x2, x3);
              if(L<0)
                std::cerr<<"negative volume :: "<<*it<<", "<<L<<std::endl;
              
              l[0] = functional.volume(p,  x1, x2, x3)/L;
              l[1] = functional.volume(x0, p,  x2, x3)/L;
              l[2] = functional.volume(x0, x1, p,  x3)/L;
              l[3] = functional.volume(x0, x1, x2, p)/L;
              
              real_t min_l = min(min(l[0], l[1]), min(l[2], l[3]));
              if(min_l>tol){
                tol = min_l;
                best_e = *ie;
              }
            }
          }
          if(inverted){
            p[0] = (_x[*it]+p[0])/2;
            p[1] = (_y[*it]+p[1])/2;
            p[2] = (_z[*it]+p[2])/2;
          }else{
            break;
          }
        }
        
        if(inverted)
          continue;
      
        {
          const index_t *n=_ENList+best_e*4;
          for(size_t i=0;i<9;i++)
            mp[i] =
              l[0]*_metric[n[0]*9+i]+
              l[1]*_metric[n[1]*9+i]+
              l[2]*_metric[n[2]*9+i]+
              l[3]*_metric[n[3]*9+i];
        }

        bool improvement=true;
        if(qconstrain){
          // Check if this positions improves the local mesh quality.
          real_t min_q_new, mean_q_new;
          {
            typename std::set<index_t>::iterator ie=NEList[*it].begin();
            {
              const index_t *n=_ENList+(*ie)*nloc;
              real_t vectors[] = {_x[n[0]], _y[n[0]], _z[n[0]],
                                  _x[n[1]], _y[n[1]], _z[n[1]],
                                  _x[n[2]], _y[n[2]], _z[n[2]],
                                  _x[n[3]], _y[n[3]], _z[n[3]]};
              
              real_t *r[4], *m[4];
              for(int iloc=0;iloc<4;iloc++)
                if(n[iloc]==(*it)){
                  r[iloc] = p;
                  m[iloc] = mp;
                }else{
                  r[iloc] = vectors+3*iloc;
                  m[iloc] = _metric+n[iloc]*9;
                }
              min_q_new = functional.calculate(r[0], r[1], r[2], r[3],
                                               m[0], m[1], m[2], m[3]);
              mean_q_new = min_q_new;
            }
            for(;ie!=NEList[*it].end();++ie){
              const index_t *n=_ENList+(*ie)*nloc;
              real_t vectors[] = {_x[n[0]], _y[n[0]], _z[n[0]],
                                  _x[n[1]], _y[n[1]], _z[n[1]],
                                  _x[n[2]], _y[n[2]], _z[n[2]],
                                  _x[n[3]], _y[n[3]], _z[n[3]]};
              
              real_t *r[4], *m[4];
              for(int iloc=0;iloc<4;iloc++)
                if(n[iloc]==(*it)){
                  r[iloc] = p;
                  m[iloc] = mp;
                }else{
                  r[iloc] = vectors+3*iloc;
                  m[iloc] = _metric+n[iloc]*9;
                }
              real_t q = functional.calculate(r[0], r[1], r[2], r[3],
                                              m[0], m[1], m[2], m[3]);
              mean_q_new += q;
              min_q_new = min(min_q_new, q);
            }
          }
          mean_q_new /= NEList[*it].size();
          
          improvement = (mean_q_new>mean_q)||(min_q_new>min_q);
          if(improvement)
            rms += pow(min_q_new-min_q, 2);
        }
        if(improvement){
          _x[*it] = p[0];
          _y[*it] = p[1];
          _z[*it] = p[2];
          _metric[(*it)*9  ] = mp[0]; _metric[(*it)*9+1] = mp[1]; _metric[(*it)*9+2] = mp[2];
          _metric[(*it)*9+3] = mp[1]; _metric[(*it)*9+4] = mp[4]; _metric[(*it)*9+5] = mp[5];
          _metric[(*it)*9+6] = mp[2]; _metric[(*it)*9+7] = mp[5]; _metric[(*it)*9+8] = mp[8];
        }
      }
    }

    std::cerr<<"Smooth loop time = "<<omp_get_wtime()-start_tic<<std::endl;
    return sqrt(rms);
  }

 private:
  int _NNodes, _NElements, _ndims, nloc;
  const index_t *_ENList, *_node_distribution;
  real_t *_x, *_y, *_z;
  std::vector< std::set<index_t> > NNList, NEList;
  std::vector<int> norder;
  std::set<index_t> surface_nodes;
#ifdef HAVE_MPI
  const MPI_Comm *_mesh_comm;
#endif
  real_t *_metric;
  Surface<real_t, index_t> *_surface;
};
#endif
