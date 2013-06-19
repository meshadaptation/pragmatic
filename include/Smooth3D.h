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

#ifndef SMOOTH3D_H
#define SMOOTH3D_H

/*! \brief Applies Laplacian smoothen in metric space.
 */
template<typename real_t>
  class Smooth3D{
 public:
  /// Default constructor.
  Smooth3D(Mesh<real_t> &mesh, Surface3D<real_t> &surface){
    _mesh = &mesh;
    _surface = &surface;

    mpi_nparts = 1;
    rank=0;
#ifdef HAVE_MPI
    MPI_Comm_size(_mesh->get_mpi_comm(), &mpi_nparts);
    MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
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

      property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                             _mesh->get_coords(n[1]),
                                             _mesh->get_coords(n[2]),
                                             _mesh->get_coords(n[3]));
      break;
    }

    kernels["Laplacian"]       = &Smooth3D<real_t>::laplacian_3d_kernel;
    kernels["smart Laplacian"] = &Smooth3D<real_t>::smart_laplacian_3d_kernel;
    //	kernels["smart Laplacian search"]	= &Smooth3D<real_t>::smart_laplacian_search_3d_kernel;
    //	kernels["optimisation Linf"]			= &Smooth3D<real_t>::optimisation_linf_3d_kernel;
  }

  /// Default destructor.
  ~Smooth3D(){
    delete property;
  }

  // Smooth the mesh using a given method. Valid methods are:
  // "Laplacian", "smart Laplacian", "smart Laplacian search", "optimisation Linf"
  void smooth(std::string method, int max_iterations=10){
    init_cache(method);

    std::vector<int> halo_elements;
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

    bool (Smooth3D<real_t>::*smooth_kernel)(index_t) = NULL;

    if(kernels.count(method)){
      smooth_kernel = kernels[method];
    }else{
      std::cerr<<"WARNING: Unknown smoothing method \""<<method<<"\"\nUsing \"smart Laplacian\"\n";
      smooth_kernel = kernels["smart Laplacian"];
    }

    // Use this to keep track of vertices that are still to be visited.
    int NNodes = _mesh->get_number_nodes();
    std::vector<int> active_vertices(NNodes, 0);

    // First sweep through all vertices. Add vertices adjacent to any
    // vertex moved into the active_vertex list.
    int max_colour = colour_sets.rbegin()->first;
#ifdef HAVE_MPI
    if(mpi_nparts>1)
      MPI_Allreduce(MPI_IN_PLACE, &max_colour, 1, MPI_INT, MPI_MAX, _mesh->get_mpi_comm());
#endif

    int nav = 0;
#pragma omp parallel
    {
      for(int ic=1;ic<=max_colour;ic++){
        if(colour_sets.count(ic)){
          int node_set_size = colour_sets[ic].size();
#pragma omp for schedule(static)
          for(int cn=0;cn<node_set_size;cn++){
            index_t node = colour_sets[ic][cn];

            if((this->*smooth_kernel)(node)){
              for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                active_vertices[*it] = 1;
              }
            }
          }
        }

#pragma omp master
        {
          _mesh->halo_update(&(_mesh->_coords[0]), ndims);
          _mesh->halo_update(&(_mesh->metric[0]), msize);

          for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
            quality[*ie] = -1;
        }
#pragma omp barrier
      }

      for(int iter=1;iter<max_iterations;iter++){
        for(int ic=1;ic<=max_colour;ic++){
          if(colour_sets.count(ic)){
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
                for(typename std::vector<index_t>::const_iterator it=_mesh->NNList[node].begin();it!=_mesh->NNList[node].end();++it){
                  active_vertices[*it] = 1;
                }
              }
            }
          }
#pragma omp master
          {
            _mesh->halo_update(&(_mesh->_coords[0]), ndims);
            _mesh->halo_update(&(_mesh->metric[0]), msize);

            for(std::vector<int>::const_iterator ie=halo_elements.begin();ie!=halo_elements.end();++ie)
              quality[*ie] = -1;
          }
#pragma omp barrier
        }

        // Count number of active vertices.
#pragma omp single
        {
          nav = 0;
        }
#pragma omp for schedule(static) reduction(+:nav)
        for(int i=0;i<NNodes;i++){
          if(_mesh->is_owned_node(i))
            nav += active_vertices[i];
        }
#ifdef HAVE_MPI
        if(mpi_nparts>1){
#pragma omp master
          {
            MPI_Allreduce(MPI_IN_PLACE, &nav, 1, MPI_INT, MPI_SUM, _mesh->get_mpi_comm());
          }
        }
#endif

#pragma omp barrier
        if(nav==0)
          break;
      }
    }

    return;
  }

  bool laplacian_3d_kernel(index_t node){
    real_t p[3];
    float mp[6];
    if(laplacian_3d_kernel(node, p, mp)){
      // Looks good so lets copy it back;
      for(size_t j=0;j<3;j++)
        _mesh->_coords[node*3+j] = p[j];

      for(size_t j=0;j<6;j++)
        _mesh->metric[node*6+j] = mp[j];

      return true;
    }
    return false;
  }

  bool laplacian_3d_kernel(index_t node, real_t *p, float *mp){
    const real_t *normal[]={NULL, NULL};
    std::vector<index_t> adj_nodes;
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
    for(typename std::vector<index_t>::const_iterator il=adj_nodes.begin();il!=adj_nodes.end();++il){
      const float *m0 = _mesh->get_metric(node);
      const float *m1 = _mesh->get_metric(*il);

      float ml00 = 0.5*(m0[0] + m1[0]);
      float ml01 = 0.5*(m0[1] + m1[1]);
      float ml02 = 0.5*(m0[2] + m1[2]);
      float ml11 = 0.5*(m0[3] + m1[3]);
      float ml12 = 0.5*(m0[4] + m1[4]);
      float ml22 = 0.5*(m0[5] + m1[5]);

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

    if(!pragmatic_isnormal(p[0]+p[1]+p[2])){
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

      for(size_t i=0;i<6;i++)
        mp[i] =
          (l[0]*_mesh->metric[n[0]*6+i]+
           l[1]*_mesh->metric[n[1]*6+i]+
           l[2]*_mesh->metric[n[2]*6+i]+
           l[3]*_mesh->metric[n[3]*6+i])/L;
    }

    return true;
  }

  bool smart_laplacian_3d_kernel(index_t node){
    real_t p[3];
    float mp[6];
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

    for(size_t j=0;j<6;j++)
      _mesh->metric[node*6+j] = mp[j];

    return true;
  }

 private:
  void init_cache(std::string method){
    colour_sets.clear();

    zoltan_graph_t graph;
    graph.rank = rank;

    int NNodes = _mesh->get_number_nodes();
    assert(NNodes==(int)_mesh->NNList.size());
    graph.nnodes = NNodes;

    int NPNodes = NNodes - _mesh->recv_halo.size();
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
      for(typename std::vector<index_t>::iterator it=_mesh->NNList[i].begin();it!=_mesh->NNList[i].end();++it){
        csr_edges[sum++] = *it;
      }
    }
    graph.csr_edges = &(csr_edges[0]);

    graph.gid = &(_mesh->lnn2gnn[0]);
    graph.owner = (int*) &(_mesh->node_owner[0]);

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
        std::vector<const real_t *> x(nloc);
        std::vector<const float *> m(nloc);
        for(size_t i=0;i<nloc;i++){
          x[i] = _mesh->get_coords(n[i]);
          m[i] = _mesh->get_metric(n[i]);
        }

        quality[*ie] = property->lipnikov(x[0], x[1], x[2], x[3],
                                          m[0], m[1], m[2], m[3]);
      }

      patch_quality = std::min(patch_quality, quality[*ie]);
    }

    return patch_quality;
  }

  real_t functional_Linf(index_t node, const real_t *p, const float *mp) const{
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

  Mesh<real_t> *_mesh;
  Surface3D<real_t> *_surface;
  ElementProperty<real_t> *property;

  const static size_t ndims=3;
  const static size_t nloc=4;
  const static size_t msize=6;

  int mpi_nparts, rank;
  real_t good_q, sigma_q;
  std::vector<real_t> quality;
  std::map<int, std::vector<index_t> > colour_sets;

  std::map<std::string, bool (Smooth3D<real_t>::*)(index_t)> kernels;
};

#endif
