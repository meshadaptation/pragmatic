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
 
#ifndef COARSEN_H
#define COARSEN_H

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#include "ElementProperty.h"
#include "Lock.h"
#include "Mesh.h"

/*! \brief Performs 2D/3D mesh coarsening.
 *
 */

template<typename real_t, int dim> class Coarsen{
 public:
  /// Default constructor.
  Coarsen(Mesh<real_t> &mesh){
    _mesh = &mesh;

    property = NULL;
    size_t NElements = _mesh->get_number_elements();
    for(size_t i=0;i<NElements;i++){
      const int *n=_mesh->get_element(i);
      if(n[0]<0)
        continue;

      if(dim==2)
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

    nnodes_reserve = 0;
    delete_slivers = false;
    surface_coarsening = false;
    quality_constrained = false;
  }

  /// Default destructor.
  ~Coarsen(){
    if(property!=NULL)
      delete property;
  }

  /*! Perform coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  void coarsen(real_t L_low, real_t L_max,
	       bool enable_surface_coarsening=false,
	       bool enable_delete_slivers=false,
	       bool enable_quality_constrained=false){
    
    surface_coarsening = enable_surface_coarsening;
    delete_slivers = enable_delete_slivers;
    quality_constrained = enable_quality_constrained;
    
    size_t NNodes = _mesh->get_number_nodes();

    _L_low = L_low;
    _L_max = L_max;
        
    std::vector< std::atomic<int> > ccount(100);
    std::fill(ccount.begin(), ccount.end(), 0);

    if(nnodes_reserve<NNodes){
      nnodes_reserve = NNodes;

      vLocks.resize(NNodes);
    }

#pragma omp parallel
    {
      // Initialize.
#pragma omp for schedule(static)
      for(int i=0;i<NNodes;i++){
        vLocks[i].unlock();
      }

      for(int citerations=0;citerations<2;citerations++){
        // Vector "retry" is used to store aborted vertices.
        // Vector "round" is used to store propagated vertices.
        std::vector<index_t> retry, next_retry;
        std::vector<index_t> locks_held;
#pragma omp for schedule(static) nowait
        for(index_t node=0; node<NNodes; ++node){ // Need to consider randomising order to avoid mesh artifacts related to numbering.
          bool abort = false;

          if(!vLocks[node].try_lock()){
            retry.push_back(node);
            continue;
          }
          locks_held.push_back(node);

          for(auto& it : _mesh->NNList[node]){
            if(!vLocks[it].try_lock()){
              abort = true;
              break;
            }
            locks_held.push_back(it);
          }

          if(!abort){
            index_t target = coarsen_identify_kernel(node, L_low, L_max);
            if(target>=0){
              coarsen_kernel(node, target);
	      ccount[citerations]++;
            }
          }else{
            retry.push_back(node);
          }

          for(auto& it : locks_held){
            vLocks[it].unlock();
          }
          locks_held.clear();
        }

        for(int iretry=0;iretry<100;iretry++){
          next_retry.clear();

          for(auto& node : retry){
            bool abort = false;

            if(!vLocks[node].try_lock()){
              next_retry.push_back(node);
              continue;
            }
            locks_held.push_back(node);

            for(auto& it : _mesh->NNList[node]){
              if(!vLocks[it].try_lock()){
                abort = true;
                break;
              }
              locks_held.push_back(it);
            }

            if(!abort){
              index_t target = coarsen_identify_kernel(node, L_low, L_max);
              if(target>=0){
                coarsen_kernel(node, target);
		ccount[citerations]++;
              }
            }else{
              next_retry.push_back(node);
            }

            for(auto& it : locks_held){
              vLocks[it].unlock();
            }
            locks_held.clear();
          }

          retry.swap(next_retry);
	  if(retry.empty())
	    break;
        }

#pragma omp barrier
	if(ccount[citerations]==0){
	  break;
        }
      }
    }
  }

 private:

  /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
   */
  inline int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const{
    // Cannot delete if already deleted.
    if(_mesh->NNList[rm_vertex].empty())
      return -1;
    
    if(_mesh->NEList[rm_vertex].size()==1)
      return -1;

    // For now, lock the halo
    if(_mesh->is_halo_node(rm_vertex))
      return -1;
    
    //
    bool delete_with_extreme_prejudice = false;
    if(delete_slivers && dim==3){
      std::set<index_t>::const_iterator ee=_mesh->NEList[rm_vertex].begin();
      double q_linf = _mesh->quality[*ee];
      ++ee;

      for(;ee!=_mesh->NEList[rm_vertex].end();++ee)
        q_linf = std::min(q_linf, _mesh->quality[*ee]);

      if(q_linf<1.0e-6)
        delete_with_extreme_prejudice = true;
    }

    /* Sort the edges according to length. We want to collapse the
       shortest. If it is not possible to collapse the edge then move
       onto the next shortest.*/
    std::multimap<real_t, index_t> short_edges;
    for(const auto &nn : _mesh->NNList[rm_vertex]){
      double length = _mesh->calc_edge_length(rm_vertex, nn);
      if(length<L_low || delete_with_extreme_prejudice)
        short_edges.insert(std::pair<real_t, index_t>(length, nn));
    }
    
    bool reject_collapse = false;
    index_t target_vertex=-1;
    while(short_edges.size()){
      // Get the next shortest edge.
      target_vertex = short_edges.begin()->second;
      short_edges.erase(short_edges.begin());

       // Assume the best.
      reject_collapse=false;
     
      if(surface_coarsening){
	std::set<index_t> compromised_boundary;
	for(const auto &element : _mesh->NEList[rm_vertex]){
          const int *n=_mesh->get_element(element);
          for(size_t i=0;i<nloc;i++){
            if(n[i]!=rm_vertex){
              if(_mesh->boundary[element*nloc+i]>0){
                compromised_boundary.insert(_mesh->boundary[element*nloc+i]);
	      }
	    }
          }
	}

	if(compromised_boundary.size()>1){
          reject_collapse=true;
	  continue;
	}else if(compromised_boundary.size()==1){
     	  // Only allow this vertex to be collapsed to a vertex on the same boundary (not to an internal vertex).
          std::set<index_t> target_boundary;
          for(const auto &element : _mesh->NEList[target_vertex]){
            const int *n=_mesh->get_element(element);
            for(size_t i=0;i<nloc;i++){
              if(n[i]!=target_vertex){
                if(_mesh->boundary[element*nloc+i]>0){
                  target_boundary.insert(_mesh->boundary[element*nloc+i]);
	        }
              }
            }
	  }

          if(target_boundary.size()==1){
	    if(*target_boundary.begin() != *compromised_boundary.begin()){
              reject_collapse=true;
              continue;
	    }

            // Restrict how many boundary facets we can collapse in one go (fewer topological issues to think about).	
            std::set<index_t> deleted_elements;
            std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                                  _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
	                          std::inserter(deleted_elements, deleted_elements.begin()));
            int scnt=0;
	    bool confirm_boundary=false;
            for(const auto& de : deleted_elements){
              // Count surface facets.
	      // Check this is not actually an internal edge.
	      const int *n = _mesh->get_element(de);
  	      for(int i=0;i<nloc;i++){
                if(_mesh->boundary[de*nloc+i]>0)
                  scnt++;

		if(!confirm_boundary){
		  if(n[i]!=rm_vertex && n[i]!=target_vertex){ 
		    std::set<index_t> paired_elements;
		    std::set_intersection(deleted_elements.begin(), deleted_elements.end(),
                                          _mesh->NEList[n[i]].begin(), _mesh->NEList[n[i]].end(),
                                          std::inserter(paired_elements, paired_elements.begin()));
		    if(paired_elements.size()==1)
                      confirm_boundary = true;
	          }
		}
	      }
	    }
            if(scnt!=2 || !confirm_boundary){
              reject_collapse=true;
	      continue;
            }
          }else{
            reject_collapse=true;
            continue;
          }
	}
      }

      /* Check the properties of new elements. If the
         new properties are not acceptable then continue. */
      
      long double total_old_av=0;
      long double total_new_av=0;
      bool better=true;
      for(const auto &ee : _mesh->NEList[rm_vertex]){
	const int *old_n=_mesh->get_element(ee);
	
	double q_linf = 0.0;
	if(quality_constrained)
	  q_linf = _mesh->quality[ee];
	
	long double old_av=0.0;
	if(!surface_coarsening){
	  if(dim==2)
	    old_av = property->area_precision(_mesh->get_coords(old_n[0]),
					      _mesh->get_coords(old_n[1]),
					      _mesh->get_coords(old_n[2]));
	  else
	    old_av = property->volume_precision(_mesh->get_coords(old_n[0]),
						_mesh->get_coords(old_n[1]),
						_mesh->get_coords(old_n[2]),
						_mesh->get_coords(old_n[3]));
	  
	  total_old_av+=old_av;
	}
	  
	// Skip checks this element would be deleted under the operation.
	if(_mesh->NEList[target_vertex].find(ee)!=_mesh->NEList[target_vertex].end())
	  continue;
	
	// Create a copy of the proposed element
	std::vector<int> n(nloc);
	for(size_t i=0;i<nloc;i++){
	  int nid = old_n[i];
	  if(nid==rm_vertex)
	    n[i] = target_vertex;
	  else
	    n[i] = nid;
	}
	
	// Check the area/volume of this new element.
	long double new_av=0.0;
	if(dim==2)
	  new_av = property->area_precision(_mesh->get_coords(n[0]),
				            _mesh->get_coords(n[1]),
				            _mesh->get_coords(n[2]));
	else
	  new_av = property->volume_precision(_mesh->get_coords(n[0]),
					      _mesh->get_coords(n[1]),
					      _mesh->get_coords(n[2]),
					      _mesh->get_coords(n[3]));
	
	// Reject if there is an inverted element.
	if(new_av<DBL_EPSILON){
          reject_collapse=true;
          break;
        }
	
	total_new_av+=new_av;
	
	double new_q=0.0;
	if(quality_constrained){
	  if(dim==2)
	    new_q = property->lipnikov(_mesh->get_coords(n[0]),
				       _mesh->get_coords(n[1]),
				       _mesh->get_coords(n[2]),
				       _mesh->get_metric(n[0]),
				       _mesh->get_metric(n[1]),
				       _mesh->get_metric(n[2]));
	  else
	    new_q = property->lipnikov(_mesh->get_coords(n[0]),
				       _mesh->get_coords(n[1]),
				       _mesh->get_coords(n[2]),
				       _mesh->get_coords(n[3]),
				       _mesh->get_metric(n[0]),
				       _mesh->get_metric(n[1]),
				       _mesh->get_metric(n[2]),
				       _mesh->get_metric(n[3]));
	  if(new_q<q_linf)
	    better=false;
	}
      }
      if(reject_collapse)
	continue;
      
      if(!surface_coarsening){
	// Check we are not removing surface features.
	if(std::abs(total_new_av-total_old_av)/std::max(total_new_av, total_old_av)>DBL_EPSILON){
          reject_collapse=true;
	  continue;
	}
      }
      
      if(!delete_with_extreme_prejudice){
	// Check if any of the new edges are longer than L_max.
        for(const auto &nn : _mesh->NNList[rm_vertex]){
          if(target_vertex==nn)
            continue;

          if(_mesh->calc_edge_length(target_vertex, nn)>L_max){
            reject_collapse=true;
            break;
          }
        }
	if(reject_collapse)
	  continue;
      }

      if(quality_constrained){
        if(!better){
	  reject_collapse=false;
        }
      }

      // If this edge is ok to collapse then break out of loop.
      if(!reject_collapse)
        break;
    }

    // If we've checked all edges and none are collapsible then return.
    if(reject_collapse)
      return -2;

    return target_vertex;
  }

  /*! Kernel for performing coarsening.
   * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
   */
  inline void coarsen_kernel(index_t rm_vertex, index_t target_vertex){
    std::set<index_t> deleted_elements;
    std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                     _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                     std::inserter(deleted_elements, deleted_elements.begin()));

    // This is the set of vertices which are common neighbours between rm_vertex and target_vertex.
    std::set<index_t> common_patch;

    std::set<index_t> compromised_boundary;
    for(const auto &element : _mesh->NEList[rm_vertex]){
      const int *n=_mesh->get_element(element);
      for(size_t i=0;i<nloc;i++){
        if(n[i]!=rm_vertex){
          if(_mesh->boundary[element*nloc+i]>0){
            compromised_boundary.insert(_mesh->boundary[element*nloc+i]);
          }
        }
      }
    }

    std::set<index_t> target_boundary;
    for(const auto &element : _mesh->NEList[target_vertex]){
      const int *n=_mesh->get_element(element);
      for(size_t i=0;i<nloc;i++){
        if(n[i]!=target_vertex){
          if(_mesh->boundary[element*nloc+i]>0){
            target_boundary.insert(_mesh->boundary[element*nloc+i]);
          }
        }
      }
    }

    // Remove deleted elements from node-element adjacency list and from element-node list.
    for(const auto &eid : deleted_elements){
      // Remove element from NEList[rm_vertex].
      _mesh->NEList[rm_vertex].erase(eid);

      // Remove element from NEList of the other two vertices.
      size_t lrm_vertex;
      std::vector<index_t> other_vertex;
      for(size_t i=0; i<nloc; ++i){
        index_t vid = _mesh->_ENList[eid*nloc+i];
        if(vid==rm_vertex){
          lrm_vertex = i;
        }else{
          _mesh->NEList[vid].erase(eid);

          // If this vertex is neither rm_vertex nor target_vertex, then it is one of the common neighbours.
          if(vid != target_vertex){
            other_vertex.push_back(vid);
            common_patch.insert(vid);
          }
        }
      }

      // Handle vertex collapsing onto boundary.
      if(compromised_boundary.empty() && _mesh->boundary[eid*nloc+lrm_vertex]>0){
	assert(target_boundary.size()==1);

        // Find element whose internal facet will be pulled into the external boundary.
        std::set<index_t> otherNE;
        if(dim==2){
          assert(other_vertex.size()==1);
          otherNE = _mesh->NEList[other_vertex[0]];
        }else{
          assert(other_vertex.size()==2);
          std::set_intersection(_mesh->NEList[other_vertex[0]].begin(), _mesh->NEList[other_vertex[0]].end(),
                                _mesh->NEList[other_vertex[1]].begin(), _mesh->NEList[other_vertex[1]].end(),
                                std::inserter(otherNE, otherNE.begin()));
        }
        std::set<index_t> new_boundary_eid;
        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                              otherNE.begin(), otherNE.end(),
                              std::inserter(new_boundary_eid, new_boundary_eid.begin()));

        // eid has been removed from NEList[rm_vertex],
        // so new_boundary_eid contains only the other element.
	if(!new_boundary_eid.empty()){
          assert(new_boundary_eid.size()==1);
          index_t target_eid = *new_boundary_eid.begin();
          for(int i=0;i<nloc;i++){
            int nid=_mesh->_ENList[target_eid*nloc+i];
            if(dim==2){
              if(nid!=rm_vertex && nid!=other_vertex[0]){
                _mesh->boundary[target_eid*nloc+i] = _mesh->boundary[eid*nloc+lrm_vertex];
                break;
              }
             }else{
              if(nid!=rm_vertex && nid!=other_vertex[0] && nid!=other_vertex[1]){
                _mesh->boundary[target_eid*nloc+i] = _mesh->boundary[eid*nloc+lrm_vertex];
                break;
              }
            }
          }
	}
      }
      /*
      else if(dim==3 && !compromised_boundary.empty()){
	assert(target_boundary.size()==1);

        // Find element whose internal edge will be pulled into an external edge.
        std::set<index_t> otherNE;
        assert(other_vertex.size()==2);
	
        std::set_intersection(_mesh->NEList[other_vertex[0]].begin(), _mesh->NEList[other_vertex[0]].end(),
                              _mesh->NEList[other_vertex[1]].begin(), _mesh->NEList[other_vertex[1]].end(),
                              std::inserter(otherNE, otherNE.begin()));

        std::set<index_t> new_boundary_eid;
	int tvertex = target_vertex;
        int boundary_id = _mesh->boundary[eid*nloc+tvertex];
        std::set_intersection(_mesh->NEList[tvertex].begin(), _mesh->NEList[tvertex].end(),
                              otherNE.begin(), otherNE.end(),
                              std::inserter(new_boundary_eid, new_boundary_eid.begin()));
	assert(new_boundary_eid.size()==1 || new_boundary_eid.size()==0);
	if(!new_boundary_eid.empty()){
	  std::cerr<<"t";
          index_t target_eid = *new_boundary_eid.begin();
          const index_t *n=_mesh->get_element(target_eid);
          for(int i=0;i<nloc;i++){
            if(n[i]!=tvertex && n[i]!=other_vertex[0] && n[i]!=other_vertex[1]){
              _mesh->boundary[target_eid*nloc+i] = boundary_id;
              break;
            }
          }
        }

        int tvertex=rm_vertex;
        int boundary_id = _mesh->boundary[eid*nloc+tvertex];
	std::set_intersection(_mesh->NEList[tvertex].begin(), _mesh->NEList[tvertex].end(),
	                      otherNE.begin(), otherNE.end(),
	                      std::inserter(new_boundary_eid, new_boundary_eid.begin()));

        // eid has been removed from NEList[rm_vertex],
        // so new_boundary_eid contains only the other element.
        for(auto& target_eid : new_boundary_eid){
	  std::cerr<<"r";
          const index_t *n=_mesh->get_element(target_eid);
          for(int i=0;i<nloc;i++){
            if(n[i]!=tvertex && n[i]!=other_vertex[0] && n[i]!=other_vertex[1]){
              _mesh->boundary[target_eid*nloc+i] = boundary_id;
              break;
            }
          }
        }
	std::cerr<<"-";
      }
      */

      // Remove element from mesh.
      _mesh->_ENList[eid*nloc] = -1;
    }

    assert((dim==2 && common_patch.size() == deleted_elements.size()) || (dim==3));

    // For all adjacent elements, replace rm_vertex with target_vertex in ENList and update quality.
    for(const auto& eid : _mesh->NEList[rm_vertex]){
      for(size_t i=0;i<nloc;i++){
        if(_mesh->_ENList[nloc*eid+i]==rm_vertex){
          _mesh->_ENList[nloc*eid+i] = target_vertex;
          break;
        }
      }

      _mesh->template update_quality<dim>(eid);

      // Add element to target_vertex's NEList.
      _mesh->NEList[target_vertex].insert(eid);
    }

    // Update surrounding NNList.
    common_patch.insert(target_vertex);
    for(const auto& nid : _mesh->NNList[rm_vertex]){
      typename std::vector<index_t>::iterator it = std::find(_mesh->NNList[nid].begin(), _mesh->NNList[nid].end(), rm_vertex);
      _mesh->NNList[nid].erase(it);

      // Find all entries pointing back to rm_vertex and update them to target_vertex.
      if(common_patch.count(nid)==0){
        if(true || !compromised_boundary.empty()){
          // Need to take extra care as the topology may have changed.
	  if(std::find(_mesh->NNList[nid].begin(), _mesh->NNList[nid].end(), target_vertex)==_mesh->NNList[nid].end())
            _mesh->NNList[nid].push_back(target_vertex);

          if(std::find(_mesh->NNList[target_vertex].begin(), _mesh->NNList[target_vertex].end(), nid)==_mesh->NNList[target_vertex].end())
            _mesh->NNList[target_vertex].push_back(nid);
        }else{
          _mesh->NNList[nid].push_back(target_vertex);
  	  _mesh->NNList[target_vertex].push_back(nid);
	}
      }
    }

    _mesh->erase_vertex(rm_vertex);
  }

  Mesh<real_t> *_mesh;
  ElementProperty<real_t> *property;

  size_t nnodes_reserve;
  std::vector<Lock> vLocks;

  real_t _L_low, _L_max;
  bool delete_slivers, surface_coarsening, quality_constrained;

  const static size_t ndims=dim;
  const static size_t nloc=dim+1;
  const static size_t msize=(dim==2?3:6);
};

#endif
