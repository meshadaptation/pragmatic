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

template<typename real_t, int dim> class Coarsen
{
public:
    /// Default constructor.
    Coarsen(Mesh<real_t> &mesh)
    {
        _mesh = &mesh;

        property = NULL;
        size_t NElements = _mesh->get_number_elements();
        for(size_t i=0; i<NElements; i++) {
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
    ~Coarsen()
    {
        if(property!=NULL)
            delete property;
    }

    /*! Perform coarsening.
     * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
     */
    void coarsen(real_t L_low, real_t L_max,
                 bool enable_surface_coarsening=false,
                 bool enable_delete_slivers=false,
                 bool enable_quality_constrained=false)
    {

        surface_coarsening = enable_surface_coarsening;
        delete_slivers = enable_delete_slivers;
        quality_constrained = enable_quality_constrained;

        size_t NNodes = _mesh->get_number_nodes();

        _L_low = L_low;
        _L_max = L_max;

        std::vector< std::atomic<int> > ccount(100);
        std::fill(ccount.begin(), ccount.end(), 0);

        if(nnodes_reserve<NNodes) {
            nnodes_reserve = NNodes;

            vLocks.resize(NNodes);
        }

        #pragma omp parallel
        {
            // Initialize.
            #pragma omp for schedule(static)
            for(int i=0; i<NNodes; i++) {
                vLocks[i].unlock();
            }

            for(int citerations=0; citerations<100; citerations++) {
                // Vector "retry" is used to store aborted vertices.
                // Vector "round" is used to store propagated vertices.
                std::vector<index_t> retry, next_retry;
                std::vector<index_t> locks_held;
                #pragma omp for schedule(static) nowait
                for(index_t node=0; node<NNodes; ++node) { // Need to consider randomising order to avoid mesh artifacts related to numbering.
                    bool abort = false;

                    if(!vLocks[node].try_lock()) {
                        retry.push_back(node);
                        continue;
                    }
                    locks_held.push_back(node);

                    for(auto& it : _mesh->NNList[node]) {
                        if(!vLocks[it].try_lock()) {
                            abort = true;
                            break;
                        }
                        locks_held.push_back(it);
                    }

                    if(!abort) {
                        index_t target = coarsen_identify_kernel(node, L_low, L_max);
                        if(target>=0) {
                            coarsen_kernel(node, target);
                            ccount[citerations]++;
                        }
                    } else {
                        retry.push_back(node);
                    }

                    for(auto& it : locks_held) {
                        vLocks[it].unlock();
                    }
                    locks_held.clear();
                }

                for(int iretry=0; iretry<100; iretry++) {
                    next_retry.clear();

                    for(auto& node : retry) {
                        bool abort = false;

                        if(!vLocks[node].try_lock()) {
                            next_retry.push_back(node);
                            continue;
                        }
                        locks_held.push_back(node);

                        for(auto& it : _mesh->NNList[node]) {
                            if(!vLocks[it].try_lock()) {
                                abort = true;
                                break;
                            }
                            locks_held.push_back(it);
                        }

                        if(!abort) {
                            index_t target = coarsen_identify_kernel(node, L_low, L_max);
                            if(target>=0) {
                                coarsen_kernel(node, target);
                                ccount[citerations]++;
                            }
                        } else {
                            next_retry.push_back(node);
                        }

                        for(auto& it : locks_held) {
                            vLocks[it].unlock();
                        }
                        locks_held.clear();
                    }

                    retry.swap(next_retry);
                    if(retry.empty())
                        break;
                }

                #pragma omp barrier
                if(ccount[citerations]==0) {
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
    inline int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const
    {
        // Cannot delete if already deleted.
        if(_mesh->NNList[rm_vertex].empty())
            return -1;

        // For now, lock the halo
        if(_mesh->is_halo_node(rm_vertex))
            return -1;

        //
        bool delete_with_extreme_prejudice = false;
        if(delete_slivers && dim==3) {
            std::set<index_t>::const_iterator ee=_mesh->NEList[rm_vertex].begin();
            double q_linf = _mesh->quality[*ee];
            ++ee;

            for(; ee!=_mesh->NEList[rm_vertex].end(); ++ee)
                q_linf = std::min(q_linf, _mesh->quality[*ee]);

            if(q_linf<1.0e-6)
                delete_with_extreme_prejudice = true;
        }

        /* Sort the edges according to length. We want to collapse the
           shortest. If it is not possible to collapse the edge then move
           onto the next shortest.*/
        std::multimap<real_t, index_t> short_edges;
        for(const auto &nn : _mesh->NNList[rm_vertex]) {
            double length = _mesh->calc_edge_length(rm_vertex, nn);
            if(length<L_low || delete_with_extreme_prejudice)
                short_edges.insert(std::pair<real_t, index_t>(length, nn));
        }

        bool reject_collapse = false;
        index_t target_vertex=-1;
        while(short_edges.size()) {
            // Get the next shortest edge.
            target_vertex = short_edges.begin()->second;
            short_edges.erase(short_edges.begin());

            // Assume the best.
            reject_collapse=false;

            if(surface_coarsening) {
                std::set<index_t> compromised_boundary;
                for(const auto &element : _mesh->NEList[rm_vertex]) {
                    const int *n=_mesh->get_element(element);
                    for(size_t i=0; i<nloc; i++) {
                        if(n[i]!=rm_vertex) {
                            if(_mesh->boundary[element*nloc+i]>0) {
                                compromised_boundary.insert(_mesh->boundary[element*nloc+i]);
                            }
                        }
                    }
                }

                if(compromised_boundary.size()>1) {
                    reject_collapse=true;
                    continue;
                }

                if(compromised_boundary.size()==1) {
                    // Only allow this vertex to be collapsed to a vertex on the same boundary (not to an internal vertex).
                    std::set<index_t> target_boundary;
                    for(const auto &element : _mesh->NEList[target_vertex]) {
                        const int *n=_mesh->get_element(element);
                        for(size_t i=0; i<nloc; i++) {
                            if(n[i]!=target_vertex) {
                                if(_mesh->boundary[element*nloc+i]>0) {
                                    target_boundary.insert(_mesh->boundary[element*nloc+i]);
                                }
                            }
                        }
                    }

                    if(target_boundary.size()==1) {
                        if(*target_boundary.begin() != *compromised_boundary.begin()) {
                            reject_collapse=true;
                            continue;
                        }

                        std::set<index_t> deleted_elements;
                        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                                              _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                                              std::inserter(deleted_elements, deleted_elements.begin()));

                        if(dim==2) {
                            if(deleted_elements.size()!=1) {
                                reject_collapse=true;
                                continue;
                            }
                        } else {
                            /*
                               for(const auto& de : deleted_elements){
                               for(int i=0;i<nloc;i++){
                               if(_mesh->boundary[de*nloc+i]>0){
                               if(_mesh->_ENList[de*nloc+i]==rm_vertex){
                               reject_collapse=true;
                               break;
                               }
                               }
                               }
                               if(reject_collapse)
                               break;
                               }
                               if(reject_collapse)
                               continue;
                               */

                            bool confirm_boundary=false;
                            int scnt=0;
                            for(const auto& de : deleted_elements) {
                                // Need to confirm that this edges does in fact lie on the boundary - and is not actually an internal edges connected at both ends to a boundary.
                                const int *n = _mesh->get_element(de);
                                for(int i=0; i<nloc; i++) {
                                    if(n[i]!=target_vertex && n[i]!=rm_vertex) {
                                        if(_mesh->boundary[de*nloc+i]>0) {
                                            confirm_boundary = true;
                                        }
                                    }
                                    if(_mesh->boundary[de*nloc+i]>0)
                                        scnt++;
                                }
                            }
                            if(!confirm_boundary || scnt>2) {
                                reject_collapse=true;
                                continue;
                            }
                        }
                    } else {
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
            for(const auto &ee : _mesh->NEList[rm_vertex]) {
                const int *old_n=_mesh->get_element(ee);

                double q_linf = 0.0;
                if(quality_constrained)
                    q_linf = _mesh->quality[ee];

                long double old_av=0.0;
                if(!surface_coarsening) {
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
                for(size_t i=0; i<nloc; i++) {
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
                if(new_av<DBL_EPSILON) {
                    reject_collapse=true;
                    break;
                }

                total_new_av+=new_av;

                double new_q=0.0;
                if(quality_constrained) {
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

            if(!surface_coarsening) {
                // Check we are not removing surface features.
                if(std::abs(total_new_av-total_old_av)/std::max(total_new_av, total_old_av)>DBL_EPSILON) {
                    reject_collapse=true;
                    continue;
                }
            }

            if(!delete_with_extreme_prejudice) {
                // Check if any of the new edges are longer than L_max.
                for(const auto &nn : _mesh->NNList[rm_vertex]) {
                    if(target_vertex==nn)
                        continue;

                    if(_mesh->calc_edge_length(target_vertex, nn)>L_max) {
                        reject_collapse=true;
                        break;
                    }
                }
                if(reject_collapse)
                    continue;
            }

            if(quality_constrained) {
                if(!better) {
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
    inline void coarsen_kernel(index_t rm_vertex, index_t target_vertex)
    {
        std::set<index_t> deleted_elements;
        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                              _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                              std::inserter(deleted_elements, deleted_elements.begin()));

        // Clean NEList, update boundary and spike ENList.
        for(const auto &eid : deleted_elements) {
            const index_t *n = _mesh->get_element(eid);

            // Find falling facet.
            int facet[ndims], pos=0;
            int inherit_boundary_id=0;
            for (int i=0; i<nloc; i++) {
                if (n[i]!=target_vertex) {
                    facet[pos++] = n[i];
                }
                if (n[i]==rm_vertex) {
                    inherit_boundary_id = _mesh->boundary[eid*nloc+i];
                }
            }

            // Find associated element.
            std::set<index_t> associated_elements;
            std::set_intersection(_mesh->NEList[facet[0]].begin(), _mesh->NEList[facet[0]].end(),
                                  _mesh->NEList[facet[1]].begin(), _mesh->NEList[facet[1]].end(),
                                  std::inserter(associated_elements, associated_elements.begin()));
            if (ndims==3) {
                std::set<index_t> associated_elements3;
                std::set_intersection(_mesh->NEList[facet[2]].begin(), _mesh->NEList[facet[2]].end(),
                                      associated_elements.begin(), associated_elements.end(),
                                      std::inserter(associated_elements3, associated_elements3.begin()));
                associated_elements.swap(associated_elements3);
            }
            if (associated_elements.size()==2) {
                int associated_element = *associated_elements.begin();
                if (associated_element==eid) {
                    associated_element = *associated_elements.rbegin();
                }

                // Locate falling facet on this element.
                int ifacet=0;
                const index_t *m = _mesh->get_element(associated_element);
                for (; ifacet<nloc; ifacet++) {
                    if (m[ifacet]==facet[0])
                        continue;
                    if (m[ifacet]==facet[1])
                        continue;
                    if (ndims==3 && m[ifacet]==facet[2])
                        continue;
                    break;
                }

                // Finally...update boundary.
                _mesh->boundary[associated_element*nloc+ifacet] = inherit_boundary_id;
            }
            for(size_t i=0; i<nloc; ++i) {
                _mesh->NEList[n[i]].erase(eid);
            }

            // Remove element from mesh.
            _mesh->_ENList[eid*nloc] = -1;
        }

        // For all adjacent elements, replace rm_vertex with target_vertex in ENList and update quality.
        std::vector<index_t> new_edges;
        for(const auto& eid : _mesh->NEList[rm_vertex]) {
            assert(_mesh->_ENList[nloc*eid]!=-1);

            for(size_t i=0; i<nloc; i++) {
                if(_mesh->_ENList[nloc*eid+i]==rm_vertex) {
                    _mesh->_ENList[nloc*eid+i] = target_vertex;
                } else {
                    new_edges.push_back(_mesh->_ENList[nloc*eid+i]);
                }
            }

            _mesh->template update_quality<dim>(eid);

            // Add element to target_vertex's NEList.
            _mesh->NEList[target_vertex].insert(eid);
        }

        // Update surrounding NNList.
        for(const auto& nid : _mesh->NNList[rm_vertex]) {
            auto it = std::find(_mesh->NNList[nid].begin(), _mesh->NNList[nid].end(), rm_vertex);
            _mesh->NNList[nid].erase(it);
        }
        for(const auto &nid : new_edges) {
            if(std::find(_mesh->NNList[nid].begin(), _mesh->NNList[nid].end(), target_vertex)==_mesh->NNList[nid].end())
                _mesh->NNList[nid].push_back(target_vertex);

            if(std::find(_mesh->NNList[target_vertex].begin(), _mesh->NNList[target_vertex].end(), nid)==_mesh->NNList[target_vertex].end())
                _mesh->NNList[target_vertex].push_back(nid);
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
