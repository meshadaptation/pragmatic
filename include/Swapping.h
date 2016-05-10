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

#ifndef SWAPPING_H
#define SWAPPING_H

#include <algorithm>
#include <set>
#include <vector>

#include "Edge.h"
#include "ElementProperty.h"
#include "Lock.h"
#include "Mesh.h"

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
typedef boost::unordered_map< index_t, std::set<index_t> > propagation_map;
#else
#include <map>
typedef std::map< index_t, std::set<index_t> > propagation_map;
#endif

/*! \brief Performs edge/face swapping.
 *
 */
template<typename real_t, int dim> class Swapping
{
public:
    /// Default constructor.
    Swapping(Mesh<real_t> &mesh)
    {
        _mesh = &mesh;

        size_t NElements = _mesh->get_number_elements();

        // Set the orientation of elements.
        property = NULL;
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
    }

    /// Default destructor.
    ~Swapping()
    {
        if(property!=NULL)
            delete property;
    }

    void swap(real_t quality_tolerance)
    {
        size_t NNodes = _mesh->get_number_nodes();
        size_t NElements = _mesh->get_number_elements();

        min_Q = quality_tolerance;

        if(nnodes_reserve<NNodes) {
            nnodes_reserve = NNodes;

            marked_edges.resize(NNodes);

            vLocks.resize(NNodes);
        }

        #pragma omp parallel
        {
            // Vector "retry" is used to store aborted vertices.
            // Vector "round" is used to store propagated vertices.
            std::vector<index_t> retry, next_retry;
            std::vector<index_t> this_round, next_round;
            std::vector<index_t> locks_held;
            #pragma omp for schedule(guided) nowait
            for(index_t node=0; node<NNodes; ++node) {
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
                    std::set< Edge<index_t> > active_edges;
                    for(auto& ele : _mesh->NEList[node]) {
                        if(_mesh->quality[ele] < min_Q) {
                            const index_t* n = _mesh->get_element(ele);
                            for(int i=0; i<nloc; ++i) {
                                if(node < n[i])
                                    active_edges.insert(Edge<index_t>(node, n[i]));
                            }
                        }
                    }

                    for(auto& edge : active_edges) {
                        marked_edges[edge.edge.first].erase(edge.edge.second);
                        propagation_map pMap;
                        bool swapped = swap_kernel(edge, pMap);

                        if(swapped) {
                            for(auto& entry : pMap) {
                                for(auto& v : entry.second) {
                                    marked_edges[entry.first].insert(v);
                                    next_round.push_back(entry.first);
                                }
                            }
                        }
                    }
                } else
                    retry.push_back(node);

                for(auto& it : locks_held) {
                    vLocks[it].unlock();
                }
                locks_held.clear();
            }

            while(retry.size()>0) {
                next_retry.clear();

                for(auto& node : retry) {
                    if(marked_edges[node].empty())
                        continue;

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
                        std::set<index_t> marked_edges_copy = marked_edges[node];
                        marked_edges[node].clear();

                        for(auto& target : marked_edges_copy) {
                            Edge<index_t> edge(node, target);
                            propagation_map pMap;
                            marked_edges[edge.edge.first].erase(edge.edge.second);
                            bool swapped = swap_kernel(edge, pMap);

                            if(swapped) {
                                for(auto& entry : pMap) {
                                    for(auto& v : entry.second) {
                                        marked_edges[entry.first].insert(v);
                                        next_round.push_back(entry.first);
                                    }
                                }
                            }
                        }
                    } else
                        next_retry.push_back(node);

                    for(auto& it : locks_held) {
                        vLocks[it].unlock();
                    }
                    locks_held.clear();
                }

                retry.swap(next_retry);
            }

            while(!next_round.empty()) {
                this_round.swap(next_round);
                next_round.clear();

                for(auto& node : this_round) {
                    if(marked_edges[node].empty())
                        continue;

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
                        std::set< Edge<index_t> > active_edges;
                        for(auto& ele : _mesh->NEList[node]) {
                            if(_mesh->quality[ele] < min_Q) {
                                const index_t* n = _mesh->get_element(ele);
                                for(int i=0; i<nloc; ++i) {
                                    if(node < n[i])
                                        active_edges.insert(Edge<index_t>(node, n[i]));
                                }
                            }
                        }

                        for(auto& edge : active_edges) {
                            marked_edges[edge.edge.first].erase(edge.edge.second);
                            propagation_map pMap;
                            bool swapped = swap_kernel(edge, pMap);

                            if(swapped) {
                                for(auto& entry : pMap) {
                                    for(auto& v : entry.second) {
                                        marked_edges[entry.first].insert(v);
                                        next_round.push_back(entry.first);
                                    }
                                }
                            }
                        }
                    } else
                        retry.push_back(node);

                    for(auto& it : locks_held) {
                        vLocks[it].unlock();
                    }
                    locks_held.clear();
                }

                while(retry.size()>0) {
                    next_retry.clear();

                    for(auto& node : retry) {
                        if(marked_edges[node].empty())
                            continue;

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
                            std::set<index_t> marked_edges_copy = marked_edges[node];
                            marked_edges[node].clear();

                            for(auto& target : marked_edges_copy) {
                                Edge<index_t> edge(node, target);
                                propagation_map pMap;
                                marked_edges[edge.edge.first].erase(edge.edge.second);
                                bool swapped = swap_kernel(edge, pMap);

                                if(swapped) {
                                    for(auto& entry : pMap) {
                                        for(auto& v : entry.second) {
                                            marked_edges[entry.first].insert(v);
                                            next_round.push_back(entry.first);
                                        }
                                    }
                                }
                            }
                        } else
                            next_retry.push_back(node);

                        for(auto& it : locks_held) {
                            vLocks[it].unlock();
                        }
                        locks_held.clear();
                    }

                    retry.swap(next_retry);
                }

                if(next_round.empty()) {
                    // TODO: Try to steal work
                }
            }
        }
    }

private:

    inline bool swap_kernel(const Edge<index_t>& edge, propagation_map& pMap)
    {
        if(dim==2)
            return swap_kernel2d(edge, pMap);
        else
            return swap_kernel3d(edge, pMap);
    }

    /*
       void swap3d(real_t Q_min){
    // Process face-to-edge swap.
    for(int c=0;c<max_colour;c++){
    for(size_t i=0;i<graph.size();i++){
    int eid0 = renumber[i];

    if(colour[i]==c && (partialEEList.count(eid0)>0)){

    // Check this is not deleted.
    const int *n=_mesh->get_element(eid0);
    if(n[0]<0)
    continue;

    assert(partialEEList[eid0].size()==4);

    // Check adjacency is not toxic.
    bool toxic = false;
    for(int j=0;j<4;j++){
    int eid1 = partialEEList[eid0][j];
    if(eid1==-1)
    continue;

    const int *m=_mesh->get_element(eid1);
    if(m[0]<0){
    toxic = true;
    break;
    }
    }
    if(toxic)
    continue;

    // Create set of nodes for quick lookup.
    std::set<int> ele0_set;
    for(int j=0;j<4;j++)
    ele0_set.insert(n[j]);

    for(int j=0;j<4;j++){
    int eid1 = partialEEList[eid0][j];
    if(eid1==-1)
    continue;

    std::vector<int> hull(5, -1);
    if(j==0){
    hull[0] = n[1];
    hull[1] = n[3];
    hull[2] = n[2];
    hull[3] = n[0];
    }else if(j==1){
    hull[0] = n[2];
    hull[1] = n[3];
    hull[2] = n[0];
    hull[3] = n[1];
    }else if(j==2){
    hull[0] = n[0];
    hull[1] = n[3];
    hull[2] = n[1];
    hull[3] = n[2];
    }else if(j==3){
    hull[0] = n[0];
    hull[1] = n[1];
    hull[2] = n[2];
    hull[3] = n[3];
    }

    const int *m=_mesh->get_element(eid1);
    assert(m[0]>=0);

    for(int k=0;k<4;k++)
    if(ele0_set.count(m[k])==0){
    hull[4] = m[k];
    break;
    }
    assert(hull[4]!=-1);

    // New element: 0143
    real_t q0 = property->lipnikov(_mesh->get_coords(hull[0]),
    _mesh->get_coords(hull[1]),
    _mesh->get_coords(hull[4]),
    _mesh->get_coords(hull[3]),
    _mesh->get_metric(hull[0]),
    _mesh->get_metric(hull[1]),
    _mesh->get_metric(hull[4]),
    _mesh->get_metric(hull[3]));

    // New element: 1243
    real_t q1 = property->lipnikov(_mesh->get_coords(hull[1]),
    _mesh->get_coords(hull[2]),
    _mesh->get_coords(hull[4]),
    _mesh->get_coords(hull[3]),
    _mesh->get_metric(hull[1]),
    _mesh->get_metric(hull[2]),
    _mesh->get_metric(hull[4]),
    _mesh->get_metric(hull[3]));

    // New element:2043
    real_t q2 = property->lipnikov(_mesh->get_coords(hull[2]),
    _mesh->get_coords(hull[0]),
    _mesh->get_coords(hull[4]),
    _mesh->get_coords(hull[3]),
    _mesh->get_metric(hull[2]),
    _mesh->get_metric(hull[0]),
    _mesh->get_metric(hull[4]),
    _mesh->get_metric(hull[3]));

    if(std::min(quality[eid0],quality[eid1]) < std::min(q0, std::min(q1, q2))){
    // Cache boundary values
    int eid0_b0, eid0_b1, eid0_b2, eid1_b0, eid1_b1, eid1_b2;
    for(int face=0; face<nloc; ++face){
    if(n[face] == hull[0])
        eid0_b0 = _mesh->boundary[eid0*nloc+face];
    else if(n[face] == hull[1])
        eid0_b1 = _mesh->boundary[eid0*nloc+face];
    else if(n[face] == hull[2])
        eid0_b2 = _mesh->boundary[eid0*nloc+face];

    if(m[face] == hull[0])
        eid1_b0 = _mesh->boundary[eid1*nloc+face];
    else if(m[face] == hull[1])
        eid1_b1 = _mesh->boundary[eid1*nloc+face];
    else if(m[face] == hull[2])
        eid1_b2 = _mesh->boundary[eid1*nloc+face];
    }

    _mesh->erase_element(eid0);
    _mesh->erase_element(eid1);

    int e0[] = {hull[0], hull[1], hull[4], hull[3]};
    int b0[] = {0, 0, eid0_b2, eid1_b2};
    int eid0 = _mesh->append_element(e0, b0);
    quality.push_back(q0);

    int e1[] = {hull[1], hull[2], hull[4], hull[3]};
    int b1[] = {0, 0, eid0_b0, eid1_b0};
    int eid1 = _mesh->append_element(e1, b1);
    quality.push_back(q1);

    int e2[] = {hull[2], hull[0], hull[4], hull[3]};
    int b2[] = {0, 0, eid0_b1, eid1_b1};
    int eid2 = _mesh->append_element(e2, b2);
    quality.push_back(q2);

    _mesh->NNList[hull[3]].push_back(hull[4]);
    _mesh->NNList[hull[4]].push_back(hull[3]);
    _mesh->NEList[hull[0]].insert(eid0);
    _mesh->NEList[hull[0]].insert(eid2);
    _mesh->NEList[hull[1]].insert(eid0);
    _mesh->NEList[hull[1]].insert(eid1);
    _mesh->NEList[hull[2]].insert(eid1);
    _mesh->NEList[hull[2]].insert(eid2);
    _mesh->NEList[hull[3]].insert(eid0);
    _mesh->NEList[hull[3]].insert(eid1);
    _mesh->NEList[hull[3]].insert(eid2);
    _mesh->NEList[hull[4]].insert(eid0);
    _mesh->NEList[hull[4]].insert(eid1);
    _mesh->NEList[hull[4]].insert(eid2);

    break;
    }
    }
    }
    }
    }
    }
    */

    inline bool swap_kernel2d(const Edge<index_t>& edge, propagation_map& pMap)
    {
        index_t i = edge.edge.first;
        index_t j = edge.edge.second;

        if(_mesh->is_halo_node(i) && _mesh->is_halo_node(j))
            return false;

        // Find the two elements sharing this edge
        index_t intersection[2];
        {
            size_t loc = 0;
            std::set<index_t>::const_iterator it=_mesh->NEList[i].begin();
            while(loc<2 && it!=_mesh->NEList[i].end()) {
                if(_mesh->NEList[j].find(*it)!=_mesh->NEList[j].end()) {
                    intersection[loc++] = *it;
                }
                ++it;
            }

            // If this is a surface edge, it cannot be swapped.
            if(loc!=2)
                return false;
        }

        index_t eid0 = intersection[0];
        index_t eid1 = intersection[1];

        if(_mesh->quality[eid0] > min_Q && _mesh->quality[eid1] > min_Q)
            return false;

        const index_t *n = _mesh->get_element(eid0);
        int n_off=-1;
        for(size_t k=0; k<3; k++) {
            if((n[k]!=i) && (n[k]!=j)) {
                n_off = k;
                break;
            }
        }
        assert(n[n_off]>=0);

        const index_t *m = _mesh->get_element(eid1);
        int m_off=-1;
        for(size_t k=0; k<3; k++) {
            if((m[k]!=i) && (m[k]!=j)) {
                m_off = k;
                break;
            }
        }
        assert(m[m_off]>=0);

        assert(n[(n_off+2)%3]==m[(m_off+1)%3] && n[(n_off+1)%3]==m[(m_off+2)%3]);

        index_t k = n[n_off];
        index_t l = m[m_off];

        if(_mesh->is_halo_node(k)&& _mesh->is_halo_node(l))
            return false;

        int n_swap[] = {n[n_off], m[m_off],       n[(n_off+2)%3]}; // new eid0
        int m_swap[] = {n[n_off], n[(n_off+1)%3], m[m_off]};       // new eid1

        real_t q0 = property->lipnikov(_mesh->get_coords(n_swap[0]),
                                       _mesh->get_coords(n_swap[1]),
                                       _mesh->get_coords(n_swap[2]),
                                       _mesh->get_metric(n_swap[0]),
                                       _mesh->get_metric(n_swap[1]),
                                       _mesh->get_metric(n_swap[2]));
        real_t q1 = property->lipnikov(_mesh->get_coords(m_swap[0]),
                                       _mesh->get_coords(m_swap[1]),
                                       _mesh->get_coords(m_swap[2]),
                                       _mesh->get_metric(m_swap[0]),
                                       _mesh->get_metric(m_swap[1]),
                                       _mesh->get_metric(m_swap[2]));
        real_t worst_q = std::min(_mesh->quality[eid0], _mesh->quality[eid1]);
        real_t new_worst_q = std::min(q0, q1);

        if(new_worst_q>worst_q) {
            // Cache new quality measures.
            _mesh->quality[eid0] = q0;
            _mesh->quality[eid1] = q1;

            // Update NNList
            typename std::vector<index_t>::iterator it;
            it = std::find(_mesh->NNList[i].begin(), _mesh->NNList[i].end(), j);
            assert(it != _mesh->NNList[i].end());
            _mesh->NNList[i].erase(it);
            it = std::find(_mesh->NNList[j].begin(), _mesh->NNList[j].end(), i);
            assert(it != _mesh->NNList[j].end());
            _mesh->NNList[j].erase(it);
            _mesh->NNList[k].push_back(l);
            _mesh->NNList[l].push_back(k);

            // Update node-element list.
            _mesh->NEList[n_swap[2]].erase(eid1);
            _mesh->NEList[m_swap[1]].erase(eid0);
            _mesh->NEList[n_swap[0]].insert(eid1);
            _mesh->NEList[n_swap[1]].insert(eid0);

            // Update element-node and boundary list for this element.
            const int *bn = &_mesh->boundary[eid0*nloc];
            const int *bm = &_mesh->boundary[eid1*nloc];
            const int bn_swap[] = {bm[(m_off+2)%3], bn[(n_off+1)%3], 0}; // boundary for n_swap
            const int bm_swap[] = {bm[(m_off+1)%3], 0, bn[(n_off+2)%3]}; // boundary for m_swap

            for(size_t cnt=0; cnt<nloc; cnt++) {
                _mesh->_ENList[eid0*nloc+cnt] = n_swap[cnt];
                _mesh->_ENList[eid1*nloc+cnt] = m_swap[cnt];
                _mesh->boundary[eid0*nloc+cnt] = bn_swap[cnt];
                _mesh->boundary[eid1*nloc+cnt] = bm_swap[cnt];
            }

            pMap[std::min(i, k)].insert(std::max(i, k));
            pMap[std::min(i, l)].insert(std::max(i, l));
            pMap[std::min(j, k)].insert(std::max(j, k));
            pMap[std::min(j, l)].insert(std::max(j, l));

            return true;
        }

        return false;
    }

    inline bool swap_kernel3d(const Edge<index_t>& edge, propagation_map& pMap)
    {
        index_t nk = edge.edge.first;
        index_t nl = edge.edge.second;

        if(_mesh->is_halo_node(nk) && _mesh->is_halo_node(nl))
            return false;

        std::set<index_t> neigh_elements;
        set_intersection(_mesh->NEList[nk].begin(), _mesh->NEList[nk].end(),
                         _mesh->NEList[nl].begin(), _mesh->NEList[nl].end(),
                         inserter(neigh_elements, neigh_elements.begin()));

        bool abort = true;
        for(auto& e : neigh_elements) {
            if(_mesh->quality[e] < min_Q) {
                abort = false;
                break;
            }
        }

        if(abort)
            return false;

        double min_quality = 1.0;
        std::vector<index_t> constrained_edges_unsorted;
        std::map<int, std::map<index_t, int> > b;
        std::vector<int> element_order, e_to_eid;

        for(auto& it : neigh_elements) {
            min_quality = std::min(min_quality, _mesh->quality[it]);

            const int *m=_mesh->get_element(it);
            if(m[0]<0) {
                return false;
            }

            e_to_eid.push_back(it);

            for(int j=0; j<4; j++) {
                if((m[j]!=nk)&&(m[j]!=nl)) {
                    constrained_edges_unsorted.push_back(m[j]);
                } else if(m[j] == nk) {
                    b[it][nk] = _mesh->boundary[nloc*(it)+j];
                } else { // if(m[j] == nl)
                    b[it][nl] = _mesh->boundary[nloc*(it)+j];
                }
            }
        }

        size_t nelements = neigh_elements.size();
        assert(nelements*2==constrained_edges_unsorted.size());
        assert(b.size() == nelements);

        // Sort edges.
        std::vector<index_t> constrained_edges;
        std::vector<bool> sorted(nelements, false);
        constrained_edges.push_back(constrained_edges_unsorted[0]);
        constrained_edges.push_back(constrained_edges_unsorted[1]);
        element_order.push_back(e_to_eid[0]);
        for(size_t j=1; j<nelements; j++) {
            for(size_t e=1; e<nelements; e++) {
                if(sorted[e])
                    continue;
                if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2]) {
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    element_order.push_back(e_to_eid[e]);
                    sorted[e]=true;
                    break;
                } else if(*constrained_edges.rbegin()==constrained_edges_unsorted[e*2+1]) {
                    constrained_edges.push_back(constrained_edges_unsorted[e*2+1]);
                    constrained_edges.push_back(constrained_edges_unsorted[e*2]);
                    element_order.push_back(e_to_eid[e]);
                    sorted[e]=true;
                    break;
                }
            }
        }

        if(*constrained_edges.begin() != *constrained_edges.rbegin()) {
            return false;
        }
        // assert(element_order.size() == nelements);
        if(element_order.size() != nelements) {
            std::cerr<<"assert(element_order.size() == nelements) would fail "<<element_order.size()<<", "<<nelements<<std::endl;
            return false;
        }

        double orig_vol = 0.0;
        for(auto& ele : neigh_elements) {
            const index_t* n = _mesh->get_element(ele);
            orig_vol += property->volume(_mesh->get_coords(n[0]), _mesh->get_coords(n[1]),
                                         _mesh->get_coords(n[2]), _mesh->get_coords(n[3]));
        }

        std::vector< std::vector<index_t> > new_elements;
        std::vector< std::vector<int> > new_boundaries;
        if(nelements==3) {
            // This is the 3-element to 2-element swap.
            new_elements.resize(1);
            new_boundaries.resize(1);

            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(b[element_order[1]][nk]);
            new_boundaries[0].push_back(b[element_order[2]][nk]);
            new_boundaries[0].push_back(b[element_order[0]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(b[element_order[2]][nl]);
            new_boundaries[0].push_back(b[element_order[1]][nl]);
            new_boundaries[0].push_back(b[element_order[0]][nl]);
            new_boundaries[0].push_back(0);
        } else if(nelements==4) {
            // This is the 4-element to 4-element swap.
            new_elements.resize(2);
            new_boundaries.resize(2);

            // Option 1.
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[3]][nk]);
            new_boundaries[0].push_back(b[element_order[0]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(b[element_order[2]][nk]);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[1]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(b[element_order[3]][nl]);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[0]][nl]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[2]][nl]);
            new_boundaries[0].push_back(b[element_order[1]][nl]);
            new_boundaries[0].push_back(0);

            // Option 2
            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(nl);
            new_boundaries[1].push_back(b[element_order[1]][nk]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(b[element_order[0]][nk]);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(nl);
            new_boundaries[1].push_back(b[element_order[2]][nk]);
            new_boundaries[1].push_back(b[element_order[3]][nk]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(nk);
            new_boundaries[1].push_back(b[element_order[1]][nl]);
            new_boundaries[1].push_back(b[element_order[0]][nl]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(nk);
            new_boundaries[1].push_back(b[element_order[2]][nl]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(b[element_order[3]][nl]);
            new_boundaries[1].push_back(0);
        } else if(nelements==5) {
            // This is the 5-element to 6-element swap.
            new_elements.resize(5);
            new_boundaries.resize(5);

            // Option 1
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(b[element_order[1]][nk]);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[0]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[2]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(constrained_edges[8]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(nl);
            new_boundaries[0].push_back(b[element_order[4]][nk]);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[3]][nk]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[2]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[1]][nl]);
            new_boundaries[0].push_back(b[element_order[0]][nl]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(constrained_edges[4]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[2]][nl]);
            new_boundaries[0].push_back(0);

            new_elements[0].push_back(constrained_edges[8]);
            new_elements[0].push_back(constrained_edges[6]);
            new_elements[0].push_back(constrained_edges[0]);
            new_elements[0].push_back(nk);
            new_boundaries[0].push_back(0);
            new_boundaries[0].push_back(b[element_order[4]][nl]);
            new_boundaries[0].push_back(b[element_order[3]][nl]);
            new_boundaries[0].push_back(0);

            // Option 2
            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[8]);
            new_elements[1].push_back(nl);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(b[element_order[4]][nk]);
            new_boundaries[1].push_back(b[element_order[0]][nk]);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(constrained_edges[8]);
            new_elements[1].push_back(nl);
            new_boundaries[1].push_back(b[element_order[3]][nk]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(nl);
            new_boundaries[1].push_back(b[element_order[2]][nk]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(b[element_order[1]][nk]);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[0]);
            new_elements[1].push_back(constrained_edges[8]);
            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(nk);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(b[element_order[0]][nl]);
            new_boundaries[1].push_back(b[element_order[4]][nl]);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[8]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(nk);
            new_boundaries[1].push_back(b[element_order[3]][nl]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);

            new_elements[1].push_back(constrained_edges[2]);
            new_elements[1].push_back(constrained_edges[6]);
            new_elements[1].push_back(constrained_edges[4]);
            new_elements[1].push_back(nk);
            new_boundaries[1].push_back(b[element_order[2]][nl]);
            new_boundaries[1].push_back(b[element_order[1]][nl]);
            new_boundaries[1].push_back(0);
            new_boundaries[1].push_back(0);

            // Option 3
            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[0]);
            new_elements[2].push_back(constrained_edges[2]);
            new_elements[2].push_back(nl);
            new_boundaries[2].push_back(b[element_order[0]][nk]);
            new_boundaries[2].push_back(b[element_order[1]][nk]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);

            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[8]);
            new_elements[2].push_back(constrained_edges[0]);
            new_elements[2].push_back(nl);
            new_boundaries[2].push_back(b[element_order[4]][nk]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);

            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[6]);
            new_elements[2].push_back(constrained_edges[8]);
            new_elements[2].push_back(nl);
            new_boundaries[2].push_back(b[element_order[3]][nk]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(b[element_order[2]][nk]);
            new_boundaries[2].push_back(0);

            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[2]);
            new_elements[2].push_back(constrained_edges[0]);
            new_elements[2].push_back(nk);
            new_boundaries[2].push_back(b[element_order[0]][nl]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(b[element_order[1]][nl]);
            new_boundaries[2].push_back(0);

            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[0]);
            new_elements[2].push_back(constrained_edges[8]);
            new_elements[2].push_back(nk);
            new_boundaries[2].push_back(b[element_order[4]][nl]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);

            new_elements[2].push_back(constrained_edges[4]);
            new_elements[2].push_back(constrained_edges[8]);
            new_elements[2].push_back(constrained_edges[6]);
            new_elements[2].push_back(nk);
            new_boundaries[2].push_back(b[element_order[3]][nl]);
            new_boundaries[2].push_back(b[element_order[2]][nl]);
            new_boundaries[2].push_back(0);
            new_boundaries[2].push_back(0);

            // Option 4
            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[2]);
            new_elements[3].push_back(constrained_edges[4]);
            new_elements[3].push_back(nl);
            new_boundaries[3].push_back(b[element_order[1]][nk]);
            new_boundaries[3].push_back(b[element_order[2]][nk]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);

            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[0]);
            new_elements[3].push_back(constrained_edges[2]);
            new_elements[3].push_back(nl);
            new_boundaries[3].push_back(b[element_order[0]][nk]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);

            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[8]);
            new_elements[3].push_back(constrained_edges[0]);
            new_elements[3].push_back(nl);
            new_boundaries[3].push_back(b[element_order[4]][nk]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(b[element_order[3]][nk]);
            new_boundaries[3].push_back(0);

            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[4]);
            new_elements[3].push_back(constrained_edges[2]);
            new_elements[3].push_back(nk);
            new_boundaries[3].push_back(b[element_order[1]][nl]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(b[element_order[2]][nl]);
            new_boundaries[3].push_back(0);

            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[2]);
            new_elements[3].push_back(constrained_edges[0]);
            new_elements[3].push_back(nk);
            new_boundaries[3].push_back(b[element_order[0]][nl]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);

            new_elements[3].push_back(constrained_edges[6]);
            new_elements[3].push_back(constrained_edges[0]);
            new_elements[3].push_back(constrained_edges[8]);
            new_elements[3].push_back(nk);
            new_boundaries[3].push_back(b[element_order[4]][nl]);
            new_boundaries[3].push_back(b[element_order[3]][nl]);
            new_boundaries[3].push_back(0);
            new_boundaries[3].push_back(0);

            // Option 5
            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[0]);
            new_elements[4].push_back(constrained_edges[2]);
            new_elements[4].push_back(nl);
            new_boundaries[4].push_back(b[element_order[0]][nk]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(b[element_order[4]][nk]);
            new_boundaries[4].push_back(0);

            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[2]);
            new_elements[4].push_back(constrained_edges[4]);
            new_elements[4].push_back(nl);
            new_boundaries[4].push_back(b[element_order[1]][nk]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);

            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[4]);
            new_elements[4].push_back(constrained_edges[6]);
            new_elements[4].push_back(nl);
            new_boundaries[4].push_back(b[element_order[2]][nk]);
            new_boundaries[4].push_back(b[element_order[3]][nk]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);

            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[2]);
            new_elements[4].push_back(constrained_edges[0]);
            new_elements[4].push_back(nk);
            new_boundaries[4].push_back(b[element_order[0]][nl]);
            new_boundaries[4].push_back(b[element_order[4]][nl]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);

            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[4]);
            new_elements[4].push_back(constrained_edges[2]);
            new_elements[4].push_back(nk);
            new_boundaries[4].push_back(b[element_order[1]][nl]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(0);

            new_elements[4].push_back(constrained_edges[8]);
            new_elements[4].push_back(constrained_edges[6]);
            new_elements[4].push_back(constrained_edges[4]);
            new_elements[4].push_back(nk);
            new_boundaries[4].push_back(b[element_order[2]][nl]);
            new_boundaries[4].push_back(0);
            new_boundaries[4].push_back(b[element_order[3]][nl]);
            new_boundaries[4].push_back(0);
        } else {
            return false;
        }
        nelements = new_elements[0].size()/4;

        // Check new minimum quality.
        std::vector<double> new_min_quality(new_elements.size());
        std::vector< std::vector<double> > newq(new_elements.size());

        for(size_t option=0; option<new_elements.size(); option++) {
            newq[option].resize(nelements);
            double new_vol = 0.0;
            for(size_t j=0; j<nelements; j++) {
                const index_t* n = &new_elements[option][j*4];
                double vol = property->volume(_mesh->get_coords(n[0]), _mesh->get_coords(n[1]),
                                              _mesh->get_coords(n[2]), _mesh->get_coords(n[3]));

                if(vol<0) {
                    vol*=-1;
                    std::swap(new_elements[option][j*4], new_elements[option][j*4+1]);
                    std::swap(new_boundaries[option][j*4], new_boundaries[option][j*4+1]);
                }

                new_vol += vol;

                newq[option][j] = property->lipnikov(_mesh->get_coords(n[0]),
                                                     _mesh->get_coords(n[1]),
                                                     _mesh->get_coords(n[2]),
                                                     _mesh->get_coords(n[3]),
                                                     _mesh->get_metric(n[0]),
                                                     _mesh->get_metric(n[1]),
                                                     _mesh->get_metric(n[2]),
                                                     _mesh->get_metric(n[3]));
            }

            if(fabs(new_vol - orig_vol) > DBL_EPSILON) {
                new_min_quality[option] = -1;
            } else {
                new_min_quality[option] = newq[option][0];
                for(size_t j=0; j<nelements; j++)
                    new_min_quality[option] = std::min(newq[option][j], new_min_quality[option]);
            }
        }

        int best_option=0;
        for(size_t option=1; option<new_elements.size(); option++) {
            if(new_min_quality[option]>new_min_quality[best_option]) {
                best_option = option;
            }
        }

        if(new_min_quality[best_option] <= min_quality)
            return false;

        // Update NNList
        std::vector<index_t>::iterator vit = std::find(_mesh->NNList[nk].begin(), _mesh->NNList[nk].end(), nl);
        assert(vit != _mesh->NNList[nk].end());
        _mesh->NNList[nk].erase(vit);
        vit = std::find(_mesh->NNList[nl].begin(), _mesh->NNList[nl].end(), nk);
        assert(vit != _mesh->NNList[nl].end());
        _mesh->NNList[nl].erase(vit);

        // Remove old elements.
        for(auto& it : neigh_elements)
            _mesh->erase_element(it);

        // Add new elements and mark edges for propagation.
        // First, recycle element IDs.
        std::deque<index_t> new_eids;
        for(auto& ele : neigh_elements)
            new_eids.push_back(ele);

        // Next, find how many new elements we have to allocate
        int extra_elements = nelements - neigh_elements.size();
        if(extra_elements > 0) {
            index_t new_eid;
            #pragma omp atomic capture
            {
                new_eid = _mesh->NElements;
                _mesh->NElements += extra_elements;
            }

            if(_mesh->_ENList.size() < (new_eid+extra_elements)*nloc) {
                ENList_lock.lock();
                if(_mesh->_ENList.size() < (new_eid+extra_elements)*nloc) {
                    _mesh->_ENList.resize(2*(new_eid+extra_elements)*nloc);
                    _mesh->boundary.resize(2*(new_eid+extra_elements)*nloc);
                    _mesh->quality.resize(2*(new_eid+extra_elements)*nloc);
                }
                ENList_lock.unlock();
            }

            for(int i=0; i<extra_elements; ++i)
                new_eids.push_back(new_eid++);
        }

        for(size_t j=0; j<nelements; j++) {
            index_t eid = new_eids[0];
            new_eids.pop_front();
            for(size_t i=0; i<nloc; i++) {
                _mesh->_ENList[eid*nloc+i]=new_elements[best_option][j*4+i];
                _mesh->boundary[eid*nloc+i]=new_boundaries[best_option][j*4+i];
            }
            _mesh->quality[eid]=newq[best_option][j];

            for(int p=0; p<nloc; ++p) {
                index_t v1 = new_elements[best_option][j*4+p];
                _mesh->NEList[v1].insert(eid);

                for(int q=p+1; q<nloc; ++q) {
                    index_t v2 = new_elements[best_option][j*4+q];
                    std::vector<index_t>::iterator vit = std::find(_mesh->NNList[v1].begin(), _mesh->NNList[v1].end(), v2);
                    if(vit == _mesh->NNList[v1].end()) {
                        _mesh->NNList[v1].push_back(v2);
                        _mesh->NNList[v2].push_back(v1);
                    }

                    pMap[std::min(v1,v2)].insert(std::max(v1,v2));
                }
            }
        }

        return true;
    }

    Mesh<real_t> *_mesh;
    ElementProperty<real_t> *property;

    size_t nnodes_reserve;
    Lock ENList_lock;
    std::vector<Lock> vLocks;

    static const size_t ndims=dim;
    static const size_t nloc=dim+1;
    static const size_t msize=(dim==2?3:6);

    std::vector< std::set<index_t> > marked_edges;
    real_t min_Q;
};

#endif
