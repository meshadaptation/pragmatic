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
#include "Mesh.h"
#include "GMFTools.h"

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
        internal_surface_coarsening = false;
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
    int coarsen(real_t L_low, real_t L_max,
                 bool enable_surface_coarsening=false,
                 bool enable_int_surface_coarsening=false,
                 bool enable_delete_slivers=false,
                 bool enable_quality_constrained=false)
    {

        surface_coarsening = enable_surface_coarsening;
        internal_surface_coarsening = enable_int_surface_coarsening;
        delete_slivers = enable_delete_slivers;
        quality_constrained = enable_quality_constrained;

        size_t NNodes = _mesh->get_number_nodes();

        _L_low = L_low;
        _L_max = L_max;

        if(nnodes_reserve<NNodes) {
            nnodes_reserve = NNodes;
        }

        int ccount_ite, ccount_tot = 0;
        for(int citerations=0; citerations<100; citerations++) {
//            printf("DEBUG  ========== citerations: %d\n", citerations);
//            if (citerations==1) {
//                int ver[3] = {340, 2283, 1434};
//                Mesh<real_t> *littlemesh = new Mesh<real_t>(*_mesh, 3, ver);
//                printf("DEBUG  POUEEEET\n");
//                GMFTools<double>::export_gmf_mesh("../data/test_ballincube-littlemesh", littlemesh);
//            }
//
//            int * bdrytags = _mesh->get_boundaryTags();
//            int iElm = 129689;
//            const int * elm = _mesh->get_element(iElm);
//            int * bdry = &bdrytags[iElm*4];
//            const double *x0 = _mesh->get_coords(elm[0]);
//            const double *x1 = _mesh->get_coords(elm[1]);
//            const double *x2 = _mesh->get_coords(elm[2]);
//            const double *x3 = _mesh->get_coords(elm[3]);
//            printf("DEBUG  tet[%d]: %d (%1.3f %1.3f %1.3f)\n", iElm, elm[0], x0[0], x0[1], x0[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm[1], x1[0], x1[1], x1[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm[2], x2[0], x2[1], x2[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm[3], x3[0], x3[1], x3[2]);
//            printf("DEBUG  boundary: (%d %d %d %d)\n", bdry[0], bdry[1], bdry[2], bdry[3]);
//            int iElm2 = 248565;
//            const int * elm2 = _mesh->get_element(iElm2);
//            int * bdry2 = &bdrytags[iElm2*4];
//            const double *x02 = _mesh->get_coords(elm2[0]);
//            const double *x12 = _mesh->get_coords(elm2[1]);
//            const double *x22 = _mesh->get_coords(elm2[2]);
//            const double *x32 = _mesh->get_coords(elm2[3]);
//            printf("DEBUG  tet[%d]: %d (%1.3f %1.3f %1.3f)\n", iElm2, elm2[0], x02[0], x02[1], x02[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm2[1], x12[0], x12[1], x12[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm2[2], x22[0], x22[1], x22[2]);
//            printf("DEBUG               %d (%1.3f %1.3f %1.3f)\n", elm2[3], x32[0], x32[1], x32[2]);
//            printf("DEBUG  boundary: (%d %d %d %d)\n", bdry2[0], bdry2[1], bdry2[2], bdry2[3]);

            ccount_ite = 0;
            for(index_t node=0; node<NNodes; ++node) { // TODO Need to consider randomising order to avoid mesh artifacts related to numbering.
                index_t target = coarsen_identify_kernel(node, L_low, L_max);
                if(target>=0) {
                    coarsen_kernel(node, target);
                    ccount_ite++;
                    ccount_tot++;
                }
            }
//            x0 = _mesh->get_coords(elm[0]);
//            x1 = _mesh->get_coords(elm[1]);
//            x2 = _mesh->get_coords(elm[2]);
//            x3 = _mesh->get_coords(elm[3]);
//            printf("DEBUG  tet[%d]: %d (%1.5f %1.5f %1.5f)\n", iElm, elm[0], x0[0], x0[1], x0[2]);
//            printf("DEBUG               %d (%1.5f %1.5f %1.5f)\n", elm[1], x1[0], x1[1], x1[2]);
//            printf("DEBUG               %d (%1.5f %1.5f %1.5f)\n", elm[2], x2[0], x2[1], x2[2]);
//            printf("DEBUG               %d (%1.5f %1.5f %1.5f)\n", elm[3], x3[0], x3[1], x3[2]);
//            printf("DEBUG  boundary: (%d %d %d %d)\n", bdry[0], bdry[1], bdry[2], bdry[3]);
//            _mesh->check_internal_boundaries_3d();
            if(ccount_ite==0) {
                break;
            }
        }
        printf("DEBUG   Number of edge collapse %d\n", ccount_tot);
        return ccount_tot;
    }

private:


    /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
     * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
     * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
     */
    inline int coarsen_identify_kernel(index_t rm_vertex, real_t L_low, real_t L_max) const
    {

        // Cannot delete if already deleted.
        if(_mesh->NNList[rm_vertex].empty()) {
            return -1;
        }

        // For now, lock the halo
        if(_mesh->is_halo_node(rm_vertex))
            return -1;

        std::set<int> regions;
        std::set<index_t>::const_iterator ee = _mesh->NEList[rm_vertex].begin();
        double q_linf = _mesh->quality[*ee];
        ee++;
        for (; ee != _mesh->NEList[rm_vertex].end(); ++ee) {
            regions.insert(_mesh->get_elementTag(*ee));
        }

        bool reject_collapse = false;
        index_t target_vertex=-1;
        std::set<index_t>::const_iterator region_it;
        for (region_it = regions.begin(); region_it != regions.end(); ++region_it) {
            int region = *region_it;
            //
            bool delete_with_extreme_prejudice = false;
            if(delete_slivers && dim==3) {
                std::set<index_t>::const_iterator ee=_mesh->NEList[rm_vertex].begin();
                double q_linf = _mesh->quality[*ee];
                ++ee;
    
                for(; ee!=_mesh->NEList[rm_vertex].end(); ++ee) {
                    if (_mesh->get_elementTag(*ee) != region)
                        continue;
                    q_linf = std::min(q_linf, _mesh->quality[*ee]);
                }
    
                if(q_linf<1.0e-6)
                    delete_with_extreme_prejudice = true;
            }

            /* Sort the edges according to length. We want to collapse the
               shortest. If it is not possible to collapse the edge then move
               onto the next shortest.*/
            std::multimap<real_t, index_t> short_edges;
            std::set<index_t>::const_iterator ee;
            for(ee = _mesh->NEList[rm_vertex].begin(); ee != _mesh->NEList[rm_vertex].end(); ++ee) {
                if (_mesh->get_elementTag(*ee) != region)
                    continue;
                const int * n = _mesh->get_element(*ee);
                for (int i = 0; i<nloc; ++i) {
                    if (n[i] != rm_vertex) {
                        double length = _mesh->calc_edge_length(rm_vertex, n[i]);
                        if(length<L_low || delete_with_extreme_prejudice)
                            short_edges.insert(std::pair<real_t, index_t>(length, n[i]));
                    }
                }

            }

            while(short_edges.size()) {
                // Get the next shortest edge.
                target_vertex = short_edges.begin()->second;
                short_edges.erase(short_edges.begin());
    
                // Assume the best.
                reject_collapse=false;
    
                int boundary_id = -10;
                if(surface_coarsening || internal_surface_coarsening) {
                    std::set<index_t> compromised_boundary;
                    for(const auto &element : _mesh->NEList[rm_vertex]) {
                        if (_mesh->get_elementTag(element) != region)
                            continue;
                        const int *n=_mesh->get_element(element);
                        for(size_t i=0; i<nloc; i++) {
                            if(n[i]!=rm_vertex) {
                                if(_mesh->boundary[element*nloc+i]>0) {
                                    compromised_boundary.insert(_mesh->boundary[element*nloc+i]);
                                    boundary_id = _mesh->boundary[element*nloc+i];
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
                            if (_mesh->get_elementTag(element) != region)
                                continue;
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

                            std::set<index_t>::const_iterator el_it;
                            for (el_it = deleted_elements.begin(); el_it != deleted_elements.end(); ){
                                if (_mesh->get_elementTag(*el_it) != region)
                                    el_it = deleted_elements.erase(el_it);
                                else
                                    el_it++;
                            }

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
                    
                    if (_mesh->get_elementTag(ee) == region) {

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

                    if (_mesh->get_elementTag(ee) == region) {
                        total_new_av+=new_av;
                    }

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

                if ((!surface_coarsening && !_mesh->is_internal_boundary(boundary_id)) || 
                    (!internal_surface_coarsening && _mesh->is_internal_boundary(boundary_id))) {
                    // Check we are not removing surface features.
                    double av_var = std::abs(total_new_av-total_old_av);
                    av_var /= std::max(total_new_av, total_old_av);
                    if (av_var > std::max(_mesh->get_ref_length(), 1.)*DBL_EPSILON) {
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

            // If this edge is ok to collapse then break out of loop.
            if(!reject_collapse)
                break;

        }

        // If we've checked all edges and none are collapsible then return.
        if(reject_collapse)
            return -2;

        return target_vertex;
    }









    /*! Kernel for identifying what vertex (if any) rm_vertex should collapse onto.
     * See Figure 15; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
     * Returns the node ID that rm_vertex should collapse onto, negative if no operation is to be performed.
     */
    inline int coarsen_identify_kernel_old(index_t rm_vertex, real_t L_low, real_t L_max) const
    {

        if (rm_vertex == 435) printf("DEBUG  rm_vertex: %d\n", rm_vertex);
        // Cannot delete if already deleted.
        if(_mesh->NNList[rm_vertex].empty()) {
            if (rm_vertex == 435)printf("DEBUG  already deleted\n");
            return -1;
        }

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

            if(q_linf<1.0e-6) {
                if (rm_vertex == 435) printf("DEBUG  exterminate\n");
                delete_with_extreme_prejudice = true;
            }
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

            int test = false;
            // Get the next shortest edge.
            target_vertex = short_edges.begin()->second;
            if (rm_vertex == 435) printf("DEBUG  target_vertex: %d\n", target_vertex);
            short_edges.erase(short_edges.begin());
            if (rm_vertex == 1126 && target_vertex == 2335) test = true;
            if (test) printf("DEBUG  *** POUET\n");

            // Assume the best.
            reject_collapse=false;


            int region = -10;
            int num_regions = 0;  
            // TODO  here we should consider intersection with target neighbour elements ?
            for(const auto &ee : _mesh->NEList[rm_vertex]) {
                const int *old_n=_mesh->get_element(ee);
                if (region != _mesh->regions[ee])
                    num_regions++;
                region = _mesh->regions[ee];
            }

            if (test) printf("DEBUG  *** num regions: %d\n",num_regions);

            if ((num_regions == 1 && surface_coarsening) ||
                (num_regions > 1 && internal_surface_coarsening)) {

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

                if (test) printf("DEBUG  *** compromised_boundary size: %d\n", compromised_boundary.size());

                if(compromised_boundary.size()>1) {
                    if (rm_vertex == 435) printf("DEBUG   POUET 0\n");
                    reject_collapse=true;
                    continue;
                }

                if(compromised_boundary.size()==1) {
                    // Only allow this vertex to be collapsed to a vertex on the same boundary (not to an internal vertex).
                    std::set<index_t> target_boundary;
                    #if 0                    
                    for(const auto &element : _mesh->NEList[target_vertex]) {
                        const int *n=_mesh->get_element(element);
                        for(size_t i=0; i<nloc; i++) {
                            if(n[i]!=target_vertex) {
                                if(_mesh->boundary[element*nloc+i]) {
                                    target_boundary.insert(_mesh->boundary[element*nloc+i]);
                                }
                            }
                        }
                    }
                    #else
                    
                    #endif
                    if (test) printf("DEBUG  *** target_boundary size: %d\n", target_boundary.size());

                    if(target_boundary.size()==1) {
                        if(*target_boundary.begin() != *compromised_boundary.begin()) {
                            if (rm_vertex == 435) printf("DEBUG   POUET 1\n");
                            reject_collapse=true;
                            continue;
                        }

                        std::set<index_t> deleted_elements;
                        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                                              _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                                              std::inserter(deleted_elements, deleted_elements.begin()));

                        if(dim==2) {
                            if(deleted_elements.size()!=1 && !_mesh->is_internal_boundary(*target_boundary.begin()) ) {
                                if (rm_vertex == 435) printf("DEBUG   POUET 2\n");
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
                            int boundary_tag = -10;
                            int scnt=0;
                            // TODO probably check if internal boundary
                            for(const auto& de : deleted_elements) {
                                // Need to confirm that this edges does in fact lie on the boundary - and is not actually an internal edges connected at both ends to a boundary.
                                const int *n = _mesh->get_element(de);
                                for(int i=0; i<nloc; i++) {
                                    if(n[i]!=target_vertex && n[i]!=rm_vertex) {
                                        if(_mesh->boundary[de*nloc+i]>0) {
                                            confirm_boundary = true;
                                            boundary_tag = _mesh->boundary[de*nloc+i];
                                        }
                                    }
                                    if(_mesh->boundary[de*nloc+i]>0)
                                        scnt++;
                                }
                            }
                            if(!confirm_boundary || (scnt>2 && !_mesh->is_internal_boundary(boundary_tag))) {
                                if (rm_vertex == 435) printf("DEBUG   POUET 3\n");
                                reject_collapse=true;
                                continue;
                            }
                        }
                    } else {
                        if (rm_vertex == 435) printf("DEBUG   POUET 4\n");
                        reject_collapse=true;
                        continue;
                    }
                }
            }

            /* Check the properties of new elements. If the
               new properties are not acceptable then continue. */

            std::map<int, long double> total_old_avs;
            std::map<int, long double> total_new_avs;

            bool better=true;
            for(const auto &ee : _mesh->NEList[rm_vertex]) {
                const int *old_n=_mesh->get_element(ee);
                int region = _mesh->regions[ee];

                double q_linf = 0.0;
                if(quality_constrained)
                    q_linf = _mesh->quality[ee];

                long double old_av=0.0;
                if ((num_regions == 1 && !surface_coarsening) ||
                    (num_regions > 1 && !internal_surface_coarsening)) {

                    if(dim==2)
                        old_av = property->area_precision(_mesh->get_coords(old_n[0]),
                                                          _mesh->get_coords(old_n[1]),
                                                          _mesh->get_coords(old_n[2]));
                    else
                        old_av = property->volume_precision(_mesh->get_coords(old_n[0]),
                                                            _mesh->get_coords(old_n[1]),
                                                            _mesh->get_coords(old_n[2]),
                                                            _mesh->get_coords(old_n[3]));

                    total_old_avs[region] += old_av; // TODO check
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
                    if (rm_vertex == 435) printf("DEBUG   POUET 5\n");
                    reject_collapse=true;
                    break;
                }

                total_new_avs[region] += new_av; // TODO check


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

            if ((num_regions == 1 && !surface_coarsening) ||
                (num_regions > 1 && !internal_surface_coarsening)) {

                // Check we are not removing surface features.
                std::map<int, long double>::const_iterator iReg;
                for (iReg = total_old_avs.begin(); iReg != total_old_avs.end(); ++iReg) {
                    int region = iReg->first;
                    double av_var = std::abs(total_new_avs[region]-total_old_avs[region]);
                    av_var /= std::max(total_new_avs[region], total_old_avs[region]);
                    if (av_var > std::max(_mesh->get_ref_length(), 1.)*DBL_EPSILON) {
                        if (rm_vertex == 435) printf("DEBUG   POUET 6\n");
                        reject_collapse=true;
                        break;
                    }
                }
            }

            if(!delete_with_extreme_prejudice) {
                // Check if any of the new edges are longer than L_max.
                for(const auto &nn : _mesh->NNList[rm_vertex]) {
                    if(target_vertex==nn)
                        continue;

                    if(_mesh->calc_edge_length(target_vertex, nn)>L_max) {
                        if (rm_vertex == 435) printf("DEBUG   POUET 7\n");
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

        if (rm_vertex == 435) printf("DEBUG  reject_collapse: %d\n", reject_collapse);

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
        bool test = false;
        const double * rm_crd = _mesh->get_coords(rm_vertex);
//        if ((target_vertex == 340 || target_vertex == 1434) && 
//            (fabs(rm_crd[0]-0.358)<0.002 && fabs(rm_crd[1]+1.975)<0.002 && fabs(rm_crd[2]+2.229)<0.002) ) {
//            test = true;
//            printf("DEBUG   HERE, rm: %d ---> target: %d\n", rm_vertex, target_vertex);
//        }        

        // TODO As we don't coarsen accross internal boudaries and don't create 
        //  new elements, no need to worry about element tags

        std::set<index_t> deleted_elements;
        std::set_intersection(_mesh->NEList[rm_vertex].begin(), _mesh->NEList[rm_vertex].end(),
                              _mesh->NEList[target_vertex].begin(), _mesh->NEList[target_vertex].end(),
                              std::inserter(deleted_elements, deleted_elements.begin()));

        // Clean NEList, update boundary and spike ENList.
        for(const auto &eid : deleted_elements) {
            const index_t *n = _mesh->get_element(eid);
            //if (test) printf("DEBUG  \n");
            if (test) printf("DEBUG  eid: %d (%d %d %d %d)\n", eid, n[0], n[1], n[2], n[3]);

            // Find falling facet.
            int falling_facet[ndims], pos=0;
            int inherit_boundary_id=0;
            int target_boundary_id=0;
            for (int i=0; i<nloc; i++) {
                if (n[i]!=target_vertex) {
                    falling_facet[pos++] = n[i];
                }
                if (n[i]==rm_vertex) {
                    inherit_boundary_id = _mesh->boundary[eid*nloc+i];
                }
                if (n[i]==target_vertex) {
                    target_boundary_id = _mesh->boundary[eid*nloc+i];
                }
            }

            if (rm_vertex == 1126 && target_vertex == 2335)
                printf("DEBUG  eid: %d falling facet: %d %d %d\n", eid, falling_facet[0], falling_facet[1], falling_facet[2]);

            // Find associated element.
            std::set<index_t> associated_elements;
            std::set_intersection(_mesh->NEList[falling_facet[0]].begin(), _mesh->NEList[falling_facet[0]].end(),
                                  _mesh->NEList[falling_facet[1]].begin(), _mesh->NEList[falling_facet[1]].end(),
                                  std::inserter(associated_elements, associated_elements.begin()));
            if (ndims==3) {
                std::set<index_t> associated_elements3;
                std::set_intersection(_mesh->NEList[falling_facet[2]].begin(), _mesh->NEList[falling_facet[2]].end(),
                                      associated_elements.begin(), associated_elements.end(),
                                      std::inserter(associated_elements3, associated_elements3.begin()));
                associated_elements.swap(associated_elements3);
            }

            if (test) printf("DEBUG    falling facet: %d %d %d, boundary id: %d, assoc. elm size: %d\n",
                falling_facet[0], falling_facet[1], falling_facet[2], inherit_boundary_id, 
                associated_elements.size());


            if (associated_elements.size()==1 || (associated_elements.size()==2 && _mesh->is_internal_boundary(target_boundary_id))) {
                // my falling facet is a boundary facet, 
                // in which case the facet onto which it is falling should inherit the falling facet tag 
                // if it is an internal facet
                int falled_faced[ndims]; // facet opposite to rm_vertex, ie the one onto which the falling facet falls
                int pos = 0;
                for (int i=0; i<nloc; i++) {
                    if (n[i]!=rm_vertex) {
                        falled_faced[pos++] = n[i];
                    }
                }
                inherit_boundary_id = target_boundary_id;
                // If rm_facet has only one neighbour: it is a boundary facet, it seems to be covered by code above
                // If rm_facet should has 2 neighbors, one of which is the deleted tet, 
                //    another one in which I must find the facet and update it boundary id
                // Find associated elements.
                std::set<index_t> associated_elements_falled_facet;
                std::set_intersection(_mesh->NEList[falled_faced[0]].begin(), _mesh->NEList[falled_faced[0]].end(),
                                      _mesh->NEList[falled_faced[1]].begin(), _mesh->NEList[falled_faced[1]].end(),
                                      std::inserter(associated_elements_falled_facet, 
                                      associated_elements_falled_facet.begin()));
                if (ndims==3) {
                    std::set<index_t> associated_elements3_falled_facet;
                    std::set_intersection(_mesh->NEList[falled_faced[2]].begin(), _mesh->NEList[falled_faced[2]].end(),
                                          associated_elements_falled_facet.begin(), associated_elements_falled_facet.end(),
                                          std::inserter(associated_elements3_falled_facet, associated_elements3_falled_facet.begin()));
                    associated_elements_falled_facet.swap(associated_elements3_falled_facet);
                }
                // find element on the other side of the facet
                if (associated_elements_falled_facet.size()==2) {
                    int associated_element_falled_facet = *associated_elements_falled_facet.begin();
                    if (associated_element_falled_facet == eid) {
                        associated_element_falled_facet = *associated_elements_falled_facet.rbegin();
                    }
                    // find facet in the element
                    const index_t *m = _mesh->get_element(associated_element_falled_facet);
                    int iFacet = 0;
                    for (iFacet=0; iFacet<nloc; iFacet++) {
                        if (m[iFacet]==falled_faced[0])
                            continue;
                        if (m[iFacet]==falled_faced[1])
                            continue;
                        if (ndims==3 && m[iFacet]==falled_faced[2])
                            continue;
                        break;
                    }
                    // Finally...update boundary.
                    _mesh->boundary[associated_element_falled_facet*nloc+iFacet] = inherit_boundary_id;
//                    if (associated_element_falled_facet == 209803 && inherit_boundary_id == 0) printf("DEBUG  BIG PROBLEM2\n");
                }
            }
            else if (associated_elements.size()==2) {
                if (test) printf("DEBUG    associated_elements: %d %d\n", *associated_elements.begin(), *associated_elements.rbegin());
                int associated_element = *associated_elements.begin();
                if (associated_element==eid) {
                    associated_element = *associated_elements.rbegin();
                }

                // Locate falling facet on this element.
                int iFacet=0;
                const index_t *m = _mesh->get_element(associated_element);
                for (; iFacet<nloc; iFacet++) {
                    if (m[iFacet]==falling_facet[0])
                        continue;
                    if (m[iFacet]==falling_facet[1])
                        continue;
                    if (ndims==3 && m[iFacet]==falling_facet[2])
                        continue;
                    break;
                }

                // Finally...update boundary.
                if (!_mesh->is_internal_boundary(_mesh->boundary[associated_element*nloc+iFacet])) {
                    _mesh->boundary[associated_element*nloc+iFacet] = inherit_boundary_id;
                    //if (inherit_boundary_id < 17 && inherit_boundary_id >0 ) printf("DEBUG  big problemo\n");
//                    if (associated_element == 209803 && inherit_boundary_id == 0) printf("DEBUG  BIG PROBLEM\n");                    
                }
            }
            for(size_t i=0; i<nloc; ++i) {
                _mesh->NEList[n[i]].erase(eid);
            }

            // Remove element from mesh.
            _mesh->_ENList[eid*nloc] = -1;
//            if (eid == 248565) printf("DEBUG   eid: %d is deleted here: rm: %d target: %d\n", eid, rm_vertex, target_vertex);
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

    real_t _L_low, _L_max;
    bool delete_slivers, surface_coarsening, internal_surface_coarsening, quality_constrained;

    const static size_t ndims=dim;
    const static size_t nloc=dim+1;
    const static size_t msize=(dim==2?3:6);
};

#endif
