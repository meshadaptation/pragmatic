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

#ifndef REFINE_H
#define REFINE_H

#include <algorithm>
#include <set>
#include <vector>

#include <string.h>
#include <inttypes.h>

#include "DeferredOperations.h"
#include "Edge.h"
#include "ElementProperty.h"
#include "Mesh.h"

/*! \brief Performs 2D/3D mesh refinement.
 *
 */
template<typename real_t, int dim> class Refine
{
public:
    /// Default constructor.
    Refine(Mesh<real_t> &mesh): nloc(dim+1), msize(dim==2?3:6), nedge(dim==2?3:6)
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
                                                       _mesh->get_coords(n[1]), _mesh->get_coords(n[2]));
            else if(dim==3)
                property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                                       _mesh->get_coords(n[1]), _mesh->get_coords(n[2]), _mesh->get_coords(n[3]));

            break;
        }

        MPI_Comm comm = _mesh->get_mpi_comm();

        nprocs = pragmatic_nprocesses(comm);
        rank = pragmatic_process_id(comm);

        nthreads = pragmatic_nthreads();

        newVertices.resize(nthreads);
        newElements.resize(nthreads);
        newBoundaries.resize(nthreads);
        newQualities.resize(nthreads);
        newCoords.resize(nthreads);
        newMetric.resize(nthreads);

        // Pre-allocate the maximum size that might be required
        allNewVertices.resize(_mesh->_ENList.size());

        threadIdx.resize(nthreads);
        splitCnt.resize(nthreads);

        def_ops = new DeferredOperations<real_t>(_mesh, nthreads, defOp_scaling_factor);
    }

    /// Default destructor.
    ~Refine()
    {
        delete property;
        delete def_ops;
    }

    /*! Perform one level of refinement See Figure 25; X Li et al, Comp
     * Methods Appl Mech Engrg 194 (2005) 4915-4950. The actual
     * templates used for 3D refinement follows Rupak Biswas, Roger
     * C. Strawn, "A new procedure for dynamic adaption of
     * three-dimensional unstructured grids", Applied Numerical
     * Mathematics, Volume 13, Issue 6, February 1994, Pages 437-452.
     */
    void refine(real_t L_max)
    {
        size_t origNElements = _mesh->get_number_elements();
        size_t origNNodes = _mesh->get_number_nodes();
        size_t edgeSplitCnt = 0;

        std::vector<int> element_tag(origNElements,0);        
        
        new_vertices_per_element.resize(nedge*origNElements);
        std::fill(new_vertices_per_element.begin(), new_vertices_per_element.end(), -1);
        

        int tid = pragmatic_thread_id();
        splitCnt[tid] = 0;

        /*
         * Average vertex degree in 2D is ~6, so there
         * are approx. (6/2)*NNodes edges in the mesh.
         * In 3D, average vertex degree is ~12.
         */
        size_t reserve_size = nedge*origNNodes/nthreads;
        newVertices[tid].clear();
        newVertices[tid].reserve(reserve_size);
        newCoords[tid].clear();
        newCoords[tid].reserve(dim*reserve_size);
        newMetric[tid].clear();
        newMetric[tid].reserve(msize*reserve_size);

        /* Loop through all edges and select them for refinement if
           its length is greater than L_max in transformed space. */
        for(size_t i=0; i<origNNodes; ++i) {
            
            if (_mesh->is_halo_node(i))
                continue;

            for(size_t it=0; it<_mesh->NNList[i].size(); ++it) {

                index_t otherVertex = _mesh->NNList[i][it];
                assert(otherVertex>=0);

                if (_mesh->is_halo_node(otherVertex))
                    continue;

                // compute neighboring elements
                std::set<index_t> intersection;
                std::set_intersection(_mesh->NEList[i].begin(), _mesh->NEList[i].end(),
                                      _mesh->NEList[otherVertex].begin(), _mesh->NEList[otherVertex].end(),
                                      std::inserter(intersection, intersection.begin()));

                // if one element is tagged => don't refine
                int skip = 0;
                for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element) {
                    index_t eid = *element;
                    if (element_tag[eid]) {
                        skip = 1;
                        break;
                    }
                }
                if (skip)
                    continue;


                /* Conditional statement ensures that the edge length is only calculated once.
                 * By ordering the vertices according to their gnn, we ensure that all processes
                 * calculate the same edge length when they fall on the halo.
                 */
                if(_mesh->lnn2gnn[i] < _mesh->lnn2gnn[otherVertex]) {
                    double length = _mesh->calc_edge_length(i, otherVertex);
                    if(length>L_max) {
                        ++splitCnt[tid];
                        refine_edge(i, otherVertex, tid);
                        for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element) {
                            index_t eid = *element;
                            element_tag[eid] = 1;
                        }
                    }
                }
            }
        }

        threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NNodes, splitCnt[tid]);
        assert(newVertices[tid].size()==splitCnt[tid]);

        
        size_t reserve = 1.1*_mesh->NNodes; // extra space is required for centroidals
        if(_mesh->_coords.size()<reserve*dim) {
            _mesh->_coords.resize(reserve*dim);
            _mesh->metric.resize(reserve*msize);
            _mesh->NNList.resize(reserve);
            _mesh->NEList.resize(reserve);
            _mesh->node_owner.resize(reserve);
            _mesh->lnn2gnn.resize(reserve);
        }
        edgeSplitCnt = _mesh->NNodes - origNNodes;
        

        // Append new coords and metric to the mesh.
        memcpy(&_mesh->_coords[dim*threadIdx[tid]], &newCoords[tid][0], dim*splitCnt[tid]*sizeof(real_t));
        memcpy(&_mesh->metric[msize*threadIdx[tid]], &newMetric[tid][0], msize*splitCnt[tid]*sizeof(double));

        // Fix IDs of new vertices
        assert(newVertices[tid].size()==splitCnt[tid]);
        for(size_t i=0; i<splitCnt[tid]; i++) {
            newVertices[tid][i].id = threadIdx[tid]+i;
        }

        // Accumulate all newVertices in a contiguous array
        memcpy(&allNewVertices[threadIdx[tid]-origNNodes], &newVertices[tid][0], newVertices[tid].size()*sizeof(DirectedEdge<index_t>));

        // Mark each element with its new vertices,
        // update NNList for all split edges.
        for(size_t i=0; i<edgeSplitCnt; ++i) {
            index_t vid = allNewVertices[i].id;
            index_t firstid = allNewVertices[i].edge.first;
            index_t secondid = allNewVertices[i].edge.second;

            // Find which elements share this edge and mark them with their new vertices.
            std::set<index_t> intersection;
            std::set_intersection(_mesh->NEList[firstid].begin(), _mesh->NEList[firstid].end(),
                                  _mesh->NEList[secondid].begin(), _mesh->NEList[secondid].end(),
                                  std::inserter(intersection, intersection.begin()));

            for(typename std::set<index_t>::const_iterator element=intersection.begin(); element!=intersection.end(); ++element) {
                index_t eid = *element;
                size_t edgeOffset = edgeNumber(eid, firstid, secondid);
                new_vertices_per_element[nedge*eid+edgeOffset] = vid;
            }

            /*
             * Update NNList for newly created vertices. This has to be done here, it cannot be
             * done during element refinement, because a split edge is shared between two elements
             * and we run the risk that these updates will happen twice, once for each element.
             */
            _mesh->NNList[vid].push_back(firstid);
            _mesh->NNList[vid].push_back(secondid);

            def_ops->remNN(firstid, secondid, tid);
            def_ops->addNN(firstid, vid, tid);
            def_ops->remNN(secondid, firstid, tid);
            def_ops->addNN(secondid, vid, tid);

            // This branch is always taken or always not taken for every vertex,
            // so the branch predictor should have no problem handling it.
            if(nprocs==1) {
                _mesh->node_owner[vid] = 0;
                _mesh->lnn2gnn[vid] = vid;
            } else {
                /*
                 * Perhaps we should introduce a system of alternating min/max assignments,
                 * i.e. one time the node is assigned to the min rank, one time to the max
                 * rank and so on, so as to avoid having the min rank accumulate the majority
                 * of newly created vertices and disturbing load balance among MPI processes.
                 */
                int owner0 = _mesh->node_owner[firstid];
                int owner1 = _mesh->node_owner[secondid];
                int owner = std::min(owner0, owner1);
                _mesh->node_owner[vid] = owner;

                if(_mesh->node_owner[vid] == rank)
                    _mesh->lnn2gnn[vid] = _mesh->gnn_offset+vid;
            }
        }

        if(dim==3) {
            // If in 3D, we need to refine facets first.
            for(index_t eid=0; eid<origNElements; ++eid) {
                // Find the 4 facets comprising the element
                const index_t *n = _mesh->get_element(eid);
                if(n[0] < 0)
                    continue;

                const index_t facets[4][3] = {{n[0], n[1], n[2]},
                    {n[0], n[1], n[3]},
                    {n[0], n[2], n[3]},
                    {n[1], n[2], n[3]}
                };

                for(int j=0; j<4; ++j) {
                    // Find which elements share this facet j
                    const index_t *facet = facets[j];
                    std::set<index_t> intersection01, EE;
                    std::set_intersection(_mesh->NEList[facet[0]].begin(), _mesh->NEList[facet[0]].end(),
                                          _mesh->NEList[facet[1]].begin(), _mesh->NEList[facet[1]].end(),
                                          std::inserter(intersection01, intersection01.begin()));
                    std::set_intersection(_mesh->NEList[facet[2]].begin(), _mesh->NEList[facet[2]].end(),
                                          intersection01.begin(), intersection01.end(),
                                          std::inserter(EE, EE.begin()));

                    assert(EE.size() <= 2 );
                    assert(EE.count(eid) == 1);

                    // Prevent facet from being refined twice:
                    // Only refine it if this is the element with the highest ID.
                    if(eid == *EE.rbegin())
                        for(size_t k=0; k<3; ++k)
                            if(new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[k], facet[(k+1)%3])] != -1) {
                                refine_facet(eid, facet, tid);
                                break;
                            }
                }
            }

            for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid) {
                for(int i=0; i<nthreads; ++i) {
                    def_ops->commit_remNN(i, vtid);
                    def_ops->commit_addNN(i, vtid);
                }
            }
        }

        // Start element refinement.
        splitCnt[tid] = 0;
        newElements[tid].clear();
        newBoundaries[tid].clear();
        newQualities[tid].clear();
        newElements[tid].reserve(dim*dim*origNElements/nthreads);
        newBoundaries[tid].reserve(dim*dim*origNElements/nthreads);
        newQualities[tid].reserve(origNElements/nthreads);

        for(size_t eid=0; eid<origNElements; ++eid) {
            //If the element has been deleted, continue.
            const index_t *n = _mesh->get_element(eid);
            if(n[0] < 0)
                continue;

            for(size_t j=0; j<nedge; ++j)
                if(new_vertices_per_element[nedge*eid+j] != -1) {
                    refine_element(eid, tid);
                    break;
                }
        }

        threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NElements, splitCnt[tid]);

        
        if(_mesh->_ENList.size()<_mesh->NElements*nloc) {
            _mesh->_ENList.resize(_mesh->NElements*nloc);
            _mesh->boundary.resize(_mesh->NElements*nloc);
            _mesh->quality.resize(_mesh->NElements);
        }
        

        // Append new elements to the mesh and commit deferred operations
        memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
        memcpy(&_mesh->boundary[nloc*threadIdx[tid]], &newBoundaries[tid][0], nloc*splitCnt[tid]*sizeof(int));
        memcpy(&_mesh->quality[threadIdx[tid]], &newQualities[tid][0], splitCnt[tid]*sizeof(double));

        // Commit deferred operations.
        for(int vtid=0; vtid<defOp_scaling_factor*nthreads; ++vtid) {
            for(int i=0; i<nthreads; ++i) {
                def_ops->commit_remNN(i, vtid);
                def_ops->commit_addNN(i, vtid);
                def_ops->commit_remNE(i, vtid);
                def_ops->commit_addNE(i, vtid);
                def_ops->commit_addNE_fix(threadIdx, i, vtid);
            }
        }

        // Update halo.
        if(nprocs>1) {
            
            std::vector< std::set< DirectedEdge<index_t> > > recv_additional(nprocs), send_additional(nprocs);

            for(size_t i=0; i<edgeSplitCnt; ++i)
            {
                DirectedEdge<index_t> *vert = &allNewVertices[i];

                if(_mesh->node_owner[vert->id] != rank) {
                    // Vertex is owned by another MPI process, so prepare to update recv and recv_halo.
                    // Only update them if the vertex is actually visible by *this* MPI process,
                    // i.e. if at least one of its neighbours is owned by *this* process.
                    bool visible = false;
                    for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[vert->id].begin(); neigh!=_mesh->NNList[vert->id].end(); ++neigh) {
                        if(_mesh->is_owned_node(*neigh)) {
                            visible = true;
                            DirectedEdge<index_t> gnn_edge(_mesh->lnn2gnn[vert->edge.first], _mesh->lnn2gnn[vert->edge.second], vert->id);
                            recv_additional[_mesh->node_owner[vert->id]].insert(gnn_edge);
                            break;
                        }
                    }
                } else {
                    // Vertex is owned by *this* MPI process, so check whether it is visible by other MPI processes.
                    // The latter is true only if both vertices of the original edge were halo vertices.
                    if(_mesh->is_halo_node(vert->edge.first) && _mesh->is_halo_node(vert->edge.second)) {
                        // Find which processes see this vertex
                        std::set<int> processes;
                        for(typename std::vector<index_t>::const_iterator neigh=_mesh->NNList[vert->id].begin(); neigh!=_mesh->NNList[vert->id].end(); ++neigh)
                            processes.insert(_mesh->node_owner[*neigh]);

                        processes.erase(rank);

                        for(typename std::set<int>::const_iterator proc=processes.begin(); proc!=processes.end(); ++proc) {
                            DirectedEdge<index_t> gnn_edge(_mesh->lnn2gnn[vert->edge.first], _mesh->lnn2gnn[vert->edge.second], vert->id);
                            send_additional[*proc].insert(gnn_edge);
                        }
                    }
                }
            }

            // Append vertices in recv_additional and send_additional to recv and send.
            // Mark how many vertices are added to each of these vectors.
            std::vector<size_t> recv_cnt(nprocs, 0), send_cnt(nprocs, 0);

            for(int i=0; i<nprocs; ++i)
            {
                recv_cnt[i] = recv_additional[i].size();
                for(typename std::set< DirectedEdge<index_t> >::const_iterator it=recv_additional[i].begin(); it!=recv_additional[i].end(); ++it) {
                    _mesh->recv[i].push_back(it->id);
                    _mesh->recv_halo.insert(it->id);
                }

                send_cnt[i] = send_additional[i].size();
                for(typename std::set< DirectedEdge<index_t> >::const_iterator it=send_additional[i].begin(); it!=send_additional[i].end(); ++it) {
                    _mesh->send[i].push_back(it->id);
                    _mesh->send_halo.insert(it->id);
                }
            }

            // Update global numbering
            _mesh->update_gappy_global_numbering(recv_cnt, send_cnt);

            // Now that the global numbering has been updated, update send_map and recv_map.
            for(int i=0; i<nprocs; ++i)
            {
                for(typename std::set< DirectedEdge<index_t> >::const_iterator it=recv_additional[i].begin(); it!=recv_additional[i].end(); ++it)
                    _mesh->recv_map[i][_mesh->lnn2gnn[it->id]] = it->id;

                for(typename std::set< DirectedEdge<index_t> >::const_iterator it=send_additional[i].begin(); it!=send_additional[i].end(); ++it)
                    _mesh->send_map[i][_mesh->lnn2gnn[it->id]] = it->id;
            }

            _mesh->trim_halo();
            
        }

#if !defined NDEBUG
        if(dim==2) {
            // Fix orientations of new elements.
            size_t NElements = _mesh->get_number_elements();

            for(size_t i=0; i<NElements; i++) {
                index_t n0 = _mesh->_ENList[i*nloc];
                if(n0<0)
                    continue;

                index_t n1 = _mesh->_ENList[i*nloc + 1];
                index_t n2 = _mesh->_ENList[i*nloc + 2];

                const real_t *x0 = &_mesh->_coords[n0*dim];
                const real_t *x1 = &_mesh->_coords[n1*dim];
                const real_t *x2 = &_mesh->_coords[n2*dim];

                real_t av = property->area(x0, x1, x2);

                if(av<=0) {
                    std::cerr<<"ERROR: inverted element in refinement"<<std::endl
                             <<"element = "<<n0<<", "<<n1<<", "<<n2<<std::endl;
                    exit(-1);
                }
            }
        }
#endif
    }

private:

    inline void refine_edge(index_t n0, index_t n1, int tid)
    {
        if(_mesh->lnn2gnn[n0] > _mesh->lnn2gnn[n1]) {
            // Needs to be swapped because we want the lesser gnn first.
            index_t tmp_n0=n0;
            n0=n1;
            n1=tmp_n0;
        }
        newVertices[tid].push_back(DirectedEdge<index_t>(n0, n1));

        // Calculate the position of the new point. From equation 16 in
        // Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950.
        real_t x, m;
        const real_t *x0 = _mesh->get_coords(n0);
        const double *m0 = _mesh->get_metric(n0);

        const real_t *x1 = _mesh->get_coords(n1);
        const double *m1 = _mesh->get_metric(n1);

        real_t weight = 1.0/(1.0 + sqrt(property->template length<dim>(x0, x1, m0)/
                                        property->template length<dim>(x0, x1, m1)));

        // Calculate position of new vertex and append it to OMP thread's temp storage
        for(size_t i=0; i<dim; i++) {
            x = x0[i]+weight*(x1[i] - x0[i]);
            newCoords[tid].push_back(x);
        }

#if 0
        // TODO HACK CAD
        double newCrd[3];
        for(size_t i=0; i<dim; i++)
            newCrd[i] = x0[i]+weight*(x1[i] - x0[i]);
        if (_mesh->get_isOnBoundary(n0) == 1 && _mesh->get_isOnBoundary(n1) == 1){
            double r = 0.5/sqrt(newCrd[0]*newCrd[0]+newCrd[1]*newCrd[1]);
            newCrd[0] *= r;
            newCrd[1] *= r;
        }
        for(size_t i=0; i<dim; i++)
            newCoords[tid].push_back(newCrd[i]);
#endif

        // Interpolate new metric and append it to OMP thread's temp storage
        for(size_t i=0; i<msize; i++) {
            m = m0[i]+weight*(m1[i] - m0[i]);
            newMetric[tid].push_back(m);
            if(pragmatic_isnan(m))
                std::cerr<<"ERROR: metric health is bad in "<<__FILE__<<std::endl
                         <<"m0[i] = "<<m0[i]<<std::endl
                         <<"m1[i] = "<<m1[i]<<std::endl
                         <<"property->length(x0, x1, m0) = "<<property->template length<dim>(x0, x1, m0)<<std::endl
                             <<"property->length(x0, x1, m1) = "<<property->template length<dim>(x0, x1, m1)<<std::endl
                                     <<"weight = "<<weight<<std::endl;
        }
    }

    inline void refine_facet(index_t eid, const index_t *facet, int tid)
    {
        const index_t *n=_mesh->get_element(eid);

        index_t newVertex[3] = {-1, -1, -1};
        newVertex[0] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[1], facet[2])];
        newVertex[1] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[0], facet[2])];
        newVertex[2] = new_vertices_per_element[nedge*eid+edgeNumber(eid, facet[0], facet[1])];

        int refine_cnt=0;
        for(size_t i=0; i<3; ++i)
            if(newVertex[i]!=-1)
                ++refine_cnt;

        switch(refine_cnt) {
        case 0:
            // Do nothing
            break;
        case 1:
            // 1:2 facet bisection
            for(int j=0; j<3; j++)
                if(newVertex[j] >= 0) {
                    def_ops->addNN(newVertex[j], facet[j], tid);
                    def_ops->addNN(facet[j], newVertex[j], tid);
                    break;
                }
            break;
        case 2:
            // 1:3 refinement with trapezoid split
            for(int j=0; j<3; j++) {
                if(newVertex[j] < 0) {
                    def_ops->addNN(newVertex[(j+1)%3], newVertex[(j+2)%3], tid);
                    def_ops->addNN(newVertex[(j+2)%3], newVertex[(j+1)%3], tid);

                    real_t ldiag1 = _mesh->calc_edge_length(newVertex[(j+1)%3], facet[(j+1)%3]);
                    real_t ldiag2 = _mesh->calc_edge_length(newVertex[(j+2)%3], facet[(j+2)%3]);
                    const int offset = ldiag1 < ldiag2 ? (j+1)%3 : (j+2)%3;

                    def_ops->addNN(newVertex[offset], facet[offset], tid);
                    def_ops->addNN(facet[offset], newVertex[offset], tid);

                    break;
                }
            }
            break;
        case 3:
            // 1:4 regular refinement
            for(int j=0; j<3; j++) {
                def_ops->addNN(newVertex[j], newVertex[(j+1)%3], tid);
                def_ops->addNN(newVertex[(j+1)%3], newVertex[j], tid);
            }
            break;
        default:
            break;
        }
    }

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    typedef boost::unordered_map<index_t, int> boundary_t;
#else
    typedef std::map<index_t, int> boundary_t;
#endif

    inline void refine_element(size_t eid, int tid)
    {
        if(dim==2) {
            /*
             *************************
             * 2D Element Refinement *
             *************************
             */

            const int *n=_mesh->get_element(eid);

            // Note the order of the edges - the i'th edge is opposite the i'th node in the element.
            index_t newVertex[3] = {-1, -1, -1};
            newVertex[0] = new_vertices_per_element[nedge*eid];
            newVertex[1] = new_vertices_per_element[nedge*eid+1];
            newVertex[2] = new_vertices_per_element[nedge*eid+2];

            int refine_cnt=0;
            for(size_t i=0; i<3; ++i)
                if(newVertex[i]!=-1)
                    ++refine_cnt;

            if(refine_cnt > 0) {
                assert (refine_cnt == 1);
                refine2D_1(newVertex, eid, tid);
            }

        } else {
            /*
             *************************
             * 3D Element Refinement *
             *************************
             */

            const int *n=_mesh->get_element(eid);

            int refine_cnt;
            std::vector< DirectedEdge<index_t> > splitEdges;
            for(int j=0, pos=0; j<4; j++)
                for(int k=j+1; k<4; k++) {
                    index_t vertexID = new_vertices_per_element[nedge*eid+pos];
                    if(vertexID >= 0) {
                        splitEdges.push_back(DirectedEdge<index_t>(n[j], n[k], vertexID));
                    }
                    ++pos;
                }
            refine_cnt=splitEdges.size();

            if(refine_cnt > 0) {
                assert (refine_cnt == 1);
                refine3D_1(splitEdges, eid, tid);
            }
        }
    }

    inline void refine2D_1(const index_t *newVertex, int eid, int tid)
    {
        // Single edge split.

        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        int rotated_ele[3];
        int rotated_boundary[3];
        index_t vertexID = -1;
        for(int j=0; j<3; j++)
            if(newVertex[j] >= 0) {
                vertexID = newVertex[j];

                rotated_ele[0] = n[j];
                rotated_ele[1] = n[(j+1)%3];
                rotated_ele[2] = n[(j+2)%3];

                rotated_boundary[0] = boundary[j];
                rotated_boundary[1] = boundary[(j+1)%3];
                rotated_boundary[2] = boundary[(j+2)%3];

                break;
            }
        assert(vertexID!=-1);

        const index_t ele0[] = {rotated_ele[0], rotated_ele[1], vertexID};
        const index_t ele1[] = {rotated_ele[0], vertexID, rotated_ele[2]};

        const index_t ele0_boundary[] = {rotated_boundary[0], 0, rotated_boundary[2]};
        const index_t ele1_boundary[] = {rotated_boundary[0], rotated_boundary[1], 0};

        index_t ele1ID;
        ele1ID = splitCnt[tid];

        // Add rotated_ele[0] to vertexID's NNList
        def_ops->addNN(vertexID, rotated_ele[0], tid);
        // Add vertexID to rotated_ele[0]'s NNList
        def_ops->addNN(rotated_ele[0], vertexID, tid);

        // ele1ID is a new ID which isn't correct yet, it has to be
        // updated once each thread has calculated how many new elements
        // it created, so put ele1ID into addNE_fix instead of addNE.
        // Put ele1 in rotated_ele[0]'s NEList
        def_ops->addNE_fix(rotated_ele[0], ele1ID, tid);

        // Put eid and ele1 in vertexID's NEList
        def_ops->addNE(vertexID, eid, tid);
        def_ops->addNE_fix(vertexID, ele1ID, tid);

        // Replace eid with ele1 in rotated_ele[2]'s NEList
        def_ops->remNE(rotated_ele[2], eid, tid);
        def_ops->addNE_fix(rotated_ele[2], ele1ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        splitCnt[tid] += 1;
    }

    inline void refine3D_1(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        // Find the opposite edge
        index_t oe[2];
        for(int j=0, pos=0; j<4; j++)
            if(!splitEdges[0].contains(n[j]))
                oe[pos++] = n[j];

        // Form and add two new edges.
        const int ele0[] = {splitEdges[0].edge.first, splitEdges[0].id, oe[0], oe[1]};
        const int ele1[] = {splitEdges[0].edge.second, splitEdges[0].id, oe[0], oe[1]};

        const int ele0_boundary[] = {0, b[splitEdges[0].edge.second], b[oe[0]], b[oe[1]]};
        const int ele1_boundary[] = {0, b[splitEdges[0].edge.first], b[oe[0]], b[oe[1]]};

        index_t ele1ID;
        ele1ID = splitCnt[tid];

        // ele1ID is a new ID which isn't correct yet, it has to be
        // updated once each thread has calculated how many new elements
        // it created, so put ele1ID into addNE_fix instead of addNE.
        // Put ele1 in oe[0] and oe[1]'s NEList
        def_ops->addNE_fix(oe[0], ele1ID, tid);
        def_ops->addNE_fix(oe[1], ele1ID, tid);

        // Put eid and ele1 in newVertex[0]'s NEList
        def_ops->addNE(splitEdges[0].id, eid, tid);
        def_ops->addNE_fix(splitEdges[0].id, ele1ID, tid);

        // Replace eid with ele1 in splitEdges[0].edge.second's NEList
        def_ops->remNE(splitEdges[0].edge.second, eid, tid);
        def_ops->addNE_fix(splitEdges[0].edge.second, ele1ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        splitCnt[tid] += 1;
    }

    inline void append_element(const index_t *elem, const int *boundary, const size_t tid)
    {
        if(dim==3) {
            // Fix orientation of new element.
            const real_t *x0 = &(_mesh->_coords[elem[0]*dim]);
            const real_t *x1 = &(_mesh->_coords[elem[1]*dim]);
            const real_t *x2 = &(_mesh->_coords[elem[2]*dim]);
            const real_t *x3 = &(_mesh->_coords[elem[3]*dim]);

            real_t av = property->volume(x0, x1, x2, x3);

            if(av<0) {
                index_t *e = const_cast<index_t *>(elem);
                int *b = const_cast<int *>(boundary);

                // Flip element
                index_t e0 = e[0];
                e[0] = e[1];
                e[1] = e0;

                // and boundary
                int b0 = b[0];
                b[0] = b[1];
                b[1] = b0;
            }
        }

        for(size_t i=0; i<nloc; ++i) {
            newElements[tid].push_back(elem[i]);
            newBoundaries[tid].push_back(boundary[i]);
        }

        double q = _mesh->template calculate_quality<dim>(elem);
        newQualities[tid].push_back(q);
    }

    inline void replace_element(const index_t eid, const index_t *n, const int *boundary)
    {
        if(dim==3) {
            // Fix orientation of new element.
            const real_t *x0 = &(_mesh->_coords[n[0]*dim]);
            const real_t *x1 = &(_mesh->_coords[n[1]*dim]);
            const real_t *x2 = &(_mesh->_coords[n[2]*dim]);
            const real_t *x3 = &(_mesh->_coords[n[3]*dim]);

            real_t av = property->volume(x0, x1, x2, x3);

            if(av<0) {
                index_t *e = const_cast<index_t *>(n);
                int *b = const_cast<int *>(boundary);

                // Flip element
                index_t e0 = e[0];
                e[0] = e[1];
                e[1] = e0;

                // and boundary
                int b0 = b[0];
                b[0] = b[1];
                b[1] = b0;
            }
        }

        for(size_t i=0; i<nloc; i++) {
            _mesh->_ENList[eid*nloc+i]=n[i];
            _mesh->boundary[eid*nloc+i]=boundary[i];
        }

        _mesh->template update_quality<dim>(eid);
    }

    inline size_t edgeNumber(index_t eid, index_t v1, index_t v2) const
    {
        const int *n=_mesh->get_element(eid);

        if(dim==2) {
            /* In 2D:
             * Edge 0 is the edge (n[1],n[2]).
             * Edge 1 is the edge (n[0],n[2]).
             * Edge 2 is the edge (n[0],n[1]).
             */
            if(n[1]==v1 || n[1]==v2) {
                if(n[2]==v1 || n[2]==v2)
                    return 0;
                else
                    return 2;
            } else
                return 1;
        } else { //if(dim=3)
            /*
             * In 3D:
             * Edge 0 is the edge (n[0],n[1]).
             * Edge 1 is the edge (n[0],n[2]).
             * Edge 2 is the edge (n[0],n[3]).
             * Edge 3 is the edge (n[1],n[2]).
             * Edge 4 is the edge (n[1],n[3]).
             * Edge 5 is the edge (n[2],n[3]).
             */
            if(n[0]==v1 || n[0]==v2) {
                if(n[1]==v1 || n[1]==v2)
                    return 0;
                else if(n[2]==v1 || n[2]==v2)
                    return 1;
                else
                    return 2;
            } else if(n[1]==v1 || n[1]==v2) {
                if(n[2]==v1 || n[2]==v2)
                    return 3;
                else
                    return 4;
            } else
                return 5;
        }
    }

    std::vector< std::vector< DirectedEdge<index_t> > > newVertices;
    std::vector< std::vector<real_t> > newCoords;
    std::vector< std::vector<double> > newMetric;
    std::vector< std::vector<index_t> > newElements;
    std::vector< std::vector<int> > newBoundaries;
    std::vector< std::vector<double> > newQualities;
    std::vector<index_t> new_vertices_per_element;

    std::vector<size_t> threadIdx, splitCnt;
    std::vector< DirectedEdge<index_t> > allNewVertices;

    DeferredOperations<real_t>* def_ops;
    static const int defOp_scaling_factor = 32;

    Mesh<real_t> *_mesh;
    ElementProperty<real_t> *property;

    const size_t nloc, msize, nedge;
    int nprocs, rank, nthreads;

    void (Refine<real_t,dim>::* refineMode2D[3])(const index_t *, int, int);
    void (Refine<real_t,dim>::* refineMode3D[6])(std::vector< DirectedEdge<index_t> >&, int, int);
};


#endif
