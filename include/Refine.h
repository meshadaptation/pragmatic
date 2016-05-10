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

#ifdef HAVE_MPI
        MPI_Comm comm = _mesh->get_mpi_comm();

        nprocs = pragmatic_nprocesses(comm);
        rank = pragmatic_process_id(comm);
#else
        nprocs = 1;
        rank = 0;
#endif

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

        cidRecv_additional.resize(nprocs);
        cidSend_additional.resize(nprocs);

        if(dim==2) {
            refineMode2D[0] = &Refine<real_t,dim>::refine2D_1;
            refineMode2D[1] = &Refine<real_t,dim>::refine2D_2;
            refineMode2D[2] = &Refine<real_t,dim>::refine2D_3;
        } else {
            refineMode3D[0] = &Refine<real_t,dim>::refine3D_1;
            refineMode3D[1] = &Refine<real_t,dim>::refine3D_2;
            refineMode3D[2] = &Refine<real_t,dim>::refine3D_3;
            refineMode3D[3] = &Refine<real_t,dim>::refine3D_4;
            refineMode3D[4] = &Refine<real_t,dim>::refine3D_5;
            refineMode3D[5] = &Refine<real_t,dim>::refine3D_6;
        }
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

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                new_vertices_per_element.resize(nedge*origNElements);
                std::fill(new_vertices_per_element.begin(), new_vertices_per_element.end(), -1);
            }

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
            #pragma omp for schedule(guided) nowait
            for(size_t i=0; i<origNNodes; ++i) {
                for(size_t it=0; it<_mesh->NNList[i].size(); ++it) {
                    index_t otherVertex = _mesh->NNList[i][it];
                    assert(otherVertex>=0);

                    /* Conditional statement ensures that the edge length is only calculated once.
                     * By ordering the vertices according to their gnn, we ensure that all processes
                     * calculate the same edge length when they fall on the halo.
                     */
                    if(_mesh->lnn2gnn[i] < _mesh->lnn2gnn[otherVertex]) {
                        double length = _mesh->calc_edge_length(i, otherVertex);
                        if(length>L_max) {
                            ++splitCnt[tid];
                            refine_edge(i, otherVertex, tid);
                        }
                    }
                }
            }

            threadIdx[tid] = pragmatic_omp_atomic_capture(&_mesh->NNodes, splitCnt[tid]);
            assert(newVertices[tid].size()==splitCnt[tid]);

            #pragma omp barrier

            #pragma omp single
            {
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
            }

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
            #pragma omp barrier
            #pragma omp for schedule(guided)
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
                #pragma omp for schedule(guided)
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

                #pragma omp for schedule(guided)
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

            #pragma omp for schedule(guided) nowait
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

            #pragma omp barrier
            #pragma omp single
            {
                if(_mesh->_ENList.size()<_mesh->NElements*nloc) {
                    _mesh->_ENList.resize(_mesh->NElements*nloc);
                    _mesh->boundary.resize(_mesh->NElements*nloc);
                    _mesh->quality.resize(_mesh->NElements);
                }
            }

            // Append new elements to the mesh and commit deferred operations
            memcpy(&_mesh->_ENList[nloc*threadIdx[tid]], &newElements[tid][0], nloc*splitCnt[tid]*sizeof(index_t));
            memcpy(&_mesh->boundary[nloc*threadIdx[tid]], &newBoundaries[tid][0], nloc*splitCnt[tid]*sizeof(int));
            memcpy(&_mesh->quality[threadIdx[tid]], &newQualities[tid][0], splitCnt[tid]*sizeof(double));

            // Commit deferred operations.
            #pragma omp for schedule(guided)
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
#ifdef HAVE_MPI
            if(nprocs>1) {
                #pragma omp single
                {
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

                    // Additional code for centroidal vertices.
                    if(dim==3)
                    {
                        for(int i=0; i<nprocs; ++i) {
                            recv_cnt[i] += cidRecv_additional[i].size();
                            for(typename std::set<Wedge>::const_iterator it=cidRecv_additional[i].begin(); it!=cidRecv_additional[i].end(); ++it) {
                                _mesh->recv[i].push_back(it->cid);
                                _mesh->recv_halo.insert(it->cid);
                            }

                            send_cnt[i] += cidSend_additional[i].size();
                            for(typename std::set<Wedge>::const_iterator it=cidSend_additional[i].begin(); it!=cidSend_additional[i].end(); ++it) {
                                _mesh->send[i].push_back(it->cid);
                                _mesh->send_halo.insert(it->cid);
                            }
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

                        // Additional code for centroidals.
                        if(dim==3) {
                            for(typename std::set<Wedge>::const_iterator it=cidRecv_additional[i].begin(); it!=cidRecv_additional[i].end(); ++it)
                                _mesh->recv_map[i][_mesh->lnn2gnn[it->cid]] = it->cid;

                            for(typename std::set<Wedge>::const_iterator it=cidSend_additional[i].begin(); it!=cidSend_additional[i].end(); ++it)
                                _mesh->send_map[i][_mesh->lnn2gnn[it->cid]] = it->cid;

                            cidRecv_additional[i].clear();
                            cidSend_additional[i].clear();
                        }
                    }

                    _mesh->trim_halo();
                }
            }
#endif

#if !defined NDEBUG
            if(dim==2) {
                #pragma omp barrier
                // Fix orientations of new elements.
                size_t NElements = _mesh->get_number_elements();

                #pragma omp for schedule(guided)
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
                        #pragma omp critical
                        std::cerr<<"ERROR: inverted element in refinement"<<std::endl
                                 <<"element = "<<n0<<", "<<n1<<", "<<n2<<std::endl;
                        exit(-1);
                    }
                }
            }
#endif
        }
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

            if(refine_cnt > 0)
                (this->*refineMode2D[refine_cnt-1])(newVertex, eid, tid);

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

            if(refine_cnt > 0)
                (this->*refineMode3D[refine_cnt-1])(splitEdges, eid, tid);
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

    inline void refine2D_2(const index_t *newVertex, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        int rotated_ele[3];
        int rotated_boundary[3];
        index_t vertexID[2];
        for(int j=0; j<3; j++) {
            if(newVertex[j] < 0) {
                vertexID[0] = newVertex[(j+1)%3];
                vertexID[1] = newVertex[(j+2)%3];

                rotated_ele[0] = n[j];
                rotated_ele[1] = n[(j+1)%3];
                rotated_ele[2] = n[(j+2)%3];

                rotated_boundary[0] = boundary[j];
                rotated_boundary[1] = boundary[(j+1)%3];
                rotated_boundary[2] = boundary[(j+2)%3];

                break;
            }
        }

        real_t ldiag0 = _mesh->calc_edge_length(rotated_ele[1], vertexID[0]);
        real_t ldiag1 = _mesh->calc_edge_length(rotated_ele[2], vertexID[1]);

        const int offset = ldiag0 < ldiag1 ? 0 : 1;

        const index_t ele0[] = {rotated_ele[0], vertexID[1], vertexID[0]};
        const index_t ele1[] = {vertexID[offset], rotated_ele[1], rotated_ele[2]};
        const index_t ele2[] = {vertexID[0], vertexID[1], rotated_ele[offset+1]};

        const index_t ele0_boundary[] = {0, rotated_boundary[1], rotated_boundary[2]};
        const index_t ele1_boundary[] = {rotated_boundary[0], (offset==0)?rotated_boundary[1]:0, (offset==0)?0:rotated_boundary[2]};
        const index_t ele2_boundary[] = {(offset==0)?rotated_boundary[2]:0, (offset==0)?0:rotated_boundary[1], 0};

        index_t ele0ID, ele2ID;
        ele0ID = splitCnt[tid];
        ele2ID = ele0ID+1;


        // NNList: Connect vertexID[0] and vertexID[1] with each other
        def_ops->addNN(vertexID[0], vertexID[1], tid);
        def_ops->addNN(vertexID[1], vertexID[0], tid);

        // vertexID[offset] and rotated_ele[offset+1] are the vertices on the diagonal
        def_ops->addNN(vertexID[offset], rotated_ele[offset+1], tid);
        def_ops->addNN(rotated_ele[offset+1], vertexID[offset], tid);

        // rotated_ele[offset+1] is the old vertex which is on the diagonal
        // Add ele2 in rotated_ele[offset+1]'s NEList
        def_ops->addNE_fix(rotated_ele[offset+1], ele2ID, tid);

        // Replace eid with ele0 in NEList[rotated_ele[0]]
        def_ops->remNE(rotated_ele[0], eid, tid);
        def_ops->addNE_fix(rotated_ele[0], ele0ID, tid);

        // Put ele0, ele1 and ele2 in vertexID[offset]'s NEList
        def_ops->addNE(vertexID[offset], eid, tid);
        def_ops->addNE_fix(vertexID[offset], ele0ID, tid);
        def_ops->addNE_fix(vertexID[offset], ele2ID, tid);

        // vertexID[(offset+1)%2] is the new vertex which is not on the diagonal
        // Put ele0 and ele2 in vertexID[(offset+1)%2]'s NEList
        def_ops->addNE_fix(vertexID[(offset+1)%2], ele0ID, tid);
        def_ops->addNE_fix(vertexID[(offset+1)%2], ele2ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
        assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);

        replace_element(eid, ele1, ele1_boundary);
        append_element(ele0, ele0_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        splitCnt[tid] += 2;
    }

    inline void refine2D_3(const index_t *newVertex, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        const index_t ele0[] = {n[0], newVertex[2], newVertex[1]};
        const index_t ele1[] = {n[1], newVertex[0], newVertex[2]};
        const index_t ele2[] = {n[2], newVertex[1], newVertex[0]};
        const index_t ele3[] = {newVertex[0], newVertex[1], newVertex[2]};

        const int ele0_boundary[] = {0, boundary[1], boundary[2]};
        const int ele1_boundary[] = {0, boundary[2], boundary[0]};
        const int ele2_boundary[] = {0, boundary[0], boundary[1]};
        const int ele3_boundary[] = {0, 0, 0};

        index_t ele1ID, ele2ID, ele3ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;

        // Update NNList
        def_ops->addNN(newVertex[0], newVertex[1], tid);
        def_ops->addNN(newVertex[0], newVertex[2], tid);
        def_ops->addNN(newVertex[1], newVertex[0], tid);
        def_ops->addNN(newVertex[1], newVertex[2], tid);
        def_ops->addNN(newVertex[2], newVertex[0], tid);
        def_ops->addNN(newVertex[2], newVertex[1], tid);

        // Update NEList
        def_ops->remNE(n[1], eid, tid);
        def_ops->addNE_fix(n[1], ele1ID, tid);
        def_ops->remNE(n[2], eid, tid);
        def_ops->addNE_fix(n[2], ele2ID, tid);

        def_ops->addNE_fix(newVertex[0], ele1ID, tid);
        def_ops->addNE_fix(newVertex[0], ele2ID, tid);
        def_ops->addNE_fix(newVertex[0], ele3ID, tid);

        def_ops->addNE(newVertex[1], eid, tid);
        def_ops->addNE_fix(newVertex[1], ele2ID, tid);
        def_ops->addNE_fix(newVertex[1], ele3ID, tid);

        def_ops->addNE(newVertex[2], eid, tid);
        def_ops->addNE_fix(newVertex[2], ele1ID, tid);
        def_ops->addNE_fix(newVertex[2], ele3ID, tid);

        assert(ele0[0]>=0 && ele0[1]>=0 && ele0[2]>=0);
        assert(ele1[0]>=0 && ele1[1]>=0 && ele1[2]>=0);
        assert(ele2[0]>=0 && ele2[1]>=0 && ele2[2]>=0);
        assert(ele3[0]>=0 && ele3[1]>=0 && ele3[2]>=0);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        splitCnt[tid] += 3;
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

    inline void refine3D_2(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        /* Here there are two possibilities. Either the two split
         * edges share a vertex (case 2(a)) or they are opposite edges
         * (case 2(b)). Case 2(a) results in a 1:3 subdivision, case 2(b)
         * results in a 1:4.
         */

        int n0=splitEdges[0].connected(splitEdges[1]);
        if(n0>=0) {
            /*
             *************
             * Case 2(a) *
             *************
             */
            int n1 = (n0 == splitEdges[0].edge.first) ? splitEdges[0].edge.second : splitEdges[0].edge.first;
            int n2 = (n0 == splitEdges[1].edge.first) ? splitEdges[1].edge.second : splitEdges[1].edge.first;

            // Opposite vertex
            int n3;
            for(int j=0; j<nloc; ++j)
                if(n[j] != n0 && n[j] != n1 && n[j] != n2) {
                    n3 = n[j];
                    break;
                }

            // Find the diagonal which has bisected the trapezoid.
            DirectedEdge<index_t> diagonal, offdiagonal;
            std::vector<index_t>::const_iterator p = std::find(_mesh->NNList[splitEdges[0].id].begin(),
                    _mesh->NNList[splitEdges[0].id].end(), n2);
            if(p != _mesh->NNList[splitEdges[0].id].end()) {
                diagonal.edge.first = splitEdges[0].id;
                diagonal.edge.second = n2;
                offdiagonal.edge.first = splitEdges[1].id;
                offdiagonal.edge.second = n1;
            } else {
                assert(std::find(_mesh->NNList[splitEdges[1].id].begin(),
                                 _mesh->NNList[splitEdges[1].id].end(), n1) != _mesh->NNList[splitEdges[1].id].end());
                diagonal.edge.first = splitEdges[1].id;
                diagonal.edge.second = n1;
                offdiagonal.edge.first = splitEdges[0].id;
                offdiagonal.edge.second = n2;
            }

            const int ele0[] = {n0, splitEdges[0].id, splitEdges[1].id, n3};
            const int ele1[] = {diagonal.edge.first, offdiagonal.edge.first, diagonal.edge.second, n3};
            const int ele2[] = {diagonal.edge.first, diagonal.edge.second, offdiagonal.edge.second, n3};

            const int ele0_boundary[] = {0, b[n1], b[n2], b[n3]};
            const int ele1_boundary[] = {b[offdiagonal.edge.second], 0, 0, b[n3]};
            const int ele2_boundary[] = {b[n0], b[diagonal.edge.second], 0, b[n3]};

            index_t ele1ID, ele2ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;

            def_ops->addNE(diagonal.edge.first, eid, tid);
            def_ops->addNE_fix(diagonal.edge.first, ele1ID, tid);
            def_ops->addNE_fix(diagonal.edge.first, ele2ID, tid);

            def_ops->remNE(diagonal.edge.second, eid, tid);
            def_ops->addNE_fix(diagonal.edge.second, ele1ID, tid);
            def_ops->addNE_fix(diagonal.edge.second, ele2ID, tid);

            def_ops->addNE(offdiagonal.edge.first, eid, tid);
            def_ops->addNE_fix(offdiagonal.edge.first, ele1ID, tid);

            def_ops->remNE(offdiagonal.edge.second, eid, tid);
            def_ops->addNE_fix(offdiagonal.edge.second, ele2ID, tid);

            def_ops->addNE_fix(n3, ele1ID, tid);
            def_ops->addNE_fix(n3, ele2ID, tid);

            replace_element(eid, ele0, ele0_boundary);
            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            splitCnt[tid] += 2;
        } else {
            /*
             *************
             * Case 2(b) *
             *************
             */
            const int ele0[] = {splitEdges[0].edge.first, splitEdges[0].id, splitEdges[1].edge.first, splitEdges[1].id};
            const int ele1[] = {splitEdges[0].edge.first, splitEdges[0].id, splitEdges[1].edge.second, splitEdges[1].id};
            const int ele2[] = {splitEdges[0].edge.second, splitEdges[0].id, splitEdges[1].edge.first, splitEdges[1].id};
            const int ele3[] = {splitEdges[0].edge.second, splitEdges[0].id, splitEdges[1].edge.second, splitEdges[1].id};

            const int ele0_boundary[] = {0, b[splitEdges[0].edge.second], 0, b[splitEdges[1].edge.second]};
            const int ele1_boundary[] = {0, b[splitEdges[0].edge.second], 0, b[splitEdges[1].edge.first]};
            const int ele2_boundary[] = {0, b[splitEdges[0].edge.first], 0, b[splitEdges[1].edge.second]};
            const int ele3_boundary[] = {0, b[splitEdges[0].edge.first], 0, b[splitEdges[1].edge.first]};

            index_t ele1ID, ele2ID, ele3ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;

            def_ops->addNN(splitEdges[0].id, splitEdges[1].id, tid);
            def_ops->addNN(splitEdges[1].id, splitEdges[0].id, tid);

            def_ops->addNE(splitEdges[0].id, eid, tid);
            def_ops->addNE_fix(splitEdges[0].id, ele1ID, tid);
            def_ops->addNE_fix(splitEdges[0].id, ele2ID, tid);
            def_ops->addNE_fix(splitEdges[0].id, ele3ID, tid);

            def_ops->addNE(splitEdges[1].id, eid, tid);
            def_ops->addNE_fix(splitEdges[1].id, ele1ID, tid);
            def_ops->addNE_fix(splitEdges[1].id, ele2ID, tid);
            def_ops->addNE_fix(splitEdges[1].id, ele3ID, tid);

            def_ops->addNE_fix(splitEdges[0].edge.first, ele1ID, tid);

            def_ops->remNE(splitEdges[0].edge.second, eid, tid);
            def_ops->addNE_fix(splitEdges[0].edge.second, ele2ID, tid);
            def_ops->addNE_fix(splitEdges[0].edge.second, ele3ID, tid);

            def_ops->addNE_fix(splitEdges[1].edge.first, ele2ID, tid);

            def_ops->remNE(splitEdges[1].edge.second, eid, tid);
            def_ops->addNE_fix(splitEdges[1].edge.second, ele1ID, tid);
            def_ops->addNE_fix(splitEdges[1].edge.second, ele3ID, tid);

            replace_element(eid, ele0, ele0_boundary);
            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            splitCnt[tid] += 3;
        }
    }

    inline void refine3D_3(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        /* There are 3 cases that need to be considered. They can
         * be distinguished by the total number of nodes that are
         * common between any pair of edges.
         * Case 3(a): there are 3 different nodes common between pairs
         * of split edges, i.e the three new vertices are on the
         * same triangle.
         * Case 3(b): The three new vertices are around the same
         * original vertex.
         * Case 3(c): There are 2 different nodes common between pairs
         * of split edges.
         */
        std::set<index_t> shared;
        for(int j=0; j<3; j++) {
            for(int k=j+1; k<3; k++) {
                index_t nid = splitEdges[j].connected(splitEdges[k]);
                if(nid>=0)
                    shared.insert(nid);
            }
        }
        size_t nshared = shared.size();

        if(nshared==3) {
            /*
             *************
             * Case 3(a) *
             *************
             */
            index_t m[] = {-1, -1, -1, -1, -1, -1, -1};

            m[0] = splitEdges[0].edge.first;
            m[1] = splitEdges[0].id;
            m[2] = splitEdges[0].edge.second;
            if(splitEdges[1].contains(m[2])) {
                m[3] = splitEdges[1].id;
                if(splitEdges[1].edge.first!=m[2])
                    m[4] = splitEdges[1].edge.first;
                else
                    m[4] = splitEdges[1].edge.second;
                m[5] = splitEdges[2].id;
            } else {
                m[3] = splitEdges[2].id;
                if(splitEdges[2].edge.first!=m[2])
                    m[4] = splitEdges[2].edge.first;
                else
                    m[4] = splitEdges[2].edge.second;
                m[5] = splitEdges[1].id;
            }
            for(int j=0; j<4; j++) {
                if((n[j]!=m[0])&&(n[j]!=m[2])&&(n[j]!=m[4])) {
                    m[6] = n[j];
                    break;
                }
            }

            const int ele0[] = {m[0], m[1], m[5], m[6]};
            const int ele1[] = {m[1], m[2], m[3], m[6]};
            const int ele2[] = {m[5], m[3], m[4], m[6]};
            const int ele3[] = {m[1], m[3], m[5], m[6]};

            const int ele0_boundary[] = {0, b[m[2]], b[m[4]], b[m[6]]};
            const int ele1_boundary[] = {b[m[0]], 0, b[m[4]], b[m[6]]};
            const int ele2_boundary[] = {b[m[0]], b[m[2]], 0, b[m[6]]};
            const int ele3_boundary[] = {0, 0, 0, b[m[6]]};

            index_t ele1ID, ele2ID, ele3ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;

            def_ops->addNE(m[1], eid, tid);
            def_ops->addNE_fix(m[1], ele1ID, tid);
            def_ops->addNE_fix(m[1], ele3ID, tid);

            def_ops->addNE(m[5], eid, tid);
            def_ops->addNE_fix(m[5], ele2ID, tid);
            def_ops->addNE_fix(m[5], ele3ID, tid);

            def_ops->addNE_fix(m[3], ele1ID, tid);
            def_ops->addNE_fix(m[3], ele2ID, tid);
            def_ops->addNE_fix(m[3], ele3ID, tid);

            def_ops->addNE_fix(m[6], ele1ID, tid);
            def_ops->addNE_fix(m[6], ele2ID, tid);
            def_ops->addNE_fix(m[6], ele3ID, tid);

            def_ops->remNE(m[2], eid, tid);
            def_ops->addNE_fix(m[2], ele1ID, tid);

            def_ops->remNE(m[4], eid, tid);
            def_ops->addNE_fix(m[4], ele2ID, tid);

            replace_element(eid, ele0, ele0_boundary);
            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            splitCnt[tid] += 3;
        } else if(nshared==1) {
            /*
             *************
             * Case 3(b) *
             *************
             */

            // Find the three bottom vertices, i.e. vertices of
            // the original elements which are part of the wedge.
            index_t top_vertex = *shared.begin();
            index_t bottom_triangle[3], top_triangle[3];
            for(int j=0; j<3; ++j) {
                if(splitEdges[j].edge.first != top_vertex) {
                    bottom_triangle[j] = splitEdges[j].edge.first;
                } else {
                    bottom_triangle[j] = splitEdges[j].edge.second;
                }
                top_triangle[j] = splitEdges[j].id;
            }

            // Boundary values of each wedge side
            int bwedge[] = {b[bottom_triangle[2]], b[bottom_triangle[0]], b[bottom_triangle[1]], 0, b[top_vertex]};
            refine_wedge(top_triangle, bottom_triangle, bwedge, NULL, eid, tid);

            const int ele0[] = {top_vertex, splitEdges[0].id, splitEdges[1].id, splitEdges[2].id};
            const int ele0_boundary[] = {0, b[bottom_triangle[0]], b[bottom_triangle[1]], b[bottom_triangle[2]]};

            def_ops->remNE(bottom_triangle[0], eid, tid);
            def_ops->remNE(bottom_triangle[1], eid, tid);
            def_ops->remNE(bottom_triangle[2], eid, tid);
            def_ops->addNE(splitEdges[0].id, eid, tid);
            def_ops->addNE(splitEdges[1].id, eid, tid);
            def_ops->addNE(splitEdges[2].id, eid, tid);

            replace_element(eid, ele0, ele0_boundary);
        } else {
            /*
             *************
             * Case 3(c) *
             *************
             */
            assert(shared.size() == 2);
            /*
             * This case results in a 1:4 or 1:5 subdivision. There are three
             * split edges, connected in a Z-like way (cf. connection of
             * diagonals in 1:3 wedge split). By convention, the topOfZ edge is
             * the one connected to middleOfZ.edge.first, bottomOfZ is the one
             * connected to middleOfZ.edge.second. Top and bottom edges are
             * flipped if necessary so that the first vertex of top and the
             * second vertex of bottom are the ones connected to the middle edge.
             */

            DirectedEdge<index_t> *topZ, *middleZ, *bottomZ;
            // Middle split edge
            for(int j=0; j<3; ++j) {
                if(splitEdges[j].contains(*shared.begin()) && splitEdges[j].contains(*shared.rbegin())) {
                    middleZ = &splitEdges[j];

                    if(splitEdges[(j+1)%3].contains(splitEdges[j].edge.first)) {
                        topZ = &splitEdges[(j+1)%3];
                        bottomZ = &splitEdges[(j+2)%3];
                    } else {
                        topZ = &splitEdges[(j+2)%3];
                        bottomZ = &splitEdges[(j+1)%3];
                    }

                    break;
                }
            }

            // Flip vertices of top and bottom edges if necessary
            if(topZ->edge.first != middleZ->edge.first) {
                topZ->edge.second = topZ->edge.first;
                topZ->edge.first = middleZ->edge.first;
            }
            if(bottomZ->edge.second != middleZ->edge.second) {
                bottomZ->edge.first = bottomZ->edge.second;
                bottomZ->edge.second = middleZ->edge.second;
            }

            /*
             * There are 3 sub-cases, depending on the way the trapezoids on the
             * facets between topZ-middleZ and middleZ-bottomZ were bisected.
             *
             * Case 3(c)(1): Both diagonals involve middleZ->id.
             * Case 3(c)(2): Only one diagonal involves middleZ->id.
             * Case 3(c)(3): No diagonal involves middleZ->id.
             */

            std::vector< DirectedEdge<index_t> > diagonals;
            for(std::vector<index_t>::const_iterator it=_mesh->NNList[middleZ->id].begin();
                    it!=_mesh->NNList[middleZ->id].end(); ++it) {
                if(*it == topZ->edge.second || *it == bottomZ->edge.first) {
                    diagonals.push_back(DirectedEdge<index_t>(middleZ->id, *it));
                }
            }

            switch(diagonals.size()) {
            case 0: {
                // Case 3(c)(2)
                const int ele0[] = {middleZ->edge.first, topZ->id, bottomZ->id, bottomZ->edge.first};
                const int ele1[] = {middleZ->id, middleZ->edge.first, topZ->id, bottomZ->id};
                const int ele2[] = {topZ->id, topZ->edge.second, bottomZ->id, bottomZ->edge.first};
                const int ele3[] = {topZ->id, topZ->edge.second, bottomZ->edge.second, bottomZ->id};
                const int ele4[] = {middleZ->id, topZ->id, bottomZ->edge.second, bottomZ->id};

                const int ele0_boundary[] = {0, b[topZ->edge.second], b[middleZ->edge.second], 0};
                const int ele1_boundary[] = {0, 0, b[topZ->edge.second], b[bottomZ->edge.first]};
                const int ele2_boundary[] = {b[middleZ->edge.first], 0, b[middleZ->edge.second], 0};
                const int ele3_boundary[] = {b[middleZ->edge.first], 0, 0, b[bottomZ->edge.first]};
                const int ele4_boundary[] = {0, b[topZ->edge.second], 0, b[bottomZ->edge.first]};

                index_t ele1ID, ele2ID, ele3ID, ele4ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;
                ele4ID = ele1ID+3;

                def_ops->addNN(topZ->id, bottomZ->id, tid);
                def_ops->addNN(bottomZ->id, topZ->id, tid);

                def_ops->addNE_fix(middleZ->edge.first, ele1ID, tid);

                def_ops->remNE(middleZ->edge.second, eid, tid);
                def_ops->addNE_fix(middleZ->edge.second, ele3ID, tid);
                def_ops->addNE_fix(middleZ->edge.second, ele4ID, tid);

                def_ops->remNE(topZ->edge.second, eid, tid);
                def_ops->addNE_fix(topZ->edge.second, ele2ID, tid);
                def_ops->addNE_fix(topZ->edge.second, ele3ID, tid);

                def_ops->addNE_fix(bottomZ->edge.first, ele2ID, tid);

                def_ops->addNE_fix(middleZ->id, ele1ID, tid);
                def_ops->addNE_fix(middleZ->id, ele4ID, tid);

                def_ops->addNE(topZ->id, eid, tid);
                def_ops->addNE_fix(topZ->id, ele1ID, tid);
                def_ops->addNE_fix(topZ->id, ele2ID, tid);
                def_ops->addNE_fix(topZ->id, ele3ID, tid);
                def_ops->addNE_fix(topZ->id, ele4ID, tid);

                def_ops->addNE(bottomZ->id, eid, tid);
                def_ops->addNE_fix(bottomZ->id, ele1ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele2ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele3ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele4ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                append_element(ele4, ele4_boundary, tid);
                splitCnt[tid] += 4;
                break;
            }
            case 1: {
                // Case 3(c)(3)

                // Re-arrange topZ and bottomZ if necessary; make topZ point to the
                // splitEdge for which edge.second is connected to middleZ->top.
                if(topZ->edge.second != diagonals[0].edge.second) {
                    assert(diagonals[0].edge.second == bottomZ->edge.first);
                    DirectedEdge<index_t> *p = topZ;
                    topZ = bottomZ;
                    bottomZ = p;

                    // Flip topZ, middleZ, bottomZ
                    index_t v = middleZ->edge.first;
                    middleZ->edge.first = middleZ->edge.second;
                    middleZ->edge.second = v;

                    v = topZ->edge.first;
                    topZ->edge.first = topZ->edge.second;
                    topZ->edge.second = v;

                    v = bottomZ->edge.first;
                    bottomZ->edge.first = bottomZ->edge.second;
                    bottomZ->edge.second = v;
                }

                const int ele0[] = {middleZ->edge.first, topZ->id, bottomZ->id, bottomZ->edge.first};
                const int ele1[] = {middleZ->id, middleZ->edge.first, topZ->id, bottomZ->id};
                const int ele2[] = {topZ->id, topZ->edge.second, bottomZ->id, bottomZ->edge.first};
                const int ele3[] = {middleZ->id, topZ->id, topZ->edge.second, bottomZ->id};
                const int ele4[] = {middleZ->id, topZ->edge.second, middleZ->edge.second, bottomZ->id};

                const int ele0_boundary[] = {0, b[topZ->edge.second], b[middleZ->edge.second], 0};
                const int ele1_boundary[] = {0, 0, b[topZ->edge.second], b[bottomZ->edge.first]};
                const int ele2_boundary[] = {b[middleZ->edge.first], 0, b[middleZ->edge.second], 0};
                const int ele3_boundary[] = {0, 0, 0, b[bottomZ->edge.first]};
                const int ele4_boundary[] = {b[middleZ->edge.first], b[topZ->edge.second], 0, b[bottomZ->edge.first]};

                index_t ele1ID, ele2ID, ele3ID, ele4ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;
                ele4ID = ele1ID+3;

                def_ops->addNN(topZ->id, bottomZ->id, tid);
                def_ops->addNN(bottomZ->id, topZ->id, tid);

                def_ops->addNE_fix(middleZ->edge.first, ele1ID, tid);

                def_ops->remNE(middleZ->edge.second, eid, tid);
                def_ops->addNE_fix(middleZ->edge.second, ele4ID, tid);

                def_ops->remNE(topZ->edge.second, eid, tid);
                def_ops->addNE_fix(topZ->edge.second, ele2ID, tid);
                def_ops->addNE_fix(topZ->edge.second, ele3ID, tid);
                def_ops->addNE_fix(topZ->edge.second, ele4ID, tid);

                def_ops->addNE_fix(bottomZ->edge.first, ele2ID, tid);

                def_ops->addNE_fix(middleZ->id, ele1ID, tid);
                def_ops->addNE_fix(middleZ->id, ele3ID, tid);
                def_ops->addNE_fix(middleZ->id, ele4ID, tid);

                def_ops->addNE(topZ->id, eid, tid);
                def_ops->addNE_fix(topZ->id, ele1ID, tid);
                def_ops->addNE_fix(topZ->id, ele2ID, tid);
                def_ops->addNE_fix(topZ->id, ele3ID, tid);

                def_ops->addNE(bottomZ->id, eid, tid);
                def_ops->addNE_fix(bottomZ->id, ele1ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele2ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele3ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele4ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                append_element(ele4, ele4_boundary, tid);
                splitCnt[tid] += 4;
                break;
            }
            case 2: {
                // Case 3(c)(1)
                const int ele0[] = {middleZ->id, bottomZ->edge.first, middleZ->edge.first, topZ->id};
                const int ele1[] = {middleZ->id, bottomZ->edge.first, topZ->id, topZ->edge.second};
                const int ele2[] = {middleZ->id, bottomZ->id, bottomZ->edge.first, topZ->edge.second};
                const int ele3[] = {middleZ->id, middleZ->edge.second, bottomZ->id, topZ->edge.second};

                const int ele0_boundary[] = {b[middleZ->edge.second], b[bottomZ->edge.first], 0, b[topZ->edge.second]};
                const int ele1_boundary[] = {b[middleZ->edge.second], b[bottomZ->edge.first], 0, 0};
                const int ele2_boundary[] = {b[middleZ->edge.first], 0, 0, b[topZ->edge.second]};
                const int ele3_boundary[] = {b[middleZ->edge.first], 0, b[bottomZ->edge.first], b[topZ->edge.second]};

                index_t ele1ID, ele2ID, ele3ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;

                def_ops->addNE(middleZ->id, eid, tid);
                def_ops->addNE_fix(middleZ->id, ele1ID, tid);
                def_ops->addNE_fix(middleZ->id, ele2ID, tid);
                def_ops->addNE_fix(middleZ->id, ele3ID, tid);

                def_ops->remNE(middleZ->edge.second, eid, tid);
                def_ops->addNE_fix(middleZ->edge.second, ele3ID, tid);

                def_ops->remNE(topZ->edge.second, eid, tid);
                def_ops->addNE_fix(topZ->edge.second, ele1ID, tid);
                def_ops->addNE_fix(topZ->edge.second, ele2ID, tid);
                def_ops->addNE_fix(topZ->edge.second, ele3ID, tid);

                def_ops->addNE_fix(bottomZ->edge.first, ele1ID, tid);
                def_ops->addNE_fix(bottomZ->edge.first, ele2ID, tid);

                def_ops->addNE(topZ->id, eid, tid);
                def_ops->addNE_fix(topZ->id, ele1ID, tid);

                def_ops->addNE_fix(bottomZ->id, ele2ID, tid);
                def_ops->addNE_fix(bottomZ->id, ele3ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                splitCnt[tid] += 3;
                break;
            }
            default:
                break;
            }
        }
    }

    inline void refine3D_4(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        /*
         * There are 2 cases here:
         *
         * Case 4(a): Three split edges are on the same triangle.
         * Case 4(b): Each of the four triangles has exactly two split edges.
         */

        std::set<index_t> shared;
        for(int j=0; j<4; ++j) {
            for(int k=j+1; k<4; ++k) {
                index_t nid = splitEdges[j].connected(splitEdges[k]);
                if(nid>=0)
                    shared.insert(nid);
            }
        }
        size_t nshared = shared.size();
        assert(nshared==3 || nshared==4);

        if(nshared==3) {
            /*
             *************
             * Case 4(a) *
             *************
             */
            DirectedEdge<index_t>* p[4];
            int pos = 0;
            for(int j=0; j<4; ++j)
                if(shared.count(splitEdges[j].edge.first)>0 && shared.count(splitEdges[j].edge.second)>0)
                    p[pos++] = &splitEdges[j];
                else
                    p[3] = &splitEdges[j];

            assert(pos==3);

            // p[0], p[1] and p[2] point to the three split edges which
            // are on the same facet, p[3] points to the other split edge.

            // Re-arrange p[] so that p[0] points to the
            // split edge which is not connected to p[3].
            if(p[3]->connected(*p[0]) >= 0) {
                for(int j=1; j<3; ++j) {
                    if(p[3]->connected(*p[j]) < 0) {
                        DirectedEdge<index_t> *swap = p[j];
                        p[j] = p[0];
                        p[0] = swap;
                        break;
                    }
                }
            }

            // Re-arrange p[3] if necessary so that edge.first
            // is the vertex on the triangle with the 3 split edges.
            if(shared.count(p[3]->edge.first)==0) {
                index_t v = p[3]->edge.first;
                p[3]->edge.first = p[3]->edge.second;
                p[3]->edge.second = v;
            }

            // Same for p[1] and p[2]; make edge.first = p[3]->edge.first.
            for(int j=1; j<=2; ++j)
                if(p[j]->edge.first != p[3]->edge.first) {
                    assert(p[j]->edge.second == p[3]->edge.first);
                    p[j]->edge.second = p[j]->edge.first;
                    p[j]->edge.first = p[3]->edge.first;
                }

            /*
             * There are 3 sub-cases, depending on the way the trapezoids
             * on the facets between p[1]-p[3] and p[2]-p[3] were bisected.
             *
             * Case 4(a)(1): No diagonal involves p[3].
             * Case 4(a)(2): Only one diagonal involves p[3].
             * Case 4(a)(3): Both diagonals involve p[3].
             */

            std::vector< DirectedEdge<index_t> > diagonals;
            for(std::vector<index_t>::const_iterator it=_mesh->NNList[p[3]->id].begin();
                    it!=_mesh->NNList[p[3]->id].end(); ++it) {
                if(*it == p[1]->edge.second || *it == p[2]->edge.second) {
                    diagonals.push_back(DirectedEdge<index_t>(p[3]->id, *it));
                }
            }

            switch(diagonals.size()) {
            case 0: {
                // Case 4(a)(1)
                const int ele0[] = {p[0]->id, p[1]->edge.second, p[1]->id, p[3]->edge.second};
                const int ele1[] = {p[0]->id, p[1]->id, p[2]->id, p[3]->edge.second};
                const int ele2[] = {p[0]->id, p[2]->id, p[2]->edge.second, p[3]->edge.second};
                const int ele3[] = {p[1]->id, p[3]->id, p[2]->id, p[3]->edge.second};
                const int ele4[] = {p[1]->id, p[2]->id, p[3]->id, p[3]->edge.first};

                const int ele0_boundary[] = {b[p[2]->edge.second], 0, b[p[3]->edge.first], b[p[3]->edge.second]};
                const int ele1_boundary[] = {0, 0, 0, b[p[3]->edge.second]};
                const int ele2_boundary[] = {b[p[1]->edge.second], b[p[2]->edge.first], 0, b[p[3]->edge.second]};
                const int ele3_boundary[] = {b[p[1]->edge.second], 0, b[p[2]->edge.second], 0};
                const int ele4_boundary[] = {b[p[1]->edge.second], b[p[2]->edge.second], b[p[3]->edge.second], 0};

                index_t ele1ID, ele2ID, ele3ID, ele4ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;
                ele4ID = ele1ID+3;

                def_ops->remNE(p[2]->edge.first, eid, tid);
                def_ops->addNE_fix(p[2]->edge.first, ele4ID, tid);

                def_ops->remNE(p[2]->edge.second, eid, tid);
                def_ops->addNE_fix(p[2]->edge.second, ele2ID, tid);

                def_ops->addNE_fix(p[3]->edge.second, ele1ID, tid);
                def_ops->addNE_fix(p[3]->edge.second, ele2ID, tid);
                def_ops->addNE_fix(p[3]->edge.second, ele3ID, tid);

                def_ops->addNE(p[0]->id, eid, tid);
                def_ops->addNE_fix(p[0]->id, ele1ID, tid);
                def_ops->addNE_fix(p[0]->id, ele2ID, tid);

                def_ops->addNE(p[1]->id, eid, tid);
                def_ops->addNE_fix(p[1]->id, ele1ID, tid);
                def_ops->addNE_fix(p[1]->id, ele3ID, tid);
                def_ops->addNE_fix(p[1]->id, ele4ID, tid);

                def_ops->addNE_fix(p[2]->id, ele1ID, tid);
                def_ops->addNE_fix(p[2]->id, ele2ID, tid);
                def_ops->addNE_fix(p[2]->id, ele3ID, tid);
                def_ops->addNE_fix(p[2]->id, ele4ID, tid);

                def_ops->addNE_fix(p[3]->id, ele3ID, tid);
                def_ops->addNE_fix(p[3]->id, ele4ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                append_element(ele4, ele4_boundary, tid);
                splitCnt[tid] += 4;
                break;
            }
            case 1: {
                // Case 4(a)(2)

                // Swap p[1] and p[2] if necessary so that p[2]->edge.second
                // is the ending point of the diagonal bisecting the trapezoid.
                if(p[2]->edge.second != diagonals[0].edge.second) {
                    DirectedEdge<index_t> *swap = p[1];
                    p[1] = p[2];
                    p[2] = swap;
                }
                assert(p[2]->edge.second == diagonals[0].edge.second);

                const int ele0[] = {p[0]->id, p[1]->edge.second, p[1]->id, p[3]->edge.second};
                const int ele1[] = {p[0]->id, p[3]->id, p[2]->edge.second, p[3]->edge.second};
                const int ele2[] = {p[0]->id, p[1]->id, p[3]->id, p[3]->edge.second};
                const int ele3[] = {p[0]->id, p[3]->id, p[2]->id, p[2]->edge.second};
                const int ele4[] = {p[0]->id, p[1]->id, p[2]->id, p[3]->id};
                const int ele5[] = {p[2]->id, p[3]->id, p[1]->id, p[3]->edge.first};

                const int ele0_boundary[] = {b[p[2]->edge.second], 0, b[p[1]->edge.first], b[p[3]->edge.second]};
                const int ele1_boundary[] = {b[p[1]->edge.second], b[p[3]->edge.first], 0, 0};
                const int ele2_boundary[] = {b[p[2]->edge.second], 0, 0, 0};
                const int ele3_boundary[] = {b[p[1]->edge.second], b[p[3]->edge.second], 0, 0};
                const int ele4_boundary[] = {0, 0, 0, b[p[3]->edge.second]};
                const int ele5_boundary[] = {b[p[2]->edge.second], b[p[3]->edge.second], b[p[1]->edge.second], 0};

                index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;
                ele4ID = ele1ID+3;
                ele5ID = ele1ID+4;

                def_ops->addNN(p[0]->id, p[3]->id, tid);
                def_ops->addNN(p[3]->id, p[0]->id, tid);

                def_ops->remNE(p[2]->edge.first, eid, tid);
                def_ops->addNE_fix(p[2]->edge.first, ele5ID, tid);

                def_ops->remNE(p[2]->edge.second, eid, tid);
                def_ops->addNE_fix(p[2]->edge.second, ele1ID, tid);
                def_ops->addNE_fix(p[2]->edge.second, ele3ID, tid);

                def_ops->addNE_fix(p[3]->edge.second, ele1ID, tid);
                def_ops->addNE_fix(p[3]->edge.second, ele2ID, tid);

                def_ops->addNE(p[0]->id, eid, tid);
                def_ops->addNE_fix(p[0]->id, ele1ID, tid);
                def_ops->addNE_fix(p[0]->id, ele2ID, tid);
                def_ops->addNE_fix(p[0]->id, ele3ID, tid);
                def_ops->addNE_fix(p[0]->id, ele4ID, tid);

                def_ops->addNE(p[1]->id, eid, tid);
                def_ops->addNE_fix(p[1]->id, ele2ID, tid);
                def_ops->addNE_fix(p[1]->id, ele4ID, tid);
                def_ops->addNE_fix(p[1]->id, ele5ID, tid);

                def_ops->addNE_fix(p[2]->id, ele3ID, tid);
                def_ops->addNE_fix(p[2]->id, ele4ID, tid);
                def_ops->addNE_fix(p[2]->id, ele5ID, tid);

                def_ops->addNE_fix(p[3]->id, ele1ID, tid);
                def_ops->addNE_fix(p[3]->id, ele2ID, tid);
                def_ops->addNE_fix(p[3]->id, ele3ID, tid);
                def_ops->addNE_fix(p[3]->id, ele4ID, tid);
                def_ops->addNE_fix(p[3]->id, ele5ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                append_element(ele4, ele4_boundary, tid);
                append_element(ele5, ele5_boundary, tid);
                splitCnt[tid] += 5;
                break;
            }
            case 2: {
                // Case 4(a)(3)
                const int ele0[] = {p[1]->edge.first, p[1]->id, p[2]->id, p[3]->id};
                const int ele1[] = {p[3]->id, p[1]->edge.second, p[0]->id, p[3]->edge.second};
                const int ele2[] = {p[3]->id, p[0]->id, p[2]->edge.second, p[3]->edge.second};
                const int ele3[] = {p[1]->id, p[1]->edge.second, p[0]->id, p[3]->id};
                const int ele4[] = {p[1]->id, p[0]->id, p[2]->id, p[3]->id};
                const int ele5[] = {p[2]->id, p[3]->id, p[0]->id, p[2]->edge.second};

                const int ele0_boundary[] = {0, b[p[1]->edge.second], b[p[2]->edge.second], b[p[3]->edge.second]};
                const int ele1_boundary[] = {b[p[3]->edge.first], 0, b[p[2]->edge.second], 0};
                const int ele2_boundary[] = {b[p[3]->edge.first], b[p[1]->edge.second], 0, 0};
                const int ele3_boundary[] = {0, 0, b[p[2]->edge.second], b[p[3]->edge.second]};
                const int ele4_boundary[] = {0, 0, 0, b[p[3]->edge.second]};
                const int ele5_boundary[] = {0, b[p[3]->edge.second], b[p[1]->edge.second], 0};

                index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID;
                ele1ID = splitCnt[tid];
                ele2ID = ele1ID+1;
                ele3ID = ele1ID+2;
                ele4ID = ele1ID+3;
                ele5ID = ele1ID+4;

                def_ops->addNN(p[0]->id, p[3]->id, tid);
                def_ops->addNN(p[3]->id, p[0]->id, tid);


                def_ops->remNE(p[1]->edge.second, eid, tid);
                def_ops->addNE_fix(p[1]->edge.second, ele1ID, tid);
                def_ops->addNE_fix(p[1]->edge.second, ele3ID, tid);

                def_ops->remNE(p[2]->edge.second, eid, tid);
                def_ops->addNE_fix(p[2]->edge.second, ele2ID, tid);
                def_ops->addNE_fix(p[2]->edge.second, ele5ID, tid);

                def_ops->remNE(p[3]->edge.second, eid, tid);
                def_ops->addNE_fix(p[3]->edge.second, ele1ID, tid);
                def_ops->addNE_fix(p[3]->edge.second, ele2ID, tid);

                def_ops->addNE_fix(p[0]->id, ele1ID, tid);
                def_ops->addNE_fix(p[0]->id, ele2ID, tid);
                def_ops->addNE_fix(p[0]->id, ele3ID, tid);
                def_ops->addNE_fix(p[0]->id, ele4ID, tid);
                def_ops->addNE_fix(p[0]->id, ele5ID, tid);

                def_ops->addNE(p[1]->id, eid, tid);
                def_ops->addNE_fix(p[1]->id, ele3ID, tid);
                def_ops->addNE_fix(p[1]->id, ele4ID, tid);

                def_ops->addNE(p[2]->id, eid, tid);
                def_ops->addNE_fix(p[2]->id, ele4ID, tid);
                def_ops->addNE_fix(p[2]->id, ele5ID, tid);

                def_ops->addNE(p[3]->id, eid, tid);
                def_ops->addNE_fix(p[3]->id, ele1ID, tid);
                def_ops->addNE_fix(p[3]->id, ele2ID, tid);
                def_ops->addNE_fix(p[3]->id, ele3ID, tid);
                def_ops->addNE_fix(p[3]->id, ele4ID, tid);
                def_ops->addNE_fix(p[3]->id, ele5ID, tid);

                replace_element(eid, ele0, ele0_boundary);
                append_element(ele1, ele1_boundary, tid);
                append_element(ele2, ele2_boundary, tid);
                append_element(ele3, ele3_boundary, tid);
                append_element(ele4, ele4_boundary, tid);
                append_element(ele5, ele5_boundary, tid);
                splitCnt[tid] += 5;
                break;
            }
            default:
                break;
            }
        } else {
            /*
             *************
             * Case 4(b) *
             *************
             */

            // In this case, the element is split into two wedges.

            // Find the top left, top right, bottom left and
            // bottom right split edges, as depicted in the paper.
            DirectedEdge<index_t> *tr, *tl, *br, *bl;
            tl = &splitEdges[0];

            // Find top right
            for(int j=1; j<4; ++j) {
                if(splitEdges[j].contains(tl->edge.first)) {
                    tr = &splitEdges[j];
                    break;
                }
            }
            // Re-arrange tr so that tl->edge.first == tr->edge.first
            if(tr->edge.first != tl->edge.first) {
                assert(tr->edge.second == tl->edge.first);
                tr->edge.second = tr->edge.first;
                tr->edge.first = tl->edge.first;
            }

            // Find bottom left
            for(int j=1; j<4; ++j) {
                if(splitEdges[j].contains(tl->edge.second)) {
                    bl = &splitEdges[j];
                    break;
                }
            }
            // Re-arrange bl so that tl->edge.second == bl->edge.second
            if(bl->edge.second != tl->edge.second) {
                assert(bl->edge.first == tl->edge.second);
                bl->edge.first = bl->edge.second;
                bl->edge.second = tl->edge.second;
            }

            // Find bottom right
            for(int j=1; j<4; ++j) {
                if(splitEdges[j].contains(bl->edge.first) && splitEdges[j].contains(tr->edge.second)) {
                    br = &splitEdges[j];
                    break;
                }
            }
            // Re-arrange br so that bl->edge.first == br->edge.first
            if(br->edge.first != bl->edge.first) {
                assert(br->edge.second == bl->edge.first);
                br->edge.second = br->edge.first;
                br->edge.first = bl->edge.first;
            }

            assert(tr->edge.second == br->edge.second);

            // Find how the trapezoids have been split
            DirectedEdge<index_t> bw1, bw2, tw1, tw2;
            std::vector<index_t>::const_iterator p;

            // For the bottom wedge:
            // 1) From tl->id to tr->edge.second or from tr->id to tl->edge.second?
            // 2) From bl->id to br->edge.second or from br->id to bl->edge.second?
            // For the top wedge:
            // 1) From tl->id to bl->edge.first or from bl->id to tl->edge.first?
            // 2) From tr->id to br->edge.first or from br->id to tr->edge.first?

            p = std::find(_mesh->NNList[tl->id].begin(), _mesh->NNList[tl->id].end(), tr->edge.second);
            if(p != _mesh->NNList[tl->id].end()) {
                bw1.edge.first = tl->id;
                bw1.edge.second = tr->edge.second;
            } else {
                bw1.edge.first = tr->id;
                bw1.edge.second = tl->edge.second;
            }

            p = std::find(_mesh->NNList[bl->id].begin(), _mesh->NNList[bl->id].end(), br->edge.second);
            if(p != _mesh->NNList[bl->id].end()) {
                bw2.edge.first = bl->id;
                bw2.edge.second = br->edge.second;
            } else {
                bw2.edge.first = br->id;
                bw2.edge.second = bl->edge.second;
            }

            p = std::find(_mesh->NNList[tl->id].begin(), _mesh->NNList[tl->id].end(), bl->edge.first);
            if(p != _mesh->NNList[tl->id].end()) {
                tw1.edge.first = tl->id;
                tw1.edge.second = bl->edge.first;
            } else {
                tw1.edge.first = bl->id;
                tw1.edge.second = tl->edge.first;
            }

            p = std::find(_mesh->NNList[tr->id].begin(), _mesh->NNList[tr->id].end(), br->edge.first);
            if(p != _mesh->NNList[tr->id].end()) {
                tw2.edge.first = tr->id;
                tw2.edge.second = br->edge.first;
            } else {
                tw2.edge.first = br->id;
                tw2.edge.second = tr->edge.first;
            }

            // If bw1 and bw2 are connected, then the third quadrilateral
            // can be split whichever way we like. Otherwise, we want to
            // choose the diagonal which will lead to a 1:3 wedge split.
            // Same for tw1 and tw2.

            DirectedEdge<index_t> bw, tw;
            bool flex_bottom, flex_top;

            if(bw1.connected(bw2) >= 0) {
                flex_bottom = true;
            } else {
                flex_bottom = false;
                bw.edge.first = bw1.edge.first;
                bw.edge.second = bw2.edge.first;
                assert(bw.connected(bw1) >= 0 && bw.connected(bw2) >= 0);
            }
            if(tw1.connected(tw2) >= 0) {
                flex_top = true;
            } else {
                flex_top = false;
                tw.edge.first = tw1.edge.first;
                tw.edge.second = tw2.edge.first;
                assert(tw.connected(tw1) >= 0 && tw.connected(tw2) >= 0);
            }

            DirectedEdge<index_t> diag;
            if(flex_top && !flex_bottom) {
                // Choose the diagonal which is the preferred one for the bottom wedge
                diag.edge.first = bw.edge.first;
                diag.edge.second = bw.edge.second;
            } else if(!flex_top && flex_bottom) {
                // Choose the diagonal which is the preferred one for the top wedge
                diag.edge.first = tw.edge.first;
                diag.edge.second = tw.edge.second;
            } else {
                if(flex_top && flex_bottom) {
                    // Choose the shortest diagonal
                    real_t ldiag1 = _mesh->calc_edge_length(tl->id, br->id);
                    real_t ldiag2 = _mesh->calc_edge_length(bl->id, tr->id);

                    if(ldiag1 < ldiag2) {
                        diag.edge.first = tl->id;
                        diag.edge.second = br->id;
                    } else {
                        diag.edge.first = bl->id;
                        diag.edge.second = tr->id;
                    }
                } else {
                    // If we reach this point, it means we are not
                    // flexible in any of the diagonals. If we are
                    // lucky enough and bw==tw, then diag=bw.
                    if(bw.contains(tw.edge.first) && bw.contains(tw.edge.second)) {
                        diag.edge.first = bw.edge.first;
                        diag.edge.second = bw.edge.second;
                    } else {
                        // Choose the shortest diagonal
                        real_t ldiag1 = _mesh->calc_edge_length(tl->id, br->id);
                        real_t ldiag2 = _mesh->calc_edge_length(bl->id, tr->id);

                        if(ldiag1 < ldiag2) {
                            diag.edge.first = tl->id;
                            diag.edge.second = br->id;
                        } else {
                            diag.edge.first = bl->id;
                            diag.edge.second = tr->id;
                        }
                    }
                }
            }

            def_ops->addNN(diag.edge.first, diag.edge.second, tid);
            def_ops->addNN(diag.edge.second, diag.edge.first, tid);

            // At this point, we have identified the wedges and how their sides
            // have been split, so we can proceed to the actual refinement.
            index_t top_triangle[3];
            index_t bottom_triangle[3];
            int bwedge[5];

            // Bottom wedge
            top_triangle[0] = tr->id;
            top_triangle[1] = tr->edge.second;
            top_triangle[2] = br->id;
            bottom_triangle[0] = tl->id;
            bottom_triangle[1] = tl->edge.second;
            bottom_triangle[2] = bl->id;
            bwedge[0] = b[bl->edge.first];
            bwedge[1] = b[tl->edge.first];
            bwedge[2] = 0;
            bwedge[3] = b[tl->edge.second];
            bwedge[4] = b[tr->edge.second];
            refine_wedge(top_triangle, bottom_triangle, bwedge, &diag, eid, tid);

            // Top wedge
            top_triangle[0] = tl->id;
            top_triangle[1] = tl->edge.first;
            top_triangle[2] = tr->id;
            bottom_triangle[0] = bl->id;
            bottom_triangle[1] = bl->edge.first;
            bottom_triangle[2] = br->id;
            bwedge[0] = b[tr->edge.second];
            bwedge[1] = b[tl->edge.second];
            bwedge[2] = 0;
            bwedge[3] = b[bl->edge.first];
            bwedge[4] = b[tl->edge.first];
            // Flip diag
            index_t swap = diag.edge.first;
            diag.edge.first = diag.edge.second;
            diag.edge.second = swap;
            refine_wedge(top_triangle, bottom_triangle, bwedge, &diag, eid, tid);

            // Remove parent element
            for(size_t j=0; j<nloc; ++j)
                def_ops->remNE(n[j], eid, tid);
            _mesh->_ENList[eid*nloc] = -1;
        }
    }

    inline void refine3D_5(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        // Find the unsplit edge
        int adj_cnt[] = {0, 0, 0, 0};
        for(int j=0; j<nloc; ++j) {
            for(int k=0; k<5; ++k)
                if(splitEdges[k].contains(n[j]))
                    ++adj_cnt[j];
        }

        // Vertices of the unsplit edge are adjacent to 2 split edges;
        // the other vertices are adjacent to 3 split edges.
        index_t ue[2];
        int pos=0;
        for(int j=0; j<nloc; ++j)
            if(adj_cnt[j] == 2)
                ue[pos++] = n[j];

        // Find the opposite edge
        DirectedEdge<index_t> *oe;
        for(int k=0; k<5; ++k)
            if(!splitEdges[k].contains(ue[0]) && !splitEdges[k].contains(ue[1])) {
                // Swap splitEdges[k] with splitEdges[4]
                if(k!=4) {
                    DirectedEdge<index_t> swap = splitEdges[4];
                    splitEdges[4] = splitEdges[k];
                    splitEdges[k] = swap;
                }
                oe = &splitEdges[4];
                break;
            }

        // Like in 4(b), find tl, tr, bl and br
        DirectedEdge<index_t> *tr, *tl, *br, *bl;
        tl = &splitEdges[0];

        // Flip tl if necessary so that tl->edge.first is part of the unsplit edge
        if(oe->contains(tl->edge.first)) {
            index_t swap = tl->edge.second;
            tl->edge.second = tl->edge.first;
            tl->edge.first = swap;
        }

        // Find top right
        for(int j=1; j<4; ++j) {
            if(splitEdges[j].contains(tl->edge.first)) {
                tr = &splitEdges[j];
                break;
            }
        }
        // Re-arrange tr so that tl->edge.first == tr->edge.first
        if(tr->edge.first != tl->edge.first) {
            assert(tr->edge.second == tl->edge.first);
            tr->edge.second = tr->edge.first;
            tr->edge.first = tl->edge.first;
        }

        // Find bottom left
        for(int j=1; j<4; ++j) {
            if(splitEdges[j].contains(tl->edge.second)) {
                bl = &splitEdges[j];
                break;
            }
        }
        // Re-arrange bl so that tl->edge.second == bl->edge.second
        if(bl->edge.second != tl->edge.second) {
            assert(bl->edge.first == tl->edge.second);
            bl->edge.first = bl->edge.second;
            bl->edge.second = tl->edge.second;
        }

        // Find bottom right
        for(int j=1; j<4; ++j) {
            if(splitEdges[j].contains(bl->edge.first) && splitEdges[j].contains(tr->edge.second)) {
                br = &splitEdges[j];
                break;
            }
        }
        // Re-arrange br so that bl->edge.first == br->edge.first
        if(br->edge.first != bl->edge.first) {
            assert(br->edge.second == bl->edge.first);
            br->edge.second = br->edge.first;
            br->edge.first = bl->edge.first;
        }

        assert(tr->edge.second == br->edge.second);
        assert(oe->contains(tl->edge.second) && oe->contains(tr->edge.second));

        // Find how the trapezoids have been split:
        // 1) From tl->id to bl->edge.first or from bl->id to tl->edge.first?
        // 2) From tr->id to br->edge.first or from br->id to tr->edge.first?
        DirectedEdge<index_t> q1, q2;
        std::vector<index_t>::const_iterator p;

        p = std::find(_mesh->NNList[tl->id].begin(), _mesh->NNList[tl->id].end(), bl->edge.first);
        if(p != _mesh->NNList[tl->id].end()) {
            q1.edge.first = tl->id;
            q1.edge.second = bl->edge.first;
        } else {
            q1.edge.first = bl->id;
            q1.edge.second = tl->edge.first;
        }

        p = std::find(_mesh->NNList[tr->id].begin(), _mesh->NNList[tr->id].end(), br->edge.first);
        if(p != _mesh->NNList[tr->id].end()) {
            q2.edge.first = tr->id;
            q2.edge.second = br->edge.first;
        } else {
            q2.edge.first = br->id;
            q2.edge.second = tr->edge.first;
        }

        DirectedEdge<index_t> diag, cross_diag;
        if(q1.connected(q2) >= 0) {
            // We are flexible in choosing how the third quadrilateral
            // will be split and we will choose the shortest diagonal.
            real_t ldiag1 = _mesh->calc_edge_length(tl->id, br->id);
            real_t ldiag2 = _mesh->calc_edge_length(bl->id, tr->id);

            if(ldiag1 < ldiag2) {
                diag.edge.first = br->id;
                diag.edge.second = tl->id;
                cross_diag.edge.first = tr->id;
                cross_diag.edge.second = bl->id;
            } else {
                diag.edge.first = bl->id;
                diag.edge.second = tr->id;
                cross_diag.edge.first = tl->id;
                cross_diag.edge.second = br->id;
            }
        } else {
            // We will choose the diagonal which leads to a 1:3 wedge refinement.
            if(q1.edge.first == bl->id) {
                assert(q2.edge.first == tr->id);
                diag.edge.first = q1.edge.first;
                diag.edge.second = q2.edge.first;
            } else {
                assert(q2.edge.first == br->id);
                diag.edge.first = q2.edge.first;
                diag.edge.second = q1.edge.first;
            }
            assert((diag.edge.first==bl->id && diag.edge.second==tr->id)
                   || (diag.edge.first==br->id && diag.edge.second==tl->id));

            cross_diag.edge.first = (diag.edge.second == tr->id ? tl->id : tr->id);
            cross_diag.edge.second = (diag.edge.first == br->id ? bl->id : br->id);
            assert((cross_diag.edge.first==tl->id && cross_diag.edge.second==br->id)
                   || (cross_diag.edge.first==tr->id && cross_diag.edge.second==bl->id));
        }

        index_t bottom_triangle[] = {br->id, br->edge.first, bl->id};
        index_t top_triangle[] = {tr->id, tr->edge.first, tl->id};
        int bwedge[] = {b[bl->edge.second], b[br->edge.second], 0, b[bl->edge.first], b[tl->edge.first]};
        refine_wedge(top_triangle, bottom_triangle, bwedge, &diag, eid, tid);

        const int ele0[] = {tl->edge.second, bl->id, tl->id, oe->id};
        const int ele1[] = {tr->edge.second, tr->id, br->id, oe->id};
        const int ele2[] = {diag.edge.first, cross_diag.edge.first, diag.edge.second, oe->id};
        const int ele3[] = {diag.edge.first, diag.edge.second, cross_diag.edge.second, oe->id};

        const int ele0_boundary[] = {0, b[bl->edge.first], b[tl->edge.first], b[tr->edge.second]};
        const int ele1_boundary[] = {0, b[tl->edge.first], b[bl->edge.first], b[tl->edge.second]};
        const int ele2_boundary[] = {b[bl->edge.first], 0, 0, 0};
        const int ele3_boundary[] = {0, b[tl->edge.first], 0, 0};

        index_t ele1ID, ele2ID, ele3ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;

        def_ops->addNN(diag.edge.first, diag.edge.second, tid);
        def_ops->addNN(diag.edge.second, diag.edge.first, tid);

        def_ops->remNE(tr->edge.first, eid, tid);
        def_ops->remNE(br->edge.first, eid, tid);
        def_ops->remNE(tr->edge.second, eid, tid);

        def_ops->addNE(tl->id, eid, tid);
        def_ops->addNE(bl->id, eid, tid);
        def_ops->addNE(oe->id, eid, tid);

        def_ops->addNE_fix(tr->edge.second, ele1ID, tid);
        def_ops->addNE_fix(tr->id, ele1ID, tid);
        def_ops->addNE_fix(br->id, ele1ID, tid);
        def_ops->addNE_fix(oe->id, ele1ID, tid);

        def_ops->addNE_fix(diag.edge.first, ele2ID, tid);
        def_ops->addNE_fix(diag.edge.second, ele2ID, tid);
        def_ops->addNE_fix(cross_diag.edge.first, ele2ID, tid);
        def_ops->addNE_fix(oe->id, ele2ID, tid);

        def_ops->addNE_fix(diag.edge.first, ele3ID, tid);
        def_ops->addNE_fix(diag.edge.second, ele3ID, tid);
        def_ops->addNE_fix(cross_diag.edge.second, ele3ID, tid);
        def_ops->addNE_fix(oe->id, ele3ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        splitCnt[tid] += 3;
    }

    inline void refine3D_6(std::vector< DirectedEdge<index_t> >& splitEdges, int eid, int tid)
    {
        const int *n=_mesh->get_element(eid);
        const int *boundary=&(_mesh->boundary[eid*nloc]);

        boundary_t b;
        for(int j=0; j<nloc; ++j)
            b[n[j]] = boundary[j];

        /*
         * There is an internal edge in this case. We choose the shortest among:
         * a) newVertex[0] - newVertex[5]
         * b) newVertex[1] - newVertex[4]
         * c) newVertex[2] - newVertex[3]
         */

        real_t ldiag0 = _mesh->calc_edge_length(splitEdges[0].id, splitEdges[5].id);
        real_t ldiag1 = _mesh->calc_edge_length(splitEdges[1].id, splitEdges[4].id);
        real_t ldiag2 = _mesh->calc_edge_length(splitEdges[2].id, splitEdges[3].id);

        std::vector<index_t> internal(2);
        std::vector<index_t> opposite(4);
        std::vector<int> bndr(4);
        if(ldiag0 < ldiag1 && ldiag0 < ldiag2) {
            // 0-5
            internal[0] = splitEdges[5].id;
            internal[1] = splitEdges[0].id;
            opposite[0] = splitEdges[3].id;
            opposite[1] = splitEdges[4].id;
            opposite[2] = splitEdges[2].id;
            opposite[3] = splitEdges[1].id;
            bndr[0] = boundary[0];
            bndr[1] = boundary[2];
            bndr[2] = boundary[1];
            bndr[3] = boundary[3];
        } else if(ldiag1 < ldiag2) {
            // 1-4
            internal[0] = splitEdges[1].id;
            internal[1] = splitEdges[4].id;
            opposite[0] = splitEdges[0].id;
            opposite[1] = splitEdges[3].id;
            opposite[2] = splitEdges[5].id;
            opposite[3] = splitEdges[2].id;
            bndr[0] = boundary[3];
            bndr[1] = boundary[0];
            bndr[2] = boundary[1];
            bndr[3] = boundary[2];
        } else {
            // 2-3
            internal[0] = splitEdges[3].id;
            internal[1] = splitEdges[2].id;
            opposite[0] = splitEdges[4].id;
            opposite[1] = splitEdges[5].id;
            opposite[2] = splitEdges[1].id;
            opposite[3] = splitEdges[0].id;
            bndr[0] = boundary[0];
            bndr[1] = boundary[1];
            bndr[2] = boundary[3];
            bndr[3] = boundary[2];
        }

        const int ele0[] = {n[0], splitEdges[0].id, splitEdges[1].id, splitEdges[2].id};
        const int ele1[] = {n[1], splitEdges[3].id, splitEdges[0].id, splitEdges[4].id};
        const int ele2[] = {n[2], splitEdges[1].id, splitEdges[3].id, splitEdges[5].id};
        const int ele3[] = {n[3], splitEdges[2].id, splitEdges[4].id, splitEdges[5].id};
        const int ele4[] = {internal[0], opposite[0], opposite[1], internal[1]};
        const int ele5[] = {internal[0], opposite[1], opposite[2], internal[1]};
        const int ele6[] = {internal[0], opposite[2], opposite[3], internal[1]};
        const int ele7[] = {internal[0], opposite[3], opposite[0], internal[1]};

        const int ele0_boundary[] = {0, boundary[1], boundary[2], boundary[3]};
        const int ele1_boundary[] = {0, boundary[2], boundary[0], boundary[3]};
        const int ele2_boundary[] = {0, boundary[0], boundary[1], boundary[3]};
        const int ele3_boundary[] = {0, boundary[0], boundary[1], boundary[2]};
        const int ele4_boundary[] = {0, 0, 0, bndr[0]};
        const int ele5_boundary[] = {bndr[1], 0, 0, 0};
        const int ele6_boundary[] = {0, 0, 0, bndr[2]};
        const int ele7_boundary[] = {bndr[3], 0, 0, 0};

        index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID, ele6ID, ele7ID;
        ele1ID = splitCnt[tid];
        ele2ID = ele1ID+1;
        ele3ID = ele1ID+2;
        ele4ID = ele1ID+3;
        ele5ID = ele1ID+4;
        ele6ID = ele1ID+5;
        ele7ID = ele1ID+6;

        def_ops->addNN(internal[0], internal[1], tid);
        def_ops->addNN(internal[1], internal[0], tid);

        def_ops->remNE(n[1], eid, tid);
        def_ops->addNE_fix(n[1], ele1ID, tid);
        def_ops->remNE(n[2], eid, tid);
        def_ops->addNE_fix(n[2], ele2ID, tid);
        def_ops->remNE(n[3], eid, tid);
        def_ops->addNE_fix(n[3], ele3ID, tid);

        def_ops->addNE(splitEdges[0].id, eid, tid);
        def_ops->addNE_fix(splitEdges[0].id, ele1ID, tid);

        def_ops->addNE(splitEdges[1].id, eid, tid);
        def_ops->addNE_fix(splitEdges[1].id, ele2ID, tid);

        def_ops->addNE(splitEdges[2].id, eid, tid);
        def_ops->addNE_fix(splitEdges[2].id, ele3ID, tid);

        def_ops->addNE_fix(splitEdges[3].id, ele1ID, tid);
        def_ops->addNE_fix(splitEdges[3].id, ele2ID, tid);

        def_ops->addNE_fix(splitEdges[4].id, ele1ID, tid);
        def_ops->addNE_fix(splitEdges[4].id, ele3ID, tid);

        def_ops->addNE_fix(splitEdges[5].id, ele2ID, tid);
        def_ops->addNE_fix(splitEdges[5].id, ele3ID, tid);

        def_ops->addNE_fix(internal[0], ele4ID, tid);
        def_ops->addNE_fix(internal[0], ele5ID, tid);
        def_ops->addNE_fix(internal[0], ele6ID, tid);
        def_ops->addNE_fix(internal[0], ele7ID, tid);
        def_ops->addNE_fix(internal[1], ele4ID, tid);
        def_ops->addNE_fix(internal[1], ele5ID, tid);
        def_ops->addNE_fix(internal[1], ele6ID, tid);
        def_ops->addNE_fix(internal[1], ele7ID, tid);

        def_ops->addNE_fix(opposite[0], ele4ID, tid);
        def_ops->addNE_fix(opposite[1], ele4ID, tid);
        def_ops->addNE_fix(opposite[1], ele5ID, tid);
        def_ops->addNE_fix(opposite[2], ele5ID, tid);
        def_ops->addNE_fix(opposite[2], ele6ID, tid);
        def_ops->addNE_fix(opposite[3], ele6ID, tid);
        def_ops->addNE_fix(opposite[3], ele7ID, tid);
        def_ops->addNE_fix(opposite[0], ele7ID, tid);

        replace_element(eid, ele0, ele0_boundary);
        append_element(ele1, ele1_boundary, tid);
        append_element(ele2, ele2_boundary, tid);
        append_element(ele3, ele3_boundary, tid);
        append_element(ele4, ele4_boundary, tid);
        append_element(ele5, ele5_boundary, tid);
        append_element(ele6, ele6_boundary, tid);
        append_element(ele7, ele7_boundary, tid);
        splitCnt[tid] += 7;
    }

    inline void refine_wedge(const index_t top_triangle[], const index_t bottom_triangle[],
                             const int bndr[], DirectedEdge<index_t>* third_diag, int eid, int tid)
    {
        /*
         * bndr[] must contain the boundary values for each side of the wedge:
         * bndr[0], bndr[1] and bndr[2]: Boundary values of Side0, Side1 and Side2
         * bndr[3]: Boundary value of top triangle
         * bndr[4]: Boundary value of bottom triangle
         */

        /*
         * third_diag is used optionally if we need to define manually what the
         * third diagonal is (used in cases 4(b) and 5). It needs to be a directed
         * edge from the bottom triangle to the top triangle and be on Side2.
         */
        if(third_diag != NULL) {
            for(int j=0; j<3; ++j) {
                if(third_diag->edge.first == bottom_triangle[j] || third_diag->edge.second == top_triangle[j]) {
                    break;
                } else if(third_diag->edge.first == top_triangle[j] || third_diag->edge.second == bottom_triangle[j]) {
                    index_t swap = third_diag->edge.first;
                    third_diag->edge.first = third_diag->edge.second;
                    third_diag->edge.second = swap;
                    break;
                }
            }
            assert((third_diag->edge.first == bottom_triangle[2] && third_diag->edge.second == top_triangle[0]) ||
                   (third_diag->edge.first == bottom_triangle[0] && third_diag->edge.second == top_triangle[2]));
        }

        /*
         * For each quadrilateral side of the wedge find
         * the diagonal which has bisected the wedge side.
         * Side0: bottom[0] - bottom[1] - top[1] - top[0]
         * Side1: bottom[1] - bottom[2] - top[2] - top[1]
         * Side2: bottom[2] - bottom[0] - top[0] - top[2]
         */
        std::vector< DirectedEdge<index_t> > diagonals, ghostDiagonals;
        for(int j=0; j<3; ++j) {
            bool fwd_connected;
            if(j==2 && third_diag != NULL) {
                fwd_connected = (bottom_triangle[j] == third_diag->edge.first ? true : false);
            } else {
                std::vector<index_t>::const_iterator p = std::find(_mesh->NNList[bottom_triangle[j]].begin(),
                        _mesh->NNList[bottom_triangle[j]].end(), top_triangle[(j+1)%3]);
                fwd_connected = (p != _mesh->NNList[bottom_triangle[j]].end() ? true : false);
            }
            if(fwd_connected) {
                diagonals.push_back(DirectedEdge<index_t>(bottom_triangle[j], top_triangle[(j+1)%3]));
                ghostDiagonals.push_back(DirectedEdge<index_t>(bottom_triangle[(j+1)%3], top_triangle[j]));
            } else {
                diagonals.push_back(DirectedEdge<index_t>(bottom_triangle[(j+1)%3], top_triangle[j]));
                ghostDiagonals.push_back(DirectedEdge<index_t>(bottom_triangle[j], top_triangle[(j+1)%3]));
            }
        }

        // Determine how the wedge will be split
        std::vector<index_t> diag_shared;
        for(int j=0; j<3; j++) {
            index_t nid = diagonals[j].connected(diagonals[(j+1)%3]);
            if(nid>=0)
                diag_shared.push_back(nid);
        }

        if(!diag_shared.empty()) {
            /*
             ***************
             * Case 1-to-3 *
             ***************
             */

            assert(diag_shared.size() == 2);

            // Here we can subdivide the wedge into 3 tetrahedra.

            // Find the "middle" diagonal, i.e. the one which
            // consists of the two vertices in diag_shared.
            int middle;
            index_t non_shared_top=-1, non_shared_bottom=-1;
            for(int j=0; j<3; ++j) {
                if(diagonals[j].contains(diag_shared[0]) && diagonals[j].contains(diag_shared[1])) {
                    middle = j;
                    for(int k=0; k<2; ++k) {
                        if(diagonals[(j+k+1)%3].edge.first != diag_shared[0] && diagonals[(j+k+1)%3].edge.first != diag_shared[1])
                            non_shared_bottom = diagonals[(j+k+1)%3].edge.first;
                        else
                            non_shared_top = diagonals[(j+k+1)%3].edge.second;
                    }
                    break;
                }
            }
            assert(non_shared_top >= 0 && non_shared_bottom >= 0);

            /*
             * 2 elements are formed by the three vertices of two connected
             * diagonals plus a fourth vertex which is the one vertex of top/
             * bottom triangle which does not belong to any diagonal.
             *
             * 1 element is formed by the four vertices of two disjoint diagonals.
             */
            index_t v_top, v_bottom;

            // diagonals[middle].edge.first is always one of the bottom vertices
            // diagonals[middle].edge.second is always one of the top vertices

            for(int j=0; j<3; ++j) {
                if(top_triangle[j]!=diagonals[middle].edge.second && top_triangle[j]!=non_shared_top) {
                    v_top = top_triangle[j];
                    assert(bottom_triangle[j] == diagonals[middle].edge.first);
                    break;
                }
            }

            for(int j=0; j<3; ++j) {
                if(bottom_triangle[j]!=diagonals[middle].edge.first && bottom_triangle[j]!=non_shared_bottom) {
                    v_bottom = bottom_triangle[j];
                    assert(top_triangle[j] == diagonals[middle].edge.second);
                    break;
                }
            }

            const int ele1[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, non_shared_top, v_top};
            const int ele2[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, v_bottom, non_shared_bottom};
            const int ele3[] = {diagonals[middle].edge.first, diagonals[middle].edge.second, non_shared_top, non_shared_bottom};

            int bv_bottom, bnsb, bfirst;
            for(int j=0; j<3; ++j) {
                if(v_bottom == bottom_triangle[j]) {
                    bv_bottom = bndr[(j+1)%3];
                } else if(non_shared_bottom == bottom_triangle[j]) {
                    bnsb = bndr[(j+1)%3];
                } else {
                    bfirst = bndr[(j+1)%3];
                }
            }

            const int ele1_boundary[] = {bndr[3], bv_bottom, bnsb, 0};
            const int ele2_boundary[] = {bfirst, bndr[4], 0, bnsb};
            const int ele3_boundary[] = {bfirst, bv_bottom, 0, 0};

            index_t ele1ID, ele2ID, ele3ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;

            def_ops->addNE_fix(diagonals[middle].edge.first, ele1ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.first, ele2ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.first, ele3ID, tid);

            def_ops->addNE_fix(diagonals[middle].edge.second, ele1ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.second, ele2ID, tid);
            def_ops->addNE_fix(diagonals[middle].edge.second, ele3ID, tid);

            def_ops->addNE_fix(non_shared_bottom, ele2ID, tid);
            def_ops->addNE_fix(non_shared_bottom, ele3ID, tid);

            def_ops->addNE_fix(non_shared_top, ele1ID, tid);
            def_ops->addNE_fix(non_shared_top, ele3ID, tid);

            def_ops->addNE_fix(v_bottom, ele2ID, tid);

            def_ops->addNE_fix(v_top, ele1ID, tid);

            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            splitCnt[tid] += 3;
        } else {
            /*
             ***************
             * Case 1-to-8 *
             ***************
             */

            /*
             * The wedge must by split into 8 tetrahedra with the introduction of
             * a new centroidal vertex. Each tetrahedron is formed by the three
             * vertices of a triangular facet (there are 8 triangular facets: 6 are
             * formed via the bisection of the 3 quadrilaterals of the wedge, 2 are
             * the top and bottom triangles and the centroidal vertex.
             */

            // Allocate space for the centroidal vertex
            index_t cid = pragmatic_omp_atomic_capture(&_mesh->NNodes, 1);

            const int ele1[] = {diagonals[0].edge.first, ghostDiagonals[0].edge.first, diagonals[0].edge.second, cid};
            const int ele2[] = {diagonals[0].edge.first, diagonals[0].edge.second, ghostDiagonals[0].edge.second, cid};
            const int ele3[] = {diagonals[1].edge.first, ghostDiagonals[1].edge.first, diagonals[1].edge.second, cid};
            const int ele4[] = {diagonals[1].edge.first, diagonals[1].edge.second, ghostDiagonals[1].edge.second, cid};
            const int ele5[] = {diagonals[2].edge.first, ghostDiagonals[2].edge.first, diagonals[2].edge.second, cid};
            const int ele6[] = {diagonals[2].edge.first, diagonals[2].edge.second, ghostDiagonals[2].edge.second, cid};
            const int ele7[] = {top_triangle[0], top_triangle[1], top_triangle[2], cid};
            const int ele8[] = {bottom_triangle[0], bottom_triangle[2], bottom_triangle[1], cid};

            const int ele1_boundary[] = {0, 0, 0, bndr[0]};
            const int ele2_boundary[] = {0, 0, 0, bndr[0]};
            const int ele3_boundary[] = {0, 0, 0, bndr[1]};
            const int ele4_boundary[] = {0, 0, 0, bndr[1]};
            const int ele5_boundary[] = {0, 0, 0, bndr[2]};
            const int ele6_boundary[] = {0, 0, 0, bndr[2]};
            const int ele7_boundary[] = {0, 0, 0, bndr[3]};
            const int ele8_boundary[] = {0, 0, 0, bndr[4]};

            index_t ele1ID, ele2ID, ele3ID, ele4ID, ele5ID, ele6ID, ele7ID, ele8ID;
            ele1ID = splitCnt[tid];
            ele2ID = ele1ID+1;
            ele3ID = ele1ID+2;
            ele4ID = ele1ID+3;
            ele5ID = ele1ID+4;
            ele6ID = ele1ID+5;
            ele7ID = ele1ID+6;
            ele8ID = ele1ID+7;

            for(int j=0; j<3; ++j) {
                _mesh->NNList[cid].push_back(top_triangle[j]);
                _mesh->NNList[cid].push_back(bottom_triangle[j]);
                def_ops->addNN(top_triangle[j], cid, tid);
                def_ops->addNN(bottom_triangle[j], cid, tid);
            }

            def_ops->addNE_fix(cid, ele1ID, tid);
            def_ops->addNE_fix(cid, ele2ID, tid);
            def_ops->addNE_fix(cid, ele3ID, tid);
            def_ops->addNE_fix(cid, ele4ID, tid);
            def_ops->addNE_fix(cid, ele5ID, tid);
            def_ops->addNE_fix(cid, ele6ID, tid);
            def_ops->addNE_fix(cid, ele7ID, tid);
            def_ops->addNE_fix(cid, ele8ID, tid);

            def_ops->addNE_fix(bottom_triangle[0], ele8ID, tid);
            def_ops->addNE_fix(bottom_triangle[1], ele8ID, tid);
            def_ops->addNE_fix(bottom_triangle[2], ele8ID, tid);

            def_ops->addNE_fix(top_triangle[0], ele7ID, tid);
            def_ops->addNE_fix(top_triangle[1], ele7ID, tid);
            def_ops->addNE_fix(top_triangle[2], ele7ID, tid);

            def_ops->addNE_fix(diagonals[0].edge.first, ele1ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.first, ele2ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.second, ele1ID, tid);
            def_ops->addNE_fix(diagonals[0].edge.second, ele2ID, tid);
            def_ops->addNE_fix(ghostDiagonals[0].edge.first, ele1ID, tid);
            def_ops->addNE_fix(ghostDiagonals[0].edge.second, ele2ID, tid);

            def_ops->addNE_fix(diagonals[1].edge.first, ele3ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.first, ele4ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.second, ele3ID, tid);
            def_ops->addNE_fix(diagonals[1].edge.second, ele4ID, tid);
            def_ops->addNE_fix(ghostDiagonals[1].edge.first, ele3ID, tid);
            def_ops->addNE_fix(ghostDiagonals[1].edge.second, ele4ID, tid);

            def_ops->addNE_fix(diagonals[2].edge.first, ele5ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.first, ele6ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.second, ele5ID, tid);
            def_ops->addNE_fix(diagonals[2].edge.second, ele6ID, tid);
            def_ops->addNE_fix(ghostDiagonals[2].edge.first, ele5ID, tid);
            def_ops->addNE_fix(ghostDiagonals[2].edge.second, ele6ID, tid);

            // Sort all 6 vertices of the wedge by their coordinates.
            // Need to do so to enforce consistency across MPI processes.
            std::map<Coords_t, index_t> coords_map;
            for(int j=0; j<3; ++j) {
                Coords_t cb(_mesh->get_coords(bottom_triangle[j]));
                coords_map[cb] = bottom_triangle[j];
                Coords_t ct(_mesh->get_coords(top_triangle[j]));
                coords_map[ct] = top_triangle[j];
            }

            real_t nc[] = {0.0, 0.0, 0.0}; // new coordinates
            double nm[msize]; // new metric
            const index_t* n = _mesh->get_element(eid);

            {
                // Calculate the coordinates of the centroidal vertex.
                // We start with a temporary location at the euclidean barycentre of the wedge.
                for(typename std::map<Coords_t, index_t>::const_iterator it=coords_map.begin(); it!=coords_map.end(); ++it) {
                    const real_t *x = _mesh->get_coords(it->second);
                    for(int j=0; j<dim; ++j)
                        nc[j] += x[j];
                }
                for(int j=0; j<dim; ++j) {
                    nc[j] /= coords_map.size();
                    _mesh->_coords[cid*dim+j] = nc[j];
                }

                // Interpolate metric at temporary location using the parent element's basis functions
                std::map<Coords_t, index_t> parent_coords;
                for(int j=0; j<nloc; ++j) {
                    Coords_t cn(_mesh->get_coords(n[j]));
                    parent_coords[cn] = n[j];
                }

                std::vector<const real_t *> x;
                std::vector<index_t> sorted_n;
                for(typename std::map<Coords_t, index_t>::const_iterator it=parent_coords.begin(); it!=parent_coords.end(); ++it) {
                    x.push_back(_mesh->get_coords(it->second));
                    sorted_n.push_back(it->second);
                }

                // Order of parent element's vertices has changed, so volume might be negative.
                real_t L = fabs(property->volume(x[0], x[1], x[2], x[3]));

                real_t ll[4];
                ll[0] = fabs(property->volume(nc  , x[1], x[2], x[3])/L);
                ll[1] = fabs(property->volume(x[0], nc  , x[2], x[3])/L);
                ll[2] = fabs(property->volume(x[0], x[1], nc  , x[3])/L);
                ll[3] = fabs(property->volume(x[0], x[1], x[2], nc  )/L);

                for(int i=0; i<msize; i++) {
                    nm[i] = ll[0] * _mesh->metric[sorted_n[0]*msize+i]+
                            ll[1] * _mesh->metric[sorted_n[1]*msize+i]+
                            ll[2] * _mesh->metric[sorted_n[2]*msize+i]+
                            ll[3] * _mesh->metric[sorted_n[3]*msize+i];
                    _mesh->metric[cid*msize+i] = nm[i];
                }
            }

            // Use the 3D laplacian smoothing kernel to find the barycentre of the wedge in metric space.
            Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A =
                Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 3);
            Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);

            for(typename std::map<Coords_t, index_t>::const_iterator it=coords_map.begin(); it!=coords_map.end(); ++it) {
                const real_t *il = _mesh->get_coords(it->second);
                real_t x = il[0]-nc[0];
                real_t y = il[1]-nc[1];
                real_t z = il[2]-nc[2];

                q[0] += nm[0]*x + nm[1]*y + nm[2]*z;
                q[1] += nm[1]*x + nm[3]*y + nm[4]*z;
                q[2] += nm[2]*x + nm[4]*y + nm[5]*z;

                A[0] += nm[0];
                A[1] += nm[1];
                A[2] += nm[2];
                A[4] += nm[3];
                A[5] += nm[4];
                A[8] += nm[5];
            }
            A[3] = A[1];
            A[6] = A[2];
            A[7] = A[5];

            // Want to solve the system Ap=q to find the new position, p.
            Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
            A.svd().solve(q, &b);

            for(int i=0; i<3; ++i) {
                nc[i] += b[i];
            }

            // Interpolate metric at new location
            real_t l[]= {-1, -1, -1, -1};
            real_t tol=-1;
            std::vector<index_t> sorted_best_e(nloc);
            const index_t *welements[] = {ele1, ele2, ele3, ele4, ele5, ele6, ele7, ele8};

            for(int ele=0; ele<8; ++ele) {
                std::map<Coords_t, index_t> local_coords;
                for(int j=0; j<nloc; ++j) {
                    Coords_t cl(_mesh->get_coords(welements[ele][j]));
                    local_coords[cl] = welements[ele][j];
                }

                std::vector<const real_t *> x;
                std::vector<index_t> sorted_n;
                for(typename std::map<Coords_t, index_t>::const_iterator it=local_coords.begin(); it!=local_coords.end(); ++it) {
                    x.push_back(_mesh->get_coords(it->second));
                    sorted_n.push_back(it->second);
                }

                real_t L = fabs(property->volume(x[0], x[1], x[2], x[3]));

                assert(L>0);

                real_t ll[4];
                ll[0] = fabs(property->volume(nc  , x[1], x[2], x[3])/L);
                ll[1] = fabs(property->volume(x[0], nc  , x[2], x[3])/L);
                ll[2] = fabs(property->volume(x[0], x[1], nc  , x[3])/L);
                ll[3] = fabs(property->volume(x[0], x[1], x[2], nc  )/L);

                real_t min_l = std::min(std::min(ll[0], ll[1]), std::min(ll[2], ll[3]));
                if(min_l>tol) {
                    tol = min_l;
                    for(int j=0; j<nloc; ++j) {
                        l[j] = ll[j];
                        sorted_best_e[j] = sorted_n[j];
                    }
                }
            }

            for(int i=0; i<dim; ++i) {
                _mesh->_coords[cid*dim+i] = nc[i];
            }

            for(int i=0; i<msize; ++i)
                _mesh->metric[cid*msize+i] = l[0]*_mesh->metric[sorted_best_e[0]*msize+i]+
                                             l[1]*_mesh->metric[sorted_best_e[1]*msize+i]+
                                             l[2]*_mesh->metric[sorted_best_e[2]*msize+i]+
                                             l[3]*_mesh->metric[sorted_best_e[3]*msize+i];

            append_element(ele1, ele1_boundary, tid);
            append_element(ele2, ele2_boundary, tid);
            append_element(ele3, ele3_boundary, tid);
            append_element(ele4, ele4_boundary, tid);
            append_element(ele5, ele5_boundary, tid);
            append_element(ele6, ele6_boundary, tid);
            append_element(ele7, ele7_boundary, tid);
            append_element(ele8, ele8_boundary, tid);
            splitCnt[tid] += 8;

            // Finally, assign a gnn and owner
            if(nprocs == 1) {
                _mesh->node_owner[cid] = 0;
                _mesh->lnn2gnn[cid] = cid;
            }
#ifdef HAVE_MPI
            else {
                int owner = nprocs;
                for(int j=0; j<nloc; ++j)
                    owner = std::min(owner, _mesh->node_owner[n[j]]);

                _mesh->node_owner[cid] = owner;

                if(_mesh->node_owner[cid] != rank) {
                    // Vertex is owned by another MPI process, so prepare to update recv and recv_halo.
                    Wedge wedge(cid, _mesh->get_coords(cid));
                    #pragma omp critical
                    cidRecv_additional[_mesh->node_owner[cid]].insert(wedge);
                } else {
                    // Vertex is owned by *this* MPI process, so check whether it is visible by other MPI processes.
                    // The latter is true only if all vertices of the original element were halo vertices.
                    if(_mesh->is_halo_node(n[0]) && _mesh->is_halo_node(n[1]) && _mesh->is_halo_node(n[2]) && _mesh->is_halo_node(n[3])) {
                        // Find which processes see this vertex
                        std::set<int> processes;
                        for(int j=0; j<nloc; ++j)
                            processes.insert(_mesh->node_owner[n[j]]);

                        processes.erase(rank);

                        Wedge wedge(cid, _mesh->get_coords(cid));
                        for(typename std::set<int>::const_iterator proc=processes.begin(); proc!=processes.end(); ++proc) {
                            #pragma omp critical
                            cidSend_additional[*proc].insert(wedge);
                        }
                    }

                    // Finally, assign a gnn
                    _mesh->lnn2gnn[cid] = _mesh->gnn_offset+cid;
                }
            }
#endif
        }
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

    // Struct used for sorting vertices by their coordinates. It's
    // meant to be used by the 1:8 wedge refinement code to enforce
    // consistent order of floating point arithmetic across MPI processes.
    struct Coords_t {
        real_t coords[3];

        Coords_t(const real_t *x)
        {
            coords[0] = x[0];
            coords[1] = x[1];
            coords[2] = x[2];
        }

        /// Less-than operator
        bool operator<(const Coords_t& in) const
        {
            bool isLess;

            for(int i=0; i<3; ++i) {
                if(coords[i] < in.coords[i]) {
                    isLess=true;
                    break;
                } else if(coords[i] > in.coords[i]) {
                    isLess = false;
                    break;
                }
            }

            return isLess;
        }
    };

    // Struct containing gnn's of the six vertices comprising a wedge. It is only
    // to be used for consistent sorting of centroidal vertices across MPI processes.
    struct Wedge {
        const Coords_t coords;
        const index_t cid;

        Wedge(const index_t id, const real_t *cid_coords) : coords(cid_coords), cid(id) {}

        /// Less-than operator
        bool operator<(const Wedge& in) const
        {
            return coords < in.coords;
        }
    };

    std::vector< std::vector< DirectedEdge<index_t> > > newVertices;
    std::vector< std::vector<real_t> > newCoords;
    std::vector< std::vector<double> > newMetric;
    std::vector< std::vector<index_t> > newElements;
    std::vector< std::vector<int> > newBoundaries;
    std::vector< std::vector<double> > newQualities;
    std::vector<index_t> new_vertices_per_element;

    std::vector<size_t> threadIdx, splitCnt;
    std::vector< DirectedEdge<index_t> > allNewVertices;
    std::vector< std::set<Wedge> > cidRecv_additional, cidSend_additional;

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
