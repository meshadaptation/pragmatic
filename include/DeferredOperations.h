/*  Copyright (C) 2013 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Georgios Rokos
 *  Software Performance Optimisation Group
 *  Department of Computing
 *  Imperial College London
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

#ifndef DEFERRED_OPERATIONS_H
#define DEFERRED_OPERATIONS_H

#include <set>
#include <vector>

#include "Mesh.h"

template <typename real_t>
class DeferredOperations
{
public:
    DeferredOperations(Mesh<real_t>* mesh, const int num_threads, const int scaling_factor)
        : nthreads(num_threads), defOp_scaling_factor(scaling_factor)
    {
        _mesh = mesh;
        deferred_operations.resize(nthreads);
        for(int i=0; i<nthreads; ++i)
            deferred_operations[i].resize(nthreads*defOp_scaling_factor);
    }

    ~DeferredOperations() {}

    inline void addNN(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNN.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNN.push_back(n);
    }

    inline void remNN(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNN.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNN.push_back(n);
    }

    inline void addNE(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE.push_back(n);
    }

    inline void addNE_fix(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE_fix.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].addNE_fix.push_back(n);
    }

    inline void repEN(const size_t pos, const index_t n, const int tid)
    {
        deferred_operations[tid][(pos/16) % (defOp_scaling_factor*nthreads)].repEN.push_back(pos);
        deferred_operations[tid][(pos/16) % (defOp_scaling_factor*nthreads)].repEN.push_back(n);
    }

    inline void remNE(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNE.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].remNE.push_back(n);
    }

    inline void propagate_coarsening(const index_t i, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].coarsening_propagation.push_back(i);
    }

    inline void propagate_refinement(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].refinement_propagation.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].refinement_propagation.push_back(n);
    }

    inline void propagate_swapping(const index_t i, const index_t n, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].swapping_propagation.push_back(i);
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].swapping_propagation.push_back(n);
    }

    inline void reset_colour(const index_t i, const int tid)
    {
        deferred_operations[tid][hash(i) % (defOp_scaling_factor*nthreads)].reset_colour.push_back(i);
    }

    inline void commit_addNN(const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].addNN.begin();
                it!=deferred_operations[tid][vtid].addNN.end(); it+=2) {
            _mesh->NNList[*it].push_back(*(it+1));
        }

        deferred_operations[tid][vtid].addNN.clear();
    }

    inline void commit_remNN(const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].remNN.begin();
                it!=deferred_operations[tid][vtid].remNN.end(); it+=2) {
            typename std::vector<index_t>::iterator position = std::find(_mesh->NNList[*it].begin(), _mesh->NNList[*it].end(), *(it+1));
            assert(position != _mesh->NNList[*it].end());
            _mesh->NNList[*it].erase(position);
        }

        deferred_operations[tid][vtid].remNN.clear();
    }

    inline void commit_addNE(const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].addNE.begin();
                it!=deferred_operations[tid][vtid].addNE.end(); it+=2) {
            _mesh->NEList[*it].insert(*(it+1));
        }

        deferred_operations[tid][vtid].addNE.clear();
    }

    inline void commit_addNE_fix(std::vector<size_t>& threadIdx, const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].addNE_fix.begin();
                it!=deferred_operations[tid][vtid].addNE_fix.end(); it+=2) {
            // Element was created by thread tid
            index_t fixedId = *(it+1) + threadIdx[tid];
            _mesh->NEList[*it].insert(fixedId);
        }

        deferred_operations[tid][vtid].addNE_fix.clear();
    }

    inline void commit_remNE(const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].remNE.begin();
                it!=deferred_operations[tid][vtid].remNE.end(); it+=2) {
            assert(_mesh->NEList[*it].count(*(it+1)) != 0);
            _mesh->NEList[*it].erase(*(it+1));
        }

        deferred_operations[tid][vtid].remNE.clear();
    }

    inline void commit_repEN(const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].repEN.begin();
                it!=deferred_operations[tid][vtid].repEN.end(); it+=2) {
            _mesh->_ENList[*it] = *(it+1);
        }

        deferred_operations[tid][vtid].repEN.clear();
    }

    inline void commit_coarsening_propagation(index_t* dynamic_vertex, const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].coarsening_propagation.begin();
                it!=deferred_operations[tid][vtid].coarsening_propagation.end(); ++it) {
            dynamic_vertex[*it] = -2;
        }

        deferred_operations[tid][vtid].coarsening_propagation.clear();
    }

    inline void commit_refinement_propagation(std::vector< std::set<index_t> >& marked_edges, const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].refinement_propagation.begin();
                it!=deferred_operations[tid][vtid].refinement_propagation.end(); it+=2) {
            marked_edges[*it].insert(*(it+1));
        }

        deferred_operations[tid][vtid].refinement_propagation.clear();
    }

    inline void commit_swapping_propagation(std::vector< std::set<index_t> >& marked_edges, const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].swapping_propagation.begin();
                it!=deferred_operations[tid][vtid].swapping_propagation.end(); it+=2) {
            marked_edges[*it].insert(*(it+1));
        }

        deferred_operations[tid][vtid].swapping_propagation.clear();
    }

    inline void commit_colour_reset(int* node_colour, const int tid, const int vtid)
    {
        for(typename std::vector<index_t>::const_iterator it=deferred_operations[tid][vtid].reset_colour.begin();
                it!=deferred_operations[tid][vtid].reset_colour.end(); ++it) {
            node_colour[*it] = 0;
        }

        deferred_operations[tid][vtid].reset_colour.clear();
    }

private:
    /*
     * Park & Miller (aka Lehmer) pseudo-random number generation. Possible bug if
     * index_t is a datatype longer than 32 bits. However, in the context of a single
     * MPI node, it is highly unlikely that index_t will ever need to be longer.
     * A 64-bit datatype makes sense only for global node numbers, not local.
     */
    inline uint32_t hash(const uint32_t id) const
    {
        return ((uint64_t)id * 279470273UL) % 4294967291UL;
    }

    struct def_op_t {
        // Mesh
        std::vector<index_t> addNN; // addNN -> [i, n] : Add node n to NNList[i].
        std::vector<index_t> remNN; // remNN -> [i, n] : Remove node n from NNList[i].
        std::vector<index_t> addNE; // addNE -> [i, n] : Add element n to NEList[i].
        std::vector<index_t> remNE; // remNE -> [i, n] : Remove element n from NEList[i].
        std::vector<index_t> addNE_fix; // addNE_fix -> [i, n] : Fix ID of element n according to
        // threadIdx[thread_which_created_n] and add it to NEList[i].
        std::vector<index_t> repEN; // remEN -> [pos, n] : Set _ENList[pos] = n.
        std::vector<index_t> coarsening_propagation; // [i] : Mark Coarseninig::dynamic_vertex[i]=-2.
        std::vector<index_t> refinement_propagation; // [i, n]: Mark Edge(i,n) for refinement.
        std::vector<index_t> swapping_propagation; // [i, n] : Mark Swapping::marked_edges[i].insert(n).
        std::vector<index_t> reset_colour; // [i] : Set Colouring::node_colour[i]=-1.
    };

    //Deferred operations main structure
    std::vector< std::vector<def_op_t> > deferred_operations;
    const int nthreads;
    const int defOp_scaling_factor;

    Mesh<real_t>* _mesh;
};

#endif
