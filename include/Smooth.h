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

#ifndef SMOOTH_H
#define SMOOTH_H

#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <vector>
#include <limits>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <errno.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "ElementProperty.h"
#include "Lock.h"
#include "Mesh.h"
#include "MetricTensor.h"


/*! \brief Applies Laplacian smoothen in metric space.
*/
template<typename real_t, int dim>
class Smooth
{
public:
    /// Default constructor.
    Smooth(Mesh<real_t> &mesh):nloc(dim+1), msize(dim==2?3:6)
    {
        _mesh = &mesh;

        mpi_nparts = 1;
        rank=0;
#ifdef HAVE_MPI
        MPI_Comm_size(_mesh->get_mpi_comm(), &mpi_nparts);
        MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
#endif

        epsilon_q = DBL_EPSILON;

        // Set the orientation of elements.
        property = NULL;
        int NElements = _mesh->get_number_elements();
        for(int i=0; i<NElements; i++) {
            const int *n=_mesh->get_element(i);
            if(n[0]<0)
                continue;

            if(dim==2) {
                property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                                       _mesh->get_coords(n[1]),
                                                       _mesh->get_coords(n[2]));
            } else {
                property = new ElementProperty<real_t>(_mesh->get_coords(n[0]),
                                                       _mesh->get_coords(n[1]),
                                                       _mesh->get_coords(n[2]),
                                                       _mesh->get_coords(n[3]));
            }

            break;
        }
    }

    /// Default destructor.
    ~Smooth()
    {
        delete property;
    }

    // Smart laplacian mesh smoothing.
    void smart_laplacian(int max_iterations=10, double quality_tol=-1.0)
    {
        int NNodes = _mesh->get_number_nodes();
        int NElements = _mesh->get_number_elements();

        if(NElements==0)
            return;

        std::vector< std::atomic<bool> > is_boundary(NNodes);
        std::vector< std::atomic<bool> > active_vertices(NNodes);
        if(vLocks.size() < NNodes)
            vLocks.resize(NNodes);

        double qsum=0;
        good_q = quality_tol;

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int n=0; n<NNodes; ++n) {
                is_boundary[n].store(false, std::memory_order_relaxed);
                if(_mesh->NNList[n].empty()) {
                    assert(_mesh->NEList[n].empty());
                    active_vertices[n].store(false, std::memory_order_relaxed);
                } else {
                    active_vertices[n].store(true, std::memory_order_relaxed);
                }
                vLocks[n].unlock();
            }

            #pragma omp for schedule(guided)
            for(int i=0; i<NElements; i++) {
                const int *n=_mesh->get_element(i);
                if(n[0]<0)
                    continue;

                for(size_t j=0; j<nloc; j++) {
                    if(_mesh->boundary[i*nloc+j]>0) {
                        for(size_t k=1; k<nloc; k++) {
                            is_boundary[n[(j+k)%nloc]].store(true, std::memory_order_relaxed);
                        }
                    }
                }
            }

            if(good_q<0) {
                #pragma omp for schedule(static) reduction(+:qsum)
                for(int i=0; i<NElements; i++) {
                    const int *n=_mesh->get_element(i);
                    if(n[0]<0)
                        continue;

                    assert(std::isfinite(_mesh->quality[i]));
                    qsum+=_mesh->quality[i];
                }
#pragma single
                {
                    good_q = qsum/NElements;
                    assert(std::isnormal(good_q));
                }
            }

            // First sweep through all vertices. Add vertices adjacent to any
            // vertex moved into the active_vertex list.
            std::vector<index_t> retry, next_retry;

            int iter=0;
            while((iter++) < max_iterations) {
                #pragma omp for schedule(guided) nowait
                for(index_t node=0; node<NNodes; ++node) {
                    if(_mesh->is_halo_node(node) || is_boundary[node].load(std::memory_order_relaxed) || !active_vertices[node].load(std::memory_order_relaxed))
                        continue;

                    if(!vLocks[node].try_lock()) {
                        retry.push_back(node);
                        continue;
                    }

                    bool abort = false;
                    for(const auto& it : _mesh->NNList[node]) {
                        if(vLocks[it].is_locked()) {
                            abort = true;
                            break;
                        }
                    }

                    if(!abort) {
                        if(smart_laplacian_kernel(node)) {
                            for(auto& it : _mesh->NNList[node]) {
                                assert(!_mesh->NNList[node].empty());
                                assert(!_mesh->NEList[node].empty());
                                active_vertices[it].store(true, std::memory_order_relaxed);
                            }
                        } else {
                            active_vertices[node].store(false, std::memory_order_relaxed);
                        }
                    } else {
                        retry.push_back(node);
                    }
                    vLocks[node].unlock();
                }

                for(int iretry=0; iretry<100; iretry++) { // Put a hard limit on the number of times we try to get a lock.
                    next_retry.clear();

                    for(const auto& node : retry) {
                        bool abort = false;

                        if(!vLocks[node].try_lock()) {
                            next_retry.push_back(node);
                            continue;
                        }

                        for(const auto& it : _mesh->NNList[node]) {
                            if(vLocks[it].is_locked()) {
                                abort = true;
                                break;
                            }
                        }

                        if(!abort) {
                            if(smart_laplacian_kernel(node)) {
                                for(auto& it : _mesh->NNList[node]) {
                                    assert(!_mesh->NNList[node].empty());
                                    assert(!_mesh->NEList[node].empty());
                                    active_vertices[it].store(true, std::memory_order_relaxed);
                                }
                            } else {
                                active_vertices[node].store(false, std::memory_order_relaxed);
                            }
                        } else {
                            next_retry.push_back(node);
                        }
                        vLocks[node].unlock();
                    }

                    retry.swap(next_retry);
                    if(retry.empty())
                        break;
                }
            }
        }

        return;
    }

    // Linf optimisation based smoothing..
    void optimisation_linf(int max_iterations=10, double quality_tol=-1.0)
    {
        int NNodes = _mesh->get_number_nodes();
        int NElements = _mesh->get_number_elements();

        if(NElements==0)
            return;

        std::vector< std::atomic<bool> > is_boundary(NNodes);
        std::vector< std::atomic<bool> > active_vertices(NNodes);
        if(vLocks.size() < NNodes)
            vLocks.resize(NNodes);

        double qsum=0;
        good_q = quality_tol;

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int n=0; n<NNodes; ++n) {
                is_boundary[n].store(false, std::memory_order_relaxed);
                if(_mesh->NNList[n].empty()) {
                    assert(_mesh->NEList[n].empty());
                    active_vertices[n].store(false, std::memory_order_relaxed);
                } else {
                    active_vertices[n].store(true, std::memory_order_relaxed);
                }
                vLocks[n].unlock();
            }

            #pragma omp for schedule(guided)
            for(int i=0; i<NElements; i++) {
                const int *n=_mesh->get_element(i);
                if(n[0]<0)
                    continue;

                for(size_t j=0; j<nloc; j++) {
                    if(_mesh->boundary[i*nloc+j]>0) {
                        for(size_t k=1; k<nloc; k++) {
                            is_boundary[n[(j+k)%nloc]].store(true, std::memory_order_relaxed);
                        }
                    }
                }
            }

            if(good_q<0) {
                #pragma omp for schedule(static) reduction(+:qsum)
                for(int i=0; i<NElements; i++) {
                    const int *n=_mesh->get_element(i);
                    if(n[0]<0)
                        continue;

                    assert(std::isfinite(_mesh->quality[i]));
                    qsum+=_mesh->quality[i];
                }
#pragma single
                {
                    good_q = qsum/NElements;
                    assert(std::isnormal(good_q));
                }
            }

            // First sweep through all vertices. Add vertices adjacent to any
            // vertex moved into the active_vertex list.
            std::vector<index_t> retry, next_retry;

            int iter=0;
            while((iter++) < max_iterations) {
                #pragma omp for schedule(guided) nowait
                for(index_t node=0; node<NNodes; ++node) {
                    if(_mesh->is_halo_node(node) || is_boundary[node].load(std::memory_order_relaxed) || !active_vertices[node].load(std::memory_order_relaxed))
                        continue;

                    if(!vLocks[node].try_lock()) {
                        retry.push_back(node);
                        continue;
                    }

                    bool abort = false;
                    for(const auto& it : _mesh->NNList[node]) {
                        if(vLocks[it].is_locked()) {
                            abort = true;
                            break;
                        }
                    }

                    if(!abort) {
                        if(optimisation_linf_kernel(node)) {
                            for(auto& it : _mesh->NNList[node]) {
                                assert(!_mesh->NNList[node].empty());
                                assert(!_mesh->NEList[node].empty());
                                active_vertices[it].store(true, std::memory_order_relaxed);
                            }
                        } else {
                            active_vertices[node].store(false, std::memory_order_relaxed);
                        }
                    } else {
                        retry.push_back(node);
                    }
                    vLocks[node].unlock();
                }

                for(int iretry=0; iretry<100; iretry++) { // Put a hard limit on the number of times we try to get a lock.
                    next_retry.clear();

                    for(const auto& node : retry) {
                        bool abort = false;

                        if(!vLocks[node].try_lock()) {
                            next_retry.push_back(node);
                            continue;
                        }

                        for(const auto& it : _mesh->NNList[node]) {
                            if(vLocks[it].is_locked()) {
                                abort = true;
                                break;
                            }
                        }

                        if(!abort) {
                            if(optimisation_linf_kernel(node)) {
                                for(auto& it : _mesh->NNList[node]) {
                                    assert(!_mesh->NNList[node].empty());
                                    assert(!_mesh->NEList[node].empty());
                                    active_vertices[it].store(true, std::memory_order_relaxed);
                                }
                            } else {
                                active_vertices[node].store(false, std::memory_order_relaxed);
                            }
                        } else {
                            next_retry.push_back(node);
                        }
                        vLocks[node].unlock();
                    }

                    retry.swap(next_retry);
                    if(retry.empty())
                        break;
                }
            }
        }

        return;
    }

    // Laplacian smoothing
    void laplacian(int max_iterations=10)
    {
        int NNodes = _mesh->get_number_nodes();
        int NElements = _mesh->get_number_elements();

        std::vector< std::atomic<bool> > is_boundary(NNodes);
        std::vector< std::atomic<bool> > active_vertices(NNodes);
        if(vLocks.size() < NNodes)
            vLocks.resize(NNodes);

        #pragma omp parallel for
        for(int n=0; n<NNodes; ++n) {
            is_boundary[n].store(false, std::memory_order_relaxed);
        }

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int n=0; n<NNodes; ++n) {
                is_boundary[n].store(false, std::memory_order_relaxed);
                if(_mesh->NNList[n].empty()) {
                    assert(_mesh->NEList[n].empty());
                    active_vertices[n].store(false, std::memory_order_relaxed);
                } else {
                    active_vertices[n].store(true, std::memory_order_relaxed);
                }
                vLocks[n].unlock();
            }

            #pragma omp for schedule(guided)
            for(int i=0; i<NElements; i++) {
                const int *n=_mesh->get_element(i);
                if(n[0]<0)
                    continue;

                for(size_t j=0; j<nloc; j++) {
                    if(_mesh->boundary[i*nloc+j]>0) {
                        for(size_t k=1; k<nloc; k++) {
                            is_boundary[n[(j+k)%nloc]].store(true, std::memory_order_relaxed);
                        }
                    }
                }
            }
            // First sweep through all vertices. Add vertices adjacent to any
            // vertex moved into the active_vertex list.
            std::vector<index_t> retry, next_retry;

            int iter=0;
            while((iter++) < max_iterations) {
                #pragma omp for schedule(guided) nowait
                for(index_t node=0; node<NNodes; ++node) {
                    if(_mesh->is_halo_node(node) || is_boundary[node].load(std::memory_order_relaxed) || !active_vertices[node].load(std::memory_order_relaxed))
                        continue;

                    if(!vLocks[node].try_lock()) {
                        retry.push_back(node);
                        continue;
                    }

                    bool abort = false;
                    for(const auto& it : _mesh->NNList[node]) {
                        if(vLocks[it].is_locked()) {
                            abort = true;
                            break;
                        }
                    }

                    if(!abort) {
                        if(laplacian_kernel(node)) {
                            for(auto& it : _mesh->NNList[node]) {
                                assert(!_mesh->NNList[node].empty());
                                assert(!_mesh->NEList[node].empty());
                                active_vertices[it].store(true, std::memory_order_relaxed);
                            }
                        } else {
                            active_vertices[node].store(false, std::memory_order_relaxed);
                        }
                    } else {
                        retry.push_back(node);
                    }
                    vLocks[node].unlock();
                }

                for(int iretry=0; iretry<100; iretry++) { // Put a hard limit on the number of times we try to get a lock.
                    next_retry.clear();

                    for(const auto& node : retry) {
                        bool abort = false;

                        if(!vLocks[node].try_lock()) {
                            next_retry.push_back(node);
                            continue;
                        }

                        for(const auto& it : _mesh->NNList[node]) {
                            if(vLocks[it].is_locked()) {
                                abort = true;
                                break;
                            }
                        }

                        if(!abort) {
                            if(laplacian_kernel(node)) {
                                for(auto& it : _mesh->NNList[node]) {
                                    assert(!_mesh->NNList[node].empty());
                                    assert(!_mesh->NEList[node].empty());
                                    active_vertices[it].store(true, std::memory_order_relaxed);
                                }
                            } else {
                                active_vertices[node].store(false, std::memory_order_relaxed);
                            }
                        } else {
                            next_retry.push_back(node);
                        }
                        vLocks[node].unlock();
                    }

                    retry.swap(next_retry);
                    if(retry.empty())
                        break;
                }
            }
        }
        return;
    }

private:

    // Laplacian smooth kernels
    inline bool laplacian_kernel(index_t node)
    {
        bool update;
        if(dim==2)
            update = laplacian_2d_kernel(node);
        else
            update = laplacian_3d_kernel(node);

        return update;
    }

    inline bool laplacian_2d_kernel(index_t node)
    {
        real_t p[2];
        laplacian_2d_kernel(node, p);

        double mp[3];
        bool valid = generate_location_2d(node, p, mp);
        if(!valid) {
            // Try the mid point.
            for(size_t j=0; j<2; j++)
                p[j] = 0.5*(p[j] +  _mesh->_coords[node*2+j]);

            valid = generate_location_2d(node, p, mp);
        }

        // Give up
        if(!valid)
            return false;

        for(size_t j=0; j<2; j++)
            _mesh->_coords[node*2+j] = p[j];

        for(size_t j=0; j<3; j++)
            _mesh->metric[node*3+j] = mp[j];

        for(auto& e : _mesh->NEList[node])
            update_quality(e);

        return true;
    }

    inline bool laplacian_3d_kernel(index_t node)
    {
        real_t p[3];
        laplacian_3d_kernel(node, p);

        double mp[6];
        bool valid = generate_location_3d(node, p, mp);
        if(!valid) {
            // Try the mid point.
            for(size_t j=0; j<3; j++)
                p[j] = 0.5*(p[j] +  _mesh->_coords[node*3+j]);

            valid = generate_location_3d(node, p, mp);
        }
        if(!valid)
            return false;

        for(size_t j=0; j<3; j++)
            _mesh->_coords[node*3+j] = p[j];

        for(size_t j=0; j<6; j++)
            _mesh->metric[node*6+j] = mp[j];

        for(auto& e : _mesh->NEList[node])
            update_quality(e);

        return true;
    }

    inline void laplacian_2d_kernel(index_t node, real_t *p)
    {
        std::set<index_t> patch(_mesh->get_node_patch(node));

        real_t x0 = get_x(node);
        real_t y0 = get_y(node);

        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
        Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);

        const real_t *m0 = _mesh->get_metric(node);
        for(const auto& il : patch) {
            real_t x = get_x(il)-x0;
            real_t y = get_y(il)-y0;

            const real_t *m1 = _mesh->get_metric(il);
            double m[] = {0.5*(m0[0]+m1[0]), 0.5*(m0[1]+m1[1]), 0.5*(m0[2]+m1[2])};

            q[0] += (m[0]*x + m[1]*y);
            q[1] += (m[1]*x + m[2]*y);

            A[0] += m[0];
            A[1] += m[1];
            A[3] += m[2];
        }
        A[2]=A[1];

        // Want to solve the system Ap=q to find the new position, p.
        Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2);
        A.svd().solve(q, &b);

        for(size_t i=0; i<2; i++)
            p[i] = b[i];

        p[0] += x0;
        p[1] += y0;

        return;
    }

    inline void laplacian_3d_kernel(index_t node, real_t *p)
    {
        std::set<index_t> patch(_mesh->get_node_patch(node));

        real_t x0 = get_x(node);
        real_t y0 = get_y(node);
        real_t z0 = get_z(node);

        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 3);
        Eigen::Matrix<real_t, Eigen::Dynamic, 1> q = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);

        const real_t *m0 = _mesh->get_metric(node);
        for(const auto& il : patch) {
            real_t x = get_x(il)-x0;
            real_t y = get_y(il)-y0;
            real_t z = get_z(il)-z0;

            const real_t *m1 = _mesh->get_metric(il);
            double m[] = {0.5*(m0[0]+m1[0]), 0.5*(m0[1]+m1[1]), 0.5*(m0[2]+m1[2]),
                          0.5*(m0[3]+m1[3]), 0.5*(m0[4]+m1[4]),
                          0.5*(m0[5]+m1[5])
                         };

            q[0] += m[0]*x + m[1]*y + m[2]*z;
            q[1] += m[1]*x + m[3]*y + m[4]*z;
            q[2] += m[2]*x + m[4]*y + m[5]*z;

            A[0] += m[0];
            A[1] += m[1];
            A[2] += m[2];
            A[4] += m[3];
            A[5] += m[4];
            A[8] += m[5];
        }
        A[3] = A[1];
        A[6] = A[2];
        A[7] = A[5];

        // Want to solve the system Ap=q to find the new position, p.
        Eigen::Matrix<real_t, Eigen::Dynamic, 1> b = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(3);
        A.svd().solve(q, &b);

        for(int i=0; i<3; i++)
            p[i] = b[i];

        p[0] += x0;
        p[1] += y0;
        p[2] += z0;

        return;
    }

    // Smart Laplacian kernels
    inline bool smart_laplacian_kernel(index_t node)
    {
        bool update;
        if(dim==2)
            update = smart_laplacian_2d_kernel(node);
        else
            update = smart_laplacian_3d_kernel(node);

        return update;
    }

    inline bool smart_laplacian_2d_kernel(index_t node)
    {
        real_t p[2];
        laplacian_2d_kernel(node, p);

        double mp[3];
        bool valid = generate_location_2d(node, p, mp);
        if(!valid) {
            // Try the mid point.
            for(size_t j=0; j<2; j++)
                p[j] = 0.5*(p[j] +  _mesh->_coords[node*2+j]);

            valid = generate_location_2d(node, p, mp);
        }

        // Give up
        if(!valid)
            return false;

        real_t functional = functional_Linf(node, p, mp);
        real_t functional_orig = functional_Linf(node);

        if(functional-functional_orig<epsilon_q)
            return false;

        for(size_t j=0; j<2; j++)
            _mesh->_coords[node*2+j] = p[j];

        for(size_t j=0; j<3; j++)
            _mesh->metric[node*3+j] = mp[j];

        for(const auto& e : _mesh->NEList[node])
            update_quality(e);

        return true;
    }

    inline bool smart_laplacian_3d_kernel(index_t node)
    {
        real_t p[3];
        laplacian_3d_kernel(node, p);

        double mp[6];
        bool valid = generate_location_3d(node, p, mp);
        if(!valid) {
            // Try the mid point.
            for(size_t j=0; j<3; j++)
                p[j] = 0.5*(p[j] +  _mesh->_coords[node*2+j]);

            valid = generate_location_3d(node, p, mp);
        }

        // Give up
        if(!valid)
            return false;

        real_t functional = functional_Linf(node, p, mp);
        real_t functional_orig = functional_Linf(node);

        if(functional-functional_orig<epsilon_q)
            return false;

        for(size_t j=0; j<3; j++)
            _mesh->_coords[node*3+j] = p[j];

        for(size_t j=0; j<6; j++)
            _mesh->metric[node*6+j] = mp[j];

        for(const auto& e : _mesh->NEList[node])
            update_quality(e);

        return true;
    }

    // l-infinity optimisation kernels
    inline bool optimisation_linf_kernel(index_t node)
    {
        bool update;
        if(dim==2)
            update = optimisation_linf_2d_kernel(node);
        else
            update = optimisation_linf_3d_kernel(node);

        return update;

    }

    inline bool optimisation_linf_2d_kernel(index_t n0)
    {
        assert(!_mesh->NNList[n0].empty());
        assert(!_mesh->NEList[n0].empty());

        const double *m0 = _mesh->get_metric(n0);
        const double *x0 = _mesh->get_coords(n0);

        // Find the worst element.
        std::pair<double, index_t> worst_element(DBL_MAX, -1);
        for(const auto& it : _mesh->NEList[n0]) {
            if(_mesh->quality[it]<worst_element.first)
                worst_element = std::pair<double, index_t>(_mesh->quality[it], it);
        }
        assert(worst_element.second!=-1);

        // Return if it is already "good enough".
        if(worst_element.first>good_q)
            return false;

        // Find direction of steepest ascent for quality of worst element.
        double search[2], grad_w[2];
        {
            const index_t *n=_mesh->get_element(worst_element.second);
            size_t loc=0;
            for(; loc<3; loc++)
                if(n[loc]==n0)
                    break;

            int n1 = n[(loc+1)%3];
            int n2 = n[(loc+2)%3];

            const double *x1 = _mesh->get_coords(n1);
            const double *x2 = _mesh->get_coords(n2);

            property->lipnikov_grad(loc, x0, x1, x2, m0, grad_w);

            double mag = sqrt(grad_w[0]*grad_w[0]+grad_w[1]*grad_w[1]);
            assert(mag!=0);

            for(int i=0; i<2; i++)
                search[i] = grad_w[i]/mag;
        }

        // Estimate how far we move along this search path until we make
        // another element of a similar quality to the current worst. This
        // is effectively a simplex method for linear programming.
        double alpha;
        {
            double bbox[] = {DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX};
            for(const auto& it : _mesh->NEList[n0]) {
                const double *x1 = _mesh->get_coords(it);

                bbox[0] = std::min(bbox[0], x1[0]);
                bbox[1] = std::max(bbox[1], x1[0]);

                bbox[2] = std::min(bbox[2], x1[1]);
                bbox[3] = std::max(bbox[3], x1[1]);
            }
            alpha = (bbox[1]-bbox[0] + bbox[3]-bbox[2])/2.0;
        }

        for(const auto& it : _mesh->NEList[n0]) {
            if(it==worst_element.second)
                continue;

            const index_t *n=_mesh->get_element(it);
            size_t loc=0;
            for(; loc<3; loc++)
                if(n[loc]==n0)
                    break;

            int n1 = n[(loc+1)%3];
            int n2 = n[(loc+2)%3];

            const double *x1 = _mesh->get_coords(n1);
            const double *x2 = _mesh->get_coords(n2);

            double grad[2];
            property->lipnikov_grad(loc, x0, x1, x2, m0, grad);

            double new_alpha =
                (_mesh->quality[it]-worst_element.first)/
                ((search[0]*grad_w[0]+search[1]*grad_w[1])-
                 (search[0]*grad[0]+search[1]*grad[1]));

            if(new_alpha>0)
                alpha = std::min(alpha, new_alpha);
        }

        bool linf_update;
        for(int isearch=0; isearch<10; isearch++) {
            linf_update = false;

            // Only want to step half that distance so we do not degrade the other elements too much.
            alpha*=0.5;

            double new_x0[2];
            for(int i=0; i<2; i++) {
                new_x0[i] = x0[i] + alpha*search[i];
                if(!std::isnormal(new_x0[i]))
                    return false;
            }

            double new_m0[3];
            bool valid = generate_location_2d(n0, new_x0, new_m0);

            if(!valid)
                continue;

            // Need to check that we have not decreased the Linf norm. Start by assuming the best.
            linf_update = true;
            std::vector<double> new_quality;
            for(const auto& it : _mesh->NEList[n0]) {
                const index_t *n=_mesh->get_element(it);
                size_t loc=0;
                for(; loc<3; loc++)
                    if(n[loc]==n0)
                        break;

                int n1 = n[(loc+1)%3];
                int n2 = n[(loc+2)%3];

                const double *x1 = _mesh->get_coords(n1);
                const double *x2 = _mesh->get_coords(n2);

                const double *m1 = _mesh->get_metric(n1);
                const double *m2 = _mesh->get_metric(n2);

                double new_q = property->lipnikov(new_x0, x1, x2, new_m0, m1, m2);
                new_quality.push_back(new_q);

                if(!std::isnormal(new_q) || new_quality.back()<worst_element.first) {
                    linf_update = false;
                    break;
                }
            }

            if(!linf_update)
                continue;

            // Update information
            // go backwards and pop quality
            assert(_mesh->NEList[n0].size()==new_quality.size());
            for(typename std::set<index_t>::const_reverse_iterator it=_mesh->NEList[n0].rbegin(); it!=_mesh->NEList[n0].rend(); ++it) {
                _mesh->quality[*it] = new_quality.back();
                new_quality.pop_back();
            }
            assert(new_quality.empty());

            for(size_t i=0; i<dim; i++)
                _mesh->_coords[n0*dim+i] = new_x0[i];

            for(size_t i=0; i<msize; i++)
                _mesh->metric[n0*msize+i] = new_m0[i];

            for(auto& e : _mesh->NEList[n0])
                update_quality(e);

            break;
        }

        return linf_update;
    }

    inline bool optimisation_linf_3d_kernel(index_t n0)
    {
        const double *m0 = _mesh->get_metric(n0);
        const double *x0 = _mesh->get_coords(n0);

        // Find the worst element.
        std::pair<double, index_t> worst_element(DBL_MAX, -1);
        for(const auto& it : _mesh->NEList[n0]) {
            if(_mesh->quality[it]<worst_element.first)
                worst_element = std::pair<double, index_t>(_mesh->quality[it], it);
        }
        assert(worst_element.second!=-1);

        // Jump out if already good enough.
        if(worst_element.first>good_q)
            return false;

        // Find direction of steepest ascent for quality of worst element.
        double grad_w[3], search[3];
        {
            const index_t *n=_mesh->get_element(worst_element.second);
            size_t loc=0;
            for(; loc<4; loc++)
                if(n[loc]==n0)
                    break;

            int n1, n2, n3;
            switch(loc) {
            case 0:
                n1 = n[1];
                n2 = n[2];
                n3 = n[3];
                break;
            case 1:
                n1 = n[2];
                n2 = n[0];
                n3 = n[3];
                break;
            case 2:
                n1 = n[0];
                n2 = n[1];
                n3 = n[3];
                break;
            case 3:
                n1 = n[0];
                n2 = n[2];
                n3 = n[1];
                break;
            }

            const double *x1 = _mesh->get_coords(n1);
            const double *x2 = _mesh->get_coords(n2);
            const double *x3 = _mesh->get_coords(n3);

            property->lipnikov_grad(loc, x0, x1, x2, x3, m0, grad_w);

            double mag = sqrt(grad_w[0]*grad_w[0] + grad_w[1]*grad_w[1] + grad_w[2]*grad_w[2]);
            if(!std::isnormal(mag)) {
                std::cout<<"mag issues "<<mag<<", "<<grad_w[0]<<", "<<grad_w[1]<<", "<<grad_w[2]<<std::endl;
                std::cout<<"This usually means that the metric field is rubbish\n";
            }
            assert(std::isnormal(mag));

            for(int i=0; i<3; i++)
                search[i] = grad_w[i]/mag;
        }

        // Estimate how far we move along this search path until we make
        // another element of a similar quality to the current worst. This
        // is effectively a simplex method for linear programming.
        double alpha;
        {
            double bbox[] = {DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX};
            for(const auto& it : _mesh->NEList[n0]) {
                const double *x1 = _mesh->get_coords(it);

                bbox[0] = std::min(bbox[0], x1[0]);
                bbox[1] = std::max(bbox[1], x1[0]);

                bbox[2] = std::min(bbox[2], x1[1]);
                bbox[3] = std::max(bbox[3], x1[1]);

                bbox[4] = std::min(bbox[4], x1[2]);
                bbox[5] = std::max(bbox[5], x1[2]);
            }
            alpha = (bbox[1]-bbox[0] + bbox[3]-bbox[2] + bbox[5]-bbox[4])/6.0;
        }
        for(const auto& it : _mesh->NEList[n0]) {
            if(it==worst_element.second)
                continue;

            const index_t *n=_mesh->get_element(it);
            size_t loc=0;
            for(; loc<4; loc++)
                if(n[loc]==n0)
                    break;

            int n1, n2, n3;
            switch(loc) {
            case 0:
                n1 = n[1];
                n2 = n[2];
                n3 = n[3];
                break;
            case 1:
                n1 = n[2];
                n2 = n[0];
                n3 = n[3];
                break;
            case 2:
                n1 = n[0];
                n2 = n[1];
                n3 = n[3];
                break;
            case 3:
                n1 = n[0];
                n2 = n[2];
                n3 = n[1];
                break;
            }

            const double *x1 = _mesh->get_coords(n1);
            const double *x2 = _mesh->get_coords(n2);
            const double *x3 = _mesh->get_coords(n3);

            double grad[3];
            property->lipnikov_grad(loc, x0, x1, x2, x3, m0, grad);

            double new_alpha =
                (_mesh->quality[it]-worst_element.first)/
                ((search[0]*grad_w[0]+search[1]*grad_w[1]+search[2]*grad_w[2])-
                 (search[0]*grad[0]+search[1]*grad[1]+search[2]*grad[2]));

            if(new_alpha>0)
                alpha = std::min(alpha, new_alpha);
        }

        bool linf_update;
        for(int isearch=0; isearch<10; isearch++) {
            linf_update = false;

            // Only want to step half that distance so we do not degrade the other elements too much.
            alpha*=0.5;

            double new_x0[3];
            for(int i=0; i<3; i++) {
                new_x0[i] = x0[i] + alpha*search[i];
            }

            double new_m0[6];
            bool valid = generate_location_3d(n0, new_x0, new_m0);

            if(!valid)
                continue;

            // Need to check that we have not decreased the Linf norm. Start by assuming the best.
            linf_update = true;
            std::vector<double> new_quality;
            for(const auto& it : _mesh->NEList[n0]) {
                const index_t *n=_mesh->get_element(it);
                size_t loc=0;
                for(; loc<4; loc++)
                    if(n[loc]==n0)
                        break;

                int n1, n2, n3;
                switch(loc) {
                case 0:
                    n1 = n[1];
                    n2 = n[2];
                    n3 = n[3];
                    break;
                case 1:
                    n1 = n[2];
                    n2 = n[0];
                    n3 = n[3];
                    break;
                case 2:
                    n1 = n[0];
                    n2 = n[1];
                    n3 = n[3];
                    break;
                case 3:
                    n1 = n[0];
                    n2 = n[2];
                    n3 = n[1];
                    break;
                }

                const double *x1 = _mesh->get_coords(n1);
                const double *x2 = _mesh->get_coords(n2);
                const double *x3 = _mesh->get_coords(n3);


                const double *m1 = _mesh->get_metric(n1);
                const double *m2 = _mesh->get_metric(n2);
                const double *m3 = _mesh->get_metric(n3);

                double new_q = property->lipnikov(new_x0, x1, x2, x3, new_m0, m1, m2, m3);

                if(new_q>worst_element.first) {
                    new_quality.push_back(new_q);
                } else {
                    // This means that the linear approximation was not sufficient.
                    linf_update = false;
                    break;
                }
            }

            if(!linf_update)
                continue;

            // Update information
            // go backwards and pop quality
            assert(_mesh->NEList[n0].size()==new_quality.size());
            for(typename std::set<index_t>::const_reverse_iterator it=_mesh->NEList[n0].rbegin(); it!=_mesh->NEList[n0].rend(); ++it) {
                _mesh->quality[*it] = new_quality.back();
                new_quality.pop_back();
            }
            assert(new_quality.empty());

            for(size_t i=0; i<dim; i++)
                _mesh->_coords[n0*dim+i] = new_x0[i];

            for(size_t i=0; i<msize; i++)
                _mesh->metric[n0*msize+i] = new_m0[i];

            for(auto& e : _mesh->NEList[n0])
                update_quality(e);

            break;
        }

        return linf_update;
    }

    inline real_t get_x(index_t nid) const
    {
        return _mesh->_coords[nid*dim];
    }

    inline real_t get_y(index_t nid) const
    {
        return _mesh->_coords[nid*dim+1];
    }

    inline real_t get_z(index_t nid) const
    {
        assert(dim==3);
        return _mesh->_coords[nid*dim+2];
    }

    inline real_t functional_Linf(index_t node) const
    {
        double patch_quality = std::numeric_limits<double>::max();

        for(const auto& ie : _mesh->NEList[node]) {
            patch_quality = std::min(patch_quality, _mesh->quality[ie]);
        }

        return patch_quality;
    }

    inline real_t functional_Linf(index_t n0, const real_t *p, const real_t *mp) const
    {
        real_t f;
        if(dim==2) {
            f = functional_Linf_2d(n0, p, mp);
        } else {
            f = functional_Linf_3d(n0, p, mp);
        }
        return f;
    }

    inline real_t functional_Linf_2d(index_t n0, const real_t *p, const real_t *mp) const
    {
        real_t functional = DBL_MAX;
        for(const auto& ie : _mesh->NEList[n0]) {
            const index_t *n=_mesh->get_element(ie);
            assert(n[0]>=0);
            int iloc = 0;

            while(n[iloc]!=(int)n0) {
                iloc++;
            }
            int loc1 = (iloc+1)%3;
            int loc2 = (iloc+2)%3;

            const real_t *x1 = _mesh->get_coords(n[loc1]);
            const real_t *x2 = _mesh->get_coords(n[loc2]);

            const double *m1 = _mesh->get_metric(n[loc1]);
            const double *m2 = _mesh->get_metric(n[loc2]);

            real_t fnl = property->lipnikov(p,  x1, x2,
                                            mp, m1, m2);
            functional = std::min(functional, fnl);
        }

        return functional;
    }

    inline real_t functional_Linf_3d(index_t n0, const real_t *p, const real_t *mp) const
    {
        real_t functional = DBL_MAX;
        for(const auto& ie : _mesh->NEList[n0]) {
            const index_t *n=_mesh->get_element(ie);
            size_t loc=0;
            for(; loc<4; loc++)
                if(n[loc]==n0)
                    break;

            int n1, n2, n3;
            switch(loc) {
            case 0:
                n1 = n[1];
                n2 = n[2];
                n3 = n[3];
                break;
            case 1:
                n1 = n[2];
                n2 = n[0];
                n3 = n[3];
                break;
            case 2:
                n1 = n[0];
                n2 = n[1];
                n3 = n[3];
                break;
            case 3:
                n1 = n[0];
                n2 = n[2];
                n3 = n[1];
                break;
            }

            const double *x1 = _mesh->get_coords(n1);
            const double *x2 = _mesh->get_coords(n2);
            const double *x3 = _mesh->get_coords(n3);

            const double *m1 = _mesh->get_metric(n1);
            const double *m2 = _mesh->get_metric(n2);
            const double *m3 = _mesh->get_metric(n3);

            real_t fnl = property->lipnikov(p, x1, x2, x3,
                                            mp,m1, m2, m3);

            functional = std::min(functional, fnl);
        }
        return functional;
    }

    inline bool generate_location_2d(index_t node, const real_t *p, double *mp) const
    {
        // Interpolate metric at this new position.
        real_t l[]= {-1, -1, -1};
        int best_e=-1;
        real_t tol=-1;

        for(const auto& ie : _mesh->NEList[node]) {
            const index_t *n=_mesh->get_element(ie);
            assert(n[0]>=0);

            const real_t *x0 = _mesh->get_coords(n[0]);
            const real_t *x1 = _mesh->get_coords(n[1]);
            const real_t *x2 = _mesh->get_coords(n[2]);

            /* Check for inversion by looking at the area
             * of the element whose node is being moved.*/
            real_t area;
            if(n[0]==node) {
                area = property->area(p, x1, x2);
            } else if(n[1]==node) {
                area = property->area(x0, p, x2);
            } else {
                area = property->area(x0, x1, p);
            }
            if(area<0)
                return false;

            real_t L = property->area(x0, x1, x2);

            real_t ll[3];
            ll[0] = property->area(p,  x1, x2)/L;
            ll[1] = property->area(x0, p,  x2)/L;
            ll[2] = property->area(x0, x1, p )/L;

            real_t min_l = std::min(ll[0], std::min(ll[1], ll[2]));
            if(best_e==-1) {
                tol = min_l;
                best_e = ie;
                for(size_t i=0; i<nloc; i++)
                    l[i] = ll[i];
            } else {
                if(min_l>tol) {
                    tol = min_l;
                    best_e = ie;
                    for(size_t i=0; i<nloc; i++)
                        l[i] = ll[i];
                }
            }
        }
        assert(best_e!=-1);
        assert(tol>-DBL_EPSILON);

        const index_t *n=_mesh->get_element(best_e);
        assert(n[0]>=0);

        for(size_t i=0; i<msize; i++)
            mp[i] =
                l[0]*_mesh->metric[n[0]*msize+i]+
                l[1]*_mesh->metric[n[1]*msize+i]+
                l[2]*_mesh->metric[n[2]*msize+i];

        return true;
    }

    inline bool generate_location_3d(index_t node, const real_t *p, double *mp) const
    {
        // Interpolate metric at this new position.
        real_t l[]= {-1, -1, -1, -1};
        int best_e=-1;
        real_t tol=-1;

        for(const auto& ie : _mesh->NEList[node]) {
            const index_t *n=_mesh->get_element(ie);
            assert(n[0]>=0);

            const real_t *x0 = _mesh->get_coords(n[0]);
            const real_t *x1 = _mesh->get_coords(n[1]);
            const real_t *x2 = _mesh->get_coords(n[2]);
            const real_t *x3 = _mesh->get_coords(n[3]);

            /* Check for inversion by looking at the volume
             * of element whose node is being moved.*/
            real_t volume;
            if(n[0]==node) {
                volume = property->volume(p, x1, x2, x3);
            } else if(n[1]==node) {
                volume = property->volume(x0, p, x2, x3);
            } else if(n[2]==node) {
                volume = property->volume(x0, x1, p, x3);
            } else {
                volume = property->volume(x0, x1, x2, p);
            }
            if(volume<0)
                return false;

            real_t L = property->volume(x0, x1, x2, x3);

            real_t ll[4];
            ll[0] = property->volume(p,  x1, x2, x3)/L;
            ll[1] = property->volume(x0, p,  x2, x3)/L;
            ll[2] = property->volume(x0, x1, p,  x3)/L;
            ll[3] = property->volume(x0, x1, x2, p )/L;

            real_t min_l = std::min(std::min(ll[0], ll[1]), std::min(ll[2], ll[3]));
            if(best_e==-1) {
                tol = min_l;
                best_e = ie;
                for(size_t i=0; i<nloc; i++)
                    l[i] = ll[i];
            } else {
                if(min_l>tol) {
                    tol = min_l;
                    best_e = ie;
                    for(size_t i=0; i<nloc; i++)
                        l[i] = ll[i];
                }
            }
        }
        assert(best_e!=-1);
#ifndef NDEBUG
        if(!(tol>-10*DBL_EPSILON)) {
            std::cerr<<__FILE__<<", "<<__LINE__<<" failing with tol="<<tol<<std::endl;
        }
        assert(tol>-10*DBL_EPSILON);
#endif

        const index_t *n=_mesh->get_element(best_e);
        assert(n[0]>=0);

        for(size_t i=0; i<msize; i++)
            mp[i] =
                l[0]*_mesh->metric[n[0]*msize+i]+
                l[1]*_mesh->metric[n[1]*msize+i]+
                l[2]*_mesh->metric[n[2]*msize+i]+
                l[3]*_mesh->metric[n[3]*msize+i];

        return true;
    }

    inline void update_quality(index_t element)
    {
        if(dim==2)
            update_quality_2d(element);
        else
            update_quality_3d(element);

        assert(std::isfinite(_mesh->quality[element]));
    }

    inline void update_quality_2d(index_t element)
    {
        const index_t *n=_mesh->get_element(element);

        assert(n[0]>=0);
        assert(n[1]>=0);
        assert(n[2]>=0);

        const double *x0 = _mesh->get_coords(n[0]);
        const double *x1 = _mesh->get_coords(n[1]);
        const double *x2 = _mesh->get_coords(n[2]);

        const double *m0 = _mesh->get_metric(n[0]);
        const double *m1 = _mesh->get_metric(n[1]);
        const double *m2 = _mesh->get_metric(n[2]);

        _mesh->quality[element] = property->lipnikov(x0, x1, x2,
                                  m0, m1, m2);

        return;
    }

    inline void update_quality_3d(index_t element)
    {
        const index_t *n=_mesh->get_element(element);

        const double *x0 = _mesh->get_coords(n[0]);
        const double *x1 = _mesh->get_coords(n[1]);
        const double *x2 = _mesh->get_coords(n[2]);
        const double *x3 = _mesh->get_coords(n[3]);

        const double *m0 = _mesh->get_metric(n[0]);
        const double *m1 = _mesh->get_metric(n[1]);
        const double *m2 = _mesh->get_metric(n[2]);
        const double *m3 = _mesh->get_metric(n[3]);

        _mesh->quality[element] = property->lipnikov(x0, x1, x2, x3,
                                  m0, m1, m2, m3);

        return;
    }

    /*
    // Compute barycentric coordinates (u, v, w) for
    // point p with respect to triangle (a, b, c)
    void Barycentric(Point a, Point b, Point c, float &u, float &v, float &w)
    {
    Vector v0 = b - a, v1 = c - a, v2 = p - a;
    den = v0.x * v1.y - v1.x * v0.y;
    inv_dev = 1.0 / den;
    v = (v2.x * v1.y - v1.x * v2.y) * inv_den;
    w = (v0.x * v2.y - v2.x * v0.y) * inv_den;
    u = 1.0f - v - w;
    }
    */

    Mesh<real_t> *_mesh;
    ElementProperty<real_t> *property;
    std::vector<Lock> vLocks;

    const size_t nloc, msize;

    int mpi_nparts, rank;
    real_t good_q, epsilon_q;
};

#endif

