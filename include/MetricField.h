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

#ifndef METRICFIELD_H
#define METRICFIELD_H

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cfloat>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "MetricTensor.h"
#include "Mesh.h"
#include "ElementProperty.h"

#include "generate_Steiner_ellipse_3d.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

/*! \brief Constructs metric tensor field which encodes anisotropic
 *  edge size information.
 *
 * Use this class to create a metric tensor field using desired error
 * tolerences and curvature of linear fields subject to a set of constraints.
 * index_t should be set as int32_t for problems with less than 2 billion nodes.
 * For larger meshes int64_t should be used.
 */
template<typename real_t, int dim> class MetricField
{
public:
    /*! Default constructor.
    */
    MetricField(Mesh<real_t>& mesh)
    {
        _NNodes = mesh.get_number_nodes();
        _NElements = mesh.get_number_elements();
        _mesh = &mesh;
        _metric = NULL;

        rank = 0;
        nprocs = 1;
#ifdef HAVE_MPI
        MPI_Comm_size(_mesh->get_mpi_comm(), &nprocs);
        MPI_Comm_rank(_mesh->get_mpi_comm(), &rank);
#endif

        double bbox[dim*2];
        for(int i=0; i<dim; i++) {
            bbox[i*2] = DBL_MAX;
            bbox[i*2+1] = -DBL_MAX;
        }
        #pragma omp parallel
        {
            double lbbox[dim*2];
            for(int i=0; i<dim; i++) {
                lbbox[i*2] = DBL_MAX;
                lbbox[i*2+1] = -DBL_MAX;
            }
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++) {
                const real_t *x = _mesh->get_coords(i);

                for(int j=0; j<dim; j++) {
                    lbbox[j*2] = std::min(lbbox[j*2], x[j]);
                    lbbox[j*2+1] = std::max(lbbox[j*2+1], x[j]);
                }
            }

            #pragma omp critical
            {
                for(int j=0; j<dim; j++) {
                    bbox[j*2] = std::min(lbbox[j*2], bbox[j*2]);
                    bbox[j*2+1] = std::max(lbbox[j*2+1], bbox[j*2+1]);
                }
            }
        }
        double max_extent = bbox[1]-bbox[0];
        for(int j=1; j<dim; j++)
            max_extent = std::max(max_extent, bbox[j*2+1]-bbox[j*2]);

        min_eigenvalue = 1.0/pow(max_extent, 2);
    }

    /*! Default destructor.
    */
    ~MetricField()
    {
        if(_metric!=NULL)
            delete [] _metric;
    }

    void generate_mesh_metric(double resolution_scaling_factor)
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];

        if(dim==2) {
            #pragma omp parallel
            {
                double alpha = pow(1.0/resolution_scaling_factor, 2);
                #pragma omp for schedule(static)
                for(int i=0; i<_NNodes; i++)
                {
                    real_t m[3];

                    fit_ellipsoid(i, m);

                    for(int j=0; j<3; j++)
                        m[j]*=alpha;

                    _metric[i].set_metric(m);
                }
            }
        } else {
            #pragma omp parallel
            {
                double alpha = pow(1.0/resolution_scaling_factor, 2);
                #pragma omp for schedule(static)
                for(int i=0; i<_NNodes; i++)
                {
                    real_t m[6];

                    fit_ellipsoid(i, m);

                    for(int j=0; j<6; j++)
                        m[j]*=alpha;

                    _metric[i].set_metric(m);
                }
            }
        }
    }

    /* Start of code generated by fit_ellipsoid_3d.py. Warning - be careful about modifying
       any of the generated code directly.  Any changes/fixes should be done
       in the code generation script generation.
       */

    void fit_ellipsoid(int i, real_t *sm)
    {
        if(dim==2) {
            Eigen::Matrix<double, 3, 3> A = Eigen::Matrix<real_t, 3, 3>::Zero(3,3);
            Eigen::Matrix<double, 3, 1> b = Eigen::Matrix<real_t, 3, 1>::Zero(3);

            std::vector<index_t> nodes = _mesh->NNList[i];
            nodes.push_back(i);

            for(typename std::vector<index_t>::const_iterator it=nodes.begin(); it!=nodes.end(); ++it) {
                const real_t *X0=_mesh->get_coords(*it);
                real_t x0=X0[0], y0=X0[1];
                assert(std::isfinite(x0));
                assert(std::isfinite(y0));

                for(typename std::vector<index_t>::const_iterator n=_mesh->NNList[*it].begin(); n!=_mesh->NNList[*it].end(); ++n) {
                    if(*n<=*it)
                        continue;

                    const real_t *X=_mesh->get_coords(*n);
                    real_t x=X[0]-x0, y=X[1]-y0;

                    assert(std::isfinite(x));
                    assert(std::isfinite(y));
                    if(x<0) {
                        x*=-1;
                        y*=-1;
                    }
                    A[0]+=pow(x, 4);
                    A[1]+=pow(x, 2)*pow(y, 2);
                    A[2]+=pow(x, 3)*y;
                    A[3]+=pow(x, 2)*pow(y, 2);
                    A[4]+=pow(y, 4);
                    A[5]+=x*pow(y, 3);
                    A[6]+=pow(x, 3)*y;
                    A[7]+=x*pow(y, 3);
                    A[8]+=pow(x, 2)*pow(y, 2);

                    b[0]+=pow(x, 2);
                    b[1]+=pow(y, 2);
                    b[2]+=x*y;
                }
            }

            Eigen::Matrix<double, 3, 1> S = Eigen::Matrix<real_t, 3, 1>::Zero(3);

            A.svd().solve(b, &S);

            if(_mesh->NNList[i].size()>=3) {
                sm[0] = S[0];
                sm[1] = S[2];
                sm[2] = S[1];
            } else {
                assert(std::isfinite(S[0]));
                assert(std::isfinite(S[1]));

                sm[0] = S[0];
                sm[1] = 0;
                sm[2] = S[1];
            }
        } else {
            Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<real_t, 6, 6>::Zero(6,6);
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<real_t, 6, 1>::Zero(6);

            std::vector<index_t> nodes = _mesh->NNList[i];
            nodes.push_back(i);

            for(typename std::vector<index_t>::const_iterator it=nodes.begin(); it!=nodes.end(); ++it) {
                const real_t *X0=_mesh->get_coords(*it);
                real_t x0=X0[0], y0=X0[1], z0=X0[2];
                assert(std::isfinite(x0));
                assert(std::isfinite(y0));
                assert(std::isfinite(z0));

                for(typename std::vector<index_t>::const_iterator n=_mesh->NNList[*it].begin(); n!=_mesh->NNList[*it].end(); ++n) {
                    if(*n<=*it)
                        continue;

                    const real_t *X=_mesh->get_coords(*n);
                    real_t x=X[0]-x0, y=X[1]-y0, z=X[2]-z0;

                    assert(std::isfinite(x));
                    assert(std::isfinite(y));
                    assert(std::isfinite(z));
                    if(x<0) {
                        x*=-1;
                        y*=-1;
                        z*=-1;
                    }
                    A[0]+=pow(x, 4);
                    A[1]+=pow(x, 2)*pow(y, 2);
                    A[2]+=pow(x, 2)*pow(z, 2);
                    A[3]+=pow(x, 2)*y*z;
                    A[4]+=pow(x, 3)*z;
                    A[5]+=pow(x, 3)*y;
                    A[6]+=pow(x, 2)*pow(y, 2);
                    A[7]+=pow(y, 4);
                    A[8]+=pow(y, 2)*pow(z, 2);
                    A[9]+=pow(y, 3)*z;
                    A[10]+=x*pow(y, 2)*z;
                    A[11]+=x*pow(y, 3);
                    A[12]+=pow(x, 2)*pow(z, 2);
                    A[13]+=pow(y, 2)*pow(z, 2);
                    A[14]+=pow(z, 4);
                    A[15]+=y*pow(z, 3);
                    A[16]+=x*pow(z, 3);
                    A[17]+=x*y*pow(z, 2);
                    A[18]+=pow(x, 2)*y*z;
                    A[19]+=pow(y, 3)*z;
                    A[20]+=y*pow(z, 3);
                    A[21]+=pow(y, 2)*pow(z, 2);
                    A[22]+=x*y*pow(z, 2);
                    A[23]+=x*pow(y, 2)*z;
                    A[24]+=pow(x, 3)*z;
                    A[25]+=x*pow(y, 2)*z;
                    A[26]+=x*pow(z, 3);
                    A[27]+=x*y*pow(z, 2);
                    A[28]+=pow(x, 2)*pow(z, 2);
                    A[29]+=pow(x, 2)*y*z;
                    A[30]+=pow(x, 3)*y;
                    A[31]+=x*pow(y, 3);
                    A[32]+=x*y*pow(z, 2);
                    A[33]+=x*pow(y, 2)*z;
                    A[34]+=pow(x, 2)*y*z;
                    A[35]+=pow(x, 2)*pow(y, 2);

                    b[0]+=pow(x, 2);
                    b[1]+=pow(y, 2);
                    b[2]+=pow(z, 2);
                    b[3]+=y*z;
                    b[4]+=x*z;
                    b[5]+=x*y;
                }
            }

            Eigen::Matrix<double, 6, 1> S = Eigen::Matrix<real_t, 6, 1>::Zero(6);

            A.svd().solve(b, &S);

            if(_mesh->NNList[i].size()>=6) {
                sm[0] = S[0];
                sm[1] = S[5];
                sm[2] = S[4];
                sm[3] = S[1];
                sm[4] = S[3];
                sm[5] = S[2];
            } else {
                sm[0] = S[0];
                sm[1] = 0;
                sm[2] = 0;
                sm[3] = S[1];
                sm[4] = 0;
                sm[5] = S[2];
            }
        }

        return;
    }

    /* End of code generated by fit_ellipsoid_3d.py. Warning - be careful about
       modifying any of the generated code directly.  Any changes/fixes
       should be done in the code generation script generation.*/

    void generate_Steiner_ellipse(double resolution_scaling_factor)
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];

        if(dim==2) {
            std::cerr<<"ERROR: void generate_Steiner_ellipse() not yet implemented in 2D.\n";
            exit(-1);
        } else {
            std::vector<double> SteinerMetricField(_NElements*6);
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for(int i=0; i<_NElements; i++) {
                    const index_t *n=_mesh->get_element(i);

                    const real_t *x0 = _mesh->get_coords(n[0]);
                    const real_t *x1 = _mesh->get_coords(n[1]);
                    const real_t *x2 = _mesh->get_coords(n[2]);
                    const real_t *x3 = _mesh->get_coords(n[3]);

                    pragmatic::generate_Steiner_ellipse(x0, x1, x2, x3, SteinerMetricField.data()+i*6);
                }

                double alpha = pow(1.0/resolution_scaling_factor, 2);
                #pragma omp for schedule(static)
                for(int i=0; i<_NNodes; i++) {
                    double sm[6];
                    for(int j=0; j<6; j++)
                        sm[j] = 0.0;

                    for(typename std::set<index_t>::const_iterator ie=_mesh->NEList[i].begin(); ie!=_mesh->NEList[i].end(); ++ie) {
                        for(int j=0; j<6; j++)
                            sm[j]+=SteinerMetricField[(*ie)*6+j];
                    }

                    double scale = alpha/_mesh->NEList[i].size();
                    for(int j=0; j<6; j++)
                        sm[j]*=scale;

                    _metric[i].set_metric(sm);
                }
            }
        }
    }

    /*! Copy back the metric tensor field.
     * @param metric is a pointer to the buffer where the metric field can be copied.
     */
    void get_metric(real_t* metric)
    {
        // Enforce first-touch policy.
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++) {
                _metric[i].get_metric(metric+i*(dim==2?3:6));
            }
        }
    }

    /*! Set the metric tensor field. It is assumed that only the top triangle of the tensors are stored.
     * @param metric is a pointer to the buffer where the metric field is to be copied from.
     */
    void set_metric(const real_t* metric)
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];

        for(int i=0; i<_NNodes; i++) {
            _metric[i].set_metric(metric+i*(dim==2?3:6));
        }
    }

    /*! Set the metric tensor field. It is assumed that the full form of the tensors are stored.
     * @param metric is a pointer to the buffer where the metric field is to be copied from.
     */
    void set_metric_full(const real_t* metric)
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];

        real_t m[dim==2?3:6];
        for(int i=0; i<_NNodes; i++) {
            if(dim==2) {
                m[0] = metric[i*4];
                m[1] = metric[i*4+1];
                m[2] = metric[i*4+3];
            } else {
                m[0] = metric[i*9];
                m[1] = metric[i*9+1];
                m[2] = metric[i*9+2];
                m[3] = metric[i*9+4];
                m[4] = metric[i*9+5];
                m[5] = metric[i*9+8];
            }
            _metric[i].set_metric(m);
        }
    }


    /*! Set the metric tensor field.
     * @param metric is a pointer to the buffer where the metric field is to be copied from.
     * @param id is the node index of the metric field being set.
     */
    void set_metric(const real_t* metric, int id)
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];

        _metric[id].set_metric(metric);
    }

    /// Update the metric field on the mesh.
    void relax_mesh(double omega)
    {
        assert(_metric!=NULL);

        size_t pNElements = (size_t)predict_nelements_part();

        // We don't really know how much addition space we'll need so this was set after some experimentation.
        size_t fudge = 5;

        if(pNElements > _mesh->NElements) {
            // Let's leave a safety margin.
            pNElements *= fudge;
        } else {
            /* The mesh can contain more elements than the predicted number, however
             * some elements may still need to be refined, therefore until the mesh
             * is coarsened and defraged we need extra space for the new vertices and
             * elements that will be created during refinement.
             */
            pNElements = _mesh->NElements * fudge;
        }

        // In 2D, the number of nodes is ~ 1/2 the number of elements.
        // In 3D, the number of nodes is ~ 1/6 the number of elements.
        size_t pNNodes = pNElements/(dim==2?2:6);

        _mesh->_ENList.resize(pNElements*(dim+1));
        _mesh->boundary.resize(pNElements*(dim+1));
        _mesh->quality.resize(pNElements);
        _mesh->_coords.resize(pNNodes*dim);
        _mesh->metric.resize(pNNodes*(dim==2?3:6));
        _mesh->NNList.resize(pNNodes);
        _mesh->NEList.resize(pNNodes);
        _mesh->node_owner.resize(pNNodes, -1);
        _mesh->lnn2gnn.resize(pNNodes, -1);

#ifdef HAVE_MPI
        // At this point we can establish a new, gappy global numbering system
        if(nprocs>1)
            _mesh->create_gappy_global_numbering(pNElements);
#endif

        // Enforce first-touch policy
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++) {
                double M[dim==2?3:6];
                _metric[i].get_metric(M);
                for(int j=0; j<(dim==2?3:6); j++)
                    _mesh->metric[i*(dim==2?3:6)+j] = (1.0-omega)*_mesh->metric[i*(dim==2?3:6)+j] + omega*M[j];
                MetricTensor<real_t,dim>::positive_definiteness(&(_mesh->metric[i*(dim==2?3:6)]));
            }
        }

#ifdef HAVE_MPI
        // Halo update if parallel
        halo_update<double, (dim==2?3:6)>(_mesh->get_mpi_comm(), _mesh->send, _mesh->recv, _mesh->metric);
#endif
    }


    /// Update the metric field on the mesh.
    void update_mesh()
    {
        assert(_metric!=NULL);

        size_t pNElements = (size_t)predict_nelements_part();

        // We don't really know how much addition space we'll need so this was set after some experimentation.
        size_t fudge = 5;

        if(pNElements > _mesh->NElements) {
            // Let's leave a safety margin.
            pNElements *= fudge;
        } else {
            /* The mesh can contain more elements than the predicted number, however
             * some elements may still need to be refined, therefore until the mesh
             * is coarsened and defraged we need extra space for the new vertices and
             * elements that will be created during refinement.
             */
            pNElements = _mesh->NElements * fudge;
        }

        // In 2D, the number of nodes is ~ 1/2 the number of elements.
        // In 3D, the number of nodes is ~ 1/6 the number of elements.
        size_t pNNodes = std::max(pNElements/(dim==2?2:6), _mesh->get_number_nodes());

        _mesh->_ENList.resize(pNElements*(dim+1));
        _mesh->boundary.resize(pNElements*(dim+1));
        _mesh->quality.resize(pNElements);
        _mesh->_coords.resize(pNNodes*dim);
        _mesh->metric.resize(pNNodes*(dim==2?3:6));
        _mesh->NNList.resize(pNNodes);
        _mesh->NEList.resize(pNNodes);
        _mesh->node_owner.resize(pNNodes, -1);
        _mesh->lnn2gnn.resize(pNNodes, -1);

#ifdef HAVE_MPI
        // At this point we can establish a new, gappy global numbering system
        if(nprocs>1)
            _mesh->create_gappy_global_numbering(pNElements);
#endif

        // Enforce first-touch policy
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++) {
                _metric[i].get_metric(&(_mesh->metric[i*(dim==2?3:6)]));
            }
            #pragma omp for schedule(static)
            for(int i=0; i<_NElements; i++) {
                _mesh->template update_quality<dim>(i);
            }
        }

#ifdef HAVE_MPI
        // Halo update if parallel
        halo_update<double, (dim==2?3:6)>(_mesh->get_mpi_comm(), _mesh->send, _mesh->recv, _mesh->metric);
#endif
    }

    /*! Add the contribution from the metric field from a new field with a target linear interpolation error.
     * @param psi is field while curvature is to be considered.
     * @param target_error is the user target error for a given norm.
     * @param p_norm Set this optional argument to a positive integer to
     * apply the p-norm scaling to the metric, as in Chen, Sun and Xu,
     * Mathematics of Computation, Volume 76, Number 257, January 2007,
     * pp. 179-204.
     */
    void add_field(const real_t* psi, const real_t target_error, int p_norm=-1)
    {
        bool add_to=true;
        if(_metric==NULL) {
            add_to = false;
            _metric = new MetricTensor<real_t,dim>[_NNodes];
        }

        real_t eta = 1.0/target_error;
        #pragma omp parallel
        {
            // Calculate Hessian at each point.
            double h[dim==2?3:6];

            if(p_norm>0) {
                #pragma omp for schedule(static) nowait
                for(int i=0; i<_NNodes; i++) {
                    hessian_qls_kernel(psi, i, h);

                    double m_det;
                    if(dim==2) {
                        /*|h[0] h[1]|
                          |h[1] h[2]|*/
                        m_det = fabs(h[0]*h[2]-h[1]*h[1]);
                    } else if(dim==3) {
                        /*|h[0] h[1] h[2]|
                          |h[1] h[3] h[4]|
                          |h[2] h[4] h[5]|

                          sympy
                          h0,h1,h2,h3,h4,h5 = symbols("h[0], h[1], h[2], h[3], h[4], h[5]")
                          M = Matrix([[h0, h1, h2],
                          [h1, h3, h4],
                          [h2, h4, h5]])
                          print_ccode(det(M))
                          */
                        m_det = fabs(h[0]*h[3]*h[5] - h[0]*pow(h[4], 2) - pow(h[1], 2)*h[5] + 2*h[1]*h[2]*h[4] - pow(h[2], 2)*h[3]);
                    }

                    double scaling_factor = eta * pow(m_det+DBL_EPSILON, -1.0 / (2.0 * p_norm + dim));

                    if(std::isnormal(scaling_factor)) {
                        for(int j=0; j<(dim==2?3:6); j++)
                            h[j] *= scaling_factor;
                    } else {
                        if(dim==2) {
                            h[0] = min_eigenvalue;
                            h[1] = 0.0;
                            h[2] = min_eigenvalue;
                        } else {
                            h[0] = min_eigenvalue;
                            h[1] = 0.0;
                            h[2] = 0.0;
                            h[3] = min_eigenvalue;
                            h[4] = 0.0;
                            h[5] = min_eigenvalue;
                        }
                    }

                    if(add_to) {
                        // Merge this metric with the existing metric field.
                        _metric[i].constrain(h);
                    } else {
                        _metric[i].set_metric(h);
                    }
                }
            } else {
                #pragma omp for schedule(static)
                for(int i=0; i<_NNodes; i++) {
                    hessian_qls_kernel(psi, i, h);

                    for(int j=0; j<(dim==2?3:6); j++)
                        h[j] *= eta;

                    if(add_to) {
                        // Merge this metric with the existing metric field.
                        _metric[i].constrain(h);
                    } else {
                        _metric[i].set_metric(h);
                    }
                }
            }
        }
    }

    /*! Apply maximum edge length constraint.
     * @param max_len specifies the maximum allowed edge length.
     */
    void apply_max_edge_length(real_t max_len)
    {
        real_t M[dim==2?3:6];
        real_t m = 1.0/(max_len*max_len);

        if(dim==2) {
            M[0] = m;
            M[1] = 0.0;
            M[2] = m;
        } else if(dim==3) {
            M[0] = m;
            M[1] = 0.0;
            M[2] = 0.0;
            M[3] = m;
            M[4] = 0.0;
            M[5] = m;
        }

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++)
                _metric[i].constrain(&(M[0]));
        }
    }

    /*! Apply minimum edge length constraint.
     * @param min_len specifies the minimum allowed edge length globally.
     */
    void apply_min_edge_length(real_t min_len)
    {
        real_t M[dim==2?3:6];
        double m = 1.0/(min_len*min_len);

        if(dim==2) {
            M[0] = m;
            M[1] = 0.0;
            M[2] = m;
        } else if(dim==3) {
            M[0] = m;
            M[1] = 0.0;
            M[2] = 0.0;
            M[3] = m;
            M[4] = 0.0;
            M[5] = m;
        }

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++)
                _metric[i].constrain(M, false);
        }
    }

    /*! Apply minimum edge length constraint.
     * @param min_len specifies the minimum allowed edge length locally at each vertex.
     */
    void apply_min_edge_length(const real_t *min_len)
    {
        #pragma omp parallel
        {
            real_t M[dim==2?3:6];
            #pragma omp for schedule(static)
            for(int n=0; n<_NNodes; n++)
            {
                double m = 1.0/(min_len[n]*min_len[n]);

                if(dim==2) {
                    M[0] = m;
                    M[1] = 0.0;
                    M[2] = m;
                } else if(dim==3) {
                    M[0] = m;
                    M[1] = 0.0;
                    M[2] = 0.0;
                    M[3] = m;
                    M[4] = 0.0;
                    M[5] = m;
                }

                _metric[n].constrain(M, false);
            }
        }
    }

    /*! Apply maximum aspect ratio constraint.
     * @param max_aspect_ratio maximum aspect ratio for elements.
     */
    void apply_max_aspect_ratio(real_t max_aspect_ratio)
    {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++)
                _metric[i].limit_aspect_ratio(max_aspect_ratio);
        }
    }

    /*! Apply maximum number of elements constraint.
     * @param nelements the maximum number of elements desired.
     */
    void apply_max_nelements(real_t nelements)
    {
        int predicted = predict_nelements();
        if(predicted>nelements)
            apply_nelements(nelements);
    }

    /*! Apply minimum number of elements constraint.
     * @param nelements the minimum number of elements desired.
     */
    void apply_min_nelements(real_t nelements)
    {
        int predicted = predict_nelements();
        if(predicted<nelements)
            apply_nelements(nelements);
    }

    /*! Apply required number of elements.
     * @param nelements is the required number of elements after adapting.
     */
    void apply_nelements(real_t nelements)
    {
        double scale_factor = nelements/predict_nelements();
        if(dim==3)
            scale_factor = pow(scale_factor, 2.0/3.0);

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<_NNodes; i++)
                _metric[i].scale(scale_factor);
        }
    }

    /*! Predict the number of elements in this partition when mesh satisfies metric tensor field.
    */
    real_t predict_nelements_part()
    {
        real_t predicted;

        if(dim==2) {
            real_t total_area_metric = 0.0;

            const real_t inv3=1.0/3.0;

            const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
            const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
            const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
            ElementProperty<real_t> property(refx0, refx1, refx2);

            #pragma omp parallel for reduction(+:total_area_metric)
            for(int i=0; i<_NElements; i++) {
                const index_t *n=_mesh->get_element(i);

                const real_t *x0 = _mesh->get_coords(n[0]);
                const real_t *x1 = _mesh->get_coords(n[1]);
                const real_t *x2 = _mesh->get_coords(n[2]);
                real_t area = property.area(x0, x1, x2);

                const real_t *m0=_metric[n[0]].get_metric();
                const real_t *m1=_metric[n[1]].get_metric();
                const real_t *m2=_metric[n[2]].get_metric();

                real_t m00 = (m0[0]+m1[0]+m2[0])*inv3;
                real_t m01 = (m0[1]+m1[1]+m2[1])*inv3;
                real_t m11 = (m0[2]+m1[2]+m2[2])*inv3;

                real_t det = std::abs(m00*m11-m01*m01);

                total_area_metric += area*sqrt(det);
            }

            /* Ideal area of equilateral triangle in metric space:
               s^2*sqrt(3)/4 where s is the length of the side. However, algorithm
               allows lengths from 1/sqrt(2) up to sqrt(2) in metric
               space. Therefore:*/
            const real_t smallest_ideal_area = 0.5*(sqrt(3.0)/4.0);
            predicted = total_area_metric/smallest_ideal_area;
        } else if(dim==3) {
            real_t total_volume_metric = 0.0;

            const real_t *refx0 = _mesh->get_coords(_mesh->get_element(0)[0]);
            const real_t *refx1 = _mesh->get_coords(_mesh->get_element(0)[1]);
            const real_t *refx2 = _mesh->get_coords(_mesh->get_element(0)[2]);
            const real_t *refx3 = _mesh->get_coords(_mesh->get_element(0)[3]);
            ElementProperty<real_t> property(refx0, refx1, refx2, refx3);

            #pragma omp parallel for reduction(+:total_volume_metric)
            for(int i=0; i<_NElements; i++) {
                const index_t *n=_mesh->get_element(i);

                const real_t *x0 = _mesh->get_coords(n[0]);
                const real_t *x1 = _mesh->get_coords(n[1]);
                const real_t *x2 = _mesh->get_coords(n[2]);
                const real_t *x3 = _mesh->get_coords(n[3]);
                real_t volume = property.volume(x0, x1, x2, x3);

                const real_t *m0=_metric[n[0]].get_metric();
                const real_t *m1=_metric[n[1]].get_metric();
                const real_t *m2=_metric[n[2]].get_metric();
                const real_t *m3=_metric[n[3]].get_metric();

                real_t m00 = (m0[0]+m1[0]+m2[0]+m3[0])*0.25;
                real_t m01 = (m0[1]+m1[1]+m2[1]+m3[1])*0.25;
                real_t m02 = (m0[2]+m1[2]+m2[2]+m3[2])*0.25;
                real_t m11 = (m0[3]+m1[3]+m2[3]+m3[3])*0.25;
                real_t m12 = (m0[4]+m1[4]+m2[4]+m3[4])*0.25;
                real_t m22 = (m0[5]+m1[5]+m2[5]+m3[5])*0.25;

                real_t det = (m11*m22 - m12*m12)*m00 - (m01*m22 - m02*m12)*m01 + (m01*m12 - m02*m11)*m02;

                assert(det>-DBL_EPSILON);
                total_volume_metric += volume*sqrt(det);
            }

            // Ideal volume of triangle in metric space.
            real_t ideal_volume = 1.0/sqrt(72.0);
            predicted = total_volume_metric/ideal_volume;
        }

        return predicted;
    }

    /*! Predict the number of elements when mesh satisfies metric tensor field.
    */
    real_t predict_nelements()
    {
        double predicted=predict_nelements_part();

#ifdef HAVE_MPI
        if(nprocs>1) {
            MPI_Allreduce(MPI_IN_PLACE, &predicted, 1, _mesh->MPI_REAL_T, MPI_SUM, _mesh->get_mpi_comm());
        }
#endif

        return predicted;
    }

private:

    /// Least squared Hessian recovery.
    void hessian_qls_kernel(const real_t *psi, int i, real_t *Hessian)
    {
        int min_patch_size = (dim==2?6:15); // In 3D, 10 is the minimum but can give crappy results.

        std::set<index_t> patch = _mesh->get_node_patch(i, min_patch_size);
        patch.insert(i);

        if(dim==2) {
            // Form quadratic system to be solved. The quadratic fit is:
            // P = a0*y^2+a1*x^2+a2*x*y+a3*y+a4*x+a5
            // A = P^TP
            Eigen::Matrix<real_t, 6, 6> A = Eigen::Matrix<real_t, 6, 6>::Zero(6,6);
            Eigen::Matrix<real_t, 6, 1> b = Eigen::Matrix<real_t, 6, 1>::Zero(6);

            real_t x0=_mesh->_coords[i*2], y0=_mesh->_coords[i*2+1];

            for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++) {
                real_t x=_mesh->_coords[(*n)*2]-x0, y=_mesh->_coords[(*n)*2+1]-y0;

                A[0]+=y*y*y*y;
                A[6]+=x*x*y*y;
                A[7]+=x*x*x*x;
                A[12]+=x*y*y*y;
                A[13]+=x*x*x*y;
                A[14]+=x*x*y*y;
                A[18]+=y*y*y;
                A[19]+=x*x*y;
                A[20]+=x*y*y;
                A[21]+=y*y;
                A[24]+=x*y*y;
                A[25]+=x*x*x;
                A[26]+=x*x*y;
                A[27]+=x*y;
                A[28]+=x*x;
                A[30]+=y*y;
                A[31]+=x*x;
                A[32]+=x*y;
                A[33]+=y;
                A[34]+=x;
                A[35]+=1;

                b[0]+=psi[*n]*y*y;
                b[1]+=psi[*n]*x*x;
                b[2]+=psi[*n]*x*y;
                b[3]+=psi[*n]*y;
                b[4]+=psi[*n]*x;
                b[5]+=psi[*n];
            }
            A[1] = A[6];
            A[2] = A[12];
            A[3] = A[18];
            A[4] = A[24];
            A[5] = A[30];
            A[8] = A[13];
            A[9] = A[19];
            A[10]= A[25];
            A[11]= A[31];
            A[15]= A[20];
            A[16]= A[26];
            A[17]= A[32];
            A[22]= A[27];
            A[23]= A[33];
            A[29]= A[34];

            Eigen::Matrix<real_t, 6, 1> a = Eigen::Matrix<real_t, 6, 1>::Zero(6);
            A.svd().solve(b, &a);

            Hessian[0] = 2*a[1]; // d2/dx2
            Hessian[1] = a[2];   // d2/dxdy
            Hessian[2] = 2*a[0]; // d2/dy2

        } else if(dim==3) {
            // Form quadratic system to be solved. The quadratic fit is:
            // P = 1 + x + y + z + x^2 + y^2 + z^2 + xy + xz + yz
            // A = P^TP
            Eigen::Matrix<real_t, 10, 10> A = Eigen::Matrix<real_t, 10, 10>::Zero(10,10);
            Eigen::Matrix<real_t, 10, 1> b = Eigen::Matrix<real_t, 10, 1>::Zero(10);

            real_t x0=_mesh->_coords[i*3], y0=_mesh->_coords[i*3+1], z0=_mesh->_coords[i*3+2];
            assert(std::isfinite(x0));
            assert(std::isfinite(y0));
            assert(std::isfinite(z0));

            for(typename std::set<index_t>::const_iterator n=patch.begin(); n!=patch.end(); n++) {
                real_t x=_mesh->_coords[(*n)*3]-x0, y=_mesh->_coords[(*n)*3+1]-y0, z=_mesh->_coords[(*n)*3+2]-z0;
                assert(std::isfinite(x));
                assert(std::isfinite(y));
                assert(std::isfinite(z));
                assert(std::isfinite(psi[*n]));

                A[0]+=1;
                A[10]+=x;
                A[11]+=x*x;
                A[20]+=y;
                A[21]+=x*y;
                A[22]+=y*y;
                A[30]+=z;
                A[31]+=x*z;
                A[32]+=y*z;
                A[33]+=z*z;
                A[40]+=x*x;
                A[41]+=x*x*x;
                A[42]+=x*x*y;
                A[43]+=x*x*z;
                A[44]+=x*x*x*x;
                A[50]+=x*y;
                A[51]+=x*x*y;
                A[52]+=x*y*y;
                A[53]+=x*y*z;
                A[54]+=x*x*x*y;
                A[55]+=x*x*y*y;
                A[60]+=x*z;
                A[61]+=x*x*z;
                A[62]+=x*y*z;
                A[63]+=x*z*z;
                A[64]+=x*x*x*z;
                A[65]+=x*x*y*z;
                A[66]+=x*x*z*z;
                A[70]+=y*y;
                A[71]+=x*y*y;
                A[72]+=y*y*y;
                A[73]+=y*y*z;
                A[74]+=x*x*y*y;
                A[75]+=x*y*y*y;
                A[76]+=x*y*y*z;
                A[77]+=y*y*y*y;
                A[80]+=y*z;
                A[81]+=x*y*z;
                A[82]+=y*y*z;
                A[83]+=y*z*z;
                A[84]+=x*x*y*z;
                A[85]+=x*y*y*z;
                A[86]+=x*y*z*z;
                A[87]+=y*y*y*z;
                A[88]+=y*y*z*z;
                A[90]+=z*z;
                A[91]+=x*z*z;
                A[92]+=y*z*z;
                A[93]+=z*z*z;
                A[94]+=x*x*z*z;
                A[95]+=x*y*z*z;
                A[96]+=x*z*z*z;
                A[97]+=y*y*z*z;
                A[98]+=y*z*z*z;
                A[99]+=z*z*z*z;

                b[0]+=psi[*n]*1;
                b[1]+=psi[*n]*x;
                b[2]+=psi[*n]*y;
                b[3]+=psi[*n]*z;
                b[4]+=psi[*n]*x*x;
                b[5]+=psi[*n]*x*y;
                b[6]+=psi[*n]*x*z;
                b[7]+=psi[*n]*y*y;
                b[8]+=psi[*n]*y*z;
                b[9]+=psi[*n]*z*z;
            }

            A[1] = A[10];
            A[2]  = A[20];
            A[3]  = A[30];
            A[4]  = A[40];
            A[5]  = A[50];
            A[6]  = A[60];
            A[7]  = A[70];
            A[8]  = A[80];
            A[9]  = A[90];
            A[12] = A[21];
            A[13] = A[31];
            A[14] = A[41];
            A[15] = A[51];
            A[16] = A[61];
            A[17] = A[71];
            A[18] = A[81];
            A[19] = A[91];
            A[23] = A[32];
            A[24] = A[42];
            A[25] = A[52];
            A[26] = A[62];
            A[27] = A[72];
            A[28] = A[82];
            A[29] = A[92];
            A[34] = A[43];
            A[35] = A[53];
            A[36] = A[63];
            A[37] = A[73];
            A[38] = A[83];
            A[39] = A[93];
            A[45] = A[54];
            A[46] = A[64];
            A[47] = A[74];
            A[48] = A[84];
            A[49] = A[94];
            A[56] = A[65];
            A[57] = A[75];
            A[58] = A[85];
            A[59] = A[95];
            A[67] = A[76];
            A[68] = A[86];
            A[69] = A[96];
            A[78] = A[87];
            A[79] = A[97];
            A[89] = A[98];

            Eigen::Matrix<real_t, 10, 1> a = Eigen::Matrix<real_t, 10, 1>::Zero(10);
            A.svd().solve(b, &a);

            Hessian[0] = a[4]*2.0; // d2/dx2
            Hessian[1] = a[5];     // d2/dxdy
            Hessian[2] = a[6];     // d2/dxdz
            Hessian[3] = a[7]*2.0; // d2/dy2
            Hessian[4] = a[8];     // d2/dydz
            Hessian[5] = a[9]*2.0; // d2/dz2
        }
    }

private:
    int rank, nprocs;
    int _NNodes, _NElements;
    MetricTensor<real_t,dim>* _metric;
    Mesh<real_t>* _mesh;
    double min_eigenvalue;
};

#endif
