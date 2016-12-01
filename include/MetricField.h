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

    /* Gradation.
     */
    void gradation(real_t gamma, real_t maxl)
    {
        for(int i=0; i<_NNodes; i++) {
            real_t Di[dim], Vi[dim*dim];
            _metric[i].eigen_decomp(Di, Vi);

            const real_t *xi=_mesh->get_coords(i);
            for(auto &n : _mesh->NNList[i]) {
                const real_t *xn=_mesh->get_coords(n);

                real_t d;
                if(dim==2)
                    d = (ElementProperty<real_t>::length2d(xi, xn, _metric[i].get_metric()) +
                         ElementProperty<real_t>::length2d(xi, xn, _metric[n].get_metric()))*0.5;

                else
                    d = (ElementProperty<real_t>::length3d(xi, xn, _metric[i].get_metric()) +
                         ElementProperty<real_t>::length3d(xi, xn, _metric[n].get_metric()))*0.5;

                double D[dim];
                for(int j=0;j<dim;j++){
                    double li = 1.0/sqrt(Di[j]);
                    double ln = li + d*gamma;
                    D[j] = 1.0/(ln*ln);
                }

                MetricTensor<real_t, dim> M;
                M.eigen_undecomp(D, Vi);

                _metric[n].constrain(M.get_metric());
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
                    A(0,0)+=pow(x, 4);
                    A(0,1)+=pow(x, 2)*pow(y, 2);
                    A(0,2)+=pow(x, 3)*y;
                    A(1,0)+=pow(x, 2)*pow(y, 2);
                    A(1,1)+=pow(y, 4);
                    A(1,2)+=x*pow(y, 3);
                    A(2,0)+=pow(x, 3)*y;
                    A(2,1)+=x*pow(y, 3);
                    A(2,2)+=pow(x, 2)*pow(y, 2);

                    b[0]+=pow(x, 2);
                    b[1]+=pow(y, 2);
                    b[2]+=x*y;
                }
            }

            Eigen::Matrix<double, 3, 1> S = Eigen::Matrix<real_t, 3, 1>::Zero(3);
            Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::HouseholderQRPreconditioner> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            S = svd.solve(b);

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
                    A(0,0)+=pow(x, 4);
                    A(0,1)+=pow(x, 2)*pow(y, 2);
                    A(0,2)+=pow(x, 2)*pow(z, 2);
                    A(0,3)+=pow(x, 2)*y*z;
                    A(0,4)+=pow(x, 3)*z;
                    A(0,5)+=pow(x, 3)*y;
                    A(1,0)+=pow(x, 2)*pow(y, 2);
                    A(1,1)+=pow(y, 4);
                    A(1,2)+=pow(y, 2)*pow(z, 2);
                    A(1,3)+=pow(y, 3)*z;
                    A(1,4)+=x*pow(y, 2)*z;
                    A(1,5)+=x*pow(y, 3);
                    A(2,0)+=pow(x, 2)*pow(z, 2);
                    A(2,1)+=pow(y, 2)*pow(z, 2);
                    A(2,2)+=pow(z, 4);
                    A(2,3)+=y*pow(z, 3);
                    A(2,4)+=x*pow(z, 3);
                    A(2,5)+=x*y*pow(z, 2);
                    A(3,0)+=pow(x, 2)*y*z;
                    A(3,1)+=pow(y, 3)*z;
                    A(3,2)+=y*pow(z, 3);
                    A(3,3)+=pow(y, 2)*pow(z, 2);
                    A(3,4)+=x*y*pow(z, 2);
                    A(3,5)+=x*pow(y, 2)*z;
                    A(4,0)+=pow(x, 3)*z;
                    A(4,1)+=x*pow(y, 2)*z;
                    A(4,2)+=x*pow(z, 3);
                    A(4,3)+=x*y*pow(z, 2);
                    A(4,4)+=pow(x, 2)*pow(z, 2);
                    A(4,5)+=pow(x, 2)*y*z;
                    A(5,0)+=pow(x, 3)*y;
                    A(5,1)+=x*pow(y, 3);
                    A(5,2)+=x*y*pow(z, 2);
                    A(5,3)+=x*pow(y, 2)*z;
                    A(5,4)+=pow(x, 2)*y*z;
                    A(5,5)+=pow(x, 2)*pow(y, 2);

                    b[0]+=pow(x, 2);
                    b[1]+=pow(y, 2);
                    b[2]+=pow(z, 2);
                    b[3]+=y*z;
                    b[4]+=x*z;
                    b[5]+=x*y;
                }
            }

            Eigen::Matrix<double, 6, 1> S = Eigen::Matrix<real_t, 6, 1>::Zero(6);
            Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>, Eigen::HouseholderQRPreconditioner> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            S = svd.solve(b);

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

    /*! Give const pointer to the metric tensor at one node.
     * @param id is the node index of the metric field being retrieved.
     */
    const real_t* get_metric(int id)
    {
        return _metric[id].get_metric();
    }

    /*! Allocate the metric tensor field. Don't fill it in.
     */
    void alloc_metric()
    {
        if(_metric==NULL)
            _metric = new MetricTensor<real_t,dim>[_NNodes];
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

                A(0,0)+=y*y*y*y;
                A(1,0)+=x*x*y*y;
                A(1,1)+=x*x*x*x;
                A(2,0)+=x*y*y*y;
                A(2,1)+=x*x*x*y;
                A(2,2)+=x*x*y*y;
                A(3,0)+=y*y*y;
                A(3,1)+=x*x*y;
                A(3,2)+=x*y*y;
                A(3,3)+=y*y;
                A(4,0)+=x*y*y;
                A(4,1)+=x*x*x;
                A(4,2)+=x*x*y;
                A(4,3)+=x*y;
                A(4,4)+=x*x;
                A(5,0)+=y*y;
                A(5,1)+=x*x;
                A(5,2)+=x*y;
                A(5,3)+=y;
                A(5,4)+=x;
                A(5,5)+=1;

                b[0]+=psi[*n]*y*y;
                b[1]+=psi[*n]*x*x;
                b[2]+=psi[*n]*x*y;
                b[3]+=psi[*n]*y;
                b[4]+=psi[*n]*x;
                b[5]+=psi[*n];
            }
            A(0,1) = A(1,0);
            A(0,2) = A(2,0);
            A(0,3) = A(3,0);
            A(0,4) = A(4,0);
            A(0,5) = A(5,0);
            A(1,2) = A(2,1);
            A(1,3) = A(3,1);
            A(1,4)= A(4,1);
            A(1,5)= A(5,1);
            A(2,3)= A(3,2);
            A(2,4)= A(4,2);
            A(2,5)= A(5,2);
            A(3,4)= A(4,3);
            A(3,5)= A(5,3);
            A(4,5)= A(5,4);

            Eigen::Matrix<real_t, 6, 1> a = Eigen::Matrix<real_t, 6, 1>::Zero(6);
            Eigen::JacobiSVD<Eigen::Matrix<real_t, 6, 6>, Eigen::HouseholderQRPreconditioner> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            a = svd.solve(b);

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

                A(0,0)+=1;
                A(1,0)+=x;
                A(1,1)+=x*x;
                A(2,0)+=y;
                A(2,1)+=x*y;
                A(2,2)+=y*y;
                A(3,0)+=z;
                A(3,1)+=x*z;
                A(3,2)+=y*z;
                A(3,3)+=z*z;
                A(4,0)+=x*x;
                A(4,1)+=x*x*x;
                A(4,2)+=x*x*y;
                A(4,3)+=x*x*z;
                A(4,4)+=x*x*x*x;
                A(5,0)+=x*y;
                A(5,1)+=x*x*y;
                A(5,2)+=x*y*y;
                A(5,3)+=x*y*z;
                A(5,4)+=x*x*x*y;
                A(5,5)+=x*x*y*y;
                A(6,0)+=x*z;
                A(6,1)+=x*x*z;
                A(6,2)+=x*y*z;
                A(6,3)+=x*z*z;
                A(6,4)+=x*x*x*z;
                A(6,5)+=x*x*y*z;
                A(6,6)+=x*x*z*z;
                A(7,0)+=y*y;
                A(7,1)+=x*y*y;
                A(7,2)+=y*y*y;
                A(7,3)+=y*y*z;
                A(7,4)+=x*x*y*y;
                A(7,5)+=x*y*y*y;
                A(7,6)+=x*y*y*z;
                A(7,7)+=y*y*y*y;
                A(8,0)+=y*z;
                A(8,1)+=x*y*z;
                A(8,2)+=y*y*z;
                A(8,3)+=y*z*z;
                A(8,4)+=x*x*y*z;
                A(8,5)+=x*y*y*z;
                A(8,6)+=x*y*z*z;
                A(8,7)+=y*y*y*z;
                A(8,8)+=y*y*z*z;
                A(9,0)+=z*z;
                A(9,1)+=x*z*z;
                A(9,2)+=y*z*z;
                A(9,3)+=z*z*z;
                A(9,4)+=x*x*z*z;
                A(9,5)+=x*y*z*z;
                A(9,6)+=x*z*z*z;
                A(9,7)+=y*y*z*z;
                A(9,8)+=y*z*z*z;
                A(9,9)+=z*z*z*z;

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

            A(0,1) = A(1,0);
            A(0,2) = A(2,0);
            A(0,3) = A(3,0);
            A(0,4) = A(4,0);
            A(0,5) = A(5,0);
            A(0,6) = A(6,0);
            A(0,7) = A(7,0);
            A(0,8) = A(8,0);
            A(0,9) = A(9,0);
            A(1,2) = A(2,1);
            A(1,3) = A(3,1);
            A(1,4) = A(4,1);
            A(1,5) = A(5,1);
            A(1,6) = A(6,1);
            A(1,7) = A(7,1);
            A(1,8) = A(8,1);
            A(1,9) = A(9,1);
            A(2,3) = A(3,2);
            A(2,4) = A(4,2);
            A(2,5) = A(5,2);
            A(2,6) = A(6,2);
            A(2,7) = A(7,2);
            A(2,8) = A(8,2);
            A(2,9) = A(9,2);
            A(3,4) = A(4,3);
            A(3,5) = A(5,3);
            A(3,6) = A(6,3);
            A(3,7) = A(7,3);
            A(3,8) = A(8,3);
            A(3,9) = A(9,3);
            A(4,5) = A(5,4);
            A(4,6) = A(6,4);
            A(4,7) = A(7,4);
            A(4,8) = A(8,4);
            A(4,9) = A(9,4);
            A(5,6) = A(6,5);
            A(5,7) = A(7,5);
            A(5,8) = A(8,5);
            A(5,9) = A(9,5);
            A(6,7) = A(7,6);
            A(6,8) = A(8,6);
            A(6,9) = A(9,6);
            A(7,8) = A(8,7);
            A(7,9) = A(9,7);
            A(8,9) = A(9,8);

            Eigen::Matrix<real_t, 10, 1> a = Eigen::Matrix<real_t, 10, 1>::Zero(10);
            Eigen::JacobiSVD<Eigen::Matrix<real_t, 10, 10>, Eigen::HouseholderQRPreconditioner> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            a = svd.solve(b);

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
