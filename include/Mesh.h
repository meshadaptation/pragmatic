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

#ifndef MESH_H
#define MESH_H

#include <algorithm>
#include <vector>
#include <set>
#include <stack>
#include <cmath>
#include <stdint.h>

#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
#include <boost/unordered_map.hpp>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "mpi_tools.h"

#include "PragmaticTypes.h"
#include "PragmaticMinis.h"

#include "ElementProperty.h"
#include "MetricTensor.h"
#include "HaloExchange.h"

/*! \brief Manages mesh data.
 *
 * This class is used to store the mesh and associated meta-data.
 */

template<typename real_t> class Mesh
{
public:

    /*! 2D triangular mesh constructor. This is for use when there is no MPI.
     *
     * @param NNodes number of nodes in the local mesh.
     * @param NElements number of nodes in the local mesh.
     * @param ENList array storing the global node number for each element.
     * @param x is the X coordinate.
     * @param y is the Y coordinate.
     */
    Mesh(int NNodes, int NElements, const index_t *ENList, const real_t *x, const real_t *y)
    {
#ifdef HAVE_MPI
        _mpi_comm = MPI_COMM_WORLD;
#endif
        _init(NNodes, NElements, ENList, x, y, NULL, NULL, NULL);
    }

#ifdef HAVE_MPI
    /*! 2D triangular mesh constructor. This is used when parallelised with MPI.
     *
     * @param NNodes number of nodes in the local mesh.
     * @param NElements number of nodes in the local mesh.
     * @param ENList array storing the global node number for each element.
     * @param x is the X coordinate.
     * @param y is the Y coordinate.
     * @param lnn2gnn mapping of local node numbering to global node numbering.
     * @param owner_range range of node id's owned by each partition.
     * @param mpi_comm the mpi communicator.
     */
    Mesh(int NNodes, int NElements, const index_t *ENList,
         const real_t *x, const real_t *y, const index_t *lnn2gnn,
         const index_t *owner_range, MPI_Comm mpi_comm)
    {
        _mpi_comm = mpi_comm;
        _init(NNodes, NElements, ENList, x, y, NULL, lnn2gnn, owner_range);
    }
#endif

    /*! 3D tetrahedra mesh constructor. This is for use when there is no MPI.
     *
     * @param NNodes number of nodes in the local mesh.
     * @param NElements number of nodes in the local mesh.
     * @param ENList array storing the global node number for each element.
     * @param x is the X coordinate.
     * @param y is the Y coordinate.
     * @param z is the Z coordinate.
     */
    Mesh(int NNodes, int NElements, const index_t *ENList,
         const real_t *x, const real_t *y, const real_t *z)
    {
#ifdef HAVE_MPI
        _mpi_comm = MPI_COMM_WORLD;
#endif
        _init(NNodes, NElements, ENList, x, y, z, NULL, NULL);
    }

#ifdef HAVE_MPI
    /*! 3D tetrahedra mesh constructor. This is used when parallelised with MPI.
     *
     * @param NNodes number of nodes in the local mesh.
     * @param NElements number of nodes in the local mesh.
     * @param ENList array storing the global node number for each element.
     * @param x is the X coordinate.
     * @param y is the Y coordinate.
     * @param z is the Z coordinate.
     * @param lnn2gnn mapping of local node numbering to global node numbering.
     * @param owner_range range of node id's owned by each partition.
     * @param mpi_comm the mpi communicator.
     */
    Mesh(int NNodes, int NElements, const index_t *ENList,
         const real_t *x, const real_t *y, const real_t *z, const index_t *lnn2gnn,
         const index_t *owner_range, MPI_Comm mpi_comm)
    {
        _mpi_comm = mpi_comm;
        _init(NNodes, NElements, ENList, x, y, z, lnn2gnn, owner_range);
    }
#endif

    /// Default destructor.
    ~Mesh()
    {
        delete property;
    }

    /// Add a new vertex
    index_t append_vertex(const real_t *x, const double *m)
    {
        for(size_t i=0; i<ndims; i++)
            _coords[ndims*NNodes+i] = x[i];

        for(size_t i=0; i<msize; i++)
            metric[msize*NNodes+i] = m[i];

        ++NNodes;

        return get_number_nodes()-1;
    }

    /// Erase a vertex
    void erase_vertex(const index_t nid)
    {
        NNList[nid].clear();
        NEList[nid].clear();
        node_owner[nid] = rank;
        lnn2gnn[nid] = -1;
    }

    /// Add a new element
    index_t append_element(const index_t *n)
    {
        if(_ENList.size() < (NElements+1)*nloc) {
            _ENList.resize(2*NElements*nloc);
            boundary.resize(2*NElements*nloc);
            quality.resize(2*NElements);
        }

        for(size_t i=0; i<nloc; i++)
            _ENList[nloc*NElements+i] = n[i];

        ++NElements;

        return get_number_elements()-1;
    }

    void create_boundary()
    {
        assert(boundary.size()==0);

        size_t NNodes = get_number_nodes();
        size_t NElements = get_number_elements();

        // Initialise the boundary array
        boundary.resize(NElements*nloc);
        std::fill(boundary.begin(), boundary.end(), -2);

        #pragma omp parallel
        {
            if(ndims==2) {
                // Check neighbourhood of each element
                #pragma omp for schedule(guided)
                for(size_t i=0; i<NElements; i++) {
                    if(_ENList[i*3]==-1)
                        continue;

                    for(int j=0; j<3; j++) {
                        int n1 = _ENList[i*3+(j+1)%3];
                        int n2 = _ENList[i*3+(j+2)%3];

                        if(is_owned_node(n1)||is_owned_node(n2)) {
                            std::set<int> neighbours;
                            set_intersection(NEList[n1].begin(), NEList[n1].end(),
                                             NEList[n2].begin(), NEList[n2].end(),
                                             inserter(neighbours, neighbours.begin()));

                            if(neighbours.size()==2) {
                                if(*neighbours.begin()==(int)i)
                                    boundary[i*3+j] = *neighbours.rbegin();
                                else
                                    boundary[i*3+j] = *neighbours.begin();
                            }
                        } else {
                            // This is a halo facet.
                            boundary[i*3+j] = -1;
                        }
                    }
                }
            } else { // ndims==3
                // Check neighbourhood of each element
                #pragma omp for schedule(guided)
                for(size_t i=0; i<NElements; i++) {
                    if(_ENList[i*4]==-1)
                        continue;

                    for(int j=0; j<4; j++) {
                        int n1 = _ENList[i*4+(j+1)%4];
                        int n2 = _ENList[i*4+(j+2)%4];
                        int n3 = _ENList[i*4+(j+3)%4];

                        if(is_owned_node(n1)||is_owned_node(n2)||is_owned_node(n3)) {
                            std::set<int> edge_neighbours;
                            set_intersection(NEList[n1].begin(), NEList[n1].end(),
                                             NEList[n2].begin(), NEList[n2].end(),
                                             inserter(edge_neighbours, edge_neighbours.begin()));

                            std::set<int> neighbours;
                            set_intersection(NEList[n3].begin(), NEList[n3].end(),
                                             edge_neighbours.begin(), edge_neighbours.end(),
                                             inserter(neighbours, neighbours.begin()));

                            if(neighbours.size()==2) {
                                if(*neighbours.begin()==(int)i)
                                    boundary[i*4+j] = *neighbours.rbegin();
                                else
                                    boundary[i*4+j] = *neighbours.begin();
                            }
                        } else {
                            // This is a halo facet.
                            boundary[i*4+j] = -1;
                        }
                    }
                }
            }
        }

        for(std::vector<int>::iterator it=boundary.begin(); it!=boundary.end(); ++it) {
            if(*it==-2)
                *it = 1;
            else if(*it>=0)
                *it = 0;
        }
    }


    void set_boundary(int nfacets, const int *facets, const int *ids)
    {
        assert(boundary.size()==0);
        create_boundary();

        // Create a map of facets to ids.
        std::map< std::set<int>, int> facet2id;
        for(int i=0; i<nfacets; i++) {
            std::set<int> facet;
            for(int j=0; j<ndims; j++) {
                facet.insert(facets[i*ndims+j]);
            }
            assert(facet2id.find(facet)==facet2id.end());
            facet2id[facet] = ids[i];
        }

        // Sweep through boundary and set ids.
        size_t NElements = get_number_elements();
        for(int i=0; i<NElements; i++) {
            for(int j=0; j<nloc; j++) {
                if(boundary[i*nloc+j]==1) {
                    std::set<int> facet;
                    for(int k=1; k<nloc; k++) {
                        facet.insert(_ENList[i*nloc+(j+k)%nloc]);
                    }
                    assert(facet2id.find(facet)!=facet2id.end());
                    boundary[i*nloc+j] = facet2id[facet];
                }
            }
        }
    }

    /// Erase an element
    void erase_element(const index_t eid)
    {
        const index_t *n = get_element(eid);

        for(size_t i=0; i<nloc; ++i)
            NEList[n[i]].erase(eid);

        _ENList[eid*nloc] = -1;
    }

    /// Flip orientation of element.
    void invert_element(size_t eid)
    {
        int tmp = _ENList[eid*nloc];
        _ENList[eid*nloc] = _ENList[eid*nloc+1];
        _ENList[eid*nloc+1] = tmp;
    }

    /// Return a pointer to the element-node list.
    inline const index_t *get_element(size_t eid) const
    {
        return &(_ENList[eid*nloc]);
    }

    /// Return copy of element-node list.
    inline void get_element(size_t eid, index_t *ele) const
    {
        for(size_t i=0; i<nloc; i++)
            ele[i] = _ENList[eid*nloc+i];
    }

    /// Return the number of nodes in the mesh.
    inline size_t get_number_nodes() const
    {
        return NNodes;
    }

    /// Return the number of elements in the mesh.
    inline size_t get_number_elements() const
    {
        return NElements;
    }

    /// Return the number of spatial dimensions.
    inline size_t get_number_dimensions() const
    {
        return ndims;
    }

    /// Return positions vector.
    inline const real_t *get_coords(index_t nid) const
    {
        return &(_coords[nid*ndims]);
    }

    /// Return copy of the coordinate.
    inline void get_coords(index_t nid, real_t *x) const
    {
        for(size_t i=0; i<ndims; i++)
            x[i] = _coords[nid*ndims+i];
        return;
    }

    /// Return metric at that vertex.
    inline const double *get_metric(index_t nid) const
    {
        assert(metric.size()>0);
        return &(metric[nid*msize]);
    }

    /// Return copy of metric.
    inline void get_metric(index_t nid, double *m) const
    {
        assert(metric.size()>0);
        for(size_t i=0; i<msize; i++)
            m[i] = metric[nid*msize+i];
        return;
    }
    
    /// Return the array of boundary facets and associated tags
    inline  int * get_boundaryTags() 
    {
      return boundary.data();
    }

    /// Returns true if the node is in any of the partitioned elements.
    inline bool is_halo_node(index_t nid) const
    {
        return (node_owner[nid]!= rank || send_halo.count(nid)>0);
    }

    /// Returns true if the node is assigned to the local partition.
    inline bool is_owned_node(index_t nid) const
    {
        return node_owner[nid] == rank;
    }

    /// Get the mean edge length metric space.
    double get_lmean()
    {
        int NNodes = get_number_nodes();
        double total_length=0;
        int nedges=0;

        #pragma omp parallel for reduction(+:total_length,nedges)
        for(int i=0; i<NNodes; i++) {
            if(is_owned_node(i) && (NNList[i].size()>0)) {
                for(typename std::vector<index_t>::const_iterator it=NNList[i].begin(); it!=NNList[i].end(); ++it) {
                    if(i<*it) { // Ensure that every edge length is only calculated once.
                        total_length += calc_edge_length(i, *it);
                        nedges++;
                    }
                }
            }
        }

#ifdef HAVE_MPI
        if(num_processes>1) {
            MPI_Allreduce(MPI_IN_PLACE, &total_length, 1, MPI_DOUBLE, MPI_SUM, _mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE, &nedges, 1, MPI_INT, MPI_SUM, _mpi_comm);
        }
#endif

        double mean = total_length/nedges;

        return mean;
    }

    /// Calculate perimeter
    double calculate_perimeter()
    {
        int NElements = get_number_elements();
        if(ndims==2) {
            long double total_length=0;

            for(int i=0; i<NElements; i++) {
                if(_ENList[i*nloc] < 0)
                    continue;

                for(int j=0; j<3; j++) {
                    int n1 = _ENList[i*nloc+(j+1)%3];
                    int n2 = _ENList[i*nloc+(j+2)%3];

                    if(boundary[i*nloc+j]>0 && (std::min(node_owner[n1], node_owner[n2])==rank)) {
                        long double dx = ((long double)_coords[n1*2  ]-(long double)_coords[n2*2  ]);
                        long double dy = ((long double)_coords[n1*2+1]-(long double)_coords[n2*2+1]);

                        total_length += std::sqrt(dx*dx+dy*dy);
                    }
                }
            }

#ifdef HAVE_MPI
            if(num_processes>1)
                MPI_Allreduce(MPI_IN_PLACE, &total_length, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
#endif
            return total_length;
        } else {
            std::cerr<<"ERROR: calculate_perimeter() cannot be used for 3D. Use calculate_area() instead if you want the total surface area.\n";
            return -1;
        }
    }


    /// Calculate area - optimise for percision rather than performance as it is only used for verification.
    double calculate_area() const
    {
        int NElements = get_number_elements();
        long double total_area=0;

        if(ndims==2) {
            for(int i=0; i<NElements; i++) {
                const index_t *n=get_element(i);
                if(n[0] < 0)
                    continue;

                if(num_processes>1) {
                    // Don't sum if it's not ours
                    if(std::min(node_owner[n[0]], std::min(node_owner[n[1]], node_owner[n[2]]))!=rank)
                        continue;
                }

                const double *x1 = get_coords(n[0]);
                const double *x2 = get_coords(n[1]);
                const double *x3 = get_coords(n[2]);

                // Use Heron's Formula
                long double a;
                {
                    long double dx = ((long double)x1[0]-(long double)x2[0]);
                    long double dy = ((long double)x1[1]-(long double)x2[1]);
                    a = std::sqrt(dx*dx+dy*dy);
                }
                long double b;
                {
                    long double dx = ((long double)x1[0]-(long double)x3[0]);
                    long double dy = ((long double)x1[1]-(long double)x3[1]);
                    b = std::sqrt(dx*dx+dy*dy);
                }
                long double c;
                {
                    long double dx = ((long double)x2[0]-(long double)x3[0]);
                    long double dy = ((long double)x2[1]-(long double)x3[1]);
                    c = std::sqrt(dx*dx+dy*dy);
                }
                long double s = (a+b+c)/2;

                total_area += std::sqrt(s*(s-a)*(s-b)*(s-c));
            }

#ifdef HAVE_MPI
            if(num_processes>1)
                MPI_Allreduce(MPI_IN_PLACE, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
#endif
        } else { // 3D
            for(int i=0; i<NElements; i++) {
                const index_t *n=get_element(i);
                if(n[0] < 0)
                    continue;

                for(int j=0; j<4; j++) {
                    if(boundary[i*nloc+j]<=0)
                        continue;

                    int n1 = n[(j+1)%4];
                    int n2 = n[(j+2)%4];
                    int n3 = n[(j+3)%4];

                    if(num_processes>1) {
                        // Don't sum if it's not ours
                        if(std::min(node_owner[n1], std::min(node_owner[n2], node_owner[n3]))!=rank)
                            continue;
                    }

                    const double *x1 = get_coords(n1);
                    const double *x2 = get_coords(n2);
                    const double *x3 = get_coords(n3);

                    // Use Heron's Formula
                    long double a;
                    {
                        long double dx = ((long double)x1[0]-(long double)x2[0]);
                        long double dy = ((long double)x1[1]-(long double)x2[1]);
                        long double dz = ((long double)x1[2]-(long double)x2[2]);
                        a = std::sqrt(dx*dx+dy*dy+dz*dz);
                    }
                    long double b;
                    {
                        long double dx = ((long double)x1[0]-(long double)x3[0]);
                        long double dy = ((long double)x1[1]-(long double)x3[1]);
                        long double dz = ((long double)x1[2]-(long double)x3[2]);
                        b = std::sqrt(dx*dx+dy*dy+dz*dz);
                    }
                    long double c;
                    {
                        long double dx = ((long double)x2[0]-(long double)x3[0]);
                        long double dy = ((long double)x2[1]-(long double)x3[1]);
                        long double dz = ((long double)x2[2]-(long double)x3[2]);
                        c = std::sqrt(dx*dx+dy*dy+dz*dz);
                    }
                    long double s = (a+b+c)/2;

                    total_area += std::sqrt(s*(s-a)*(s-b)*(s-c));
                }
            }

#ifdef HAVE_MPI
            if(num_processes>1)
                MPI_Allreduce(MPI_IN_PLACE, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
#endif
        }
        return total_area;
    }

    /// Calculate volume
    double calculate_volume() const
    {
        int NElements = get_number_elements();
        long double total_volume=0;

        if(ndims==2) {
            std::cerr<<"ERROR: Cannot calculate volume in 2D\n";
        } else { // 3D
            if(num_processes>1) {
                #pragma omp parallel for reduction(+:total_volume)
                for(int i=0; i<NElements; i++) {
                    const index_t *n=get_element(i);
                    if(n[0] < 0)
                        continue;

                    // Don't sum if it's not ours
                    if(std::min(std::min(node_owner[n[0]], node_owner[n[1]]), std::min(node_owner[n[2]], node_owner[n[3]]))!=rank)
                        continue;

                    const double *x0 = get_coords(n[0]);
                    const double *x1 = get_coords(n[1]);
                    const double *x2 = get_coords(n[2]);
                    const double *x3 = get_coords(n[3]);

                    long double x01 = (x0[0] - x1[0]);
                    long double x02 = (x0[0] - x2[0]);
                    long double x03 = (x0[0] - x3[0]);

                    long double y01 = (x0[1] - x1[1]);
                    long double y02 = (x0[1] - x2[1]);
                    long double y03 = (x0[1] - x3[1]);

                    long double z01 = (x0[2] - x1[2]);
                    long double z02 = (x0[2] - x2[2]);
                    long double z03 = (x0[2] - x3[2]);

                    total_volume += (-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - x01*(z03*y02 - z02*y03));
                }

#ifdef HAVE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &total_volume, 1, MPI_LONG_DOUBLE, MPI_SUM, _mpi_comm);
#endif
            } else {
                #pragma omp parallel for reduction(+:total_volume)
                for(int i=0; i<NElements; i++) {
                    const index_t *n=get_element(i);
                    if(n[0] < 0)
                        continue;

                    const double *x0 = get_coords(n[0]);
                    const double *x1 = get_coords(n[1]);
                    const double *x2 = get_coords(n[2]);
                    const double *x3 = get_coords(n[3]);

                    long double x01 = (x0[0] - x1[0]);
                    long double x02 = (x0[0] - x2[0]);
                    long double x03 = (x0[0] - x3[0]);

                    long double y01 = (x0[1] - x1[1]);
                    long double y02 = (x0[1] - x2[1]);
                    long double y03 = (x0[1] - x3[1]);

                    long double z01 = (x0[2] - x1[2]);
                    long double z02 = (x0[2] - x2[2]);
                    long double z03 = (x0[2] - x3[2]);

                    total_volume += (-x03*(z02*y01 - z01*y02) + x02*(z03*y01 - z01*y03) - x01*(z03*y02 - z02*y03));
                }
            }
        }
        return total_volume/6;
    }

    /// Get the element mean quality in metric space.
    double get_qmean() const
    {
        double sum=0;
        int nele=0;

        #pragma omp parallel for reduction(+:sum, nele)
        for(size_t i=0; i<NElements; i++) {
            const index_t *n=get_element(i);
            if(n[0]<0)
                continue;

            double q;
            if(ndims==2) {
                q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]),
                                       get_metric(n[0]), get_metric(n[1]), get_metric(n[2]));
            } else {
                q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]),
                                       get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3]));
            }

            sum+=q;
            nele++;
        }

#ifdef HAVE_MPI
        if(num_processes>1) {
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, _mpi_comm);
            MPI_Allreduce(MPI_IN_PLACE, &nele, 1, MPI_INT, MPI_SUM, _mpi_comm);
        }
#endif

        if(nele>0)
            return sum/nele;
        else
            return 0;
    }

    /// Print out the qualities. Useful if you want to plot a histogram of element qualities.
    void print_quality() const
    {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(size_t i=0; i<NElements; i++)
            {
                const index_t *n=get_element(i);
                if(n[0]<0)
                    continue;

                double q;
                if(ndims==2) {
                    q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]),
                    get_metric(n[0]), get_metric(n[1]), get_metric(n[2]));
                } else {
                    q = property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]),
                                           get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3]));
                }
                #pragma omp critical
                std::cout<<"Quality[ele="<<i<<"] = "<<q<<std::endl;
            }
        }
    }

    /// Get the element minimum quality in metric space.
    double get_qmin() const
    {
        if(ndims==2)
            return get_qmin_2d();
        else
            return get_qmin_3d();
    }

    double get_qmin_2d() const
    {
        double qmin=1; // Where 1 is ideal.

        #pragma omp parallel for reduction(min:qmin)
        for(size_t i=0; i<NElements; i++) {
            const index_t *n=get_element(i);
            if(n[0]<0)
                continue;

            qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]),
                            get_metric(n[0]), get_metric(n[1]), get_metric(n[2])));
        }

#ifdef HAVE_MPI
        if(num_processes>1)
            MPI_Allreduce(MPI_IN_PLACE, &qmin, 1, MPI_DOUBLE, MPI_MIN, _mpi_comm);
#endif

        return qmin;
    }

    double get_qmin_3d() const
    {
        double qmin=1; // Where 1 is ideal.

        #pragma omp parallel for reduction(min:qmin)
        for(size_t i=0; i<NElements; i++) {
            const index_t *n=get_element(i);
            if(n[0]<0)
                continue;

            qmin = std::min(qmin, property->lipnikov(get_coords(n[0]), get_coords(n[1]), get_coords(n[2]), get_coords(n[3]),
                            get_metric(n[0]), get_metric(n[1]), get_metric(n[2]), get_metric(n[3])));
        }

#ifdef HAVE_MPI
        if(num_processes>1)
            MPI_Allreduce(MPI_IN_PLACE, &qmin, 1, MPI_DOUBLE, MPI_MIN, _mpi_comm);
#endif

        return qmin;
    }

#ifdef HAVE_MPI
    /// Return the MPI communicator.
    MPI_Comm get_mpi_comm() const
    {
        return _mpi_comm;
    }
#endif

    /// Return the node id's connected to the specified node_id
    std::set<index_t> get_node_patch(index_t nid) const
    {
        assert(nid<(index_t)NNodes);
        std::set<index_t> patch;
        for(typename std::vector<index_t>::const_iterator it=NNList[nid].begin(); it!=NNList[nid].end(); ++it)
            patch.insert(patch.end(), *it);
        return patch;
    }

    /// Grow a node patch around node id's until it reaches a minimum size.
    std::set<index_t> get_node_patch(index_t nid, size_t min_patch_size)
    {
        std::set<index_t> patch = get_node_patch(nid);

        if(patch.size()<min_patch_size) {
            std::set<index_t> front = patch, new_front;
            for(;;) {
                for(typename std::set<index_t>::const_iterator it=front.begin(); it!=front.end(); it++) {
                    for(typename std::vector<index_t>::const_iterator jt=NNList[*it].begin(); jt!=NNList[*it].end(); jt++) {
                        if(patch.find(*jt)==patch.end()) {
                            new_front.insert(*jt);
                            patch.insert(*jt);
                        }
                    }
                }

                if(patch.size()>=std::min(min_patch_size, NNodes))
                    break;

                front.swap(new_front);
            }
        }

        return patch;
    }

    /// Calculates the edge lengths in metric space.
    real_t calc_edge_length(index_t nid0, index_t nid1) const
    {
        real_t length=-1.0;
        if(ndims==2) {
            double m[3];
            m[0] = (metric[nid0*3  ]+metric[nid1*3  ])*0.5;
            m[1] = (metric[nid0*3+1]+metric[nid1*3+1])*0.5;
            m[2] = (metric[nid0*3+2]+metric[nid1*3+2])*0.5;

            length = ElementProperty<real_t>::length2d(get_coords(nid0), get_coords(nid1), m);
        } else {
            double m[6];
            m[0] = (metric[nid0*msize  ]+metric[nid1*msize  ])*0.5;
            m[1] = (metric[nid0*msize+1]+metric[nid1*msize+1])*0.5;
            m[2] = (metric[nid0*msize+2]+metric[nid1*msize+2])*0.5;

            m[3] = (metric[nid0*msize+3]+metric[nid1*msize+3])*0.5;
            m[4] = (metric[nid0*msize+4]+metric[nid1*msize+4])*0.5;

            m[5] = (metric[nid0*msize+5]+metric[nid1*msize+5])*0.5;

            length = ElementProperty<real_t>::length3d(get_coords(nid0), get_coords(nid1), m);
        }
        return length;
    }

    real_t maximal_edge_length() const
    {
        double L_max = 0.0;

        #pragma omp parallel for reduction(max:L_max)
        for(index_t i=0; i<(index_t) NNodes; i++) {
            for(typename std::vector<index_t>::const_iterator it=NNList[i].begin(); it!=NNList[i].end(); ++it) {
                if(i<*it) { // Ensure that every edge length is only calculated once.
                    L_max = std::max(L_max, calc_edge_length(i, *it));
                }
            }
        }

#ifdef HAVE_MPI
        if(num_processes>1)
            MPI_Allreduce(MPI_IN_PLACE, &L_max, 1, MPI_DOUBLE, MPI_MAX, _mpi_comm);
#endif

        return L_max;
    }

    /*! Defragment mesh. This compresses the storage of internal data
      structures. This is useful if the mesh has been significantly
      coarsened. */
    void defragment()
    {
        // Discover which vertices and elements are active.
        std::vector<index_t> active_vertex_map(NNodes);

        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<NNodes; i++) {
            active_vertex_map[i] = -1;
            NNList[i].clear();
            NEList[i].clear();
        }

        // Identify active elements.
        std::vector<index_t> active_element;

        active_element.reserve(NElements);

        std::map<index_t, std::set<int> > new_send_set, new_recv_set;
        for(size_t e=0; e<NElements; e++) {
            index_t nid = _ENList[e*nloc];

            // Check if deleted.
            if(nid<0)
                continue;

            // Check if wholly owned by another process or if halo node.
            bool local=false, halo_element=false;
            for(size_t j=0; j<nloc; j++) {
                nid = _ENList[e*nloc+j];
                if(recv_halo.count(nid)) {
                    halo_element = true;
                } else {
                    local = true;
                }
            }
            if(!local)
                continue;

            // Need these mesh entities.
            active_element.push_back(e);

            std::set<int> neigh;
            for(size_t j=0; j<nloc; j++) {
                nid = _ENList[e*nloc+j];
                active_vertex_map[nid]=0;

                if(halo_element) {
                    for(int k=0; k<num_processes; k++) {
                        if(recv_map[k].count(lnn2gnn[nid])) {
                            new_recv_set[k].insert(nid);
                            neigh.insert(k);
                        }
                    }
                }
            }
            for(size_t j=0; j<nloc; j++) {
                nid = _ENList[e*nloc+j];
                for(std::set<int>::iterator kt=neigh.begin(); kt!=neigh.end(); ++kt) {
                    if(send_map[*kt].count(lnn2gnn[nid]))
                        new_send_set[*kt].insert(nid);
                }
            }
        }

        // Create a new numbering.
        index_t cnt=0;
        for(size_t i=0; i<NNodes; i++) {
            if(active_vertex_map[i]<0)
                continue;

            active_vertex_map[i] = cnt++;
        }

        // Renumber elements
        int active_nelements = active_element.size();
        std::map< std::set<index_t>, index_t > ordered_elements;
        for(int i=0; i<active_nelements; i++) {
            index_t old_eid = active_element[i];
            std::set<index_t> sorted_element;
            for(size_t j=0; j<nloc; j++) {
                index_t new_nid = active_vertex_map[_ENList[old_eid*nloc+j]];
                sorted_element.insert(new_nid);
            }
            if(ordered_elements.find(sorted_element)==ordered_elements.end()) {
                ordered_elements[sorted_element] = old_eid;
            } else {
                std::cerr<<"dup! "
                         <<active_vertex_map[_ENList[old_eid*nloc]]<<" "
                         <<active_vertex_map[_ENList[old_eid*nloc+1]]<<" "
                         <<active_vertex_map[_ENList[old_eid*nloc+2]]<<std::endl;
            }
        }
        std::vector<index_t> element_renumber;
        element_renumber.reserve(ordered_elements.size());
        for(typename std::map< std::set<index_t>, index_t >::const_iterator it=ordered_elements.begin(); it!=ordered_elements.end(); ++it) {
            element_renumber.push_back(it->second);
        }

        // Compress data structures.
        NNodes = cnt;
        NElements = ordered_elements.size();

        std::vector<index_t> defrag_ENList(NElements*nloc);
        std::vector<real_t> defrag_coords(NNodes*ndims);
        std::vector<double> defrag_metric(NNodes*msize);
        std::vector<int> defrag_boundary(NElements*nloc);
        std::vector<double> defrag_quality(NElements);

        // This first touch is to bind memory locally.
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<(int)NElements; i++) {
                defrag_ENList[i*nloc] = 0;
                defrag_boundary[i*nloc] = 0;
            }

            #pragma omp for schedule(static)
            for(int i=0; i<(int)NNodes; i++) {
                defrag_coords[i*ndims] = 0.0;
                defrag_metric[i*msize] = 0.0;
            }
        }

        // Second sweep writes elements with new numbering.
        for(int i=0; i<NElements; i++) {
            index_t old_eid = element_renumber[i];
            index_t new_eid = i;
            for(size_t j=0; j<nloc; j++) {
                index_t new_nid = active_vertex_map[_ENList[old_eid*nloc+j]];
                assert(new_nid<(index_t)NNodes);
                defrag_ENList[new_eid*nloc+j] = new_nid;
                defrag_boundary[new_eid*nloc+j] = boundary[old_eid*nloc+j];
            }
            defrag_quality[new_eid] = quality[old_eid];
        }

        // Second sweep writes node data with new numbering.
        for(size_t old_nid=0; old_nid<active_vertex_map.size(); ++old_nid) {
            index_t new_nid = active_vertex_map[old_nid];
            if(new_nid<0)
                continue;

            for(size_t j=0; j<ndims; j++)
                defrag_coords[new_nid*ndims+j] = _coords[old_nid*ndims+j];
            for(size_t j=0; j<msize; j++)
                defrag_metric[new_nid*msize+j] = metric[old_nid*msize+j];
        }

        memcpy(&_ENList[0], &defrag_ENList[0], NElements*nloc*sizeof(index_t));
        memcpy(&boundary[0], &defrag_boundary[0], NElements*nloc*sizeof(int));
        memcpy(&quality[0], &defrag_quality[0], NElements*sizeof(double));
        memcpy(&_coords[0], &defrag_coords[0], NNodes*ndims*sizeof(real_t));
        memcpy(&metric[0], &defrag_metric[0], NNodes*msize*sizeof(double));

        // Renumber halo, fix lnn2gnn and node_owner.
        if(num_processes>1) {
            std::vector<index_t> defrag_lnn2gnn(NNodes);
            std::vector<int> defrag_owner(NNodes);

            for(size_t old_nid=0; old_nid<active_vertex_map.size(); ++old_nid) {
                index_t new_nid = active_vertex_map[old_nid];
                if(new_nid<0)
                    continue;

                defrag_lnn2gnn[new_nid] = lnn2gnn[old_nid];
                defrag_owner[new_nid] = node_owner[old_nid];
            }

            lnn2gnn.swap(defrag_lnn2gnn);
            node_owner.swap(defrag_owner);

            for(int k=0; k<num_processes; k++) {
                std::vector<int> new_halo;
                send_map[k].clear();
                for(std::vector<int>::iterator jt=send[k].begin(); jt!=send[k].end(); ++jt) {
                    if(new_send_set[k].count(*jt)) {
                        index_t new_lnn = active_vertex_map[*jt];
                        new_halo.push_back(new_lnn);
                        send_map[k][lnn2gnn[new_lnn]] = new_lnn;
                    }
                }
                send[k].swap(new_halo);
            }

            for(int k=0; k<num_processes; k++) {
                std::vector<int> new_halo;
                recv_map[k].clear();
                for(std::vector<int>::iterator jt=recv[k].begin(); jt!=recv[k].end(); ++jt) {
                    if(new_recv_set[k].count(*jt)) {
                        index_t new_lnn = active_vertex_map[*jt];
                        new_halo.push_back(new_lnn);
                        recv_map[k][lnn2gnn[new_lnn]] = new_lnn;
                    }
                }
                recv[k].swap(new_halo);
            }

            {
                send_halo.clear();
                for(int k=0; k<num_processes; k++) {
                    for(std::vector<int>::iterator jt=send[k].begin(); jt!=send[k].end(); ++jt) {
                        send_halo.insert(*jt);
                    }
                }
            }

            {
                recv_halo.clear();
                for(int k=0; k<num_processes; k++) {
                    for(std::vector<int>::iterator jt=recv[k].begin(); jt!=recv[k].end(); ++jt) {
                        recv_halo.insert(*jt);
                    }
                }
            }
        } else {
            for(size_t i=0; i<NNodes; ++i) {
                lnn2gnn[i] = i;
                node_owner[i] = 0;
            }
        }

        #pragma omp parallel
        create_adjacency();
    }

    /// This is used to verify that the mesh and its metadata is correct.
    bool verify() const
    {
        bool state = true;

        std::vector<int> mpi_node_owner(NNodes, rank);
        if(num_processes>1)
            for(int p=0; p<num_processes; p++)
                for(std::vector<int>::const_iterator it=recv[p].begin(); it!=recv[p].end(); ++it) {
                    mpi_node_owner[*it] = p;
                }
        std::vector<int> mpi_ele_owner(NElements, rank);
        if(num_processes>1)
            for(size_t i=0; i<NElements; i++) {
                if(_ENList[i*nloc]<0)
                    continue;
                int owner = mpi_node_owner[_ENList[i*nloc]];
                for(size_t j=1; j<nloc; j++)
                    owner = std::min(owner, mpi_node_owner[_ENList[i*nloc+j]]);
                mpi_ele_owner[i] = owner;
            }

        // Check for the correctness of NNList and NEList.
        std::vector< std::set<index_t> > local_NEList(NNodes);
        std::vector< std::set<index_t> > local_NNList(NNodes);
        for(size_t i=0; i<NElements; i++) {
            if(_ENList[i*nloc]<0)
                continue;

            for(size_t j=0; j<nloc; j++) {
                index_t nid_j = _ENList[i*nloc+j];

                local_NEList[nid_j].insert(i);
                for(size_t k=j+1; k<nloc; k++) {
                    index_t nid_k = _ENList[i*nloc+k];
                    local_NNList[nid_j].insert(nid_k);
                    local_NNList[nid_k].insert(nid_j);
                }
            }
        }
        {
            if(rank==0) std::cout<<"VERIFY: NNList..................";
            if(NNList.size()==0) {
                if(rank==0) std::cout<<"empty\n";
            } else {
                bool valid_nnlist=true;
                for(size_t i=0; i<NNodes; i++) {
                    size_t active_cnt=0;
                    for(size_t j=0; j<NNList[i].size(); j++) {
                        if(NNList[i][j]>=0)
                            active_cnt++;
                    }
                    if(active_cnt!=local_NNList[i].size()) {
                        valid_nnlist=false;

                        std::cerr<<std::endl<<"active_cnt!=local_NNList[i].size() "<<active_cnt<<", "<<local_NNList[i].size()<<std::endl;
                        std::cerr<<"NNList["
                                 <<i
                                 <<"("
                                 <<get_coords(i)[0]<<", "
                                 <<get_coords(i)[1]<<", ";
                        if(ndims==3) std::cerr<<get_coords(i)[2]<<", ";
                        std::cerr<<send_halo.count(i)<<", "<<recv_halo.count(i)<<")] =       ";
                        for(size_t j=0; j<NNList[i].size(); j++)
                            std::cerr<<NNList[i][j]<<"("<<NNList[NNList[i][j]].size()<<", "
                                     <<send_halo.count(NNList[i][j])<<", "
                                     <<recv_halo.count(NNList[i][j])<<") ";
                        std::cerr<<std::endl;
                        std::cerr<<"local_NNList["<<i<<"] = ";
                        for(typename std::set<index_t>::iterator kt=local_NNList[i].begin(); kt!=local_NNList[i].end(); ++kt)
                            std::cerr<<*kt<<" ";
                        std::cerr<<std::endl;

                        state = false;
                    }
                }
                if(rank==0) {
                    if(valid_nnlist) {
                        std::cout<<"pass\n";
                    } else {
                        state = false;
                        std::cout<<"warn\n";
                    }
                }
            }
        }
        {
            if(rank==0) std::cout<<"VERIFY: NEList..................";
            std::string result="pass\n";
            if(NEList.size()==0) {
                result = "empty\n";
            } else {
                for(size_t i=0; i<NNodes; i++) {
                    if(local_NEList[i].size()!=NEList[i].size()) {
                        result = "fail (NEList[i].size()!=local_NEList[i].size())\n";
                        state = false;
                        break;
                    }
                    if(local_NEList[i].size()==0)
                        continue;
                    if(local_NEList[i]!=NEList[i]) {
                        result = "fail (local_NEList[i]!=NEList[i])\n";
                        state = false;
                        break;
                    }
                }
            }
            if(rank==0) std::cout<<result;
        }
        if(ndims==2) {
            long double area=0, min_ele_area=0, max_ele_area=0;

            size_t i=0;
            for(; i<NElements; i++) {
                const index_t *n=get_element(i);
                if((mpi_ele_owner[i]!=rank) || (n[0]<0))
                    continue;

                area = property->area(get_coords(n[0]),
                                      get_coords(n[1]),
                                      get_coords(n[2]));
                min_ele_area = area;
                max_ele_area = area;
                i++;
                break;
            }
            for(; i<NElements; i++) {
                const index_t *n=get_element(i);
                if((mpi_ele_owner[i]!=rank) || (n[0]<0))
                    continue;

                long double larea = property->area(get_coords(n[0]),
                                                   get_coords(n[1]),
                                                   get_coords(n[2]));
                if(pragmatic_isnan(larea)) {
                    std::cerr<<"ERROR: Bad element "<<n[0]<<", "<<n[1]<<", "<<n[2]<<std::endl;
                }

                area += larea;
                min_ele_area = std::min(min_ele_area, larea);
                max_ele_area = std::max(max_ele_area, larea);
            }

#ifdef HAVE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_LONG_DOUBLE, MPI_SUM, get_mpi_comm());
            MPI_Allreduce(MPI_IN_PLACE, &min_ele_area, 1, MPI_LONG_DOUBLE, MPI_MIN, get_mpi_comm());
            MPI_Allreduce(MPI_IN_PLACE, &max_ele_area, 1, MPI_LONG_DOUBLE, MPI_MAX, get_mpi_comm());
#endif

            if(rank==0) {
                std::cout<<"VERIFY: total area  ............"<<area<<std::endl;
                std::cout<<"VERIFY: minimum element area...."<<min_ele_area<<std::endl;
                std::cout<<"VERIFY: maximum element area...."<<max_ele_area<<std::endl;
            }
        } else {
            long double volume=0, min_ele_vol=0, max_ele_vol=0;
            size_t i=0;
            for(; i<NElements; i++) {
                const index_t *n=get_element(i);
                if((mpi_ele_owner[i]!=rank) || (n[0]<0))
                    continue;

                volume = property->volume(get_coords(n[0]),
                                          get_coords(n[1]),
                                          get_coords(n[2]),
                                          get_coords(n[3]));
                min_ele_vol = volume;
                max_ele_vol = volume;
                i++;
                break;
            }
            for(; i<NElements; i++) {
                const index_t *n=get_element(i);
                if((mpi_ele_owner[i]!=rank) || (n[0]<0))
                    continue;

                long double lvolume = property->volume(get_coords(n[0]),
                                                       get_coords(n[1]),
                                                       get_coords(n[2]),
                                                       get_coords(n[3]));
                volume += lvolume;
                min_ele_vol = std::min(min_ele_vol, lvolume);
                max_ele_vol = std::max(max_ele_vol, lvolume);
            }

#ifdef HAVE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_LONG_DOUBLE, MPI_SUM, get_mpi_comm());
            MPI_Allreduce(MPI_IN_PLACE, &min_ele_vol, 1, MPI_LONG_DOUBLE, MPI_MIN, get_mpi_comm());
            MPI_Allreduce(MPI_IN_PLACE, &max_ele_vol, 1, MPI_LONG_DOUBLE, MPI_MAX, get_mpi_comm());
#endif

            if(rank==0) {
                std::cout<<"VERIFY: total volume.............."<<volume<<std::endl;
                std::cout<<"VERIFY: minimum element volume...."<<min_ele_vol<<std::endl;
                std::cout<<"VERIFY: maximum element volume...."<<max_ele_vol<<std::endl;
            }
        }

        double cachedq=0;
        for(size_t i=0; i<NElements; i++) {
            const index_t *n=get_element(i);
            if(n[0]<0)
                continue;
            cachedq += quality[i];
        }

#ifdef HAVE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &cachedq, 1, MPI_DOUBLE, MPI_SUM, get_mpi_comm());
#endif

        double qmean = get_qmean();
        double qmin = get_qmin();
        if(rank==0) {
            std::cout<<"VERIFY: mean quality......."<<qmean<<std::endl;
            std::cout<<"VERIFY: min quality........"<<qmin<<std::endl;
            std::cout<<"VERIFY: cached quality....."<<cachedq<<std::endl;
        }

#ifdef HAVE_MPI
        int false_cnt = state?0:1;
        if(num_processes>1) {
            MPI_Allreduce(MPI_IN_PLACE, &false_cnt, 1, MPI_INT, MPI_SUM, _mpi_comm);
        }
        state = (false_cnt == 0);
#endif

        return state;
    }

    void send_all_to_all(std::vector< std::vector<index_t> > send_vec,
                         std::vector< std::vector<index_t> > *recv_vec)
    {
#ifdef HAVE_MPI
        int ierr, recv_size, tag = 123456;
        std::vector<MPI_Status> status(num_processes);
        std::vector<MPI_Request> send_req(num_processes);
        std::vector<MPI_Request> recv_req(num_processes);

        for (int proc=0; proc<num_processes; proc++) {
            if (proc == rank) {
                send_req[proc] = MPI_REQUEST_NULL;
                continue;
            }

            ierr = MPI_Isend(send_vec[proc].data(), send_vec[proc].size(), MPI_INDEX_T,
                             proc, tag, _mpi_comm, &send_req[proc]);
            assert(ierr==0);
        }

        /* Receive send list from remote proc */
        for (int proc=0; proc<num_processes; proc++) {
            if (proc == rank) {
                recv_req[proc] = MPI_REQUEST_NULL;
                continue;
            }

            ierr = MPI_Probe(proc, tag, _mpi_comm, &(status[proc]));
            assert(ierr==0);
            ierr = MPI_Get_count(&(status[proc]), MPI_INT, &recv_size);
            assert(ierr==0);
            (*recv_vec)[proc].resize(recv_size);
            MPI_Irecv((*recv_vec)[proc].data(), recv_size, MPI_INT, proc,
                      tag, _mpi_comm, &recv_req[proc]);
            assert(ierr==0);
        }

        MPI_Waitall(num_processes, &(send_req[0]), &(status[0]));
        MPI_Waitall(num_processes, &(recv_req[0]), &(status[0]));
#endif
    }

private:
    template<typename _real_t, int _dim> friend class MetricField;
    template<typename _real_t, int _dim> friend class Smooth;
    template<typename _real_t, int _dim> friend class Swapping;
    template<typename _real_t, int _dim> friend class Coarsen;
    template<typename _real_t, int _dim> friend class Refine;
    template<typename _real_t> friend class DeferredOperations;
    template<typename _real_t> friend class VTKTools;
    template<typename _real_t> friend class CUDATools;

    void _init(int _NNodes, int _NElements, const index_t *globalENList,
               const real_t *x, const real_t *y, const real_t *z,
               const index_t *lnn2gnn, const index_t *owner_range)
    {
        num_processes = 1;
        rank=0;

        NElements = _NElements;
        NNodes = _NNodes;

#ifdef HAVE_MPI
        MPI_Comm_size(_mpi_comm, &num_processes);
        MPI_Comm_rank(_mpi_comm, &rank);

        // Assign the correct MPI data type to MPI_INDEX_T and MPI_REAL_T
        mpi_type_wrapper<index_t> mpi_index_t_wrapper;
        MPI_INDEX_T = mpi_index_t_wrapper.mpi_type;
        mpi_type_wrapper<real_t> mpi_real_t_wrapper;
        MPI_REAL_T = mpi_real_t_wrapper.mpi_type;
#endif

        nthreads = pragmatic_nthreads();

        if(z==NULL) {
            nloc = 3;
            ndims = 2;
            msize = 3;
        } else {
            nloc = 4;
            ndims = 3;
            msize = 6;
        }

        // From the globalENList, create the halo and a local ENList if num_processes>1.
        const index_t *ENList;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
        boost::unordered_map<index_t, index_t> gnn2lnn;
#else
        std::map<index_t, index_t> gnn2lnn;
#endif
        if(num_processes==1) {
            ENList = globalENList;
        } else {
#ifdef HAVE_MPI
            assert(lnn2gnn!=NULL);
            for(size_t i=0; i<(size_t)NNodes; i++) {
                gnn2lnn[lnn2gnn[i]] = i;
            }

            std::vector< std::set<index_t> > recv_set(num_processes);
            index_t *localENList = new index_t[NElements*nloc];
            for(size_t i=0; i<(size_t)NElements*nloc; i++) {
                index_t gnn = globalENList[i];
                for(int j=0; j<num_processes; j++) {
                    if(gnn<owner_range[j+1]) {
                        if(j!=rank)
                            recv_set[j].insert(gnn);
                        break;
                    }
                }
                localENList[i] = gnn2lnn[gnn];
            }
            std::vector<int> recv_size(num_processes);
            recv.resize(num_processes);
            recv_map.resize(num_processes);
            for(int j=0; j<num_processes; j++) {
                for(typename std::set<int>::const_iterator it=recv_set[j].begin(); it!=recv_set[j].end(); ++it) {
                    recv[j].push_back(*it);
                }
                recv_size[j] = recv[j].size();
            }
            std::vector<int> send_size(num_processes);
            MPI_Alltoall(&(recv_size[0]), 1, MPI_INT,
                         &(send_size[0]), 1, MPI_INT, _mpi_comm);

            // Setup non-blocking receives
            send.resize(num_processes);
            send_map.resize(num_processes);
            std::vector<MPI_Request> request(num_processes*2);
            for(int i=0; i<num_processes; i++) {
                if((i==rank)||(send_size[i]==0)) {
                    request[i] =  MPI_REQUEST_NULL;
                } else {
                    send[i].resize(send_size[i]);
                    MPI_Irecv(&(send[i][0]), send_size[i], MPI_INDEX_T, i, 0, _mpi_comm, &(request[i]));
                }
            }

            // Non-blocking sends.
            for(int i=0; i<num_processes; i++) {
                if((i==rank)||(recv_size[i]==0)) {
                    request[num_processes+i] =  MPI_REQUEST_NULL;
                } else {
                    MPI_Isend(&(recv[i][0]), recv_size[i], MPI_INDEX_T, i, 0, _mpi_comm, &(request[num_processes+i]));
                }
            }

            std::vector<MPI_Status> status(num_processes*2);
            MPI_Waitall(num_processes, &(request[0]), &(status[0]));
            MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

            for(int j=0; j<num_processes; j++) {
                for(int k=0; k<recv_size[j]; k++) {
                    index_t gnn = recv[j][k];
                    index_t lnn = gnn2lnn[gnn];
                    recv_map[j][gnn] = lnn;
                    recv[j][k] = lnn;
                }

                for(int k=0; k<send_size[j]; k++) {
                    index_t gnn = send[j][k];
                    index_t lnn = gnn2lnn[gnn];
                    send_map[j][gnn] = lnn;
                    send[j][k] = lnn;
                }
            }

            ENList = localENList;
#endif
        }

        _ENList.resize(NElements*nloc);
        quality.resize(NElements);
        _coords.resize(NNodes*ndims);
        metric.resize(NNodes*msize);
        NNList.resize(NNodes);
        NEList.resize(NNodes);
        node_owner.resize(NNodes);
        this->lnn2gnn.resize(NNodes);

        // TODO I don't know whether this method makes sense anymore.
        // Enforce first-touch policy
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int i=0; i<(int)NElements; i++) {
                for(size_t j=0; j<nloc; j++) {
                    _ENList[i*nloc+j] = ENList[i*nloc+j];
                }
            }
            if(ndims==2) {
                #pragma omp for schedule(static)
                for(int i=0; i<(int)NNodes; i++) {
                    _coords[i*2  ] = x[i];
                    _coords[i*2+1] = y[i];
                }
            } else {
                #pragma omp for schedule(static)
                for(int i=0; i<(int)NNodes; i++) {
                    _coords[i*3  ] = x[i];
                    _coords[i*3+1] = y[i];
                    _coords[i*3+2] = z[i];
                }
            }

            #pragma omp single nowait
            {
                if(num_processes>1) {
                    // Take into account renumbering for halo.
                    for(int j=0; j<num_processes; j++) {
                        for(size_t k=0; k<recv[j].size(); k++) {
                            recv_halo.insert(recv[j][k]);
                        }
                        for(size_t k=0; k<send[j].size(); k++) {
                            send_halo.insert(send[j][k]);
                        }
                    }
                }
            }

            // Set the orientation of elements.
            #pragma omp single
            {
                const int *n=get_element(0);
                assert(n[0]>=0);

                if(ndims==2)
                    property = new ElementProperty<real_t>(get_coords(n[0]),
                                                           get_coords(n[1]),
                                                           get_coords(n[2]));
                else
                    property = new ElementProperty<real_t>(get_coords(n[0]),
                                                           get_coords(n[1]),
                                                           get_coords(n[2]),
                                                           get_coords(n[3]));
            }

            if(ndims==2) {
                #pragma omp for schedule(static)
                for(size_t i=0; i<(size_t)NElements; i++) {
                    const int *n=get_element(i);
                    assert(n[0]>=0);

                    double volarea = property->area(get_coords(n[0]),
                                                    get_coords(n[1]),
                                                    get_coords(n[2]));

                    if(volarea<0)
                        invert_element(i);

                    update_quality<2>(i);
                }
            } else {
                #pragma omp for schedule(static)
                for(size_t i=0; i<(size_t)NElements; i++) {
                    const int *n=get_element(i);
                    assert(n[0]>=0);

                    double volarea = property->volume(get_coords(n[0]),
                                                      get_coords(n[1]),
                                                      get_coords(n[2]),
                                                      get_coords(n[3]));

                    if(volarea<0)
                        invert_element(i);

                    update_quality<3>(i);
                }
            }

            // create_adjacency is meant to be called from inside a parallel region
            create_adjacency();
        }

        create_global_node_numbering();
    }

    /// Create required adjacency lists.
    void create_adjacency()
    {
        int tid = pragmatic_thread_id();

        #pragma omp for schedule(static)
        for(size_t i=0; i<NNodes; i++) {
            NNList[i].clear();
            NEList[i].clear();
        }

        for(size_t i=0; i<NElements; i++) {
            if(_ENList[i*nloc]<0)
                continue;

            for(size_t j=0; j<nloc; j++) {
                index_t nid_j = _ENList[i*nloc+j];
                if((nid_j%nthreads)==tid) {
                    NEList[nid_j].insert(NEList[nid_j].end(), i);
                    for(size_t k=0; k<nloc; k++) {
                        if(j!=k) {
                            index_t nid_k = _ENList[i*nloc+k];
                            NNList[nid_j].push_back(nid_k);
                        }
                    }
                }
            }
        }

        #pragma omp barrier

        // Finalise
        #pragma omp for schedule(static)
        for(size_t i=0; i<NNodes; i++) {
            if(NNList[i].empty())
                continue;

            std::vector<index_t> *nnset = new std::vector<index_t>();

            std::sort(NNList[i].begin(),NNList[i].end());
            std::unique_copy(NNList[i].begin(), NNList[i].end(), std::inserter(*nnset, nnset->begin()));

            NNList[i].swap(*nnset);
            delete nnset;
        }
    }

    void trim_halo()
    {
        std::set<index_t> recv_halo_temp, send_halo_temp;

        // Traverse all vertices V in all recv[i] vectors. Vertices in send[i] belong by definition to *this* MPI process,
        // so all elements adjacent to them either belong exclusively to *this* process or cross partitions.
        for(int i=0; i<num_processes; i++) {
            if(recv[i].size()==0)
                continue;

            std::vector<index_t> recv_temp;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
            boost::unordered_map<index_t, index_t> recv_map_temp;
#else
            std::map<index_t, index_t> recv_map_temp;
#endif

            for(typename std::vector<index_t>::const_iterator vit = recv[i].begin(); vit != recv[i].end(); ++vit) {
                // For each vertex, traverse a copy of the vertex's NEList.
                // We need a copy because erase_element modifies the original NEList.
                std::set<index_t> NEList_copy = NEList[*vit];
                for(typename std::set<index_t>::const_iterator eit = NEList_copy.begin(); eit != NEList_copy.end(); ++eit) {
                    // Check whether all vertices comprising the element belong to another MPI process.
                    std::vector<index_t> n(nloc);
                    get_element(*eit, &n[0]);
                    if(n[0] < 0)
                        continue;

                    // If one of the vertices belongs to *this* partition, the element should be retained.
                    bool to_be_deleted = true;
                    for(size_t j=0; j<nloc; ++j)
                        if(is_owned_node(n[j])) {
                            to_be_deleted = false;
                            break;
                        }

                    if(to_be_deleted) {
                        erase_element(*eit);

                        // Now check whether one of the edges must be deleted
                        for(size_t j=0; j<nloc; ++j) {
                            for(size_t k=j+1; k<nloc; ++k) {
                                std::set<index_t> intersection;
                                std::set_intersection(NEList[n[j]].begin(), NEList[n[j]].end(), NEList[n[k]].begin(), NEList[n[k]].end(),
                                                      std::inserter(intersection, intersection.begin()));

                                // If these two vertices have no element in common anymore,
                                // then the corresponding edge does not exist, so update NNList.
                                if(intersection.empty()) {
                                    typename std::vector<index_t>::iterator it;
                                    it = std::find(NNList[n[j]].begin(), NNList[n[j]].end(), n[k]);
                                    NNList[n[j]].erase(it);
                                    it = std::find(NNList[n[k]].begin(), NNList[n[k]].end(), n[j]);
                                    NNList[n[k]].erase(it);
                                }
                            }
                        }
                    }
                }

                // If this vertex is no longer part of any element, then it is safe to be removed.
                if(NEList[*vit].empty()) {
                    // Update NNList of all neighbours
                    for(typename std::vector<index_t>::const_iterator neigh_it = NNList[*vit].begin(); neigh_it != NNList[*vit].end(); ++neigh_it) {
                        typename std::vector<index_t>::iterator it = std::find(NNList[*neigh_it].begin(), NNList[*neigh_it].end(), *vit);
                        NNList[*neigh_it].erase(it);
                    }

                    erase_vertex(*vit);
                } else {
                    // We will keep this vertex, so put it into recv_halo_temp.
                    recv_temp.push_back(*vit);
                    recv_map_temp[lnn2gnn[*vit]] = *vit;
                    recv_halo_temp.insert(*vit);
                }
            }

            recv[i].swap(recv_temp);
            recv_map[i].swap(recv_map_temp);
        }

        // Once all recv[i] have been traversed, update recv_halo.
        recv_halo.swap(recv_halo_temp);

        // Traverse all vertices V in all send[i] vectors.
        // If none of V's neighbours are owned by the i-th MPI process, it means that the i-th process
        // has removed V from its recv_halo, so remove the vertex from send[i].
        for(int i=0; i<num_processes; i++) {
            if(send[i].size()==0)
                continue;

            std::vector<index_t> send_temp;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
            boost::unordered_map<index_t, index_t> send_map_temp;
#else
            std::map<index_t, index_t> send_map_temp;
#endif

            for(typename std::vector<index_t>::const_iterator vit = send[i].begin(); vit != send[i].end(); ++vit) {
                bool to_be_deleted = true;
                for(typename std::vector<index_t>::const_iterator neigh_it = NNList[*vit].begin(); neigh_it != NNList[*vit].end(); ++neigh_it)
                    if(node_owner[*neigh_it] == i) {
                        to_be_deleted = false;
                        break;
                    }

                if(!to_be_deleted) {
                    send_temp.push_back(*vit);
                    send_map_temp[lnn2gnn[*vit]] = *vit;
                    send_halo_temp.insert(*vit);
                }
            }

            send[i].swap(send_temp);
            send_map[i].swap(send_map_temp);
        }

        // Once all send[i] have been traversed, update send_halo.
        send_halo.swap(send_halo_temp);
    }

    void create_global_node_numbering()
    {
        if(num_processes>1) {
#ifdef HAVE_MPI
            // Calculate the global numbering offset for this partition.
            int gnn_offset;
            int NPNodes = NNodes - recv_halo.size();
            MPI_Scan(&NPNodes, &gnn_offset, 1, MPI_INT, MPI_SUM, get_mpi_comm());
            gnn_offset-=NPNodes;

            // Write global node numbering and ownership for nodes assigned to local process.
            for(index_t i=0; i < (index_t) NNodes; i++) {
                if(recv_halo.count(i)) {
                    lnn2gnn[i] = 0;
                } else {
                    lnn2gnn[i] = gnn_offset++;
                    node_owner[i] = rank;
                }
            }

            // Update GNN's for the halo nodes.
            halo_update<int, 1>(_mpi_comm, send, recv, lnn2gnn);

            // Finish writing node ownerships.
            for(int i=0; i<num_processes; i++) {
                for(std::vector<int>::const_iterator it=recv[i].begin(); it!=recv[i].end(); ++it) {
                    node_owner[*it] = i;
                }
            }
#endif
        } else {
            memset(&node_owner[0], 0, NNodes*sizeof(int));
            for(index_t i=0; i < (index_t) NNodes; i++)
                lnn2gnn[i] = i;
        }
    }

    void create_gappy_global_numbering(size_t pNElements)
    {
#ifdef HAVE_MPI
        // We expect to have NElements_predict/2 nodes in the partition,
        // so let's reserve 10 times more space for global node numbers.
        index_t gnn_reserve = 5*pNElements;
        MPI_Scan(&gnn_reserve, &gnn_offset, 1, MPI_INDEX_T, MPI_SUM, _mpi_comm);
        gnn_offset -= gnn_reserve;

        for(size_t i=0; i<NNodes; ++i) {
            if(node_owner[i] == rank)
                lnn2gnn[i] = gnn_offset+i;
            else
                lnn2gnn[i] = -1;
        }

        halo_update<int, 1>(_mpi_comm, send, recv, lnn2gnn);

        for(int i=0; i<num_processes; i++) {
            send_map[i].clear();
            for(std::vector<int>::const_iterator it=send[i].begin(); it!=send[i].end(); ++it) {
                assert(node_owner[*it]==rank);
                send_map[i][lnn2gnn[*it]] = *it;
            }

            recv_map[i].clear();
            for(std::vector<int>::const_iterator it=recv[i].begin(); it!=recv[i].end(); ++it) {
                node_owner[*it] = i;
                recv_map[i][lnn2gnn[*it]] = *it;
            }
        }
#endif
    }

    void update_gappy_global_numbering(std::vector<size_t>& recv_cnt, std::vector<size_t>& send_cnt)
    {
#ifdef HAVE_MPI
        // MPI_Requests for all non-blocking communications.
        std::vector<MPI_Request> request(num_processes*2);

        // Setup non-blocking receives.
        std::vector< std::vector<index_t> > recv_buff(num_processes);
        for(int i=0; i<num_processes; i++) {
            if(recv_cnt[i]==0) {
                request[i] =  MPI_REQUEST_NULL;
            } else {
                recv_buff[i].resize(recv_cnt[i]);
                MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), MPI_INDEX_T, i, 0, _mpi_comm, &(request[i]));
            }
        }

        // Non-blocking sends.
        std::vector< std::vector<index_t> > send_buff(num_processes);
        for(int i=0; i<num_processes; i++) {
            if(send_cnt[i]==0) {
                request[num_processes+i] = MPI_REQUEST_NULL;
            } else {
                for(typename std::vector<index_t>::const_iterator it=send[i].end()-send_cnt[i]; it!=send[i].end(); ++it)
                    send_buff[i].push_back(lnn2gnn[*it]);

                MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), MPI_INDEX_T, i, 0, _mpi_comm, &(request[num_processes+i]));
            }
        }

        std::vector<MPI_Status> status(num_processes*2);
        MPI_Waitall(num_processes, &(request[0]), &(status[0]));
        MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

        for(int i=0; i<num_processes; i++) {
            int k=0;
            for(typename std::vector<index_t>::const_iterator it=recv[i].end()-recv_cnt[i]; it!=recv[i].end(); ++it, ++k)
                lnn2gnn[*it] = recv_buff[i][k];
        }
#endif
    }

    template<int dim>
    inline double calculate_quality(const index_t* n)
    {
        if(dim==2) {
            const double *x0 = get_coords(n[0]);
            const double *x1 = get_coords(n[1]);
            const double *x2 = get_coords(n[2]);

            const double *m0 = get_metric(n[0]);
            const double *m1 = get_metric(n[1]);
            const double *m2 = get_metric(n[2]);

            return property->lipnikov(x0, x1, x2, m0, m1, m2);
        } else {
            const double *x0 = get_coords(n[0]);
            const double *x1 = get_coords(n[1]);
            const double *x2 = get_coords(n[2]);
            const double *x3 = get_coords(n[3]);

            const double *m0 = get_metric(n[0]);
            const double *m1 = get_metric(n[1]);
            const double *m2 = get_metric(n[2]);
            const double *m3 = get_metric(n[3]);

            return property->lipnikov(x0, x1, x2, x3, m0, m1, m2, m3);
        }
    }

    template<int dim>
    inline void update_quality(index_t element)
    {
        const index_t *n=get_element(element);

        if(dim==2) {
            quality[element] = calculate_quality<2>(n);
        } else {
            quality[element] = calculate_quality<3>(n);
        }
    }

    size_t ndims, nloc, msize;
    std::vector<index_t> _ENList;
    std::vector<real_t> _coords;

    size_t NNodes, NElements;

    // Boundary Label
    std::vector<int> boundary;

    // Quality
    std::vector<double> quality;

    // Adjacency lists
    std::vector< std::set<index_t> > NEList;
    std::vector< std::vector<index_t> > NNList;

    ElementProperty<real_t> *property;

    // Metric tensor field.
    std::vector<double> metric;

    // Parallel support.
    int rank, num_processes, nthreads;
    std::vector< std::vector<index_t> > send, recv;
#ifdef HAVE_BOOST_UNORDERED_MAP_HPP
    std::vector< boost::unordered_map<index_t, index_t> > send_map, recv_map;
#else
    std::vector< std::map<index_t, index_t> > send_map, recv_map;
#endif
    std::set<index_t> send_halo, recv_halo;
    std::vector<int> node_owner;
    std::vector<index_t> lnn2gnn;

    index_t gnn_offset;
#ifdef HAVE_MPI
    MPI_Comm _mpi_comm;

    // MPI data type for index_t and real_t
    MPI_Datatype MPI_INDEX_T;
    MPI_Datatype MPI_REAL_T;
#endif
};

#endif
