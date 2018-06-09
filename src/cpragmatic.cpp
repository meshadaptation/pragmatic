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

#include <cassert>

#include "Mesh.h"
#include "MetricField.h"
#include "Coarsen.h"
#include "Refine.h"
#include "Swapping.h"
#include "Smooth.h"

#ifdef HAVE_VTK
#include "VTKTools.h"
#endif


static void *_pragmatic_mesh=NULL;
static void *_pragmatic_metric_field=NULL;

extern "C" {
#ifdef HAVE_VTK
    void pragmatic_dump(const char *filename)
    {
        VTKTools<double>::export_vtu(filename, (Mesh<double>*)_pragmatic_mesh);
    }

    void pragmatic_dump_debug()
    {
        pragmatic_dump("dump\0");
    }
#endif

    /** Initialise pragmatic with mesh to be adapted. pragmatic_finalize must
      be called before this can be called again, i.e. cannot adapt
      multiple meshes at the same time.

      @param [in] NNodes Number of nodes
      @param [in] NElements Number of elements
      @param [in] enlist Element-node list
      @param [in] x x coordinate array
      @param [in] y y coordinate array
      */
    void pragmatic_2d_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y)
    {
        if(_pragmatic_mesh!=NULL) {
            throw new std::string("PRAgMaTIc: only one mesh can be adapted at a time");
        }

        Mesh<double> *mesh = new Mesh<double>(*NNodes, *NElements, enlist, x, y);

        _pragmatic_mesh = mesh;
    }

    /** Initialise pragmatic with mesh to be adapted. pragmatic_finalize must
      be called before this can be called again, i.e. cannot adapt
      multiple meshes at the same time.

      @param [in] NNodes Number of nodes.
      @param [in] NElements Number of elements.
      @param [in] enlist Element-node list.
      @param [in] x x coordinate array.
      @param [in] y y coordinate array.
      @param [in] lnn2gnn local-to-global numbering.
      @param [in] NPNodes Number of owned nodes on the local processor.
      @param [in] mpi_comm is the mpi comm.
      */
    void pragmatic_2d_mpi_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y,
                               const int *lnn2gnn, const int NPNodes, MPI_Comm mpi_comm)
    {
        if(_pragmatic_mesh!=NULL) {
            throw new std::string("PRAgMaTIc: only one mesh can be adapted at a time");
        }

        Mesh<double> *mesh = new Mesh<double>(*NNodes, *NElements, enlist, x, y, lnn2gnn, NPNodes, mpi_comm);

        _pragmatic_mesh = mesh;
    }

    /** Initialise pragmatic with mesh to be adapted. pragmatic_finalize must
      be called before this can be called again, i.e. cannot adapt
      multiple meshes at the same time.

      @param [in] NNodes Number of nodes
      @param [in] NElements Number of elements
      @param [in] enlist Element-node list
      @param [in] x x coordinate array
      @param [in] y y coordinate array
      @param [in] z z coordinate array
      */
    void pragmatic_3d_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z)
    {
        assert(_pragmatic_mesh==NULL);
        assert(_pragmatic_metric_field==NULL);

        Mesh<double> *mesh = new Mesh<double>(*NNodes, *NElements, enlist, x, y, z);

        _pragmatic_mesh = mesh;
    }

    /** Initialise pragmatic with mesh to be adapted. pragmatic_finalize must
      be called before this can be called again, i.e. cannot adapt
      multiple meshes at the same time.

      @param [in] NNodes Number of nodes
      @param [in] NElements Number of elements
      @param [in] enlist Element-node list
      @param [in] x x coordinate array
      @param [in] y y coordinate array
      @param [in] z z coordinate array
      @param [in] lnn2gnn local-to-global numbering.
      @param [in] NPNodes Number of nodes owned by local processor.
      @param [in] mpi_comm is the mpi comm.

      */
    void pragmatic_3d_mpi_init(const int *NNodes, const int *NElements, const int *enlist, const double *x, const double *y, const double *z,
                           const int *lnn2gnn, const int NPNodes, MPI_Comm mpi_comm)
    {
        assert(_pragmatic_mesh==NULL);
        assert(_pragmatic_metric_field==NULL);

        Mesh<double> *mesh = new Mesh<double>(*NNodes, *NElements, enlist, x, y, z, lnn2gnn, NPNodes, mpi_comm);

        _pragmatic_mesh = mesh;
    }

    /** Initialise pragmatic with name of VTK file to be adapted.
    */
#ifdef HAVE_VTK
    void pragmatic_vtk_init(const char *filename)
    {
        assert(_pragmatic_mesh==NULL);
        assert(_pragmatic_metric_field==NULL);

        Mesh<double> *mesh=VTKTools<double>::import_vtu(filename);
        mesh->create_boundary();

        _pragmatic_mesh = mesh;
    }
#endif

    void pragmatic_init_light(void * mesh){
        _pragmatic_mesh = mesh;
    }

    /** Add field which should be adapted to.

      @param [in] psi Node centred field variable
      @param [in] error Error target
      @param [in] pnorm P-norm value for error measure. Applies the
      p-norm scaling to the metric, as in Chen, Sun and Xu,
      Mathematics of Computation, Volume 76, Number 257, January
      2007. Set to -1 to default to absolute error measure.
      */
    void pragmatic_add_field(const double *psi, const double *error, int *pnorm)
    {
        assert(_pragmatic_mesh!=NULL);

        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        if(_pragmatic_metric_field==NULL) {
            if(((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions()==2) {
                MetricField<double,2> *metric_field = new MetricField<double,2>(*mesh);
                metric_field->add_field(psi, *error, *pnorm);
                metric_field->update_mesh();

                _pragmatic_metric_field = metric_field;
            } else {
                MetricField<double,3> *metric_field = new MetricField<double,3>(*mesh);
                metric_field->add_field(psi, *error, *pnorm);
                metric_field->update_mesh();

                _pragmatic_metric_field = metric_field;
            }
        } else {
            std::cerr<<"WARNING: Fortran interface currently only supports adding a single field.\n";
        }
    }

    /** Set the node centred metric field

      @param [in] metric Metric tensor field.
      */
    void pragmatic_set_metric(const double *metric)
    {
        assert(_pragmatic_mesh!=NULL);
        assert(_pragmatic_metric_field==NULL);

        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        mesh->defragment();

        if(_pragmatic_metric_field==NULL) {
            if(((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions()==2) {
                MetricField<double,2> *metric_field = new MetricField<double,2>(*mesh);
                _pragmatic_metric_field = metric_field;
            } else {
                MetricField<double,3> *metric_field = new MetricField<double,3>(*mesh);
                _pragmatic_metric_field = metric_field;
            }
        }

        if(((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions()==2) {
            ((MetricField<double,2> *)_pragmatic_metric_field)->set_metric_full(metric);
            ((MetricField<double,2> *)_pragmatic_metric_field)->update_mesh();
        } else {
            ((MetricField<double,3> *)_pragmatic_metric_field)->set_metric_full(metric);
            ((MetricField<double,3> *)_pragmatic_metric_field)->update_mesh();
        }
    }

    /** Set the domain boundary.

      @param [in] nfacets Number of boundary facets
      @param [in] facets Facet list
      @param [in] ids Boundary ids
      */
    void pragmatic_set_boundary(const int *nfacets, const int *facets, const int *ids)
    {
        assert(_pragmatic_mesh!=NULL);

        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;
        mesh->set_boundary(*nfacets, facets, ids);
    }

    /** Set the internal regions.

      @param [in] element_tags list of element tags
      */
    void pragmatic_set_regions(const int *element_tags)
    {
        assert(_pragmatic_mesh!=NULL);

        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;
        mesh->set_regions(element_tags);
    }

    /** Reconstruct internal boundaries separing regions if not provided already

      */
    void pragmatic_set_internal_boundaries()
    {
        assert(_pragmatic_mesh!=NULL);

        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;
        mesh->set_internal_boundaries();
    }

    /** Adapt the mesh.
    */
    void pragmatic_adapt(int coarsen_surface)
    {
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        const size_t ndims = mesh->get_number_dimensions();

        // See Eqn 7; X Li et al, Comp Methods Appl Mech Engrg 194 (2005) 4915-4950
        double L_up = sqrt(2.0);
        double L_low = L_up*0.5;

        if(ndims==2) {
            Coarsen<double, 2> coarsen(*mesh);
            Smooth<double, 2> smooth(*mesh);
            Refine<double, 2> refine(*mesh);
            Swapping<double, 2> swapping(*mesh);

            double L_max = mesh->maximal_edge_length();

            double alpha = sqrt(2.0)/2.0;
            bool stop = false;
            for(size_t i=0; i<30; i++) {
                double L_ref = std::max(alpha*L_max, L_up);

                int cnt_coars, cnt_split;
                cnt_coars= coarsen.coarsen(L_low, L_ref, (bool) coarsen_surface);
                swapping.swap(0.7);
                cnt_split = refine.refine(L_ref);

                if (cnt_split == 0 && cnt_coars == 0 && stop)
                    break;
                if (cnt_split == 0 && cnt_coars == 0)
                    stop = true;
                else
                    stop = false;

                L_max = mesh->maximal_edge_length();
            }

            mesh->defragment();

            smooth.smart_laplacian(20);
            smooth.optimisation_linf(20);
        } else {
            Coarsen<double, 3> coarsen(*mesh);
            Smooth<double, 3> smooth(*mesh);
            Refine<double, 3> refine(*mesh);
            Swapping<double, 3> swapping(*mesh);

#if 0
            // TODO HACK CAD
            mesh->set_isOnBoundarySize();
            for (int j=0; j<mesh->get_number_nodes(); ++j) mesh->set_isOnBoundary(j, 0);
            int * boundary = mesh->get_boundaryTags();
            index_t * ENList = mesh->get_ENList();
            for (int j=0; j<mesh->get_number_elements()*4; ++j) {
              if (boundary[j] == 5) {
                int iElem = j/4;
                int iEdg = j % 4;
                mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+1)%4], 1);
                mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+2)%4], 1);
                mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+3)%4], 1);
              }
            }
#endif

            coarsen.coarsen(L_low, L_up, (bool) coarsen_surface);

            double L_max = mesh->maximal_edge_length();

            double alpha = sqrt(2.0)/2.0;
            bool stop = false;
            // give more time to converge with new refinement, but stop before if possible
            // TODO write a cycle detector and stop if there is a cycle
            for(size_t i=0; i<30; i++) {
                double L_ref = std::max(alpha*L_max, L_up);

                int cnt_coars, cnt_split;
                cnt_split = refine.refine(L_ref);
                cnt_coars = coarsen.coarsen(L_low, L_ref, (bool) coarsen_surface);
                swapping.swap(0.95);

                if (cnt_split == 0 && cnt_coars == 0 && stop)
                    break;
                if (cnt_split == 0 && cnt_coars == 0)
                    stop = true;
                else
                    stop = false;

                L_max = mesh->maximal_edge_length();

#if 0                
                // TODO HACK CAD
                mesh->set_isOnBoundarySize();
                for (int j=0; j<mesh->get_number_nodes(); ++j) mesh->set_isOnBoundary(j, 0);
                int * boundary = mesh->get_boundaryTags();
                index_t * ENList = mesh->get_ENList();
                for (int j=0; j<mesh->get_number_elements()*4; ++j) {
                  if (boundary[j] == 5) {
                    int iElem = j/4;
                    int iEdg = j % 4;
                    mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+1)%4], 1);
                    mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+2)%4], 1);
                    mesh->set_isOnBoundary(ENList[4*iElem+(iEdg+3)%4], 1);
                  }
                }
#endif
            }

            mesh->defragment();

            smooth.smart_laplacian(10);
            smooth.optimisation_linf(10);
        }

        mesh->remove_overlap_elements();
    }

    /** Coarsen the mesh.
    */
    void pragmatic_coarsen(int coarsen_surface)
    {
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        const size_t ndims = mesh->get_number_dimensions();

        double L_up = sqrt(2.0);

        if(ndims==2) {
            Coarsen<double, 2> coarsen(*mesh);
            Swapping<double, 2> swapping(*mesh);

            for(size_t i=0; i<5; i++) {
                coarsen.coarsen(L_up, L_up, (bool) coarsen_surface);
                swapping.swap(0.1);
            }
        } else {
            Coarsen<double, 3> coarsen(*mesh);
            Swapping<double, 3> swapping(*mesh);

            for(size_t i=0; i<5; i++) {
                coarsen.coarsen(L_up, L_up, (bool) coarsen_surface);
                swapping.swap(0.1);
            }
        }
        mesh->defragment();
    }


    /** Get size of mesh.

      @param [out] Number of nodes of the proc  (owned and not owned)
      @param [out] NElements of this proc (with or without halo depending on previous actions)
      */
    void pragmatic_get_info(int *NNodes, int *NElements)
    {
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        *NNodes = mesh->get_number_nodes();
        *NElements = mesh->get_number_elements();
    }
    
    /** Get size of mesh.

      @param [out] Number of nodes owned by current proc
      @param [out] NElements of this proc (with or without halo depending on previous actions)
      */
    void pragmatic_get_info_mpi(int *NNodes, int *NElements)
    {
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;

        *NNodes = mesh->get_number_owned_nodes();
        *NElements = mesh->get_number_elements();
    }
    
    

    void pragmatic_get_coords_2d(double *x, double *y)
    {
        size_t NNodes = ((Mesh<double> *)_pragmatic_mesh)->get_number_nodes();
        for(size_t i=0; i<NNodes; i++) {
            x[i] = ((Mesh<double> *)_pragmatic_mesh)->get_coords(i)[0];
            y[i] = ((Mesh<double> *)_pragmatic_mesh)->get_coords(i)[1];
        }
    }

    void pragmatic_get_coords_3d(double *x, double *y, double *z)
    {
        size_t NNodes = ((Mesh<double> *)_pragmatic_mesh)->get_number_nodes();
        for(size_t i=0; i<NNodes; i++) {
            x[i] = ((Mesh<double> *)_pragmatic_mesh)->get_coords(i)[0];
            y[i] = ((Mesh<double> *)_pragmatic_mesh)->get_coords(i)[1];
            z[i] = ((Mesh<double> *)_pragmatic_mesh)->get_coords(i)[2];
        }
    }
    
    /** Fills x,y arrays with coordinates of owned nodes in global numbering (minus the offset)
      */
    void pragmatic_get_coords_2d_mpi(double *x, double *y)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;
        
        size_t NNodes = mesh->get_number_nodes();
        int offset = mesh->get_gnn_offset();
        
        for(size_t i=0; i<NNodes; i++) {
            if (mesh->is_owned_node(i)) {
                int gnn = mesh->get_global_numbering(i) - offset;
                x[gnn] = mesh->get_coords(i)[0];
                y[gnn] = mesh->get_coords(i)[1];
            }
        }
    }

    /** Fills x,y,z arrays with coordinates of owned nodes in global numbering (minus the offset)
      */
    void pragmatic_get_coords_3d_mpi(double *x, double *y, double *z)
    {
        
        Mesh<double> *mesh = (Mesh<double> *)_pragmatic_mesh;
        
        size_t NNodes = mesh->get_number_nodes();
        int offset = mesh->get_gnn_offset();
        for(size_t i=0; i<NNodes; i++) {
            if (mesh->is_owned_node(i)) {
                int gnn = mesh->get_global_numbering(i) - offset;
                x[gnn] = mesh->get_coords(i)[0];
                y[gnn] = mesh->get_coords(i)[1];
                z[gnn] = mesh->get_coords(i)[2];
            }
        }
    }

    /** Fills elements array with vertices indices of the elements (local or global numbering depending on what was done before)
      */
    void pragmatic_get_elements(int *elements)
    {
        const size_t ndims = ((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions();
        const size_t NElements = ((Mesh<double> *)_pragmatic_mesh)->get_number_elements();
        const size_t nloc = (ndims==2)?3:4;

        for(size_t i=0; i<NElements; i++) {
            const int *n=((Mesh<double> *)_pragmatic_mesh)->get_element(i);

            for(size_t j=0; j<nloc; j++) {
                assert(n[j]>=0);
                elements[i*nloc+j] = n[j];
            }
        }
    }
    /*
       void pragmatic_get_lnn2gnn(int *nodes_per_partition, int *lnn2gnn){
       std::vector<int> _NPNodes, _lnn2gnn;
       ((Mesh<double> *)_pragmatic_mesh)->get_global_node_numbering(_NPNodes, _lnn2gnn);
       size_t len0 = _NPNodes.size();
       for(size_t i=0;i<len0;i++)
       nodes_per_partition[i] = _NPNodes[i];

       size_t len1 = _lnn2gnn.size();
       for(size_t i=0;i<len1;i++)
       lnn2gnn[i] = _lnn2gnn[i];
       }
       */
    void pragmatic_get_metric(double *metric)
    {
        if(((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions()==2) {
            ((MetricField<double,2> *)_pragmatic_metric_field)->get_metric(metric);
        } else {
            ((MetricField<double,3> *)_pragmatic_metric_field)->get_metric(metric);
        }
    }
    
    void pragmatic_get_boundaryTags(int ** tags)
    {
      *tags = ((Mesh<double> *)_pragmatic_mesh)->get_boundaryTags();
    }

    void pragmatic_get_elementTags(int ** regions)
    {
      *regions = ((Mesh<double> *)_pragmatic_mesh)->get_elementTags();
    }    

    void pragmatic_finalize()
    {
        if(((Mesh<double> *)_pragmatic_mesh)->get_number_dimensions()==2) {
            if(_pragmatic_metric_field!=NULL)
                delete (MetricField<double,2> *)_pragmatic_metric_field;
        } else {
            if(_pragmatic_metric_field!=NULL)
                delete (MetricField<double,3> *)_pragmatic_metric_field;
        }
        _pragmatic_metric_field=NULL;

        delete (Mesh<double> *)_pragmatic_mesh;
        _pragmatic_mesh=NULL;
    }
}
