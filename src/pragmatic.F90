!  Copyright (C) 2010 Imperial College London and others.
!
!  Please see the AUTHORS file in the main source directory for a
!  full list of copyright holders.
!
!  Gerard Gorman
!  Applied Modelling and Computation Group
!  Department of Earth Science and Engineering
!  Imperial College London
!
!  g.gorman@imperial.ac.uk
!
!  Redistribution and use in source and binary forms, with or without
!  modification, are permitted provided that the following conditions
!  are met:
!  1. Redistributions of source code must retain the above copyright
!  notice, this list of conditions and the following disclaimer.
!  2. Redistributions in binary form must reproduce the above
!  copyright notice, this list of conditions and the following
!  disclaimer in the documentation and/or other materials provided
!  with the distribution.
!
!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
!  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
!  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
!  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
!  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
!  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
!  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
!  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
!  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
!  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
!  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
!  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
!  SUCH DAMAGE.

!> This module merely contains explicit interfaces to allow the
!> convenient use of vtkfortran in fortran.
module pragmatic
  use iso_c_binding
  
  private
  public :: &
       pragmatic_begin, &
       pragmatic_add_field, &
       pragmatic_set_surface, &
       pragmatic_set_metric, &
       pragmatic_adapt, &
       pragmatic_get_info, &
       pragmatic_get_coords, &
       pragmatic_get_elements, &
       pragmatic_get_surface, &
       pragmatic_get_lnn2gnn, &
       pragmatic_get_metric, &
       pragmatic_dump, &
       pragmatic_end
  
   !> Initialises metric calculation. This must be called first.
   !
   !> @param[in] NNodes Number of nodes in the mesh.
   !> @param[in] NElements Number of elements in the mesh.
   !> @param[in] enlists Element-node list.
   !> @param[in] x X-cooardinates.
   !> @param[in] y Y-cooardinates.
  interface pragmatic_begin
     subroutine pragmatic_2d_begin(NNodes, NElements, enlist, x, y) bind(c,name="pragmatic_2d_begin")
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: NNodes
       integer(kind=c_int) :: NElements
       integer(kind=c_int) :: enlist(*)
       real(c_double) :: x(*)
       real(c_double) :: y(*)
     end subroutine pragmatic_2d_begin
     subroutine pragmatic_3d_begin(NNodes, NElements, enlist, x, y, z) bind(c,name="pragmatic_3d_begin")
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: NNodes
       integer(kind=c_int) :: NElements
       integer(kind=c_int) :: enlist(*)
       real(c_double) :: x(*)
       real(c_double) :: y(*)
       real(c_double) :: z(*)
     end subroutine pragmatic_3d_begin
     subroutine pragmatic_vtk_begin(filename) bind(c,name="pragmatic_vtk_begin")
       use iso_c_binding
       implicit none
       character(kind=c_char), intent(in) :: filename
     end subroutine pragmatic_vtk_begin
  end interface pragmatic_begin
  
   !> Add field to be included in the metric.
   !
   !> @param[in] psi Solutional variables stored at the mesh verticies.
   !> @param[in] error Target error.
   !> @param[in] p_norm Set this optional argument to a positive integer to apply the p-norm scaling to the metric.
  interface pragmatic_add_field
     subroutine pragmatic_add_field(psi, error, pnorm) bind(c,name="pragmatic_add_field")
       use iso_c_binding
       implicit none
       real(c_double) :: psi(*)
       real(c_double) :: error
       integer(c_int) :: pnorm
     end subroutine pragmatic_add_field
  end interface pragmatic_add_field

  !> Set the surface mesh.
  !
  !> @param[in] nfacets Number of surface elements.
  !> @param[in] facets Surface elements.
  !> @param[in] boundary_ids Boundary ID's.
  !> @param[in] coplanar_ids Co-planar ID's.
  interface pragmatic_set_surface
     subroutine pragmatic_set_surface(nfacets, facets, boundary_ids, coplanar_ids) bind(c,name="pragmatic_set_surface")
       use iso_c_binding
       implicit none
       integer(c_int) :: nfacets
       integer(c_int) :: facets(*)
       integer(c_int) :: boundary_ids(*)
       integer(c_int) :: coplanar_ids(*)
     end subroutine pragmatic_set_surface
  end interface pragmatic_set_surface
  
  !> Set the metric tensor field.
  !
  !> @param[in] metric Metric tensor field.
  interface pragmatic_set_metric
     subroutine pragmatic_set_metric(metric) bind(c,name="pragmatic_set_metric")
       use iso_c_binding
       implicit none
       real(c_double) :: metric(*)
     end subroutine pragmatic_set_metric
  end interface pragmatic_set_metric

  !> Adapt mesh.
  interface pragmatic_adapt
     subroutine pragmatic_adapt() bind(c,name="pragmatic_adapt")
       use iso_c_binding
     end subroutine pragmatic_adapt
  end interface pragmatic_adapt

  !> Free internal data structures.
  !
  !> @param[out] NNodes Number of nodes.
  !> @param[out] NElements Number of elements.
  !> @param[out] NSElements Number of surface elements.
  interface pragmatic_get_info
     subroutine pragmatic_get_info(NNodes, NElements, NSElements) bind(c,name="pragmatic_get_info")
       use iso_c_binding
       integer(kind=c_int) :: NNodes
       integer(kind=c_int) :: NElements
       integer(kind=c_int) :: NSElements
     end subroutine pragmatic_get_info
  end interface pragmatic_get_info

  !> Get the coordinates of the mesh vertices.
  !
  !> @param[out] x X-cooardinates.
  !> @param[out] y Y-cooardinates.
  !> @param[out] z Z-cooardinates.
  interface pragmatic_get_coords
     subroutine pragmatic_get_coords_2d(x, y) bind(c,name="pragmatic_get_coords_2d")
       use iso_c_binding
       implicit none
       real(c_double) :: x(*)
       real(c_double) :: y(*)
     end subroutine pragmatic_get_coords_2d
     subroutine pragmatic_get_coords_3d(x, y, z) bind(c,name="pragmatic_get_coords_3d")
       use iso_c_binding
       implicit none
       real(c_double) :: x(*)
       real(c_double) :: y(*)
       real(c_double) :: z(*)
     end subroutine pragmatic_get_coords_3d
  end interface pragmatic_get_coords

  !> Get the mesh elements.
  !
  !> @param[out] elements List of elements.
  interface pragmatic_get_elements
     subroutine pragmatic_get_elements(elements) bind(c,name="pragmatic_get_elements")
       use iso_c_binding
       implicit none
       integer(c_int) :: elements(*)
     end subroutine pragmatic_get_elements
  end interface pragmatic_get_elements

  !> Get the surface mesh.
  !
  !> @param[out] facets Surface elements.
  !> @param[out] boundary_ids Boundary ID's.
  !> @param[out] coplanar_ids Co-planar ID's.
  interface pragmatic_get_surface
     subroutine pragmatic_get_surface(facets, boundary_ids, coplanar_ids) bind(c,name="pragmatic_get_surface")
       use iso_c_binding
       implicit none
       integer(c_int) :: facets(*)
       integer(c_int) :: boundary_ids(*)
       integer(c_int) :: coplanar_ids(*)
     end subroutine pragmatic_get_surface
  end interface pragmatic_get_surface

  !> Get the global node numbering.
  !
  !> @param[out] nodes_per_partition Array with number of owned nodes per partition.
  !> @param[out] lnn2gnn Local node numbering to global node numbering mapping.
  interface pragmatic_get_lnn2gnn
     subroutine pragmatic_get_lnn2gnn(nodes_per_partition, lnn2gnn) bind(c,name="pragmatic_get_lnn2gnn")
       use iso_c_binding
       implicit none
       integer(c_int) :: nodes_per_partition(*)
       integer(c_int) :: lnn2gnn(*)
     end subroutine pragmatic_get_lnn2gnn
  end interface pragmatic_get_lnn2gnn

  !> Get final metric field.
  !
  !> @param[out] metric Metric tensor field.
  interface pragmatic_get_metric
     subroutine pragmatic_get_metric(metric) bind(c,name="pragmatic_get_metric")
       use iso_c_binding
       implicit none
       real(c_double) :: metric(*)
     end subroutine pragmatic_get_metric
  end interface pragmatic_get_metric

  !> Dump a VTK file with the current mesh along with quality diagnostics.
  !
  !> @param[in] filename Name of VTK file.
  interface pragmatic_dump
     subroutine pragmatic_dump(filename) bind(c,name="pragmatic_dump")
       use iso_c_binding
       implicit none
       character(kind=c_char), intent(in) :: filename
     end subroutine pragmatic_dump
  end interface pragmatic_dump

  !> Free internal data structures.
  interface pragmatic_end
     subroutine pragmatic_end() bind(c,name="pragmatic_end")
       use iso_c_binding
     end subroutine pragmatic_end
  end interface pragmatic_end

end module pragmatic
