!  Copyright (C) 2010 Imperial College London and others.
!  
!  Please see the AUTHORS file in the main source directory for a full list
!  of copyright holders.
!
!  Gerard Gorman
!  Applied Modelling and Computation Group
!  Department of Earth Science and Engineering
!  Imperial College London
!
!  amcgsoftware@imperial.ac.uk
!  
!  This library is free software; you can redistribute it and/or
!  modify it under the terms of the GNU Lesser General Public
!  License as published by the Free Software Foundation,
!  version 2.1 of the License.
!
!  This library is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!  Lesser General Public License for more details.
!
!  You should have received a copy of the GNU Lesser General Public
!  License along with this library; if not, write to the Free Software
!  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!  USA

!> This module merely contains explicit interfaces to allow the
!> convenient use of vtkfortran in fortran.
module pragmatic
  use iso_c_binding
  
  private
  public :: pragmatic_begin, pragmatic_addfield, pragmatic_get_metric, &
       pragmatic_set_metric, pragmatic_end
  
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
  end interface pragmatic_begin
  
   !> Add field to be included in the metric.
   !
   !> @param[in] psi Solutional variables stored at the mesh verticies.
   !> @param[in] error Target error.
  interface pragmatic_addfield
     subroutine pragmatic_addfield(psi, error) bind(c,name="pragmatic_addfield")
       use iso_c_binding
       implicit none
       real(c_double) :: psi(*)
       real(c_double) :: error
     end subroutine pragmatic_addfield
  end interface pragmatic_addfield

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
  interface pragmatic_end
     subroutine pragmatic_end() bind(c,name="pragmatic_end")
       use iso_c_binding
     end subroutine pragmatic_end
  end interface pragmatic_end
end module pragmatic
