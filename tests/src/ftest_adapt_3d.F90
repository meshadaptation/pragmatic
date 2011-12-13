!   Copyright (C) 2010 Imperial College London and others.
!   
!   Please see the AUTHORS file in the main source directory for a full list
!   of copyright holders.
!
!   Gerard Gorman
!   Applied Modelling and Computation Group
!   Department of Earth Science and Engineering
!   Imperial College London
!
!   amcgsoftware@imperial.ac.uk
!   
!   This library is free software; you can redistribute it and/or
!   modify it under the terms of the GNU Lesser General Public
!   License as published by the Free Software Foundation,
!   version 2.1 of the License.
!
!   This library is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!   Lesser General Public License for more details.
!
!   You should have received a copy of the GNU Lesser General Public
!   License along with this library; if not, write to the Free Software
!   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!   USA

program test_adapt
  use pragmatic
  use iso_c_binding
  implicit none

  integer(kind=c_int) :: NNodes, NElements, NSElements
  real(c_double), allocatable, dimension(:) :: xv, yv, zv, metric
  real(c_double), parameter :: eta=0.1, dh=0.02, maxh=1.0
  
  integer :: i
  real :: hx, hy, hz

  call pragmatic_begin("../data/box10x10x10.vtu"//C_NULL_CHAR)
  
  call pragmatic_get_mesh_info(NNodes, NElements, NSElements)

  allocate(xv(NNodes), yv(NNodes), zv(NNodes))
  call pragmatic_get_mesh_coords(xv, yv, zv)

  allocate(metric(9*NNodes))
  do i=1, NNodes
     hx=0.025 + 0.09*xv(i)
     hy=0.025 + 0.09*yv(i)
     hz=0.025 + 0.09*zv(i)
     
     metric((i-1)*9+1) = 1.0/hx**2 ; metric((i-1)*9+2) = 0         ; metric((i-1)*9+3) = 0
     metric((i-1)*9+4) = 0         ; metric((i-1)*9+5) = 1.0/hy**2 ; metric((i-1)*9+6) = 0
     metric((i-1)*9+7) = 0         ; metric((i-1)*9+8) = 0         ; metric((i-1)*9+9) = 1.0/hz**2
  end do  

  call pragmatic_set_metric(metric, dh, maxh)

  call pragmatic_adapt()

  call pragmatic_dump("../data/ftest_adapt_3d"//C_NULL_CHAR)

  call pragmatic_end()

  ! For now just be happy we get this far without dieing.
  print*, "pass"
end program test_adapt
