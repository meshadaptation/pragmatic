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
  real(c_double), allocatable, dimension(:) :: xv, yv, psi
  real(c_double), parameter :: eta=0.002
  
  integer :: i
  real :: x, y

  call pragmatic_begin("../data/smooth_2d.vtu"//C_NULL_CHAR)
  
  call pragmatic_get_info(NNodes, NElements, NSElements)

  allocate(xv(NNodes), yv(NNodes))
  call pragmatic_get_coords(xv, yv)

  allocate(psi(NNodes))
  do i=1, NNodes
     x = 2*xv(i)-1
     y = 2*yv(i)-1
     
     psi(i) = 0.100000000000000*sin(50*x) + atan2(-0.100000000000000, 2*x - sin(5*y))
  end do  
  call pragmatic_add_field(psi, eta, 1);

  call pragmatic_adapt()

  call pragmatic_dump("../data/ftest_adapt_2d"//C_NULL_CHAR)

  call pragmatic_end()

  ! For now just be happy we get this far without dieing.
  print*, "pass"
end program test_adapt
