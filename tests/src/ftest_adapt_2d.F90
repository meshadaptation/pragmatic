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
  real(c_double), allocatable, dimension(:) :: xv, yv, metric
  real(c_double), parameter :: eta=0.1, dh=0.02, maxh=1.0
  
  integer :: i
  real :: x, y, d2fdx2, d2fdy2, d2fdxdy

  call pragmatic_begin("../data/box50x50.vtu"//C_NULL_CHAR)
  
  call pragmatic_get_mesh_info(NNodes, NElements, NSElements)

  allocate(xv(NNodes), yv(NNodes))
  call pragmatic_get_mesh_coords(xv, yv)

  allocate(metric(4*NNodes))
  do i=1, NNodes
     x = 2*xv(i)-1
     y = 2*yv(i)-1

     d2fdx2 = -0.8/((0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(2*x - sin(5*y))**3) + &
          0.008/((0.01/(2*x - sin(5*y))* &
          (2*x - sin(5*y)) + 1)*(0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)* &
          (2*x - sin(5*y))**5) - 250.0*sin(50*x)
     
     d2fdy2 = 2.5*sin(5*y)/((0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)* &
          (2*x - sin(5*y))**2) - 5.0*cos(5*y)*(5*y)/((0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)* &
          (2*x - sin(5*y))**3) + 0.05*cos(5*y)*(5*y)/((0.01/(2*x - sin(5*y))* &
          (2*x - sin(5*y)) + 1)*(0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(2*x - sin(5*y))**5)
     
     d2fdxdy = 2.0*cos(5*y)/((0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)* &
          (2*x - sin(5*y))**3) - 0.02*cos(5*y)/((0.01/(2*x - sin(5*y))* &
          (2*x - sin(5*y)) + 1)*(0.01/(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(2*x - sin(5*y))**5)
     
     metric((i-1)*4+1) = d2fdx2/eta
     metric((i-1)*4+2) = d2fdxdy/eta
     metric((i-1)*4+3) = d2fdxdy/eta
     metric((i-1)*4+4) = d2fdy2/eta
  end do  
  call pragmatic_set_metric(metric, dh, maxh)

  call pragmatic_adapt()

  call pragmatic_dump("../data/ftest_adapt_2d"//C_NULL_CHAR)

  call pragmatic_end()

  ! For now just be happy we get this far without dieing.
  print*, "pass"
end program test_adapt
