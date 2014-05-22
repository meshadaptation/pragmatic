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

program test_adapt
  use pragmatic
  use iso_c_binding
  implicit none

  include 'mpif.h' 

  integer(kind=c_int) :: NNodes, NElements, NSElements
  real(c_double), allocatable, dimension(:) :: xv, yv, psi
  real(c_double), parameter :: eta=0.002

  integer :: i, ierr
  real :: x, y

  call mpi_init(ierr)

  call pragmatic_init("../data/smooth_2d.vtu"//C_NULL_CHAR)

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

  call pragmatic_finalize()

  ! For now just be happy we get this far without dieing.
  print*, "pass"

  call mpi_finalize(ierr)
end program test_adapt
