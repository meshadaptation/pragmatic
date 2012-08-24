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

  integer(kind=c_int) :: NNodes, NElements, NSElements
  real(c_double), allocatable, dimension(:) :: xv, yv, zv, metric
  real(c_double), parameter :: eta=0.1, dh=0.02, maxh=1.0
  
  integer :: i
  real :: hx, hy, hz

  call pragmatic_begin("../data/box10x10x10.vtu"//C_NULL_CHAR)
  
  call pragmatic_get_info(NNodes, NElements, NSElements)

  allocate(xv(NNodes), yv(NNodes), zv(NNodes))
  call pragmatic_get_coords(xv, yv, zv)

  allocate(metric(6*NNodes))
  do i=1, NNodes
     hx=0.025 + 0.09*xv(i)
     hy=0.025 + 0.09*yv(i)
     hz=0.025 + 0.09*zv(i)
     
     metric((i-1)*6+1) = 1.0/hx**2 ; metric((i-1)*6+2) = 0         ; metric((i-1)*6+3) = 0
                                     metric((i-1)*6+4) = 1.0/hy**2 ; metric((i-1)*6+5) = 0
                                                                     metric((i-1)*6+6) = 1.0/hz**2
  end do  

  call pragmatic_set_metric(metric)

  call pragmatic_adapt()

  call pragmatic_dump("../data/ftest_adapt_3d"//C_NULL_CHAR)

  call pragmatic_end()

  ! For now just be happy we get this far without dieing.
  print*, "pass"
end program test_adapt
