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

  integer :: NNodes, NElements, NSElements
  real(c_double), allocatable, dimension(:) :: x, y

  call pragmatic_begin("../data/box200x200.vtu"//C_NULL_CHAR)
  
  call pragmatic_get_mesh_info(NNodes, NElements, NSElements)

  allocate(x(NNodes), y(NNodes))
  call pragmatic_get_mesh_coords(x, y);

  !size_t NNodes = mesh->get_number_nodes();
  !double eta=0.1;
  !double dh=0.01;
  !for(size_t i=0;i<NNodes;i++){
  !  double x = 2*mesh->get_coords(i)[0]-1;
  !  double y = 2*mesh->get_coords(i)[1]-1;
  !  
  !  double d2fdx2 = -0.800000000000000/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 3)) + 0.00800000000000000/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 5)) - 250.000000000000*sin(50*x);
  !  double d2fdy2 = 2.50000000000000*sin(5*y)/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 2)) - 5.00000000000000*cos(5*y)*(5*y)/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 3)) + 0.0500000000000000*cos(5*y)*(5*y)/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 5));
  !  double d2fdxdy = 2.00000000000000*cos(5*y)/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 3)) - 0.0200000000000000*cos(5*y)/(double)((0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*(0.0100000000000000/(double)(2*x - sin(5*y))*(2*x - sin(5*y)) + 1)*pow((2*x - sin(5*y)), 5));

   ! if(isnan(d2fdx2)){
   !   double m[] =
   !     {1/(dh*dh), 0.0,
   !      0.0,       1/(dh*dh)};
   !   metric_field.set_metric(m, i);
   ! }else{
   !   double m[] =
   !     {d2fdx2/eta,  d2fdxdy/eta,
   !      d2fdxdy/eta, d2fdy2/eta};
   !   metric_field.set_metric(m, i);
   ! } 
  !}
  !metric_field.apply_min_edge_length(dh);
  !metric_field.apply_max_edge_length(1.0);
  ! subroutine pragmatic_addfield(psi, error)

 !pragmatic_set_metric(metric)

  !subroutine pragmatic_adapt()
  
  call pragmatic_end()

  ! For now just be happy we get this far without dieing.
  print*, "pass"
end program test_adapt
