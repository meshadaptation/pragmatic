#include "plex_mesh.h"
#include "Mesh.h"
#include "MetricField.h"
#include "Coarsen2D.h"

#include "VTKTools.h"

int main(int argv, char **argc){
  PetscErrorCode ierr = PetscInitialize(&argv, &argc, NULL, "spoon feeding"); CHKERRQ(ierr);

  /* Build and dump a 2D unit square */
  DM unit_square_mesh;
  ierr = create_unit_square(10, 10, MPI_COMM_WORLD, &unit_square_mesh); CHKERRQ(ierr);
  ierr = DMView(unit_square_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  Mesh<double> mesh2d(unit_square_mesh, MPI_COMM_WORLD);
  VTKTools<double>::export_vtu("../data/test_plex_2d", &mesh2d);

  long double perimeter = mesh2d.calculate_perimeter();
  long double area = mesh2d.calculate_area();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank==0){
    std::cout<<"Expecting perimeter == 4: ";
    if(fabs(perimeter-4)<=2*DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (perimeter="<<perimeter<<")"<<std::endl;

    std::cout<<"Expecting area == 1: ";
    if(fabs(area-1)<=2*DBL_EPSILON)
      std::cout<<"pass"<<std::endl;
    else
      std::cout<<"fail (area="<<area<<")"<<std::endl;
  }

  /* Add a metric field to the mesh */
  MetricField2D<double> metric_field_2d(mesh2d);
  for(size_t i=0;i<mesh2d.get_number_nodes();i++){
    double m[] = {0.5, 0.0, 0.5};
    metric_field_2d.set_metric(m, i);
  }
  metric_field_2d.update_mesh();

  /* Now perform a 2D adapt */
  Coarsen2D<double> adapt(mesh2d);
  double L_up = 0.3;
  double L_low = 0.2;
  adapt.coarsen(L_low, L_up);
  mesh2d.defragment();

  VTKTools<double>::export_vtu("../data/test_plex_2d_coarse", &mesh2d);

  /* Export adapted mesh to DMPlex */
  DM coarse_square;
  mesh2d.export_dmplex(&coarse_square);
  ierr = DMView(coarse_square, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);


  /* Build and dump a 3D unit cube */
  DM unit_cube_mesh;
  ierr = create_unit_cube(6, 6, 6, MPI_COMM_WORLD, &unit_cube_mesh); CHKERRQ(ierr);
  ierr = DMView(unit_cube_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  Mesh<double> mesh3d(unit_cube_mesh, MPI_COMM_WORLD);

  MetricField3D<double> metric_field(mesh3d);
  size_t NNodes = mesh3d.get_number_nodes();
  for(size_t i=0;i<NNodes;i++){
    double m[] = {0.5, 0.0, 0.0,
                  0.5, 0.0,
                  0.5};
    metric_field.set_metric(m, i);
  }
  metric_field.update_mesh();

  VTKTools<double>::export_vtu("../data/test_plex_3d", &mesh3d);

  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

