#include "plex_mesh.h"
#include "Mesh.h"
#include "VTKTools.h"

int main(int argv, char **argc){
  PetscErrorCode ierr = PetscInitialize(&argv, &argc, NULL, "spoon feeding"); CHKERRQ(ierr);

  /* Build and dump a 2D unit square */
  DM unit_square_mesh;
  ierr = create_unit_square(10, 10, MPI_COMM_WORLD, &unit_square_mesh); CHKERRQ(ierr);
  ierr = DMView(unit_square_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  Mesh<double> mesh2d(unit_square_mesh, MPI_COMM_WORLD);
  VTKTools<double>::export_vtu("data/test_plex_2d", &mesh2d);


  /* Build and dump a 3D unit cube */
  DM unit_cube_mesh;
  ierr = create_unit_cube(6, 6, 6, MPI_COMM_WORLD, &unit_cube_mesh); CHKERRQ(ierr);
  ierr = DMView(unit_cube_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  Mesh<double> mesh3d(unit_cube_mesh, MPI_COMM_WORLD);
  VTKTools<double>::export_vtu("data/test_plex_3d", &mesh3d);

  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

