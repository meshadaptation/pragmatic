#include "plex_mesh.h"
#include "Mesh.h"
#include "VTKTools.h"

int main(int argv, char **argc){
  PetscErrorCode ierr = PetscInitialize(&argv, &argc, NULL, "spoon feeding"); CHKERRQ(ierr);
  
  DM unit_square_mesh;
  ierr = create_unit_square(10, 10, MPI_COMM_WORLD, &unit_square_mesh); CHKERRQ(ierr);
  ierr = DMView(unit_square_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  Mesh<double> mesh(unit_square_mesh, MPI_COMM_WORLD);

  VTKTools<double>::export_vtu("../data/test_plex_2d", &mesh);

  /*
  int NVerts, NElements;
  ///....

  std::vector<double> x(NVerts), y(NVerts);

  std::vector<int> ENList(NElements*3); // {n0, n1, n2,  ...
  std::vector<int> boundary(NElements*3); // {b0, b1, b2,  ...


  DM cube_mesh = create_unit_cube(10, 10, 10);
  ierr = DMView(square_mesh, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  */

  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

