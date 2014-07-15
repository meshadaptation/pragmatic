#include <cfloat>
#include <cassert>

#include "plex_mesh.h"

PetscErrorCode create_unit_square(int I, int J, MPI_Comm comm, DM *mesh){
  // Create the boundary mesh first.
  DM boundary;
  PetscErrorCode ierr = DMPlexCreate(comm, &boundary); CHKERRQ(ierr);
  ierr = DMSetType(boundary, DMPLEX); CHKERRQ(ierr);
  ierr = DMPlexSetDimension(boundary, 1); CHKERRQ(ierr);
  
  PetscReal lower[] = {0, 0};
  PetscReal upper[] = {1, 1};
  PetscInt edges[] = {I, J};
  ierr = DMPlexCreateSquareBoundary(boundary, lower, upper, edges); CHKERRQ(ierr);
  
  // Mesh area
  DM lmesh;
  ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, &lmesh); CHKERRQ(ierr);

  /* Apply boundary IDs. Boundaries are labeled as:

     1: plane x == 0
     2: plane x == 1
     3: plane y == 0
     4: plane y == 1
   */

  ierr = DMPlexCreateLabel(lmesh, "boundary_ids"); CHKERRQ(ierr);
  
  DMLabel label;
  ierr = DMPlexGetLabel(lmesh, "boundary_ids", &label); CHKERRQ(ierr);

  ierr = DMPlexMarkBoundaryFaces(lmesh, label); CHKERRQ(ierr);

  Vec coords;
  ierr = DMGetCoordinates(lmesh, &coords); CHKERRQ(ierr);

  PetscSection coord_sec;
  ierr = DMGetCoordinateSection(lmesh, &coord_sec); CHKERRQ(ierr);

  PetscInt size;
  ierr = DMPlexGetStratumSize(lmesh, "boundary_ids", 1, &size); CHKERRQ(ierr);

  assert(size>0);

  IS regionIS;
  ierr = DMPlexGetStratumIS(lmesh, "boundary_ids", 1, &regionIS); CHKERRQ(ierr);

  const PetscInt *boundary_faces;
  ierr = ISGetIndices(regionIS, &boundary_faces); CHKERRQ(ierr);

  for(int i=0;i<size;i++){
    PetscInt face = boundary_faces[i];

    PetscInt csize;
    PetscScalar *face_coords;
    ierr = DMPlexVecGetClosure(lmesh, coord_sec, coords, face, &csize, &face_coords); CHKERRQ(ierr);

    assert(csize==4);

    if(fabs(face_coords[0])<DBL_EPSILON && fabs(face_coords[2])<DBL_EPSILON){
      DMPlexSetLabelValue(lmesh, "boundary_ids", face, 1);
    }else if(fabs(face_coords[0]-1.0)<DBL_EPSILON && fabs(face_coords[2]-1.0)<DBL_EPSILON){
      DMPlexSetLabelValue(lmesh, "boundary_ids", face, 2);
    }else if(fabs(face_coords[1])<DBL_EPSILON && fabs(face_coords[3])<DBL_EPSILON){
      DMPlexSetLabelValue(lmesh, "boundary_ids", face, 3);
    }else if(fabs(face_coords[1]-1.0)<DBL_EPSILON && fabs(face_coords[3]-1.0)<DBL_EPSILON){
      DMPlexSetLabelValue(lmesh, "boundary_ids", face, 4);
    }
 }

  int nranks;
  MPI_Comm_size(comm, &nranks);
  if(nranks>1){
    ierr = DMPlexDistribute(lmesh, NULL, 1, NULL, mesh);
    DMDestroy(&lmesh);
  }else{
     *mesh = lmesh;
  }

  ierr = DMDestroy(&boundary); CHKERRQ(ierr);

  return ierr;
}
/*
DM create_unit_cube(int I, int J, size_t K, MPI_Comm comm){
  DM boundary;
  PetscErrorCode ierr = DMPlexCreate(comm, &boundary); CHKERRQ(ierr);
  ierr = DMSetType(boundary, DMPLEX); CHKERRQ(ierr);
  ierr = DMPlexSetDimension(boundary, 2); CHKERRQ(ierr);
  
  PetscReal lower[] = {0, 0, 0};
  PetscReal upper[] = {1, 1, 1};
  PetscInt edges[] = {I, J, K};
  ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, edges); CHKERRQ(ierr);
  
  DM mesh;
  ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, &mesh); CHKERRQ(ierr);
  
  ierr = DMDestroy(&boundary); CHKERRQ(ierr);

  int nranks;
  MPI_Comm_size(comm, &nranks);
  if(nranks>1){
    DM pmesh;

    ierr = DMPlexDistribute(mesh, NULL, 1, NULL, &pmesh);
    // Need to check with valgrind if I need to destroy mesh

    mesh = pmesh;
  }

  return mesh;
}
*/
