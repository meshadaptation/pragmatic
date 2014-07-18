#include <cfloat>
#include <cassert>

#include "plex_mesh.h"

PetscErrorCode create_unit_square(int I, int J, MPI_Comm comm, DM *mesh){
  int rank;
  MPI_Comm_rank(comm, &rank);

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
  ierr = DMDestroy(&boundary); CHKERRQ(ierr);

  if (rank == 0) {
    /* Apply boundary IDs. Boundaries are labeled as:

       1: plane x == 0
       2: plane x == 1
       3: plane y == 0
       4: plane y == 1
    */
    DMLabel label;
    ierr = DMPlexCreateLabel(lmesh, "boundary_faces"); CHKERRQ(ierr);
    ierr = DMPlexGetLabel(lmesh, "boundary_faces", &label); CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(lmesh, label); CHKERRQ(ierr);

    Vec coords;
    ierr = DMGetCoordinatesLocal(lmesh, &coords); CHKERRQ(ierr);

    PetscSection coord_sec;
    ierr = DMGetCoordinateSection(lmesh, &coord_sec); CHKERRQ(ierr);

    PetscInt size;
    ierr = DMPlexGetStratumSize(lmesh, "boundary_faces", 1, &size); CHKERRQ(ierr);

    assert(rank==0 ? size>0 : true);

    IS regionIS;
    ierr = DMPlexGetStratumIS(lmesh, "boundary_faces", 1, &regionIS); CHKERRQ(ierr);

    const PetscInt *boundary_faces;
    ierr = ISGetIndices(regionIS, &boundary_faces); CHKERRQ(ierr);

    ierr = DMPlexCreateLabel(lmesh, "boundary_ids"); CHKERRQ(ierr);
    PetscInt face, csize;
    PetscScalar *face_coords = NULL;
    for(int i=0;i<size;i++){
      face = boundary_faces[i];
      ierr = DMPlexVecGetClosure(lmesh, coord_sec, coords, face,
                                 &csize, &face_coords); CHKERRQ(ierr);

      assert(rank==0 ? csize==4 : true);

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
    ierr = DMPlexVecRestoreClosure(lmesh, coord_sec, coords, face,
                                   &csize, &face_coords); CHKERRQ(ierr);
  }

  int nranks;
  MPI_Comm_size(comm, &nranks);
  if(nranks>1){
    ierr = DMPlexDistribute(lmesh, NULL, 1, NULL, mesh);
    DMDestroy(&lmesh);
  }else{
     *mesh = lmesh;
  }

  return ierr;
}

PetscErrorCode create_unit_cube(int I, int J, int K, MPI_Comm comm, DM *mesh){
  PetscErrorCode ierr;
  DM boundary;
  ierr = DMPlexCreate(comm, &boundary); CHKERRQ(ierr);
  ierr = DMSetType(boundary, DMPLEX); CHKERRQ(ierr);
  ierr = DMPlexSetDimension(boundary, 2); CHKERRQ(ierr);

  DM lmesh;
  PetscReal lower[] = {0, 0, 0};
  PetscReal upper[] = {1, 1, 1};
  PetscInt edges[] = {I, J, K};
  ierr = DMPlexCreateCubeBoundary(boundary, lower, upper, edges); CHKERRQ(ierr);
  ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, &lmesh); CHKERRQ(ierr);
  ierr = DMDestroy(&boundary); CHKERRQ(ierr);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    /* The boundary surface are numbered as follows:

     * 1: plane x == 0
     * 2: plane x == 1
     * 3: plane y == 0
     * 4: plane y == 1
     * 5: plane z == 0
     * 6: plane z == 1
     */
    DMLabel label;
    ierr = DMPlexCreateLabel(lmesh, "boundary_faces"); CHKERRQ(ierr);
    ierr = DMPlexGetLabel(lmesh, "boundary_faces", &label); CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(lmesh, label); CHKERRQ(ierr);

    Vec coords;
    ierr = DMGetCoordinatesLocal(lmesh, &coords); CHKERRQ(ierr);

    PetscSection coord_sec;
    ierr = DMGetCoordinateSection(lmesh, &coord_sec); CHKERRQ(ierr);

    PetscInt size;
    ierr = DMPlexGetStratumSize(lmesh, "boundary_faces", 1, &size); CHKERRQ(ierr);

    assert(rank==0 ? size>0 : true);

    IS regionIS;
    ierr = DMPlexGetStratumIS(lmesh, "boundary_faces", 1, &regionIS); CHKERRQ(ierr);

    const PetscInt *boundary_faces;
    ierr = ISGetIndices(regionIS, &boundary_faces); CHKERRQ(ierr);

    ierr = DMPlexCreateLabel(lmesh, "boundary_ids"); CHKERRQ(ierr);
    PetscInt face, csize;
    PetscScalar *face_coords= NULL;
    for(int i=0;i<size;i++){
      face = boundary_faces[i];
      ierr = DMPlexVecGetClosure(lmesh, coord_sec, coords, face,
                                 &csize, &face_coords); CHKERRQ(ierr);

      assert(rank==0 ? csize==9 : true);

      if(fabs(face_coords[0])<DBL_EPSILON && fabs(face_coords[3])<DBL_EPSILON &&
         fabs(face_coords[6])<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 1);
      }else if(fabs(face_coords[0]-1.0)<DBL_EPSILON && fabs(face_coords[3]-1.0)<DBL_EPSILON &&
               fabs(face_coords[6]-1.0)<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 2);
      }else if(fabs(face_coords[1])<DBL_EPSILON && fabs(face_coords[4])<DBL_EPSILON &&
               fabs(face_coords[7])<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 3);
      }else if(fabs(face_coords[1]-1.0)<DBL_EPSILON && fabs(face_coords[4]-1.0)<DBL_EPSILON &&
               fabs(face_coords[7]-1.0)<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 4);
      }else if(fabs(face_coords[2])<DBL_EPSILON && fabs(face_coords[5])<DBL_EPSILON &&
               fabs(face_coords[8])<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 5);
      }else if(fabs(face_coords[2]-1.0)<DBL_EPSILON && fabs(face_coords[5]-1.0)<DBL_EPSILON &&
               fabs(face_coords[8]-1.0)<DBL_EPSILON){ DMPlexSetLabelValue(lmesh, "boundary_ids", face, 6);
      }
    }
    ierr = DMPlexVecRestoreClosure(lmesh, coord_sec, coords, face,
                                   &csize, &face_coords); CHKERRQ(ierr);
  }

  int nranks;
  MPI_Comm_size(comm, &nranks);
  if(nranks>1){
    ierr = DMPlexDistribute(lmesh, NULL, 1, NULL, mesh); CHKERRQ(ierr);
    ierr = DMDestroy(&lmesh); CHKERRQ(ierr);
  } else {
    *mesh = lmesh;
  }

  return 0;
}
