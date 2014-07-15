#ifndef PLEX_MESH_H
#define PLEX_MESH_H

#include <petsc-private/dmpleximpl.h>
#include <petsc-private/isimpl.h>

#include <petscdmda.h>
#include <petscsf.h>

#include <petsc.h>
#include <petscsys.h>
#include <petscerror.h>
#include <petscdmplex.h>
#include <petscviewertypes.h>
#include <petscsf.h>
#include <petscdm.h>
#include "petscdmplex.h"   

PetscErrorCode create_unit_square(int I, int J, MPI_Comm comm, DM *mesh);
PetscErrorCode create_unit_cube(int I, int J, int K, MPI_Comm comm, DM *mesh);

#endif
