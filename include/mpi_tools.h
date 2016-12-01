/*  Copyright (C) 2012 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Georgios Rokos
 *  Software Performance Optimisation Group
 *  Department of Computing
 *  Imperial College London
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials provided
 *  with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 *  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 *  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 *  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */

#ifndef MPI_TOOLS_H
#define MPI_TOOLS_H

#ifdef HAVE_MPI
#include <mpi.h>

// Template MPI data types.
// Method by Justin Holewinski: http://jholewinski.org/blog/the-beauty-of-c-templates/

template <typename TYPE>
struct mpi_type_wrapper {
    const MPI_Datatype mpi_type;
    mpi_type_wrapper();
};

// Explicit instantiation for `char'
template <> mpi_type_wrapper<char>::mpi_type_wrapper();

// Explicit instantiation for `float'
template <> mpi_type_wrapper<float>::mpi_type_wrapper();

// Explicit instantiation for `double'
template <> mpi_type_wrapper<double>::mpi_type_wrapper();

// Explicit instantiation for `short'
template <> mpi_type_wrapper<short>::mpi_type_wrapper();

// Explicit instantiation for `int'
template <> mpi_type_wrapper<int>::mpi_type_wrapper();

// Explicit instantiation for `long'
template <> mpi_type_wrapper<long>::mpi_type_wrapper();

// Explicit instantiation for `unsigned long'
template <> mpi_type_wrapper<unsigned long>::mpi_type_wrapper();

// Explicit instantiation for `long long'
template <> mpi_type_wrapper<long long>::mpi_type_wrapper();
#endif

#endif
