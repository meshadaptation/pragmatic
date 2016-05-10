/*  Copyright (C) 2010 Imperial College London and others.
 *
 *  Please see the AUTHORS file in the main source directory for a
 *  full list of copyright holders.
 *
 *  Gerard Gorman
 *  Applied Modelling and Computation Group
 *  Department of Earth Science and Engineering
 *  Imperial College London
 *
 *  g.gorman@imperial.ac.uk
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

#ifndef PRAGMATICMINIS_H
#define PRAGMATICMINIS_H

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

// Definition of size_t
#include <cstdlib>
#include <atomic>

int pragmatic_nthreads()
{
#ifdef HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int pragmatic_thread_id()
{
#ifdef HAVE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

#ifdef HAVE_MPI
int pragmatic_nprocesses(MPI_Comm comm)
{
    int nprocesses;
    MPI_Comm_size(comm, &nprocesses);
    return nprocesses;
}

int pragmatic_process_id(MPI_Comm comm)
{
    int id;
    MPI_Comm_rank(comm, &id);
    return id;
}
#endif

// Returns the original value of shared, while incrementing *shared by inc.
size_t pragmatic_omp_atomic_capture(size_t* shared, size_t inc)
{
    size_t old;
#if __FUJITSU
    /*
     * 'add' vs 'addx' --> 'addx' performs addition with carry,
     * what we need is 'add'.
     *
     * 'casx' --> Register %0 is updated with the current value of the shared
     * variable, no matter whether the comparison succeeds. Following that, it is
     * obvious that we don't need to reload the shared variable from RAM into %g1
     * at the next iteration of the loop, so "ldx [%1], %%g1" can be pushed
     * outside the loop, therefore avoiding a redundant access to main memory per
     * iteration of the loop. We just need to copy %0 to %g1.
     *
     * 'bne,pt' --> "pt" is a hint for the branch predictor meaning that the
     * branch is expected NOT to be taken. The following 'mov' instruction is
     * called "branch delayed instruction" and is executed TOGETHER with the
     * preceeding branch instruction; we take advantage of this "instruction
     * slot" to copy the updated value of the shared variable from %0 to %g1
     * where it is expected to be found by the 'add'.
     */
    asm volatile(
        "ldx [%1], %%g1;"
        "retry:"
        "add %%g1, %2, %0;"
        "casx [%1], %%g1, %0;"
        "cmp %0, %%g1;"
        "bne,pn %%xcc, retry;"
        " mov %0, %%g1;"
        :"=&r"(old)
        :"p"(shared), "r"(inc)
        :"%g1"
    );
#elif HAVE_OPENMP >= 201107
    #pragma omp atomic capture
    {
        old = *shared;
        *shared += inc;
    }
#else
    old = __sync_fetch_and_add(shared, inc);
#endif
    return old;
}

#define pragmatic_isnormal std::isnormal
#define pragmatic_isnan std::isnan

#endif
