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
#ifndef HALOEXCHANGE_H
#define HALOEXCHANGE_H

#ifdef HAVE_MPI

#include <vector>
#include <cassert>

#include "PragmaticTypes.h"
#include "mpi_tools.h"

template <typename DATATYPE, int block>
void halo_update(MPI_Comm comm,
                 const std::vector< std::vector<index_t> > &send,
                 const std::vector< std::vector<index_t> > &recv,
                 std::vector<DATATYPE> &vec)
{
    int num_processes;
    MPI_Comm_size(comm, &num_processes);
    if(num_processes<2)
        return;

    assert(num_processes==send.size());
    assert(num_processes==recv.size());

    int rank;
    MPI_Comm_rank(comm, &rank);

    mpi_type_wrapper<DATATYPE> wrap;

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);

    // Setup non-blocking receives.
    std::vector< std::vector<DATATYPE> > recv_buff(num_processes);
    for(int i=0; i<num_processes; i++) {
        if((i==rank)||(recv[i].size()==0)) {
            request[i] =  MPI_REQUEST_NULL;
        } else {
            recv_buff[i].resize(recv[i].size()*block);
            MPI_Irecv(&(recv_buff[i][0]), recv_buff[i].size(), wrap.mpi_type, i, 0, comm, &(request[i]));
        }
    }

    // Non-blocking sends.
    std::vector< std::vector<DATATYPE> > send_buff(num_processes);
    for(int i=0; i<num_processes; i++) {
        if((i==rank)||(send[i].size()==0)) {
            request[num_processes+i] = MPI_REQUEST_NULL;
        } else {
            for(typename std::vector<index_t>::const_iterator it=send[i].begin(); it!=send[i].end(); ++it)
                for(int j=0; j<block; j++) {
                    send_buff[i].push_back(vec[(*it)*block+j]);
                }
            MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), wrap.mpi_type, i, 0, comm, &(request[num_processes+i]));
        }
    }

    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

    for(int i=0; i<num_processes; i++) {
        int k=0;
        for(typename std::vector<index_t>::const_iterator it=recv[i].begin(); it!=recv[i].end(); ++it, ++k)
            for(int j=0; j<block; j++)
                vec[(*it)*block+j] = recv_buff[i][k*block+j];
    }

    return;
}

template <typename DATATYPE, int block0, int block1>
void halo_update(MPI_Comm comm,
                 const std::vector< std::vector<index_t> > &send,
                 const std::vector< std::vector<index_t> > &recv,
                 std::vector<DATATYPE> &vec0, std::vector<DATATYPE> &vec1)
{
    int num_processes;
    MPI_Comm_size(comm, &num_processes);
    if(num_processes<2)
        return;

    assert(num_processes==send.size());
    assert(num_processes==recv.size());

    int rank;
    MPI_Comm_rank(comm, &rank);

    mpi_type_wrapper<DATATYPE> wrap;

    // MPI_Requests for all non-blocking communications.
    std::vector<MPI_Request> request(num_processes*2);

    // Setup non-blocking receives.
    std::vector< std::vector<DATATYPE> > recv_buff(num_processes);
    for(int i=0; i<num_processes; i++) {
        int msg_size = recv[i].size()*(block0+block1);
        if((i==rank)||(msg_size==0)) {
            request[i] =  MPI_REQUEST_NULL;
        } else {
            recv_buff[i].resize(msg_size);
            MPI_Irecv(&(recv_buff[i][0]), msg_size, wrap.mpi_type, i, 0, comm, &(request[i]));
        }
    }

    // Non-blocking sends.
    std::vector< std::vector<DATATYPE> > send_buff(num_processes);
    for(int i=0; i<num_processes; i++) {
        if((i==rank)||(send[i].size()==0)) {
            request[num_processes+i] = MPI_REQUEST_NULL;
        } else {
            for(typename std::vector<index_t>::const_iterator it=send[i].begin(); it!=send[i].end(); ++it) {
                for(int j=0; j<block0; j++) {
                    send_buff[i].push_back(vec0[(*it)*block0+j]);
                }
                for(int j=0; j<block1; j++) {
                    send_buff[i].push_back(vec1[(*it)*block1+j]);
                }
            }
            MPI_Isend(&(send_buff[i][0]), send_buff[i].size(), wrap.mpi_type, i, 0, comm, &(request[num_processes+i]));
        }
    }

    std::vector<MPI_Status> status(num_processes*2);
    MPI_Waitall(num_processes, &(request[0]), &(status[0]));
    MPI_Waitall(num_processes, &(request[num_processes]), &(status[num_processes]));

    int block01 = block0+block1;
    for(int i=0; i<num_processes; i++) {
        int k=0;
        for(typename std::vector<index_t>::const_iterator it=recv[i].begin(); it!=recv[i].end(); ++it, ++k) {
            for(int j=0; j<block0; j++)
                vec0[(*it)*block0+j] = recv_buff[i][k*block01+j];
            for(int j=0; j<block1; j++)
                vec1[(*it)*block1+j] = recv_buff[i][k*block01+block0+j];
        }
    }

    return;
}

#endif

#endif
