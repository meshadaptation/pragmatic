/*  Copyright (C) 2010 Imperial College London and others.
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

#ifndef CUDATOOLS_H
#define CUDATOOLS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <string>
#include <map>
#include <set>
#include <vector>

#include <iostream>
#include <stdint.h>

#include <cuda.h>

#include "Mesh.h"
#include "Surface.h"

template<typename real_t, typename index_t> class CUDATools
{
public:

    CUDATools()
    {
        enabled = false;
    }

    bool isEnabled()
    {
        return enabled;
    }

    void initialize()
    {
        enabled = false;

        if(cuInit(0) != CUDA_SUCCESS) {
            std::cout << "Error initializing CUDA driver" << std::endl;;
            return;
        }

        int deviceCount = 0;
        cuDeviceGetCount(&deviceCount);
        if(deviceCount == 0) {
            std::cout << "No CUDA-enabled devices found" << std::endl;
            return;
        }

        if(cuDeviceGet(&cuDevice, 0) != CUDA_SUCCESS) {
            std::cout << "Cannot get CUDA device" << std::endl;
            return;
        }

        if(cuCtxCreate(&cuContext, 0, cuDevice) != CUDA_SUCCESS) {
            std::cout << "Error creating CUDA context" << std::endl;
            return;
        }

        if(cuModuleLoad(&smoothModule, "CUDA/Smooth.cubin") != CUDA_SUCCESS) {
            std::cout << "Error loading CUDA module \"Smooth\"" << std::endl;
            return;
        }
        //cuModuleLoad(&coarsenModule, "CUDA/Coarsen.ptx");
        //cuModuleLoad(&refineModule, "CUDA/Refine.ptx");

        enabled = true;
    }

    void copyMeshDataToDevice(Mesh<real_t, index_t> * mesh, Surface<real_t, index_t> * surface,
                              std::map<int, std::deque<index_t> > & colour_sets, std::vector<real_t>  & quality,
                              int orientation, size_t dimensions)
    {
        ndims = dimensions;
        nloc = ndims+1;
        NNodes = mesh->get_number_nodes();
        NElements = mesh->_NElements;
        NSElements = surface->get_number_facets();

        // convert pragmatic data-structures to C-style arrays
        NNListToArray(mesh->NNList);
        colourSetsToArray(colour_sets);
        NEListToArray(mesh->NEList);
        SNEListToArray(surface->SNEList);
        surfaceNodesToArray(surface->surface_nodes);

        // and copy everything to the device
        copyArrayToDevice<real_t>(mesh->get_coords(0), CUDA_coords, NNodes * ndims);
        copyArrayToDevice<real_t>(mesh->get_metric(0), CUDA_metric, NNodes * ndims * ndims);
        copyArrayToDevice<real_t>(surface->get_normal(0), CUDA_normals, NSElements * ndims);
        copyArrayToDevice<real_t>(&quality[0], CUDA_quality, NElements);
        copyArrayToDevice<index_t>(&mesh->_ENList[0], CUDA_ENList, NElements * nloc);
        copyArrayToDevice<index_t>(surface->get_coplanar_ids(), CUDA_coplanar_ids, NSElements);
        copyArrayToDevice<index_t>(&surface->SENList[0], CUDA_SENList, NSElements * ndims);
        copyArrayToDevice<index_t>(NNListArray, CUDA_NNListArray, NNListArray_size);
        copyArrayToDevice<index_t>(NNListIndex, CUDA_NNListIndex, NNodes+1);
        copyArrayToDevice<index_t>(colourArray, CUDA_colourArray, NNodes);
        copyArrayToDevice<index_t>(NEListArray, CUDA_NEListArray, NEListArray_size);
        copyArrayToDevice<index_t>(NEListIndex, CUDA_NEListIndex, NNodes+1);
        copyArrayToDevice<index_t>(SNEListArray, CUDA_SNEListArray, NSElements * ndims);
        copyArrayToDevice<index_t>(SNEListIndex, CUDA_SNEListIndex, NNodes);
        copyArrayToDevice<uint32_t>(surfaceNodesArray, CUDA_surfaceNodesArray, surfaceNodesArray_size);

        //set the constant symbols of the smoothing-kernel, i.e. the addresses of all arrays copied above
        CUdeviceptr address;
        size_t symbol_size;

#define SET_CONSTANT(SYMBOL_NAME) \
            cuModuleGetGlobal(&address, &symbol_size, smoothModule, #SYMBOL_NAME); \
            cuMemcpyHtoD(address, &CUDA_ ## SYMBOL_NAME, symbol_size);

        SET_CONSTANT(coords)
        SET_CONSTANT(metric)
        SET_CONSTANT(normals)
        SET_CONSTANT(quality)
        SET_CONSTANT(ENList)
        SET_CONSTANT(SENList)
        SET_CONSTANT(NNListArray)
        SET_CONSTANT(NNListIndex)
        SET_CONSTANT(NEListArray)
        SET_CONSTANT(NEListIndex)
        SET_CONSTANT(SNEListArray)
        SET_CONSTANT(SNEListIndex)
        SET_CONSTANT(coplanar_ids)
        SET_CONSTANT(surfaceNodesArray)
        SET_CONSTANT(smoothStatus)

        // set element orientation in CUDA smoothing kernel
        cuModuleGetGlobal(&CUDA_orientation, &symbol_size, smoothModule, "orientation");
        cuMemcpyHtoD(CUDA_orientation, &orientation, symbol_size);
    }

    void copyCoordinatesToDevice(Mesh<real_t, index_t> * mesh)
    {
        copyArrayToDeviceNoAlloc<real_t>((real_t *) &mesh->_coords[0], CUDA_coords, NNodes * ndims);
    }

    void copyMetricToDevice(Mesh<real_t, index_t> * mesh)
    {
        copyArrayToDeviceNoAlloc<real_t>((real_t *) &mesh->metric[0], CUDA_metric, NNodes * ndims * ndims);
    }

    void copyCoordinatesFromDevice(Mesh<real_t, index_t> * mesh)
    {
        copyArrayFromDevice<real_t>((real_t *) &mesh->_coords[0], CUDA_coords, NNodes * ndims);
    }

    void copyMetricFromDevice(Mesh<real_t, index_t> * mesh)
    {
        copyArrayFromDevice<real_t>((real_t *) &mesh->metric[0], CUDA_metric, NNodes * ndims * ndims);
    }

    void reserveSmoothStatusMemory()
    {
        if(cuMemAlloc(&CUDA_smoothStatus, NNodes * sizeof(unsigned char)) != CUDA_SUCCESS) {
            std::cout << "Error allocating CUDA memory" << std::endl;
            exit(1);
        }

        // set the constant symbol in CUDA smoothing kernel
        CUdeviceptr address;
        size_t symbol_size;
        cuModuleGetGlobal(&address, &symbol_size, smoothModule, "smoothStatus");
        cuMemcpyHtoD(address, &CUDA_smoothStatus, symbol_size);
    }

    void reserveActiveVerticesMemory()
    {
        if(cuMemAlloc(&CUDA_activeVertices, NNodes * sizeof(unsigned char)) != CUDA_SUCCESS) {
            std::cout << "Error allocating CUDA memory" << std::endl;
            exit(1);
        }

        // set the constant symbol in CUDA smoothing kernel
        CUdeviceptr address;
        size_t symbol_size;
        cuModuleGetGlobal(&address, &symbol_size, smoothModule, "activeVertices");
        cuMemcpyHtoD(address, &CUDA_activeVertices, symbol_size);
    }

    void retrieveSmoothStatus(std::vector<unsigned char> & status)
    {
        copyArrayFromDevice<unsigned char>( (unsigned char *) &status[0], CUDA_smoothStatus, NNodes);
    }

    void freeResources()
    {
        cuMemFree(CUDA_coords);
        cuMemFree(CUDA_metric);
        cuMemFree(CUDA_normals);
        cuMemFree(CUDA_quality);
        cuMemFree(CUDA_ENList);
        cuMemFree(CUDA_coplanar_ids);
        cuMemFree(CUDA_SENList);
        cuMemFree(CUDA_NNListArray);
        cuMemFree(CUDA_NNListIndex);
        cuMemFree(CUDA_colourArray);
        cuMemFree(CUDA_SNEListArray);
        cuMemFree(CUDA_SNEListIndex);
        cuMemFree(CUDA_NEListArray);
        cuMemFree(CUDA_NEListIndex);
        cuMemFree(CUDA_surfaceNodesArray);
        cuMemFree(CUDA_smoothStatus);

        delete[] NNListArray;
        delete[] NNListIndex;
        delete[] colourArray;
        delete[] colourIndex;
        delete[] surfaceNodesArray;
        delete[] SNEListArray;
        delete[] SNEListIndex;
        delete[] NEListArray;
        delete[] NEListIndex;

        cuCtxDestroy(cuContext);
    }

    void setSmoothingKernel(std::string method, std::vector<unsigned char> & status)
    {
        if(cuModuleGetFunction(&smoothKernel, smoothModule, method.c_str()) != CUDA_SUCCESS) {
            std::cout << "Error loading CUDA kernel " << method << std::endl;
            enabled = false;
        }
    }

    void launchSmoothingKernel(int colour)
    {
        CUdeviceptr CUDA_ColourSetAddr = CUDA_colourArray + colourIndex[--colour] * sizeof(index_t);
        index_t NNodesInSet = colourIndex[colour+1] - colourIndex[colour];
        threadsPerBlock = 32;
        blocksPerGrid = (NNodesInSet + threadsPerBlock - 1) / threadsPerBlock;

        void * args[] = {&CUDA_ColourSetAddr, &NNodesInSet};

        CUresult result = cuLaunchKernel(smoothKernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, NULL);
        if(result != CUDA_SUCCESS) {
            std::cout << "Error launching CUDA kernel for colour " << colour << std::endl;
            return;
        }

        result = cuCtxSynchronize();
        if(result != CUDA_SUCCESS)
            std::cout << "Sync result " << result << std::endl;
    }

private:
    void NNListToArray(const std::vector< std::deque<index_t> > & NNList)
    {
        typename std::vector< std::deque<index_t> >::const_iterator vec_it;
        typename std::deque<index_t>::const_iterator deque_it;
        index_t offset = 0;
        index_t index = 0;

        for(vec_it = NNList.begin(); vec_it != NNList.end(); vec_it++)
            offset += vec_it->size();

        NNListArray_size = offset;

        NNListIndex = new index_t[NNodes+1];
        NNListArray = new index_t[NNListArray_size];

        offset = 0;

        for(vec_it = NNList.begin(); vec_it != NNList.end(); vec_it++) {
            NNListIndex[index++] = offset;

            for(deque_it = vec_it->begin(); deque_it != vec_it->end(); deque_it++)
                NNListArray[offset++] = *deque_it;
        }

        assert(index == NNList.size());
        NNListIndex[index] = offset;
    }

    void colourSetsToArray(const std::map< int, std::deque<index_t> > & colour_sets)
    {
        typename std::map< int, std::deque<index_t> >::const_iterator map_it;
        typename std::deque<index_t>::const_iterator deque_it;

        NColours = colour_sets.size();

        colourIndex = new index_t[NColours+1];
        colourArray = new index_t[NNodes];

        index_t offset = 0;

        for(map_it = colour_sets.begin(); map_it != colour_sets.end(); map_it++) {
            colourIndex[map_it->first - 1] = offset;

            for(deque_it = map_it->second.begin(); deque_it != map_it->second.end(); deque_it++)
                colourArray[offset++] = *deque_it;
        }

        colourIndex[colour_sets.size()] = offset;
    }

    void surfaceNodesToArray(const std::vector<bool> & surface_nodes)
    {
        const size_t nbits = sizeof(uint32_t) * 8;

        surfaceNodesArray_size = NNodes / nbits + (NNodes % nbits ? 1 : 0);
        surfaceNodesArray = new uint32_t[surfaceNodesArray_size];
        memset(surfaceNodesArray, 0, surfaceNodesArray_size * sizeof(uint32_t));

        for(index_t i = 0; i < (int) surface_nodes.size(); i++)
            if(surface_nodes[i] == true)
                surfaceNodesArray[i / nbits] |= 1 << i % nbits;
    }

    void SNEListToArray(const std::map<int, std::set<index_t> > & SNEList)
    {
        typename std::map< int, std::set<index_t> >::const_iterator map_it;
        typename std::set<index_t>::const_iterator set_it;

        SNEListArray = new index_t[NSElements*ndims];
        SNEListIndex = new index_t[NNodes];
        memset(SNEListIndex, 0, NNodes * sizeof(index_t));

        index_t offset = 0;
        for(map_it = SNEList.begin(); map_it != SNEList.end(); map_it++) {
            SNEListIndex[map_it->first] = offset;
            for(set_it = map_it->second.begin(); set_it != map_it->second.end(); set_it++)
                SNEListArray[offset++] = *set_it;
        }
    }

    void NEListToArray(const std::vector< std::set<index_t> > & NEList)
    {
        typename std::vector< std::set<index_t> >::const_iterator vec_it;
        typename std::set<index_t>::const_iterator set_it;
        index_t offset = 0;
        index_t index = 0;

        for(vec_it = NEList.begin(); vec_it != NEList.end(); vec_it++)
            offset += vec_it->size();

        NEListArray_size = offset;

        NEListIndex = new index_t[NNodes+1];
        NEListArray = new index_t[NEListArray_size];

        offset = 0;

        for(vec_it = NEList.begin(); vec_it != NEList.end(); vec_it++) {
            NEListIndex[index++] = offset;

            for(set_it = vec_it->begin(); set_it != vec_it->end(); set_it++)
                NEListArray[offset++] = *set_it;
        }

        assert(index == NEList.size());
        NEListIndex[index] = offset;
    }

    template<typename type>
    inline void copyArrayToDevice(const type * array, CUdeviceptr & CUDA_array, index_t array_size)
    {
        if(cuMemAlloc(&CUDA_array, array_size * sizeof(type)) != CUDA_SUCCESS) {
            std::cout << "Error allocating CUDA memory" << std::endl;
            exit(1);
        }

        cuMemcpyHtoD(CUDA_array, array, array_size * sizeof(type));
    }

    template<typename type>
    inline void copyArrayToDeviceNoAlloc(const type * array, CUdeviceptr & CUDA_array, index_t array_size)
    {
        cuMemcpyHtoD(CUDA_array, array, array_size * sizeof(type));
    }

    template<typename type>
    inline void copyArrayFromDevice(type * array, CUdeviceptr & CUDA_array, index_t array_size)
    {
        cuMemcpyDtoH(array, CUDA_array, array_size * sizeof(type));
    }

    bool enabled;

    CUdevice cuDevice;
    CUcontext cuContext;

    CUmodule smoothModule;
    CUmodule coarsenModule;
    CUmodule refineModule;

    CUfunction smoothKernel;
    CUfunction coarsenKernel;
    CUfunction refineKernel;

    unsigned int threadsPerBlock, blocksPerGrid;

    index_t NNodes, NElements, NSElements, ndims, nloc;

    CUdeviceptr CUDA_coords;
    CUdeviceptr CUDA_metric;
    CUdeviceptr CUDA_coplanar_ids;
    CUdeviceptr CUDA_normals;
    CUdeviceptr CUDA_ENList;
    CUdeviceptr CUDA_SENList;
    CUdeviceptr CUDA_quality;
    CUdeviceptr CUDA_smoothStatus;
    CUdeviceptr CUDA_activeVertices;

    index_t * NNListArray;
    index_t * NNListIndex;
    CUdeviceptr CUDA_NNListArray;
    CUdeviceptr CUDA_NNListIndex;
    index_t NNListArray_size;

    index_t * NEListArray;
    index_t * NEListIndex;
    CUdeviceptr CUDA_NEListArray;
    CUdeviceptr CUDA_NEListIndex;
    index_t NEListArray_size;

    index_t * colourArray;
    index_t* colourIndex;
    CUdeviceptr CUDA_colourArray;
    index_t NColours;

    uint32_t * surfaceNodesArray;
    index_t surfaceNodesArray_size;
    CUdeviceptr CUDA_surfaceNodesArray;

    index_t * SNEListArray;
    index_t * SNEListIndex;
    CUdeviceptr CUDA_SNEListArray;
    CUdeviceptr CUDA_SNEListIndex;

    CUdeviceptr CUDA_orientation;
};

#endif
