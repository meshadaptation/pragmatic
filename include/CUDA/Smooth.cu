/*
 *    Copyright (C) 2011 Imperial College London and others.
 *
 *    Please see the AUTHORS file in the main source directory for a full list
 *    of copyright holders.
 *
 *    Georgios Rokos
 *    Software Performance Optimisation Group
 *    Department of Computing
 *    Imperial College London
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation,
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
 *    USA
 */

#ifndef CUDASMOOTH_H
#define CUDASMOOTH_H

#include "ElementProperty.cuh"
#include "SVD.cuh"

#include <stdint.h>

extern "C" {

__constant__ real_t * coords;
__constant__ real_t * metric;
__constant__ real_t * normals;
__constant__ real_t * quality;
__constant__ index_t * ENList;
__constant__ index_t * SENList;
__constant__ index_t * NNListArray;
__constant__ index_t * NNListIndex;
__constant__ index_t * NEListArray;
__constant__ index_t * NEListIndex;
__constant__ index_t * SNEListArray;
__constant__ index_t * SNEListIndex;
__constant__ index_t * coplanar_ids;
__constant__ uint32_t * surfaceNodesArray;
__constant__ unsigned char * smoothStatus;

const real_t relax = (real_t) 1.0;
const size_t nbits = sizeof(uint32_t) * 8;

__device__ bool laplacian_2d_kernel(index_t node, real_t * p)
{
	uint32_t isOnSurface = surfaceNodesArray[node / nbits];
	isOnSurface >>= node % nbits;
	isOnSurface &= 0x1;

	const real_t * m0 = &metric[4*node];

	real_t x0 = coords[2*node];
	real_t y0 = coords[2*node+1];

	real_t A[4] = {0.0, 0.0, 0.0, 0.0};
	real_t q[2] = {0.0, 0.0};

	/* By default, a node's neighbours are the ones contained
	 * in the node-adjacency list. However, if the node is on
	 * the surface, its neighbours are found using the
	 * SNEList/SENList/SNeighbours combo. */
	const index_t * neighbours;
	index_t NNeighbours;
	index_t SNeighbours[2];

	const real_t * normal = NULL;
	if(isOnSurface)
	{
		index_t SElements[] = { SNEListArray[SNEListIndex[node]],
								SNEListArray[SNEListIndex[node]+1] };

		if(coplanar_ids[SElements[0]] == coplanar_ids[SElements[1]])
		{
			normal = &normals[2*SElements[0]];

			/* Visit both surface elements adjacent to the node
			 * and find the nodes that define the elements. This
			 * way, we can find the node's neighbours. */
			for(index_t i = 0, pos = 0; i < 2; i++)
				for(index_t j = 0; j < 2; j++)
					if(SENList[2*SElements[i] + j] != node)
						SNeighbours[pos++] = SENList[2*SElements[i] + j];

			NNeighbours = 2;
			neighbours = SNeighbours;
		}
		else // Corner node, in which case it cannot be moved.
			return false;
	}
	else
	{
		NNeighbours = NNListIndex[node+1] - NNListIndex[node];
		neighbours = &NNListArray[NNListIndex[node]];
	}

	for(index_t i = 0; i < NNeighbours; i++)
	{
		index_t il = neighbours[i];

		const real_t * m1 = &metric[4*il];

		real_t ml00 = 0.5*(m0[0] + m1[0]);
		real_t ml01 = 0.5*(m0[1] + m1[1]);
		real_t ml11 = 0.5*(m0[3] + m1[3]);

		real_t x = coords[2*il] - x0;
		real_t y = coords[2*il+1] - y0;

		q[0] += (ml00*x + ml01*y);
		q[1] += (ml01*x + ml11*y);

		A[0] += ml00;
		A[1] += ml01;
		A[3] += ml11;
	}

	A[2]=A[1];

	svd_solve2d(A, p, q);

	if(isOnSurface)
	{
		p[0] -= p[0]*fabs(normal[0]);
		p[1] -= p[1]*fabs(normal[1]);
	}

	return true;
}

__device__ bool generate_location_2d(index_t node, const real_t * p, real_t * mp)
{
	// Interpolate metric at this new position.
	real_t l[3];
	index_t best_e = -1;
	real_t tol = -1.0;

	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];
		const index_t * n = &ENList[3*eid];

		const real_t * x0 = &coords[ 2*n[0] ];
		const real_t * x1 = &coords[ 2*n[1] ];
		const real_t * x2 = &coords[ 2*n[2] ];

		/*
		 * Check for inversion by looking at the area of the
		 * element whose node is being moved.
		 */
		real_t element_area;
		if(n[0]==node)
			element_area = area(p, x1, x2);
		else if(n[1]==node)
			element_area = area(x0, p, x2);
		else
			element_area = area(x0, x1, p);

		if(element_area < 0.0)
		  return false;

		real_t L = area(x0, x1, x2);

		real_t ll[3];
		ll[0] = area(p,  x1, x2) / L;
		ll[1] = area(x0, p,  x2) / L;
		ll[2] = area(x0, x1, p ) / L;

		real_t min_l = min(ll[0], min(ll[1], ll[2]));

		if(best_e == -1 || min_l > tol)
		{
			tol = min_l;
			best_e = eid;
			for(int i = 0; i < 3; i++)
				l[i] = ll[i];
		}
	}

	const index_t * n = &ENList[3*best_e];

	for(size_t i = 0 ; i < 4; i++)
		mp[i] = l[0] * metric[4*n[0] + i] +
		        l[1] * metric[4*n[1] + i] +
		        l[2] * metric[4*n[2] + i];

	return true;
}

__device__ real_t functional_Linf_s(index_t node)
{
	real_t patch_quality = 1.0;

	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];

		if(quality[eid] < patch_quality)
			patch_quality = quality[eid];
	}

	return patch_quality;
}

__device__ real_t functional_Linf(index_t node, const real_t * p, const real_t * mp)
{
	real_t functional = 1.0;

	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];
		const index_t * n = &ENList[3*eid];
		index_t iloc = 0;

		while(n[iloc] != (index_t) node)
			iloc++;

		index_t loc1 = (iloc+1)%3;
		index_t loc2 = (iloc+2)%3;

		const real_t * x1 = &coords[ 2*n[loc1] ];
		const real_t * x2 = &coords[ 2*n[loc2] ];

		real_t fnl = lipnikov2d(p, x1, x2, mp, &metric[4*n[loc1]], &metric[4*n[loc2]]);

		if(fnl < functional)
			functional = fnl;
	}

	return functional;
}

__global__ void laplacian_2d(index_t * colourSet, index_t NNodesInSet)
{
	const index_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID >= NNodesInSet)
		return;

	index_t node = colourSet[threadID];
	smoothStatus[node] = 0;

	/*
	 * p is the vertex displacement vector.
	 * laplacian_2d_kernel only computes the displacement,
	 * it does not update the mesh with the new coordinates.
	 */
	real_t p[2];

	bool relocatable = laplacian_2d_kernel(node, p);

	if(!relocatable)
		return;

	p[0] += coords[2*node];
	p[1] += coords[2*node+1];
    real_t mp[4];
    bool valid = generate_location_2d(node, p, mp);
    if(!valid)
    {
    	/* Some verticies cannot be moved without causing inverted
    	 * elements. To try to free up this element we inform the outter
    	 * loop that the vertex has indeed moved so that the local
    	 * verticies are flagged for further smoothing. This gives the
    	 * chance of arriving at a new configuration where a valid
    	 * smooth can be performed.
    	 */
    	smoothStatus[node] = 1;
    	return;
    }

	for(size_t j=0; j < 4; j++)
		metric[4*node + j] = mp[j];

	coords[2*node] = p[0];
	coords[2*node+1] = p[1];

	// Designate that the vertex was relocated
	smoothStatus[node] = 1;
}

__global__ void smart_laplacian_2d(index_t * colourSet, index_t NNodesInSet)
{
	const index_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID >= NNodesInSet)
		return;

	index_t node = colourSet[threadID];
	smoothStatus[node] = 0;

	/*
	 * p is the vertex displacement vector.
	 * laplacian_2d_kernel only computes the displacement,
	 * it does not update the mesh with the new coordinates.
	 */
	real_t p[2];

	bool relocatable = laplacian_2d_kernel(node, p);

	if(!relocatable)
		return;

	/*
	 * Up to here, we have found the displacement of the vertex.
	 * Now comes the smart-Laplacian part which evaluates the
	 * quality of the cavity if the vertex is relocated.
	 */
	real_t mag = sqrt(p[0]*p[0] + p[1]*p[1]);
	real_t hat[] = {p[0]/mag, p[1]/mag};

	// This can happen if there is zero mag.
	if(!isnormal(hat[0]+hat[1]))
		return;

	real_t x0 = coords[2*node];
	real_t y0 = coords[2*node+1];
	real_t mp[4];

	// Find a valid location along the line of displacement.
	bool valid = false;
	real_t alpha = mag, functional;
	for(int rb = 0; rb < 10; rb++)
	{
		p[0] = x0 + alpha*hat[0];
		p[1] = y0 + alpha*hat[1];

		valid = generate_location_2d(node, p, mp);

		if(valid)
		{
			functional = functional_Linf(node, p, mp);
			break;
		}
		else
		{
			alpha *= 0.5;
			continue;
		}
	}
	if(!valid)
		return;

	// Recursive bisection search along line.
	const real_t orig_functional = functional_Linf_s(node);
	real_t alpha_lower = 0;
	real_t alpha_lower_func = orig_functional;
	real_t alpha_upper = alpha;
	real_t alpha_upper_func = functional;

	for(int rb = 0; rb < 10; rb++)
	{
		alpha = (alpha_lower + alpha_upper) * 0.5;
		p[0] = x0 + alpha*hat[0];
		p[1] = y0 + alpha*hat[1];

		valid = generate_location_2d(node, p, mp);

		// Check if this position improves the L-infinity norm.
		functional = functional_Linf(node, p, mp);

		if(alpha_lower_func < functional)
		{
			alpha_lower = alpha;
			alpha_lower_func = functional;
		}
		else
		{
			if(alpha_upper_func < functional)
			{
				alpha_upper = alpha;
				alpha_upper_func = functional;
			}
			else
			{
				alpha = alpha_upper;
				functional = alpha_upper_func;
				p[0] = x0 + alpha*hat[0];
				p[1] = y0 + alpha*hat[1];
				break;
			}
		}
	}

	if(functional < relax*orig_functional)
		return;

    // Recalculate qualities.
	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];
		const index_t * n = &ENList[3*eid];

		if(n[0]<0)
			continue;

		index_t iloc = 0;
		while(n[iloc] != (index_t) node)
			iloc++;

		index_t loc1 = (iloc+1)%3;
		index_t loc2 = (iloc+2)%3;

		const real_t * x1 = &coords[ 2*n[loc1] ];
		const real_t * x2 = &coords[ 2*n[loc2] ];

		quality[eid] = lipnikov2d(p, x1, x2, mp, &metric[4*n[loc1]], &metric[4*n[loc2]]);
    }

	for(size_t j=0; j < 4; j++)
		metric[4*node + j] = mp[j];

	coords[2*node] = p[0];
	coords[2*node+1] = p[1];

	// Designate that the vertex was relocated
	smoothStatus[node] = 1;
}

}

#endif
