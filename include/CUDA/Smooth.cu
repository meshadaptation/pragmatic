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

#ifndef CUDASMOOTH_H
#define CUDASMOOTH_H

#include "defs.cuh"
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

const real_t sigma_q = 0.0001;
const real_t good_q = 0.7;
const size_t nbits = sizeof(uint32_t) * 8;

// Function prototypes
__device__ bool laplacian_2d_kernel(const index_t node, real_t * p);
__device__ bool smart_laplacian_2d_kernel(const index_t node);
__device__ bool generate_location_2d(const index_t node, const real_t * p, real_t * mp);
__device__ real_t functional_Linf_node_2d(const index_t node);
__device__ real_t functional_Linf_2d(const index_t node, const real_t * p, const real_t * mp);
__device__ void calculate_local_gradient_2d(const index_t node, const index_t eid, real_t * lg);
__device__ void grad_r(index_t node, const real_t *r1, const real_t *m1,
            const real_t *r2, const real_t *m2, real_t * grad);

__device__ bool laplacian_2d_kernel(const index_t node, real_t * p)
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

__device__ bool smart_laplacian_2d_kernel(const index_t node)
{
	real_t p[2];
	bool valid = laplacian_2d_kernel(node, p);

	if(!valid)
		return false;

	p[0] += coords[2*node];
	p[1] += coords[2*node+1];
    real_t mp[4];

    valid = generate_location_2d(node, p, mp);
    if(!valid)
      return false;

    real_t functional = functional_Linf_2d(node, p, mp);
    real_t orig_functional = functional_Linf_node_2d(node);

    if(functional - orig_functional < sigma_q)
      return false;

    // Reset quality cache.
	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];
		quality[eid] = -1.0;
    }

	for(size_t j = 0; j < 4; j++)
		metric[4*node + j] = mp[j];

	coords[2*node] = p[0];
	coords[2*node+1] = p[1];

	return true;
}

__device__ bool generate_location_2d(const index_t node, const real_t * p, real_t * mp)
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

__device__ real_t functional_Linf_node_2d(const index_t node)
{
	real_t patch_quality = 1.0;

	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];

		if(quality[eid] < 0.0)
		{
			const index_t * n = &ENList[3*eid];
			const real_t * x[3];
			const real_t * m[3];
			for(size_t i = 0; i < 3; i++)
			{
				x[i] = &coords[2*n[i]];
				m[i] = &metric[4*n[i]];
			}
			quality[eid] = lipnikov2d(x[0], x[1], x[2], m[0], m[1], m[2]);
		}

		if(quality[eid] < patch_quality)
			patch_quality = quality[eid];
	}

	return patch_quality;
}

__device__ real_t functional_Linf_2d(const index_t node, const real_t * p, const real_t * mp)
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

		const real_t * m1 = &metric[ 4*n[loc1] ];
		const real_t * m2 = &metric[ 4*n[loc2] ];

		real_t fnl = lipnikov2d(p, x1, x2, mp, m1, m2);

		if(fnl < functional)
			functional = fnl;
	}

	return functional;
}

__device__ void calculate_local_gradient_2d(const index_t node, const index_t eid, real_t * grad)
{
	// Differentiate quality functional for elements with respect to x,y
	const index_t * n = &ENList[3*eid];
	size_t loc = 0;

	while(n[loc] != node)
		loc++;

	index_t loc1 = (loc+1)%3;
	index_t loc2 = (loc+2)%3;

	const real_t * r1 = &coords[2*n[loc1]];
	const real_t * r2 = &coords[2*n[loc2]];

	const real_t * m1 = &metric[4*n[loc1]];
	const real_t * m2 = &metric[4*n[loc2]];

	grad_r(node, r1, m1, r2, m2, grad);
}

__device__ void grad_r(index_t node, const real_t *r1, const real_t *m1,
            const real_t *r2, const real_t *m2, real_t * grad)
{
	grad[0] = 0.0;
	grad[1] = 0.0;

	const real_t * r0 = &coords[2*node];

	real_t linf_x = max(fabs(r1[0]-r0[0]), fabs(r2[0]-r0[0]));
	real_t delta_x = linf_x * 1.0e-2;

	real_t linf_y = max(fabs(r1[1]-r0[1]), fabs(r2[1]-r0[1]));
	real_t delta_y = linf_y * 1.0e-1;

	real_t p[2];
	real_t mp[4];

	bool valid_move_minus_x = false, valid_move_plus_x = false;
	real_t functional_minus_dx = 0.0, functional_plus_dx = 0.0;

	for(int i = 0; (i < 5) && (!valid_move_minus_x) && (!valid_move_plus_x); i++)
	{
		p[0] = r0[0] - delta_x / 2;
		p[1] = r0[1];
		valid_move_minus_x = generate_location_2d(node, p, mp);
		if(valid_move_minus_x)
			functional_minus_dx = lipnikov2d(p, r1, r2, mp, m1, m2);

		p[0] = r0[0] + delta_x / 2;
		p[1] = r0[1];
		valid_move_plus_x = generate_location_2d(node, p, mp);
		if(valid_move_plus_x)
			functional_plus_dx = lipnikov2d(p, r1, r2, mp, m1, m2);

		if((!valid_move_minus_x) && (!valid_move_plus_x))
			delta_x /= 2;
	}

	bool valid_move_minus_y = false, valid_move_plus_y = false;
	real_t functional_minus_dy = 0, functional_plus_dy = 0;
	for(int i = 0; (i < 5) && (!valid_move_minus_y) && (!valid_move_plus_y); i++)
	{
		p[0] = r0[0];
		p[1] = r0[1] - delta_y / 2;
		valid_move_minus_y = generate_location_2d(node, p, mp);
		if(valid_move_minus_y)
			functional_minus_dy = lipnikov2d(p, r1, r2, mp, m1, m2);

		p[0] = r0[0];
		p[1] = r0[1] + delta_y / 2;
		valid_move_plus_y = generate_location_2d(node, p, mp);
		if(valid_move_plus_y)
			functional_plus_dy = lipnikov2d(p, r1, r2, mp, m1, m2);

		if((!valid_move_minus_y) && (!valid_move_plus_y))
			delta_y /= 2;
	}

	if(valid_move_minus_x && valid_move_plus_x)
		grad[0] = (functional_plus_dx - functional_minus_dx) / delta_x;
	else if(valid_move_minus_x)
		grad[0] = (quality[node] - functional_minus_dx) / (delta_x * 0.5);
	else if(valid_move_plus_x)
		grad[0] = (functional_plus_dx - quality[node]) / (delta_x * 0.5);

	if(valid_move_minus_y && valid_move_plus_y)
		grad[1] = (functional_plus_dy - functional_minus_dy) / delta_y;
	else if(valid_move_minus_y)
		grad[1] = (quality[node] - functional_minus_dy) / (delta_y * 0.5);
	else if(valid_move_plus_y)
		grad[1] = (functional_plus_dy - quality[node]) / (delta_y * 0.5);
}

__global__ void laplacian_2d(const index_t * colourSet, const index_t NNodesInSet)
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

	bool valid = laplacian_2d_kernel(node, p);

	if(!valid)
		return;

	p[0] += coords[2*node];
	p[1] += coords[2*node+1];
    real_t mp[4];

    valid = generate_location_2d(node, p, mp);
    if(!valid)
    	return;

	for(size_t j = 0; j < 4; j++)
		metric[4*node + j] = mp[j];

	coords[2*node] = p[0];
	coords[2*node+1] = p[1];

	// Designate that the vertex was relocated
	smoothStatus[node] = 1;
}

__global__ void smart_laplacian_2d(const index_t * colourSet, const index_t NNodesInSet)
{
	const index_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID >= NNodesInSet)
		return;

	index_t node = colourSet[threadID];
	smoothStatus[node] = 0;

	/*
	 * If the move is valid, smart_laplacian_2d_kernel
	 * updates the coordinates and the metric.
	 */
	bool valid = smart_laplacian_2d_kernel(node);
	if(!valid)
		return;

	// Designate that the vertex was relocated
	smoothStatus[node] = 1;
}

__global__ void smart_laplacian_search_2d(const index_t * colourSet, const index_t NNodesInSet)
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

	bool valid = laplacian_2d_kernel(node, p);

	if(!valid)
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
	valid = false;
	real_t alpha = mag, functional;
	for(int rb = 0; rb < 5; rb++)
	{
		p[0] = x0 + alpha*hat[0];
		p[1] = y0 + alpha*hat[1];

		valid = generate_location_2d(node, p, mp);

		if(valid)
		{
			functional = functional_Linf_2d(node, p, mp);
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
	const real_t orig_functional = functional_Linf_node_2d(node);
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
		functional = functional_Linf_2d(node, p, mp);

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

	if(functional - orig_functional < sigma_q)
		return;

    // Reset quality cache.
	for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
	{
		index_t eid = NEListArray[i];
		quality[eid] = -1.0;
    }

	for(size_t j = 0; j < 4; j++)
		metric[4*node + j] = mp[j];

	coords[2*node] = p[0];
	coords[2*node+1] = p[1];

	// Designate that the vertex was relocated
	smoothStatus[node] = 1;
}

__global__ void optimisation_linf_2d(const index_t * colourSet, const index_t NNodesInSet)
{
	const index_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID >= NNodesInSet)
		return;

	index_t node = colourSet[threadID];
	smoothStatus[node] = 0;

	/*
	 * smart-Laplacian kernel updates coordinates
	 * when it finds a better location for the vertex
	 */
	bool update = smart_laplacian_2d_kernel(node);

	uint32_t isOnSurface = surfaceNodesArray[node / nbits];
	isOnSurface >>= node % nbits;
	isOnSurface &= 0x1;

	if(isOnSurface)
	{
		smoothStatus[node] = update;
		return;
	}

	for(int hill_climb_iteration = 0; hill_climb_iteration < 5; hill_climb_iteration++)
	{
		// As soon as the tolerance quality is reached, break.
		const real_t functional_0 = functional_Linf_node_2d(node);
		if(functional_0 > good_q)
			break;

		// Focusing on improving the worst element
		index_t target_element;
		real_t worst_quality = 1.0;
		for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
		{
			index_t eid = NEListArray[i];
			if(quality[eid] < worst_quality)
			{
				worst_quality = quality[eid];
				target_element = eid;
			}
		}

		// Find the distance we have to step to reach the local quality maximum.
		real_t alpha = -1.0;

		real_t hat0[2];
		calculate_local_gradient_2d(node, target_element, hat0);

		real_t mag0 = sqrt(hat0[0]*hat0[0] + hat0[1]*hat0[1]);
		hat0[0] /= mag0;
		hat0[1] /= mag0;

		index_t i = NEListIndex[node];
		for( ; i < NEListIndex[node+1]; ++i)
		{
			index_t eid = NEListArray[i];
			if(eid == target_element)
				continue;

			real_t hat1[2];
			calculate_local_gradient_2d(node, eid, hat1);

			real_t mag1 = sqrt(hat1[0]*hat1[0] + hat1[1]*hat1[1]);
			hat1[0] /= mag1;
			hat1[1] /= mag1;

			alpha = (quality[eid] - quality[target_element]) /
					(mag0 - (hat0[0]*hat1[0] + hat0[1]*hat1[1]) * mag1);

			if((!isnormal(alpha)) || (alpha<0))
			{
				alpha = -1.0;
				continue;
			}

			break;
		}

		// Adjust alpha to the nearest point where the patch functional intersects with another.
		for( ; i < NEListIndex[node+1]; ++i)
		{
			index_t eid = NEListArray[i];
			if(eid == target_element)
				continue;

			real_t hat1[2];
			calculate_local_gradient_2d(node, eid, hat1);

			real_t mag1 = sqrt(hat1[0]*hat1[0] + hat1[1]*hat1[1]);
			hat1[0] /= mag1;
			hat1[1] /= mag1;

			real_t new_alpha = (quality[eid] - quality[target_element]) /
					(mag0 - (hat0[0]*hat1[0] + hat0[1]*hat1[1]) * mag1);

			if((!isnormal(new_alpha)) || (new_alpha<0))
				continue;

			if(new_alpha < alpha)
				alpha = new_alpha;
		}

		// If there is no viable direction, break.
		if((!isnormal(alpha)) || (alpha <= 0.0))
			break;

		real_t p[2], gp[2], mp[4];
		bool valid_move = false;
		for(int i = 0; i < 10; i++)
		{
			// If the predicted improvement is less than sigma, break;
			if(mag0*alpha < sigma_q)
				break;

			p[0] = alpha*hat0[0];
			p[1] = alpha*hat0[1];

			// This can happen if there is zero gradient.
			if(!isnormal(p[0]+p[1]))
				break;

			const real_t * r0 = &coords[2*node];
			gp[0] = r0[0] + p[0];
			gp[1] = r0[1] + p[1];

			valid_move = generate_location_2d(node, gp, mp);
			if(!valid_move)
			{
				alpha /= 2;
				continue;
			}

			// Check if this position improves the local mesh quality.
			for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
			{
				index_t eid = NEListArray[i];
				const index_t * n = &ENList[3*eid];
				if(n[0]<0)
					continue;

				index_t iloc = 0;
				while(n[iloc] != node)
					iloc++;

				index_t loc1 = (iloc+1)%3;
				index_t loc2 = (iloc+2)%3;

				const real_t * x1 = &coords[2*n[loc1]];
				const real_t * x2 = &coords[2*n[loc2]];

				real_t functional = lipnikov2d(gp, x1, x2, mp, &metric[4*n[loc1]], &metric[4*n[loc2]]);
				if(functional - functional_0 < sigma_q)
				{
					alpha /= 2;
					valid_move = false;
					break;
				}
			}

			if(valid_move)
				break;
		}

		if(valid_move)
			update = true;
		else
			break;

		// Looks good so lets copy it back;
		for(index_t i = NEListIndex[node]; i < NEListIndex[node+1]; ++i)
		{
			index_t eid = NEListArray[i];
			const index_t * n = &ENList[3*eid];
			if(n[0]<0)
				continue;

			index_t iloc = 0;
			while(n[iloc] != node)
				iloc++;

			index_t loc1 = (iloc+1)%3;
			index_t loc2 = (iloc+2)%3;

			const real_t * x1 = &coords[2*n[loc1]];
			const real_t * x2 = &coords[2*n[loc2]];

			quality[eid] = lipnikov2d(gp, x1, x2, mp, &metric[4*n[loc1]], &metric[4*n[loc2]]);
		}

		for(size_t j = 0; j < 4; j++)
			metric[4*node + j] = mp[j];

		coords[2*node] = gp[0];
		coords[2*node+1] = gp[1];
	}

	smoothStatus[node] = update;
}

}

#endif
