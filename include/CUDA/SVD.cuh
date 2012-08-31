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

#ifndef SVD_H
#define SVD_H

#include "defs.cuh"

#include <math.h>

/*
 * Calculates the eigenvalues of a 2x2 matrix A
 * eigenvalues[0] contains the largest eigenvalue
 * eigenvalues[1] contains the smallest eigenvalue
 */
__device__ void eigenvalues2x2(const real_t A[4], real_t eigenvalues[2])
{
	real_t b, discriminant;

	b = A[0]+A[3];

	discriminant = sqrt(b*b - (real_t) 4*(A[0]*A[3] - A[1]*A[2]));

	eigenvalues[0] = (b + discriminant) * (real_t) 0.5;
	eigenvalues[1] = (b - discriminant) * (real_t) 0.5;
}

/*
 * Calculates the eigenvectors of a 2x2 matrix A
 * given the sorted (largest to smallest) array of eigenvalues
 */
__device__ void eigenvectors2x2(const real_t A[4],
		const real_t eigenvalues[2], real_t eigenvectors[4])
{
	real_t D[4];
	real_t proportion;

	/*
	 * D[0] = A[0] - λ
	 * D[3] = A[3] - λ
	 */
	D[1] = A[1];
	D[2] = A[2];

	/*
	 * We want to solve the homogeneous system:
	 * ┌─            ─┐   ┌    ┐   ┌   ┐
	 * │A[0]-λ   A[1] │   │X[0]│   │ 0 │
	 * │              │ x │    │ = │   │
	 * │ A[2]   A[3]-λ│   │X[1]│   │ 0 │
	 * └─            ─┘   └    ┘   └   ┘
	 * In order to solve it, we impose one additional restriction:
	 * X[0]^2 + X[1]^2 = 1
	 */

	/*
	 * First round: eigenvector corresponding to the largest eigenvalue.
	 * eigenvector comprises the first column of eigenvectors,
	 * i.e. eigenvector = {eigenvectors[0], eigenvectors[2]}
	 */
	D[0] = A[0] - eigenvalues[0];
	D[3] = A[3] - eigenvalues[0];

	if(D[1] != (real_t) 0.0)
	{
		proportion = -(D[0]/D[1]);
		eigenvectors[0] = sqrt((real_t) 1 / (1 + proportion*proportion));
		eigenvectors[2] = proportion * eigenvectors[0];
	}
	else if(D[3] != (real_t) 0.0)
	{
		proportion = -(D[2]/D[3]);
		eigenvectors[0] = sqrt((real_t) 1 / (1 + proportion*proportion));
		eigenvectors[2] = proportion * eigenvectors[0];
	}
	else if(D[0] == (real_t) 0.0)
	{
		eigenvectors[0] = (real_t) 1.0;
		eigenvectors[2] = (real_t) 0;
	}
	else if(D[2] == (real_t) 0.0)
	{
		eigenvectors[2] = (real_t) 1.0;
		eigenvectors[0] = (real_t) 0;
	}

	/*
	 * Second round: eigenvector corresponding to the smallest eigenvalue.
	 * eigenvector comprises the second column of eigenvectors,
	 * i.e. eigenvector = {eigenvectors[1], eigenvectors[3]}
	 */
	D[0] = A[0] - eigenvalues[1];
	D[3] = A[3] - eigenvalues[1];

	if(D[1] != (real_t) 0.0)
	{
		proportion = -(D[0]/D[1]);
		eigenvectors[1] = sqrt((real_t) 1 / (1 + proportion*proportion));
		eigenvectors[3] = proportion * eigenvectors[1];
	}
	else if(D[3] != (real_t) 0.0)
	{
		proportion = -(D[2]/D[3]);
		eigenvectors[1] = sqrt((real_t) 1 / (1 + proportion*proportion));
		eigenvectors[3] = proportion * eigenvectors[1];
	}
	else if(D[0] == (real_t) 0.0)
	{
		eigenvectors[1] = (real_t) 1.0;
		eigenvectors[3] = (real_t) 0;
	}
	else if(D[2] == (real_t) 0.0)
	{
		eigenvectors[3] = (real_t) 1.0;
		eigenvectors[1] = (real_t) 0;
	}
}

/*
 * Solves the 2D linear system Ap=q using SVD
 */
__device__ void svd_solve2d(const real_t A[4], real_t p[2], const real_t q[2])
{
	/*
	 * If A is decomposed as A = U * Σ * Vtransp, where:
	 *
	 * U: the left singular vector
	 * Σ: the diagonal matrix containing A's singular values,
	 * V: the right singular vector,
	 * Vtransp: the transpose of V,
	 *
	 * then the solution to the linear system is:
	 *
	 * p = V * Σinv * Utransp * q
	 *
	 * where:
	 * Σinv: the inverse of Σ,
	 * Utransp: the transpose of U
	 */

	real_t AAT[4]; // This will be used to store either A*Atransp or Atransp*A
	real_t eigenvalues[2];
	real_t U[4];
	real_t V[4];

	// Caclulate Atransp*A
	AAT[0] = A[0]*A[0] + A[2]*A[2];
	AAT[1] = A[0]*A[1] + A[2]*A[3];
	AAT[2] = AAT[1];
	AAT[3] = A[1]*A[1] + A[3]*A[3];

	// Calculate the eigenvalues of AT*A
	eigenvalues2x2(AAT, eigenvalues);

	// Calculate the right singular vector V:
	eigenvectors2x2(AAT, eigenvalues, V);

	/*
	 * Using these eigenvalues, with λ0 > λ1, the diagonal matrix Σ is:
	 *     ┌                  ┐
	 *     │sqrt(λ0)      0   │
	 * Σ = │                  │
	 *     │   0      sqrt(λ1)│
	 *     └                  ┘
	 *
	 * Σ is diagonal, so its inverse is simply:
	 * σ_inv(ij) = 1 / σ(ij), if i == j
	 *
	 * If an eigenvalue is very small or close to zero,
	 * then set the appropriate diagonal entry to zero instead
	 * of 1 / sqrt(eigenvalue).
	 *
	 * In order to spare memory, we will use the existing array
	 * of eigenvalues - the original eigenvalues are not needed anymore.
	 */
	if(eigenvalues[0] < 1E-12)
		eigenvalues[0] = (real_t) 0.0;
	else
		eigenvalues[0] = (real_t) 1.0 / sqrt(eigenvalues[0]);
	if(eigenvalues[1] < 1E-12)
		eigenvalues[1] = (real_t) 0.0;
	else
		eigenvalues[1] = (real_t) 1.0 / sqrt(eigenvalues[1]);

	/*
	 * Calculate the left singular vector U:
	 * Singular vectors U and V are unique, up to a free choice of sign. Once V
	 * has been defined, the choice for the sign of U is restricted. So, instead
	 * of calculating the eigenvectors of A*AT, U is formed using the formula:
	 * uj = σ_inv(j)*A*vj, where uj (resp. vj) is the j-th column of U (resp. V).
	 *
	 * U will be used in its transposed form, so we transpose it here directly...
	 */
	U[0] = eigenvalues[0] * (A[0]*V[0] + A[1]*V[2]);
	U[1] = eigenvalues[0] * (A[2]*V[0] + A[3]*V[2]);
	U[2] = eigenvalues[1] * (A[0]*V[1] + A[1]*V[3]);
	U[3] = eigenvalues[1] * (A[2]*V[1] + A[3]*V[3]);

	/*
	 * Use AAT to hold intermediate results - spare memory
	 *
	 * AAT = V * Σinv
	 */
	AAT[0] = V[0]*eigenvalues[0];
	AAT[1] = V[1]*eigenvalues[1];
	AAT[2] = V[2]*eigenvalues[0];
	AAT[3] = V[3]*eigenvalues[1];

	/*
	 * V is no longer needed, use it to calculate:
	 *
	 * V = AAT * Utransp
	 */
	V[0] = AAT[0]*U[0] + AAT[1]*U[2];
	V[1] = AAT[0]*U[1] + AAT[1]*U[3];
	V[2] = AAT[2]*U[0] + AAT[3]*U[2];
	V[3] = AAT[2]*U[1] + AAT[3]*U[3];

	/*
	 * Finally, calculate the solution of the system:
	 *
	 * p = V * q
	 */
	p[0] = V[0]*q[0] + V[1]*q[1];
	p[1] = V[2]*q[0] + V[3]*q[1];
}

#endif
