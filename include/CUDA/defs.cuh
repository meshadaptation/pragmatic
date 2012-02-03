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

#ifndef DEFS_H
#define DEFS_H

extern "C" {

#define true 1
#define false 0
#define bool unsigned int

typedef double real_t;
typedef int index_t;

__device__ bool isnormal(real_t value)
{
	return !(value == 0.0 || isinf(value) || isnan(value));
}

}

#endif
