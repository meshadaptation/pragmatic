#!/usr/bin/env sage

import GenerateC

# Consider optimising a single element.

# Define triangle vertices
x, y = var('x y') # the vertex position we are trying to optimise.
x1, y1 = var('x1 y1')
x2, y2 = var('x2 y2')

# A quadric function has been fit to the metric tensors in the patch
# around (x, y).
P = vector(SR, 6, [y**2, x**2, x*y, y, x, 1])
a = vector(SR, 6, [var('a0'), var('a1'), var('a2'), var('a3'), var('a4'), var('a5')])
b = vector(SR, 6, [var('b0'), var('b1'), var('b2'), var('b3'), var('b4'), var('b5')])
c = vector(SR, 6, [var('c0'), var('c1'), var('c2'), var('c3'), var('c4'), var('c5')])
# This M for the patch is:
# M = (m00 m01)
#     (m01 m11)
# where
# m00 = P.dot_product(a)
# m01 = P.dot_product(b)
# m11 = P.dot_product(c)
# M = matrix(SR, 2, 2, [m00, m01, m01, m11])

# For element i:
m00 = (P.dot_product(a) + var('m00_1') + var('m00_2'))/3
m01 = (P.dot_product(b) + var('m01_1') + var('m01_2'))/3
m11 = (P.dot_product(c) + var('m11_1') + var('m11_2'))/3
M = matrix(SR, 2, 2, [m00, m01, m01, m11])

# Define parameter in metric space.
l0 = vector(SR, 2, [x1-x,    y1-y])
l1 = vector(SR, 2, [x2-x1, y2-y1])
l2 = vector(SR, 2, [x-x2,    y-y2])

dl0 = sqrt((l0*M*l0.transpose()).get(0))
dl1 = sqrt((l1*M*l1.transpose()).get(0))
dl2 = sqrt((l2*M*l2.transpose()).get(0))

l = dl0+dl1+dl2

# Calculate area in metric space.
a_m = sqrt(M.det())*matrix(SR, 2, 2, [x1-x, y1-y, x2-x, y2-y]).det()/2

# f = min(l/3, 3/l)
# However, cannot differentiate min(). Therefore, we choose a branch depending on the outcome of the min.
f0 = l/3.0
f1 = 3.0/l

F0 = (f0 * (2.0 - f0))**3
F1 = (f1 * (2.0 - f1))**3

q0 = 12*sqrt(3)*a_m*F0/(l*l)
dq0dx = diff(q0, x)
dq0dy = diff(q0, y)

q1 = 12*sqrt(3)*a_m*F1/(l*l)
dq1dx = diff(q1, x)
dq1dy = diff(q1, y)
