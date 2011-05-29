#!/usr/bin/env python
from pylab import *
from numpy import *
from scipy.optimize import leastsq
from math import pi

# Data to fit
num = 100
x = linspace(0.01, 10.0, num=num)
y = [(min(i, 1.0/i) * (2.0 - min(i, 1.0/i)))**3  for i in x]

# Fit polynomial
# Parametric function: 'v' is the parameter vector, 'x' the independent varible
#fp = lambda p, x: p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*x**5 \
#    + p[6]*x**6 + p[7]*x**7 + p[8]*x**8
#p0 = [-1.53379107e-01, 7.99105505e-01, -2.35352386e-01, 2.94488939e-02, -1.85329222e-03, 5.76841376e-05,
#       -7.05937003e-07, 0.1, 0.1]

# Fit gaussian
#x0 = 1.0
#fp = lambda p, x: exp(-p[0]*(x-x0)**2)
#p0 = [0.1]

# Levy distribution: u, c==p[0]
#u = 0
#fp = lambda p, x: sqrt(p[0]/(2*pi))*exp(-p[0]/(2*(x-u)))/(x-u)**(3./2.)
#p0 = [1.13156567]

# Cauchy distribution **best so far
# x0 = 1.0
# fp = lambda p, x: (p[1]/pi)*(p[0]/((x-x0)**2 + p[0]**2))
# p0 = [0.6, 1.9]

# Gamma distribution
#fp = lambda p, x: x*exp(-x/p[0])/p[0]
#p0 = [2.0]

# Inverse-gamma distribution
fp = lambda p, x: p[1]*p[0]**3*x**-4*exp(-p[0]/x)/2
p0 = [3.0, 1.0]

# Error function
e = lambda p, x, y: (fp(p,x)-y)

p, success = leastsq(e, p0, args=(x,y), maxfev=10000)

## Plot
def plot_fit():
    print 'Estimater parameters: ', p
    plot(x,y,'ro', x, fp(p,x))

plot_fit()
show()
