# Description
PRAgMaTIc (Parallel anisotRopic Adaptive Mesh ToolkIt) provides 2D/3D
anisotropic mesh adaptivity for meshes of simplexes. The target
applications are finite element and finite volume methods although
it can also be used as a lossy compression algorithm for 2 and 3D data
(e.g. image compression). It takes as its input the mesh and a metric
tensor field which encodes desired mesh element size
anisotropically.

The toolkit is written in C++ but also provides interfaces for C. It has been 
integrated with [FEniCS/Dolfin](http://fenicsproject.org), with 
[PETSc/DMPlex](https://www.mcs.anl.gov/petsc) and integration with 
[Firedrake](http://www.firedrakeproject.org/) is ongoing.  One of the design 
goals of PRAgMaTIc is to develop highly scalable algorithms for clusters of 
multi-core and many-core nodes. PRAgMaTIc uses OpenMP for thread parallelism 
and MPI for domain decomposition parallelisation.

# Publications
*A thread-parallel algorithm for anisotropic mesh adaptation*
Under review: [arXiv:1308.2480](http://arxiv.org/abs/1308.2480)

*Hybrid OpenMP/MPI anisotropic mesh smoothing*
DOI: [10.1016/j.procs.2012.04.166](http://dx.doi.org/10.1016/j.procs.2012.04.166)

*Accelerating Anisotropic Mesh Adaptivity on nVIDIA's CUDA Using Texture Interpolation*
DOI: [10.1007/978-3-642-23397-5_38](http://dx.doi.org/10.1007/978-3-642-23397-5_38)

# Requirements
The following libraries are required to build PRAgMaTIc:
- Eigen3
- MPI
- METIS (on debian this can be installed with sudo apt-get install libmetis-dev)
- VTK (optional)
- LibMeshb (optional).

# Configure
PRAgMaTIc can be configured with a number of custom options for IOs, which are shown in the following.

ENABLE_VTK=TRUE|FALSE

ENABLE_LIBMESHB=TRUE|FALSE

These can be set as env variables, or given as command line arguments to cmake, e.g. cmake -DENABLE_VTK=FALSE would disable MPI support. If both of the same option are given, the command line argument is given a higher priority over the env variable.

If neither is given the default value is used for the configuration, which is TRUE for all configure options.

Let <SRCDIR> be the path to the source directory of PRAgMaTIc, <BUILDDIR> the build directory in which you want to build PRAgMaTIc, and <INSTALLDIR> the directory in which you wish to install PRAgMaTIc. E.g. execute

$ mkdir <BUILDDIR>

$ cd <BUILDDIR>

Then do

$ cmake <SRCDIR>

If you want to install PRAgMaTIc to a specific location then specify the target location.

$ cmake -DCMAKE_INSTALL_PREFIX=<INSTALLDIR> <SRCDIR>

The default location for <INSTALLDIR> is /usr/local.

By default, PRAgMaTIc is built in debug mode. To build in release mode, set CMAKE_BUILD_TYPE=Release.

# Build
PRAgMaTIc is now ready to be built. In your <BUILDDIR> execute

$ make

# Testing
To build and run tests, execute

$ make test

# Install
In your <BUILDDIR>, execute

$ make install

Depending on <INSTALLDIR> you might have to run with sudo rights.

$ sudo make install

# Examples
Some examples can be found in <SRCDIR>/python. Dolfin is required to run them.

Note: PRAgMaTIc needs to be built prior to running the examples.

# DEBIAN

A Debian package is provided. To build a deb simply cd to the source
directory and execute:

$ dpkg-buildpackage -us -uc
