# Description
PRAgMaTIc (Parallel anisotRopic Adaptive Mesh ToolkIt) provides 2D/3D
anisotropic mesh adaptivity for meshes of simplexes. The target
applications are finite element and finite volume methods although
it can also be used as a lossy compression algorithm for 2 and 3D data
(e.g. image compression). It takes as its input the mesh and a metric
tensor field which encodes desired mesh element size
anisotropically.

The toolkit is written in C++ but also provides interfaces for Fortran. It 
has been integrated with [FEniCS/Dolfin](http://fenicsproject.org) and
integration with PETSc/DMPlex is planned.  One of the design goals of PRAgMaTIc
is to develop highly scalable algorithms for clusters of multi-core and
many-core nodes. PRAgMaTIc uses OpenMP for thread parallelism and MPI for
domain decomposition parallelisation.

# Publications
*A thread-parallel algorithm for anisotropic mesh adaptation*
Under review: [arXiv:1308.2480](http://arxiv.org/abs/1308.2480)

*Hybrid OpenMP/MPI anisotropic mesh smoothing*
DOI: [10.1016/j.procs.2012.04.166](http://dx.doi.org/10.1016/j.procs.2012.04.166)

*Accelerating Anisotropic Mesh Adaptivity on nVIDIA's CUDA Using Texture Interpolation*
DOI: [10.1007/978-3-642-23397-5_38](http://dx.doi.org/10.1007/978-3-642-23397-5_38)

# Configure
PRAgMaTIc can be configured with a number of custom options, which are shown in the following.

ENABLE_VTK=TRUE|FALSE

ENABLE_MPI=TRUE|FALSE

ENABLE_OPENMP=TRUE|FALSE

These can be set as env variables, or given as command line arguments to cmake, e.g. cmake -DENABLE_MPI=FALSE would disable MPI support. If both of the same option are given, the command line argument is given a higher priority over the env variable.

If neither is given the default value is used for the configuration, which is TRUE for all configure options.

If you want to build PRAgMaTIc in the source directory, do

$ cmake .

If you want to build outside the source directory, first change into the directory in which you want to build

$ cd BUILDDIR

then do

$ cmake path/to/source/directory

where BUILDDIR is the directory in which you want to build PRAgMaTIc, and path/to/source/directory is the absolute path of, or relative path to the source directory.

# Build
PRAgMaTIc is now ready to be built, simply do

$ make

# Testing
$ make test

# Install
If you want to install to a specific location then specify the
argument --prefix=/target_location for ./configure. The default
location is /usr/local.

$ make install

Depending on the target_location you might have to run with sudo rights.

$ sudo make install

# RPM
A RPM spec file is provided in the source root directory. Typically
you need to create a tar ball of the source directory and then execute
rpmbuild, e.g.:

$ rpmbuild -ta pragmatic-0.1.tar.gz 

See rpmbuild documentation for more details.

# DEBIAN

A Debian package is provided. To build a deb simply cd to the source
directory and execute:

$ dpkg-buildpackage -us -uc
