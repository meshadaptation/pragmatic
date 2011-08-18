Summary:   Anisotropic mesh adaptivity
Name:	   pragmatic
Version:   0.1
Release:   0
License:   LGPL
Group:	   Applications/Programming
Vendor:    Applied Modelling and Computation Group (AMCG), Imperial College London
Packager:  Gerard Gorman <g.gorman@imperial.ac.uk>
URL:	   http://amcg.ese.ic.ac.uk
Source0:   %{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root
Prefix:	   /usr


%description
PRAgMaTIc (Parallel anisotRopic Adaptive Mesh ToolkIt) provides 2D/3D
anisotropic mesh adaptivity for meshes of simplexes. The target
applications are finite element and finite volume methods although the
it can also be used as a lossy compression algorithm for 2 and 3D data
(e.g. image compression). It takes as its input the mesh and a metric
tensor field which encodes desired mesh element size
anisotropically. The toolkit is written in C++ and has OpenMP and MPI
parallel support.

%prep
rm -rf $RPM_BUILD_DIR/%{name}-%{version}
zcat $RPM_SOURCE_DIR/%{name}-%{version}.tar.gz | tar -xvf -

%setup -q

%build
%configure
make

%install
rm -rf $RPM_BUILD_ROOT
make DESTDIR=$RPM_BUILD_ROOT install

%clean
rm -rf $RPM_BUILD_ROOT

%files -f INSTALLED_FILES

%defattr(-,root,root,-)

%changelog
* Wed Aug 17 2011 Gerard Gorman <g.gorman@imperial.ac.uk> - 
- Beta release.

