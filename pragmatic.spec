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

%files
/usr/include/pragmatic/pragmatic_config.h
/usr/include/pragmatic/Edge.h
/usr/include/pragmatic/Swapping.h
/usr/include/pragmatic/MetricTensor.h
/usr/include/pragmatic/Colour.h
/usr/include/pragmatic/Surface.h
/usr/include/pragmatic/Smooth.h
/usr/include/pragmatic/VTKTools.h
/usr/include/pragmatic/Coarsen.h
/usr/include/pragmatic/Refine.h
/usr/include/pragmatic/Mesh.h
/usr/include/pragmatic/MetricField.h
/usr/include/pragmatic/Metis.h
/usr/include/pragmatic/ElementProperty.h
/usr/include/pragmatic/Eigen/Core
/usr/include/pragmatic/Eigen/src/Core/Coeffs.h
/usr/include/pragmatic/Eigen/src/Core/Flagged.h
/usr/include/pragmatic/Eigen/src/Core/util/Memory.h
/usr/include/pragmatic/Eigen/src/Core/util/DisableMSVCWarnings.h
/usr/include/pragmatic/Eigen/src/Core/util/EnableMSVCWarnings.h
/usr/include/pragmatic/Eigen/src/Core/util/Macros.h
/usr/include/pragmatic/Eigen/src/Core/util/ForwardDeclarations.h
/usr/include/pragmatic/Eigen/src/Core/util/XprHelper.h
/usr/include/pragmatic/Eigen/src/Core/util/StaticAssert.h
/usr/include/pragmatic/Eigen/src/Core/util/Meta.h
/usr/include/pragmatic/Eigen/src/Core/util/Constants.h
/usr/include/pragmatic/Eigen/src/Core/Assign.h
/usr/include/pragmatic/Eigen/src/Core/Swap.h
/usr/include/pragmatic/Eigen/src/Core/MatrixStorage.h
/usr/include/pragmatic/Eigen/src/Core/Block.h
/usr/include/pragmatic/Eigen/src/Core/NumTraits.h
/usr/include/pragmatic/Eigen/src/Core/arch/AltiVec/PacketMath.h
/usr/include/pragmatic/Eigen/src/Core/arch/SSE/PacketMath.h
/usr/include/pragmatic/Eigen/src/Core/MapBase.h
/usr/include/pragmatic/Eigen/src/Core/Matrix.h
/usr/include/pragmatic/Eigen/src/Core/DiagonalMatrix.h
/usr/include/pragmatic/Eigen/src/Core/Visitor.h
/usr/include/pragmatic/Eigen/src/Core/Product.h
/usr/include/pragmatic/Eigen/src/Core/IO.h
/usr/include/pragmatic/Eigen/src/Core/MathFunctions.h
/usr/include/pragmatic/Eigen/src/Core/CwiseNullaryOp.h
/usr/include/pragmatic/Eigen/src/Core/CwiseUnaryOp.h
/usr/include/pragmatic/Eigen/src/Core/Redux.h
/usr/include/pragmatic/Eigen/src/Core/GenericPacketMath.h
/usr/include/pragmatic/Eigen/src/Core/NestByValue.h
/usr/include/pragmatic/Eigen/src/Core/DiagonalProduct.h
/usr/include/pragmatic/Eigen/src/Core/Cwise.h
/usr/include/pragmatic/Eigen/src/Core/CoreInstantiations.cpp
/usr/include/pragmatic/Eigen/src/Core/SolveTriangular.h
/usr/include/pragmatic/Eigen/src/Core/Transpose.h
/usr/include/pragmatic/Eigen/src/Core/CommaInitializer.h
/usr/include/pragmatic/Eigen/src/Core/Dot.h
/usr/include/pragmatic/Eigen/src/Core/DiagonalCoeffs.h
/usr/include/pragmatic/Eigen/src/Core/Minor.h
/usr/include/pragmatic/Eigen/src/Core/Fuzzy.h
/usr/include/pragmatic/Eigen/src/Core/CwiseBinaryOp.h
/usr/include/pragmatic/Eigen/src/Core/Map.h
/usr/include/pragmatic/Eigen/src/Core/Sum.h
/usr/include/pragmatic/Eigen/src/Core/Functors.h
/usr/include/pragmatic/Eigen/src/Core/Part.h
/usr/include/pragmatic/Eigen/src/Core/MatrixBase.h
/usr/include/pragmatic/Eigen/src/Core/CacheFriendlyProduct.h
/usr/include/pragmatic/Eigen/src/Cholesky/CholeskyInstantiations.cpp
/usr/include/pragmatic/Eigen/src/Cholesky/LLT.h
/usr/include/pragmatic/Eigen/src/Cholesky/LDLT.h
/usr/include/pragmatic/Eigen/src/Array/Random.h
/usr/include/pragmatic/Eigen/src/Array/Norms.h
/usr/include/pragmatic/Eigen/src/Array/BooleanRedux.h
/usr/include/pragmatic/Eigen/src/Array/Select.h
/usr/include/pragmatic/Eigen/src/Array/PartialRedux.h
/usr/include/pragmatic/Eigen/src/Array/CwiseOperators.h
/usr/include/pragmatic/Eigen/src/Array/Functors.h
/usr/include/pragmatic/Eigen/src/LeastSquares/LeastSquares.h
/usr/include/pragmatic/Eigen/src/QR/Tridiagonalization.h
/usr/include/pragmatic/Eigen/src/QR/SelfAdjointEigenSolver.h
/usr/include/pragmatic/Eigen/src/QR/EigenSolver.h
/usr/include/pragmatic/Eigen/src/QR/HessenbergDecomposition.h
/usr/include/pragmatic/Eigen/src/QR/QR.h
/usr/include/pragmatic/Eigen/src/QR/QrInstantiations.cpp
/usr/include/pragmatic/Eigen/src/LU/Inverse.h
/usr/include/pragmatic/Eigen/src/LU/Determinant.h
/usr/include/pragmatic/Eigen/src/LU/LU.h
/usr/include/pragmatic/Eigen/src/Geometry/ParametrizedLine.h
/usr/include/pragmatic/Eigen/src/Geometry/EulerAngles.h
/usr/include/pragmatic/Eigen/src/Geometry/Transform.h
/usr/include/pragmatic/Eigen/src/Geometry/AngleAxis.h
/usr/include/pragmatic/Eigen/src/Geometry/Quaternion.h
/usr/include/pragmatic/Eigen/src/Geometry/AlignedBox.h
/usr/include/pragmatic/Eigen/src/Geometry/OrthoMethods.h
/usr/include/pragmatic/Eigen/src/Geometry/Rotation2D.h
/usr/include/pragmatic/Eigen/src/Geometry/RotationBase.h
/usr/include/pragmatic/Eigen/src/Geometry/Hyperplane.h
/usr/include/pragmatic/Eigen/src/Geometry/Scaling.h
/usr/include/pragmatic/Eigen/src/Geometry/Translation.h
/usr/include/pragmatic/Eigen/src/Sparse/DynamicSparseMatrix.h
/usr/include/pragmatic/Eigen/src/Sparse/TaucsSupport.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseMatrix.h
/usr/include/pragmatic/Eigen/src/Sparse/CompressedStorage.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseBlock.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseCwise.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseMatrixBase.h
/usr/include/pragmatic/Eigen/src/Sparse/TriangularSolver.h
/usr/include/pragmatic/Eigen/src/Sparse/RandomSetter.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseTranspose.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseCwiseBinaryOp.h
/usr/include/pragmatic/Eigen/src/Sparse/SuperLUSupport.h
/usr/include/pragmatic/Eigen/src/Sparse/MappedSparseMatrix.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseProduct.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseLDLT.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseFuzzy.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseRedux.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseAssign.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseLU.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseCwiseUnaryOp.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseVector.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseUtil.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseDot.h
/usr/include/pragmatic/Eigen/src/Sparse/UmfPackSupport.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseDiagonalProduct.h
/usr/include/pragmatic/Eigen/src/Sparse/CholmodSupport.h
/usr/include/pragmatic/Eigen/src/Sparse/CoreIterators.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseLLT.h
/usr/include/pragmatic/Eigen/src/Sparse/SparseFlagged.h
/usr/include/pragmatic/Eigen/src/Sparse/AmbiVector.h
/usr/include/pragmatic/Eigen/src/SVD/SVD.h
/usr/include/pragmatic/Eigen/Cholesky
/usr/include/pragmatic/Eigen/StdVector
/usr/include/pragmatic/Eigen/Array
/usr/include/pragmatic/Eigen/Eigen
/usr/include/pragmatic/Eigen/NewStdVector
/usr/include/pragmatic/Eigen/QtAlignedMalloc
/usr/include/pragmatic/Eigen/LeastSquares
/usr/include/pragmatic/Eigen/QR
/usr/include/pragmatic/Eigen/LU
/usr/include/pragmatic/Eigen/Geometry
/usr/include/pragmatic/Eigen/Sparse
/usr/include/pragmatic/Eigen/Dense
/usr/include/pragmatic/Eigen/SVD

%defattr(-,root,root,-)

%changelog
* Wed Aug 17 2011 Gerard Gorman <g.gorman@imperial.ac.uk> - 
- Beta release.

