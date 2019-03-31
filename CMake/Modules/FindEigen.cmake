# - Try to find Eigen lib
# Once done this will define
#
#  EIGEN_FOUND - system has eigen lib
#  EIGEN_INCLUDE_DIR - the eigen include directory

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

# modified by N. Barral <n.barral@imperial.ac.uk>

if (EIGEN_INCLUDE_DIR)

  # in cache already
  set(EIGEN_FOUND TRUE)

else (EIGEN_INCLUDE_DIR)

find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Core
     PATH_SUFFIXES eigen3
     HINTS
     ${INCLUDE_INSTALL_DIR}
   )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen DEFAULT_MSG EIGEN_INCLUDE_DIR )


mark_as_advanced(EIGEN_INCLUDE_DIR)

endif(EIGEN_INCLUDE_DIR)

