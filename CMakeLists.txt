CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)
PROJECT(PRAgMaTIc)

# The version number.
set (PRAgMaTIc_VERSION_MAJOR 1)
set (PRAgMaTIc_VERSION_MINOR 3)

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake/Modules ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_language(CXX)
enable_language(C)

set (PRAGMATIC_LIBRARIES)
# Use env variable iff it exists and command line arg was not given:
if (NOT (DEFINED ENABLE_VTK) AND (NOT (x$ENV{ENABLE_VTK} STREQUAL x)))
  set(ENABLE_VTK $ENV{ENABLE_VTK})
else()
  option(ENABLE_VTK "Enable VTK input/output." ON)
endif()
if (ENABLE_VTK)
  FIND_PACKAGE(VTK NO_MODULE)
  if(VTK_FOUND)
    message(STATUS "Found VTK: ${VTK_DIR} (found version \"${VTK_VERSION}\")")
    add_definitions(-DHAVE_VTK)
    include(${VTK_USE_FILE})
    set (PRAGMATIC_LIBRARIES ${VTK_LIBRARIES} ${PRAGMATIC_LIBRARIES})
  endif()
endif()
if (NOT ENABLE_VTK OR NOT VTK_FOUND)
  message(STATUS "Configured without VTK support.")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-literal-suffix -Wno-deprecated")

FIND_PACKAGE(MPI)
if(MPI_FOUND)
  add_definitions(-DHAVE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})

  set (CMAKE_CXX_FLAGS ${MPI_CXX_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS})
  set (CMAKE_C_FLAGS ${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS})
  set (CMAKE_LINK_FLAGS ${MPI_C_LINK_FLAGS} ${CMAKE_LINK_FLAGS})
  set (PRAGMATIC_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${PRAGMATIC_LIBRARIES})
  set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(-DHAVE_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

FIND_PACKAGE(Metis REQUIRED)
if(METIS_FOUND)
  include_directories(${METIS_INCLUDE_DIR})
  set (PRAGMATIC_LIBRARIES ${METIS_LIBRARIES} ${PRAGMATIC_LIBRARIES})
endif()

include_directories(include)

# ADD_EXECUTABLE( ${PROJECT_NAME} main.cpp )
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${PRAgMaTIc_LIBS} )
# ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND cp ${PROJECT_NAME} ${PROJECT_NAME}.o )
file(GLOB C_SOURCES src/*.c)
file(GLOB CXX_SOURCES src/*.cpp)

# target_link_library
add_library(pragmatic SHARED ${C_SOURCES} ${CXX_SOURCES})
target_link_libraries(pragmatic ${PRAGMATIC_LIBRARIES})

add_subdirectory(tests EXCLUDE_FROM_ALL)

install(DIRECTORY include/ DESTINATION include/pragmatic FILES_MATCHING PATTERN *.h)
install(FILES libpragmatic.so DESTINATION lib)

ADD_EXECUTABLE(coarsen_mesh_3d ./tools/coarsen_mesh_3d.cpp ./src/generate_Steiner_ellipse_3d.cpp  ./src/mpi_tools.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(coarsen_mesh_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(jpeg2mesh ./tools/jpeg2mesh ./src/mpi_tools.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(jpeg2mesh ${PRAGMATIC_LIBRARIES})

file(COPY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_BINARY_DIR}/)
