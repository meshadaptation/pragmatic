set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin)
set(PRAGMATIC_TEST_SRC ${CMAKE_SOURCE_DIR}/tests/src)

if (NOT(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR}))
  add_custom_target(test
  make
  COMMAND mkdir -p data
  COMMAND cp ${CMAKE_SOURCE_DIR}/tests/data/* data/
  COMMAND cp ${CMAKE_SOURCE_DIR}/tests/src/*.mpi bin/
  COMMAND python ${CMAKE_SOURCE_DIR}/tests/unittest bin)
else()
  add_custom_target(test
  make
  COMMAND cp -u ${CMAKE_SOURCE_DIR}/tests/src/*.mpi bin/
  COMMAND python ${CMAKE_SOURCE_DIR}/tests/unittest bin)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/include)
file(GLOB lib_src "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB src_lite "${CMAKE_SOURCE_DIR}/src/*.cpp")
FOREACH(item ${src_lite})
  IF(${item} MATCHES "cpragmatic.cpp")
    LIST(REMOVE_ITEM src_lite ${item})
  ENDIF(${item} MATCHES "cpragmatic.cpp")
ENDFOREACH(item)
MESSAGE(STATUS "str_cite ${src_lite}")

# tests without VTK:
ADD_EXECUTABLE(test_ElementProperty ${PRAGMATIC_TEST_SRC}/test_ElementProperty.cpp ${src_lite})
TARGET_LINK_LIBRARIES(test_ElementProperty ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_generate_Steiner_ellipse_3d ${PRAGMATIC_TEST_SRC}/test_generate_Steiner_ellipse_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(test_generate_Steiner_ellipse_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_eigen ${PRAGMATIC_TEST_SRC}/test_eigen.cpp ${src_lite})
TARGET_LINK_LIBRARIES(test_eigen ${PRAGMATIC_LIBRARIES})

if (ENABLE_LIBMESHB)
  ADD_EXECUTABLE(test_gmf ${PRAGMATIC_TEST_SRC}/test_gmf.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_gmf ${PRAGMATIC_LIBRARIES} ${LIBRT_LIBRARIES})
endif()

# tests with VTK:
if (ENABLE_VTK)

  ADD_EXECUTABLE(test_adapt_2d ${PRAGMATIC_TEST_SRC}/test_adapt_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_adapt_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_adapt_3d ${PRAGMATIC_TEST_SRC}/test_adapt_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_adapt_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_coarsen_2d ${PRAGMATIC_TEST_SRC}/test_coarsen_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_coarsen_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_coarsen_3d ${PRAGMATIC_TEST_SRC}/test_coarsen_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_coarsen_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_gradation_3d ${PRAGMATIC_TEST_SRC}/test_gradation_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_gradation_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_coarsen_boundary_2d ${PRAGMATIC_TEST_SRC}/test_coarsen_boundary_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_coarsen_boundary_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_coarsen_boundary_3d ${PRAGMATIC_TEST_SRC}/test_coarsen_boundary_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_coarsen_boundary_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_refine_2d ${PRAGMATIC_TEST_SRC}/test_refine_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_refine_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_refine_3d ${PRAGMATIC_TEST_SRC}/test_refine_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_refine_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_swap_2d ${PRAGMATIC_TEST_SRC}/test_swap_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_swap_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_smooth_2d ${PRAGMATIC_TEST_SRC}/test_smooth_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_smooth_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_hessian_2d ${PRAGMATIC_TEST_SRC}/test_hessian_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_hessian_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_hessian_3d ${PRAGMATIC_TEST_SRC}/test_hessian_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_hessian_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(benchmark_adapt_2d ${PRAGMATIC_TEST_SRC}/benchmark_adapt_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(benchmark_adapt_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(benchmark_adapt_3d ${PRAGMATIC_TEST_SRC}/benchmark_adapt_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(benchmark_adapt_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_mpi_coarsen_2d ${PRAGMATIC_TEST_SRC}/test_mpi_coarsen_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_mpi_coarsen_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_mpi_coarsen_3d ${PRAGMATIC_TEST_SRC}/test_mpi_coarsen_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_mpi_coarsen_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_mpi_refine_2d ${PRAGMATIC_TEST_SRC}/test_mpi_refine_2d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_mpi_refine_2d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_mpi_refine_3d ${PRAGMATIC_TEST_SRC}/test_mpi_refine_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_mpi_refine_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_smooth_3d ${PRAGMATIC_TEST_SRC}/test_smooth_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_smooth_3d ${PRAGMATIC_LIBRARIES})

  ADD_EXECUTABLE(test_mpi_adapt_3d ${PRAGMATIC_TEST_SRC}/test_mpi_adapt_3d.cpp ${src_lite})
  TARGET_LINK_LIBRARIES(test_mpi_adapt_3d ${PRAGMATIC_LIBRARIES})

endif()
