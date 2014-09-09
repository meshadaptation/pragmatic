
include_directories(../include)

ADD_EXECUTABLE(./bin/test_adapt_2d ./src/test_adapt_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_2d_cuda ./src/test_adapt_2d_cuda.cpp)
TARGET_LINK_LIBRARIES(./bin/test_adapt_2d_cuda pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_3d ./src/test_adapt_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_adapt_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_2d ./src/test_coarsen_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_coarsen_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_3d ./src/test_coarsen_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_coarsen_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_dolfin_2d ./src/test_dolfin_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_dolfin_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_eigen ./src/test_eigen.cpp)
TARGET_LINK_LIBRARIES(./bin/test_eigen pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_ElementProperty ./src/test_ElementProperty.cpp)
TARGET_LINK_LIBRARIES(./bin/test_ElementProperty pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_gradation_2d ./src/test_gradation_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_gradation_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_2d ./src/test_hessian_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_hessian_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_3d ./src/test_hessian_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_hessian_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_adapt_3d ./src/test_mpi_adapt_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_adapt_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_2d ./src/test_mpi_coarsen_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_3d ./src/test_mpi_coarsen_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_2d ./src/test_mpi_refine_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_3d ./src/test_mpi_refine_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_page_faults ./src/test_page_faults.cpp)
TARGET_LINK_LIBRARIES(./bin/test_page_faults pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_refine_2d ./src/test_refine_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_refine_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_refine_3d ./src/test_refine_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_refine_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_2d ./src/test_smooth_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_smooth_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_3d ./src/test_smooth_3d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_smooth_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_swap_2d ./src/test_swap_2d.cpp)
TARGET_LINK_LIBRARIES(./bin/test_swap_2d pragmatic ${PRAGMATIC_LIBRARIES})

