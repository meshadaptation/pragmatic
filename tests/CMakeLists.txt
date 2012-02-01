include_directories(../include)

ADD_EXECUTABLE(test_adapt_2d src/test_adapt_2d.cpp)
TARGET_LINK_LIBRARIES(test_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_adapt_3d src/test_adapt_3d.cpp)
TARGET_LINK_LIBRARIES(test_adapt_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_chaste_mesh src/test_chaste_mesh.cpp)
TARGET_LINK_LIBRARIES(test_chaste_mesh pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_coarsen_2d src/test_coarsen_2d.cpp)
TARGET_LINK_LIBRARIES(test_coarsen_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_coarsen_3d src/test_coarsen_3d.cpp)
TARGET_LINK_LIBRARIES(test_coarsen_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_colour_3d src/test_colour_3d.cpp)
TARGET_LINK_LIBRARIES(test_colour_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_colour src/test_colour.cpp)
TARGET_LINK_LIBRARIES(test_colour pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_eigen src/test_eigen.cpp)
TARGET_LINK_LIBRARIES(test_eigen pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_gradation_2d src/test_gradation_2d.cpp)
TARGET_LINK_LIBRARIES(test_gradation_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_hessian_2d src/test_hessian_2d.cpp)
TARGET_LINK_LIBRARIES(test_hessian_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_hessian_3d src/test_hessian_3d.cpp)
TARGET_LINK_LIBRARIES(test_hessian_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_adapt_3d src/test_mpi_adapt_3d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_adapt_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_coarsen_2d src/test_mpi_coarsen_2d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_coarsen_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_coarsen_3d src/test_mpi_coarsen_3d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_coarsen_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_refine_2d src/test_mpi_refine_2d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_refine_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_refine_3d src/test_mpi_refine_3d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_refine_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_mpi_smooth_3d src/test_mpi_smooth_3d.cpp)
TARGET_LINK_LIBRARIES(test_mpi_smooth_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_page_faults src/test_page_faults.cpp)
TARGET_LINK_LIBRARIES(test_page_faults pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_refine_2d src/test_refine_2d.cpp)
TARGET_LINK_LIBRARIES(test_refine_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_refine_3d src/test_refine_3d.cpp)
TARGET_LINK_LIBRARIES(test_refine_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_smart_Laplacian_3d src/test_smart_Laplacian_3d.cpp)
TARGET_LINK_LIBRARIES(test_smart_Laplacian_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_smooth_2d src/test_smooth_2d.cpp)
TARGET_LINK_LIBRARIES(test_smooth_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_surface_2d src/test_surface_2d.cpp)
TARGET_LINK_LIBRARIES(test_surface_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(test_surface_3d src/test_surface_3d.cpp)
TARGET_LINK_LIBRARIES(test_surface_3d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(ftest_adapt_2d src/ftest_adapt_2d.F90)
TARGET_LINK_LIBRARIES(ftest_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(ftest_adapt_3d src/ftest_adapt_3d.F90)
TARGET_LINK_LIBRARIES(ftest_adapt_3d pragmatic ${PRAGMATIC_LIBRARIES})

