add_custom_target(test
make
COMMAND cp src/*.mpi bin/
COMMAND python ./unittest bin)

include_directories(../include ./include)
file(GLOB lib_src "../src/*.cpp")
file(GLOB src_lite "../src/*.cpp")
FOREACH(item ${src_lite})
  IF(${item} MATCHES "cpragmatic.cpp")
    LIST(REMOVE_ITEM src_lite ${item})
  ENDIF(${item} MATCHES "cpragmatic.cpp")
ENDFOREACH(item)
list(APPEND src_lite "./src/ticker.cpp")
MESSAGE(STATUS "str_cite ${src_lite}")

ADD_EXECUTABLE(./bin/test_generate_Steiner_ellipse_3d ./src/test_generate_Steiner_ellipse_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_generate_Steiner_ellipse_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_refine_3d ./src/test_refine_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_refine_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_ElementProperty ./src/test_ElementProperty.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_ElementProperty ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_swap_2d ./src/test_swap_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_swap_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_3d ./src/test_mpi_coarsen_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_3d ./src/test_coarsen_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_coarsen_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_refine_2d ./src/test_refine_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_refine_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_eigen ./src/test_eigen.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_eigen ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_3d ./src/test_hessian_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_hessian_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_3d ./src/test_mpi_refine_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_2d ./src/test_coarsen_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_coarsen_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_2d ./src/test_adapt_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_adapt_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_2d ./src/test_mpi_refine_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_2d ./src/test_hessian_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_hessian_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_2d ./src/test_smooth_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_smooth_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_adapt_3d ./src/test_mpi_adapt_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_mpi_adapt_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_2d ./src/test_mpi_coarsen_2d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_3d ./src/test_smooth_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_smooth_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_3d ./src/test_adapt_3d.cpp ${src_lite})
TARGET_LINK_LIBRARIES(./bin/test_adapt_3d ${PRAGMATIC_LIBRARIES})

