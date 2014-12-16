add_custom_target(test
make
COMMAND cp src/*.mpi bin/
COMMAND python ./unittest bin)

include_directories(../include ./include)

ADD_EXECUTABLE(./bin/test_refine_3d ./src/test_refine_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_refine_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_ElementProperty ./src/test_ElementProperty.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_ElementProperty ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_swap_2d ./src/test_swap_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_swap_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_3d ./src/test_mpi_coarsen_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_3d ./src/test_coarsen_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_coarsen_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_refine_2d ./src/test_refine_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_refine_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_eigen ./src/test_eigen.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_eigen ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_3d ./src/test_hessian_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_hessian_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_3d ./src/test_mpi_refine_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_coarsen_2d ./src/test_coarsen_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_coarsen_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_2d ./src/test_adapt_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_adapt_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_refine_2d ./src/test_mpi_refine_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_refine_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_hessian_2d ./src/test_hessian_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_hessian_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_2d ./src/test_smooth_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_smooth_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_adapt_3d ./src/test_mpi_adapt_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_adapt_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_mpi_coarsen_2d ./src/test_mpi_coarsen_2d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_mpi_coarsen_2d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_smooth_3d ./src/test_smooth_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_smooth_3d ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(./bin/test_adapt_3d ./src/test_adapt_3d.cpp ./src/ticker.cpp)
TARGET_LINK_LIBRARIES(./bin/test_adapt_3d ${PRAGMATIC_LIBRARIES})

