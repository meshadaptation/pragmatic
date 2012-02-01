include_directories(../include)

ADD_EXECUTABLE(test_adapt_2d src/test_adapt_2d.cpp)
TARGET_LINK_LIBRARIES(test_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})

ADD_EXECUTABLE(ftest_adapt_2d src/ftest_adapt_2d.F90)
TARGET_LINK_LIBRARIES(ftest_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})
# SET_TARGET_PROPERTIES(ftest_adapt_2d PROPERTIES LINKER_LANGUAGE Fortran)

