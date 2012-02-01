include_directories(../include)

ADD_EXECUTABLE(test_adapt_2d src/test_adapt_2d.cpp)
message(PRAGMATIC_LIBRARIES ${PRAGMATIC_LIBRARIES})
TARGET_LINK_LIBRARIES(test_adapt_2d pragmatic ${PRAGMATIC_LIBRARIES})

