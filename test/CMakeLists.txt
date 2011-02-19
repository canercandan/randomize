###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

INCLUDE(${CMAKE_MODULE_PATH}/UseCUDA.cmake)

SET(SOURCES
  t-docexample
  )

FOREACH(current ${SOURCES})
  CUDA_ADD_EXECUTABLE(${current} ${current}.cu)
  ADD_TEST(${current} ${current})
  CUDA_ADD_CURAND_TO_TARGET(${current})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

###############################################################################
