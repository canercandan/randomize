###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################

#LINK_LIBRARIES(${PROJECT_NAME} ${PROJECT_NAME}_kernel)

SET(SOURCES
  t-docexample
  t-sampler
  )

FOREACH(current ${SOURCES})
  CUDA_ADD_EXECUTABLE(${current} ${current}.cu)
  ADD_TEST(${current} ${current})
  CUDA_ADD_CURAND_TO_TARGET(${current})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/${PROJECT_NAME}/test COMPONENT test)
ENDFOREACH()

###############################################################################
