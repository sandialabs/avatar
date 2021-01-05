include_tribits_build()

IF(NOT DEFINED Check_INSTALL_DIR)
  MESSAGE(WARNING "Avatar's unit tests depend on the Check library.  If you want to build and run Avatar's unit tests, then you must set the CMake option Check_INSTALL_DIR to the installation directory of the Check library.  You must configure, build, and install the Check library yourself.  We have only tested Check version 0.9.0.")
ELSE()

  SET(Check_INCLUDE_DIR "${Check_INSTALL_DIR}/include")
  SET(Check_LIBRARY_DIR "${Check_INSTALL_DIR}/lib")

  FIND_LIBRARY(Check_LIBRARY
    NAMES check compat
    PATHS ${Check_LIBRARY_DIR}
    NO_DEFAULT_PATH
    )

  MACRO(APPEND_SET VARNAME)
    LIST(APPEND ${VARNAME} ${ARGN})
  ENDMACRO()

  MACRO(APPEND_GLOB VAR)
    FILE(GLOB LOCAL_TMP_VAR ${ARGN})
    APPEND_SET(${VAR} ${LOCAL_TMP_VAR})
  ENDMACRO()

  INCLUDE_DIRECTORIES(${Check_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(".")

  SET(Check_SRCS
    checkapi.c
    checkarray.c
    checkbagging.c
    checkblearning.c
    checkboost.c
    checkcrossval_util.c
    checkdistinct.c
    checkdiversity.c
    checkeval.c
    checkgain.c
    checkivote.c
    checkknn.c
    checkmajbagging.c
    checkmissing.c
    checkproximity.c
    checkrwdata.c
    checksmote.c
    checktree.c
    checkutil.c
    util.c
    )

  SET(TOOL_SRCS
    ../src/avatar_api.c
    ../src/array.c
    ../src/att_noising.c
    ../src/bagging.c
    ../src/balanced_learning.c
    ../src/boost.c
    ../src/crossval_util.c
    ../src/distinct_values.c
    ../src/evaluate.c
    ../src/gain.c
    ../src/heartbeat.c
    ../src/ivote.c
    ../src/knn.c
    ../src/memory.c
    ../src/missing_values.c
    ../src/options.c
    ../src/rw_data.c
    ../src/safe_memory.c
    ../src/schema.c
    ../src/skew.c
    ../src/smote.c
    ../src/subspaces.c
    ../src/tree.c
    ../src/util.c
    ../src/version_info.c
    ../tools/diversity_measures.c
    ../tools/proximity_utils.c
    )

  ADD_EXECUTABLE(checkall checkall.c ${Check_SRCS} ${TOOL_SRCS})

  TARGET_LINK_LIBRARIES(checkall avatar ${Check_LIBRARY})

  INSTALL(TARGETS checkall
    RUNTIME DESTINATION bin
    )

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ENDIF()
