IF(NOT DEFINED Check_INCLUDE_DIRS OR NOT DEFINED Check_LIBRARY_DIRS)
  MESSAGE(WARNING "Avatar's unit tests depend on the Check library.  If you want to build and run Avatar's unit tests, then you must set the CMake options Check_INCLUDE_DIRS and Check_LIBRARY_DIRS to the installation path of the Check library.  You must configure, build, and install the Check library yourself.  We have only tested Check version 0.9.0.")
ELSEIF(NOT HAVE_FCLIB)
  MESSAGE(WARNING "Avatar's unit tests require fclib.  If you want to build and run Avatar's unit tests, then you must install the fclib 1.6.1 source tree in avatar/util.")
ELSE()

  INCLUDE_DIRECTORIES(${FCLIB_DIR})
  INCLUDE_DIRECTORIES(${Check_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(".")

  SET(Check_SRCS
    checkall.c
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

  TRIBITS_COPY_FILES_TO_BINARY_DIR(AvatarUnitTestData_cp
    DEST_FILES 
      api_testing.data   diversity_test_kappa.truth  diversity_test.trees  proximity_test.trees  smote_5d.data   smote_6d.names
      api_testing.names  diversity_test.names        proximity_test.names  smote_4d.data         smote_5d.names
      api_testing.trees  diversity_test.test         proximity_test.test   smote_4d.names        smote_6d.data
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/data
  )

  TRIBITS_ADD_EXECUTABLE(
    UnitTests
    SOURCES ${Check_SRCS} ${TOOL_SRCS}
    COMM serial mpi
  )

  TRIBITS_ADD_TEST(
    UnitTests
    NAME UnitTests
    COMM serial mpi
    NUM_MPI_PROCS 1
  )

ENDIF()
