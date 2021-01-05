if(${HAVE_FCLIB})
  include_directories(${FCLIB_DIR})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(HEADERS "")
SET(SOURCES "")

APPEND_SET(HEADERS
  avatar_api.h
)

APPEND_SET(SOURCES
  avatar_api.c
  array.c
  avatar_dt.c
  att_noising.c
  attr_stats.c
  bagging.c
  balanced_learning.c
  boost.c
  crossval_util.c
  distinct_values.c
  evaluate.c
  gain.c
  heartbeat.c
  ivote.c
  knn.c
  memory.c
  missing_values.c
  options.c
  read_line_from_string_or_file.c
  rw_data.c
  safe_memory.c
  schema.c
  skew.c
  smote.c
  subspaces.c
  tree.c
  util.c
  version_info.c
  av_rng.c
  av_stats.c
  av_utils.c
)

TRIBITS_ADD_LIBRARY(
  avatar
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
)

tribits_add_executable(avatardt SOURCES dt.c INSTALLABLE)
tribits_add_executable(crossvalfc SOURCES crossval.c INSTALLABLE)
tribits_add_executable(rfFeatureValue SOURCES rffv.c INSTALLABLE)
