include_tribits_build()

if(${HAVE_FCLIB})
  include_directories(${FCLIB_DIR})
endif()

add_library(avatar STATIC
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
  reset.c
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
  av_utils.c)

add_executable(avatardt dt.c)
target_link_libraries(avatardt avatar ${FC_LIBRARIES})

add_executable(crossvalfc crossval.c)
target_link_libraries(crossvalfc avatar ${FC_LIBRARIES})

add_executable(rfFeatureValue rffv.c)
target_link_libraries(rfFeatureValue avatar ${FC_LIBRARIES})

# TODO: Add support for avatarmpi, crossvalmpi

install(TARGETS avatardt crossvalfc rfFeatureValue
  RUNTIME DESTINATION bin
  )
