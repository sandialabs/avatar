include_tribits_build()

include_directories(${CMAKE_SOURCE_DIR}/util/fclib-1.6.1/modules)

add_executable(diversity diversity.c diversity_measures.c)
target_link_libraries(diversity avatar ${FC_LIBRARIES})

add_executable(proximity proximity.c proximity_utils.c)
target_link_libraries(proximity avatar ${FC_LIBRARIES})

add_executable(remoteness remoteness.c proximity_utils.c)
target_link_libraries(remoteness avatar ${FC_LIBRARIES})

add_executable(tree_stats tree_stats.c)
target_link_libraries(tree_stats avatar ${FC_LIBRARIES})

install(PROGRAMS data_inspector extract-class-stats tree2dot DESTINATION bin)

install(TARGETS diversity proximity remoteness tree_stats
  RUNTIME DESTINATION bin
  )

