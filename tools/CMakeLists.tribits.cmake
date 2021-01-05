include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util/fclib-1.6.1/modules)

tribits_add_executable(diversity SOURCES diversity.c diversity_measures.c INSTALLABLE)

tribits_add_executable(proximity SOURCES proximity.c proximity_utils.c INSTALLABLE)

tribits_add_executable(remoteness SOURCES remoteness.c proximity_utils.c INSTALLABLE)

tribits_add_executable(tree_stats SOURCES tree_stats.c INSTALLABLE)

install(PROGRAMS data_inspector extract-class-stats tree2dot DESTINATION bin)


