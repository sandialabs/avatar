# Support for TriBITS build as part of Trilinos
macro(include_tribits_build)
  if (${PROJECT_NAME}_TRIBITS_DIR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.tribits.cmake")
    return()
  endif()
endmacro()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/util/fclib-1.6.1)
  set(HAVE_FCLIB ON)
  set(FCLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/util/fclib-1.6.1/modules)
  set(FC_LIBRARIES fclib)
  add_definitions(-DHAVE_AVATAR_FCLIB)
else()
  set(HAVE_FCLIB OFF)
endif()

IF (${PROJECT_NAME}_TRIBITS_DIR)
  # Configure Avatar with TriBITS
  TRIBITS_PACKAGE(AvatarT DISABLE_CIRCULAR_REF_DETECTION_FAILURE)

  add_subdirectory(src)
  add_subdirectory(tools)
#  add_subdirectory(man)
  add_subdirectory(python)

  TRIBITS_ADD_TEST_DIRECTORIES(unittest)    

  TRIBITS_EXCLUDE_AUTOTOOLS_FILES()
  TRIBITS_PACKAGE_POSTPROCESS()

else()
  # To support standalone build
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)

  # Configure standalone Avatar 
  set(PROJECT_NAME Avatar)
  project(${PROJECT_NAME})
  if(NOT MSVC)
    add_compile_options("-Wno-format-security")
    link_libraries("-lm -lpthread")
  endif()

  add_subdirectory(man)
  add_subdirectory(python)
  add_subdirectory(src)
  add_subdirectory(tools)
  add_subdirectory(unittest)
  if(HAVE_FCLIB)
    add_subdirectory(util)
  endif()
  
  mark_as_advanced(FORCE CMAKE_BACKWARDS_COMPATIBILITY)
  mark_as_advanced(FORCE CMAKE_OSX_ARCHITECTURES)
  mark_as_advanced(FORCE CMAKE_OSX_DEPLOYMENT_TARGET)
  mark_as_advanced(FORCE CMAKE_OSX_SYSROOT)
  mark_as_advanced(FORCE EXECUTABLE_OUTPUT_PATH)
  mark_as_advanced(FORCE LIBRARY_OUTPUT_PATH)  
endif()


