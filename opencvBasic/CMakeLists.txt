cmake_minimum_required(VERSION 2.8)
project(basic)

###############################################
#
# run following command on windows 
# cmake .. -G "Visual Studio 14 2015 Win64"
#
###############################################


message ("-- CAMKE_GENERATOR: "  ${CMAKE_GENERATOR})
message ("-- CMAKE_CXX_FLAGS: "  ${CMAKE_CXX_FLAGS})
message (STATUS "CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS})
message (STATUS "CMAKE_STATIC_LINKER_FLAGS:"  ${CMAKE_STATIC_LINKER_FLAGS})
message (STATUS "CMAKE_EXE_LINKER_FLAGS: "    ${CMAKE_EXE_LINKER_FLAGS})

set(CMAKE_BUILD_TYPE Debug)
message("-- Set CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
	
if (WIN32)
	#set(OpenCV_STATIC true)
	message ("-- Win32 Platform ")
	
	set (OpenCV_DIR "c:/Users/ding/Documents/opencv/build/")
	message ("-- Set  OpenCV_DIR: ${OpenCV_DIR}")

	
	add_definitions("-D_SCL_SECURE_NO_WARNINGS")
	
endif()

set (OpenCV_FIND_QUIETLY False)
message ("-- set  OpenCV_FIND_QUIETLY: "  ${OpenCV_FIND_QUIETLY})
find_package(OpenCV REQUIRED)
message ("-- Opencv OpenCV_STATIC: " ${OpenCV_STATIC})
message ("-- Opencv OpenCV_SHARED: " ${OpenCV_SHARED})

message ("-- OpenCV_INCLUDE_DIRS: ")
foreach(inc ${OpenCV_INCLUDE_DIRS})
message ("   ${inc}")        # Prints each item on a separate line
endforeach()

message ("-- OpenCV_LIBRARIES: ")
foreach(item ${OpenCV_LIBRARIES})
#message ("   ${item}")        # Prints each item on a separate line
endforeach()

message ("-- OpenCV_LIBS: ")
foreach(item ${OpenCV_LIBS})
message ("   ${item}")        # Prints each item on a separate line
endforeach()

#https://cmake.org/cmake/help/v2.8.8/cmake.html#prop_dir:INCLUDE_DIRECTORIES
include_directories(${OpenCV_INCLUDE_DIRS} )
get_property (inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message ("-- INCLUDE_DIRECTORIES: ")
foreach(item ${inc_dirs})
message ("   ${item}")        # Prints each item on a separate line
endforeach()

if(UNIX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  message ("-- Enable c++11 on Unix like platform.")
endif(UNIX)

add_executable(basic src/basic.cpp)
target_link_libraries(basic ${OpenCV_LIBRARIES})
