cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(mmdet_deploy)

# handle debug vs release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Use C++11
set (CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  darknet_ros_msgs
  
)



set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED )
find_package(OpenCV REQUIRED )

find_package(PkgConfig REQUIRED)
# libyaml-cpp (required)
# find_package (yaml-cpp)
# if (yaml-cpp_FOUND)
#   message("YAML Libs: ${YAML_CPP_LIBRARIES}")
#   message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
# else()
#   message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
# endif (yaml-cpp_FOUND)


# add_message_files(
#   FILES
#     BoundingBox.msg
#     BoundingBoxes.msg
#     ObjectCount.msg
# )

# add_service_files(
#   FILES
#     CheckForObjects.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    sensor_msgs
    message_runtime
    std_msgs
)

## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS 
  include
 LIBRARIES 
 CATKIN_DEPENDS 
  cv_bridge 
  image_transport 
  message_generation 
  roscpp 
  rospy 
  sensor_msgs 
  std_msgs
 DEPENDS 
 Boost
)

###########
## Build ##
###########

# explicit include dirs
include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${TORCH_INCLUDE_DIRS}
)

#############

# if(CATKIN_ENABLE_TESTING)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#   ## Add gtest based cpp test target and link libraries!
# endif()

# set(EXECUTABLE_NAME mmdet_node)
# add_executable(${EXECUTABLE_NAME})
# target_sources(${EXECUTABLE_NAME} PRIVATE src/mmdet_node.cpp
#         src/mmdet_handler.cpp
#         include/mmdet_deploy/NetTorch.cpp
#         include/mmdet_deploy/utils.cpp
# )
# target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${Torch_INCLUDE_DIRS} )
# include_directories(${YAML_CPP_INCLUDE_DIR})

# target_link_libraries(${EXECUTABLE_NAME}  ${catkin_LIBRARIES})
# target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBRARIES} )
# target_link_libraries(${EXECUTABLE_NAME}  ${YAML_CPP_LIBRARIES})
# target_link_libraries(${EXECUTABLE_NAME}  ${BOOST_LIBRARIES})

# set_target_properties(${EXECUTABLE_NAME} PROPERTIES
#   CXX_STANDARD 14
#   CXX_STANDARD_REQUIRED YES
# )