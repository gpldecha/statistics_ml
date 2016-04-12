cmake_minimum_required(VERSION 2.8.3)
project(statistics_ml)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED roscpp)


find_package(OpenCV)
find_package(Armadillo)
find_package(Boost REQUIRED COMPONENTS system)

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} mlpack
  CATKIN_DEPENDS roscpp
)


set(CMAKE_CXX_FLAGS "-O2 -O3 -std=c++11 -isystem -mtune=native -fPIC -Wall -Wno-unused-function -Wno-deprecated-declarations")        ## Optimize
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

include_directories(include ${catkin_INCLUDE_DIRS} ${INCLUDE_DIR})


set(INC  ${INCLUDE_DIR}/statistics/distributions/distributions.h
         ${INCLUDE_DIR}/statistics/distributions/gmm.h
         ${INCLUDE_DIR}/statistics/distributions/gaussian.h
         ${INCLUDE_DIR}/statistics/clustering.h
         ${INCLUDE_DIR}/statistics/meanshift.h
         ${INCLUDE_DIR}/statistics/information.h
         ${INCLUDE_DIR}/statistics/metric/distance_metric.hpp
         ${INCLUDE_DIR}/statistics/impl/distance_metric_impl.hpp
         ${INCLUDE_DIR}/statistics/initialise.h
         ${INCLUDE_DIR}/statistics/decision_functions.h
         ${INCLUDE_DIR}/algorithms/find_closest_points.h
         ${INCLUDE_DIR}/algorithms/convolution.h


)

set(SRC  ${SRC_DIR}/distributions.cpp
         ${SRC_DIR}/distributions/gmm.cpp
         ${SRC_DIR}/distributions/gaussian.cpp
         ${SRC_DIR}/meanshift.cpp
         ${SRC_DIR}/clustering.cpp
         ${SRC_DIR}/initialise.cpp
         ${SRC_DIR}/information.cpp

)

add_library(${PROJECT_NAME} SHARED ${SRC} ${INC})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} boost_filesystem boost_system mlpack ${OpenCV_LIBRARIES} ${ARMADILLO_LIBRARIES} stdc++)
                                
add_executable(gmm_test ${SRC_DIR}/test/gmm_test.cpp )
target_link_libraries(gmm_test ${PROJECT_NAME})

add_executable(test ${SRC_DIR}/test/test.cpp )
target_link_libraries(test ${PROJECT_NAME})


