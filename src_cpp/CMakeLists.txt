cmake_minimum_required(VERSION 3.16)
project(fourws_vehicle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
add_library(vehicle STATIC
    src/vehicle.cpp
)

target_include_directories(vehicle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dynamics STATIC
    src/dynamics.cpp
)
target_include_directories(dynamics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(planner STATIC
    src/planner.cpp
)
target_include_directories(planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dof_utils STATIC
    src/dof_utils.cpp
)
target_include_directories(dof_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dynamics2d STATIC
    src/dynamics2d.cpp
)
target_include_directories(dynamics2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(mpc STATIC
    src/mpc.cpp
)
target_include_directories(mpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mpc PUBLIC Eigen3::Eigen)

add_library(mppi STATIC
    src/mppi.cpp
)
target_include_directories(mppi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mppi PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(vehicle_demo test/vehicle_test.cpp)
target_link_libraries(vehicle_demo PRIVATE vehicle dynamics dynamics2d planner mpc mppi)

add_executable(dynamics_test test/dynamics_test.cpp)
target_link_libraries(dynamics_test PRIVATE dynamics dynamics2d vehicle planner mpc mppi)

add_executable(planner_test test/planner_test.cpp)
target_link_libraries(planner_test PRIVATE planner)

add_executable(dof_utils_test test/dof_utils_test.cpp)
target_link_libraries(dof_utils_test PRIVATE dof_utils)

add_executable(mpc_test test/mpc_test.cpp)
target_link_libraries(mpc_test PRIVATE mpc dynamics2d dynamics vehicle)

add_executable(mppi_test ${CMAKE_CURRENT_SOURCE_DIR}/test/mppi_test.cpp)
target_link_libraries(mppi_test PRIVATE mppi Eigen3::Eigen OpenMP::OpenMP_CXX)
