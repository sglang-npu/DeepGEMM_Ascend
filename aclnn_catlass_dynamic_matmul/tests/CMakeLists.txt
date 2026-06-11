cmake_minimum_required(VERSION 3.10)
project(CatlassTilingTests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../op_host/op_tiling
    ${PROJECT_SOURCE_DIR}/../../op_host
)

add_executable(utils_test utils_test.cpp)
target_link_libraries(utils_test GTest::gtest GTest::gtest_main)

add_executable(do_tiling_test do_tiling_test.cpp)
target_link_libraries(do_tiling_test GTest::gtest GTest::gtest_main)

add_executable(select_kernel_test select_kernel_test.cpp)
target_link_libraries(select_kernel_test GTest::gtest GTest::gtest_main)

add_executable(cache_test cache_test.cpp)
target_link_libraries(cache_test GTest::gtest GTest::gtest_main)

add_executable(csv_test csv_test.cpp)
target_link_libraries(csv_test GTest::gtest GTest::gtest_main)

add_executable(predictor_test predictor_test.cpp)
target_link_libraries(predictor_test GTest::gtest GTest::gtest_main)

enable_testing()

add_test(NAME UtilsTests COMMAND utils_test)
add_test(NAME DoTilingTests COMMAND do_tiling_test)
add_test(NAME SelectKernelTests COMMAND select_kernel_test)
add_test(NAME CacheTests COMMAND cache_test)
add_test(NAME CsvTests COMMAND csv_test)
add_test(NAME PredictorTests COMMAND predictor_test)