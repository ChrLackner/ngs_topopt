
include(CTest)

add_test(NAME pytest COMMAND pytest . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(pytest COMMAND pytest . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
