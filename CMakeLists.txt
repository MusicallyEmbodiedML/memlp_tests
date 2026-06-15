# ============================================================================
#  memlp_tests — unit tests for the MEMLP library
#
#  Two build modes from one CMakeLists:
#    * Host (default):   command-line test runner, integrates with ctest.
#    * Pico (RP2040/RP2350): firmware that streams results over USB serial.
#
#  Host:
#    cmake -S . -B build && cmake --build build && ./build/memlp_tests
#
#  Pico RP2040:
#    cmake -S . -B build-rp2040 -DMEMLP_TESTS_PICO=ON -DPICO_BOARD=pico
#    cmake --build build-rp2040
#
#  Pico RP2350:
#    cmake -S . -B build-rp2350 -DMEMLP_TESTS_PICO=ON -DPICO_BOARD=pico2
#    cmake --build build-rp2350
# ============================================================================

cmake_minimum_required(VERSION 3.13)

option(MEMLP_TESTS_PICO "Build tests for RP2040/RP2350 using the Raspberry Pi Pico SDK" OFF)

set(MEMLP_DIR ${CMAKE_CURRENT_LIST_DIR}/memlp)

# Shared test sources (self-registering via the TEST() macro).
set(MEMLP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_loss.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_layer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_mlp.cpp
)

# Library sources: everything else is header-only; MLP.cpp holds the
# explicit MLP<float> instantiation, Utils.cpp the (platform) helpers.
set(MEMLP_LIB_SOURCES
    ${MEMLP_DIR}/MLP.cpp
    ${MEMLP_DIR}/Utils.cpp
)

if(NOT EXISTS ${MEMLP_DIR}/MLP.cpp)
    message(FATAL_ERROR
        "memlp submodule not found. Run:\n"
        "  git submodule update --init --recursive")
endif()

# Layer.h uses std::span -> C++20 is required on every target.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MEMLP_TESTS_PICO)
    # ---- Embedded build (RP2040 / RP2350) ----------------------------------
    set(CMAKE_C_STANDARD 11)
    set(PICO_BOARD pico CACHE STRING "Target board: 'pico' (RP2040) or 'pico2' (RP2350)")

    include(pico_sdk_import.cmake)
    project(memlp_tests C CXX ASM)
    pico_sdk_init()

    add_executable(memlp_tests
        ${CMAKE_CURRENT_LIST_DIR}/pico_main.cpp
        ${MEMLP_TEST_SOURCES}
        ${MEMLP_LIB_SOURCES}
    )
    target_include_directories(memlp_tests PRIVATE
        ${MEMLP_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/tests
    )
    pico_set_program_name(memlp_tests "memlp_tests")
    pico_enable_stdio_usb(memlp_tests 1)
    pico_enable_stdio_uart(memlp_tests 0)
    target_link_libraries(memlp_tests pico_stdlib pico_rand)
    pico_add_extra_outputs(memlp_tests)   # .uf2, .elf, .bin, map, dis

    message(STATUS "memlp_tests: Pico build for board '${PICO_BOARD}'")
else()
    # ---- Host build --------------------------------------------------------
    project(memlp_tests C CXX)
    enable_testing()

    add_executable(memlp_tests
        ${CMAKE_CURRENT_LIST_DIR}/host_main.cpp
        ${MEMLP_TEST_SOURCES}
        ${MEMLP_LIB_SOURCES}
    )
    target_include_directories(memlp_tests PRIVATE
        ${MEMLP_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/tests
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(memlp_tests PRIVATE -Wall -Wextra -g -O0)
    endif()

    add_test(NAME memlp_tests COMMAND memlp_tests)
    message(STATUS "memlp_tests: host build")
endif()
