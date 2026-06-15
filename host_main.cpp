/**
 * @file host_main.cpp
 * @brief Command-line entry point. Runs every registered test and returns a
 *        non-zero exit code if any test failed (so it works with ctest / CI).
 */

#include "test_framework.h"

int main() {
    return tf::run_all_tests() == 0 ? 0 : 1;
}
