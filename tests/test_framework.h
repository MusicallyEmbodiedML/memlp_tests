/**
 * @file test_framework.h
 * @brief Tiny zero-dependency unit-test framework for the MEMLP library.
 *
 * Designed to run unchanged on a host PC and on bare-metal targets (RP2040 /
 * RP2350 via the Pico SDK). No exceptions, no RTTI, no heap-heavy machinery —
 * just enough to pin the current behaviour of the public API before refactoring.
 *
 * Tests register themselves with the TEST(suite, name) macro and are executed
 * by run_all_tests(), which is called from host_main.cpp / pico_main.cpp.
 *
 * Example:
 *   TEST(loss, mse_zero_when_equal) {
 *       std::vector<float> e{1.f}, a{1.f}, d(1);
 *       CHECK_CLOSE(loss::MSE(e, a, d, 1.f), 0.f, 1e-6);
 *   }
 *
 * Assertion macros:
 *   CHECK(cond)            - boolean assertion
 *   CHECK_FALSE(cond)
 *   CHECK_EQ(a, b)         - == comparison
 *   CHECK_CLOSE(a, b, tol) - |a-b| <= tol  (floating point)
 *
 * A failing CHECK records the failure and lets the test continue, so a single
 * run reports every problem rather than stopping at the first.
 */

#ifndef MEMLP_TEST_FRAMEWORK_H
#define MEMLP_TEST_FRAMEWORK_H

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

namespace tf {

struct TestCase {
    const char * suite;
    const char * name;
    std::function<void()> fn;
};

// Function-local statics so everything lives in the header (no separate .cpp).
inline std::vector<TestCase> & registry() {
    static std::vector<TestCase> r;
    return r;
}
inline int & checks_run()    { static int c = 0; return c; }
inline int & checks_failed() { static int c = 0; return c; }
inline int & current_fails() { static int c = 0; return c; }

struct Registrar {
    Registrar(const char * suite, const char * name, std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

inline void report_fail(const char * file, int line, const char * msg) {
    ++checks_failed();
    ++current_fails();
    std::printf("    FAIL %s:%d: %s\n", file, line, msg);
}

/**
 * @brief Run every registered test, printing a per-test and final summary.
 * @return Number of tests that had at least one failed check (0 == all passed).
 */
inline int run_all_tests() {
    const int total = (int)registry().size();
    std::printf("==== MEMLP unit tests: running %d tests ====\n", total);

    int failed_tests = 0;
    for (auto & tc : registry()) {
        current_fails() = 0;
        std::printf("[ RUN  ] %s.%s\n", tc.suite, tc.name);
        tc.fn();
        if (current_fails() == 0) {
            std::printf("[  OK  ] %s.%s\n", tc.suite, tc.name);
        } else {
            ++failed_tests;
            std::printf("[ FAIL ] %s.%s (%d checks failed)\n",
                        tc.suite, tc.name, current_fails());
        }
    }

    std::printf("==== %d/%d tests passed | %d checks run | %d checks failed ====\n",
                total - failed_tests, total, checks_run(), checks_failed());
    return failed_tests;
}

} // namespace tf

#define TF_CONCAT_(a, b) a##b
#define TF_CONCAT(a, b)  TF_CONCAT_(a, b)

#define TEST(suite, name)                                                      \
    static void TF_CONCAT(test_, TF_CONCAT(suite, TF_CONCAT(_, name)))();      \
    static tf::Registrar TF_CONCAT(reg_, TF_CONCAT(suite, TF_CONCAT(_, name))) \
        (#suite, #name,                                                        \
         &TF_CONCAT(test_, TF_CONCAT(suite, TF_CONCAT(_, name))));             \
    static void TF_CONCAT(test_, TF_CONCAT(suite, TF_CONCAT(_, name)))()

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++tf::checks_run();                                                    \
        if (!(cond)) tf::report_fail(__FILE__, __LINE__, #cond);               \
    } while (0)

#define CHECK_FALSE(cond) CHECK(!(cond))

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        ++tf::checks_run();                                                    \
        if (!((a) == (b)))                                                     \
            tf::report_fail(__FILE__, __LINE__, #a " == " #b);                 \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                                 \
    do {                                                                       \
        ++tf::checks_run();                                                    \
        double _da = (double)(a), _db = (double)(b);                           \
        if (std::fabs(_da - _db) > (double)(tol)) {                            \
            char _buf[256];                                                    \
            std::snprintf(_buf, sizeof(_buf),                                  \
                          "%s (=%g) ~= %s (=%g) within %g",                    \
                          #a, _da, #b, _db, (double)(tol));                    \
            tf::report_fail(__FILE__, __LINE__, _buf);                         \
        }                                                                      \
    } while (0)

#endif // MEMLP_TEST_FRAMEWORK_H
