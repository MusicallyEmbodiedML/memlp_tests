/**
 * @file test_utils.cpp
 * @brief Characterisation tests for Utils.h — activation functions, softmax,
 *        argmax and the random generators. These pin current numeric behaviour
 *        (note: ReLU here is a *leaky* ReLU with slope kReLUSlope = 0.01).
 */

#include "test_framework.h"
#include "Utils.h"

#include <vector>
#include <cstdlib>

TEST(utils, sigmoid) {
    CHECK_CLOSE(utils::sigmoid(0.0f), 0.5f, 1e-6);
    CHECK(utils::sigmoid(20.0f) > 0.99f);
    CHECK(utils::sigmoid(-20.0f) < 0.01f);
    // derivative at 0 is 0.25
    CHECK_CLOSE(utils::deriv_sigmoid(0.0f), 0.25f, 1e-6);
}

TEST(utils, tanh) {
    CHECK_CLOSE(utils::hyperbolic_tan(0.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hyperbolic_tan(100.0f), 1.0f, 1e-5);
    CHECK_CLOSE(utils::hyperbolic_tan(-100.0f), -1.0f, 1e-5);
    // d/dx tanh(0) = 1
    CHECK_CLOSE(utils::deriv_hyperbolic_tan(0.0f), 1.0f, 1e-6);
}

TEST(utils, linear) {
    CHECK_CLOSE(utils::linear(3.5f), 3.5f, 1e-6);
    CHECK_CLOSE(utils::linear(-2.0f), -2.0f, 1e-6);
    CHECK_CLOSE(utils::deriv_linear(123.0f), 1.0f, 1e-6);
}

TEST(utils, relu_is_leaky) {
    // Positive side is identity.
    CHECK_CLOSE(utils::relu(2.0f), 2.0f, 1e-6);
    CHECK_CLOSE(utils::deriv_relu(2.0f), 1.0f, 1e-6);
    // Negative side uses the leaky slope (0.01), NOT a hard zero.
    CHECK_CLOSE(utils::relu(-2.0f), -0.02f, 1e-6);
    CHECK_CLOSE(utils::deriv_relu(-2.0f), 0.01f, 1e-6);
}

TEST(utils, hardsigmoid) {
    CHECK_CLOSE(utils::hardsigmoid(0.0f), 0.5f, 1e-6);
    CHECK_CLOSE(utils::hardsigmoid(-3.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hardsigmoid(3.0f), 1.0f, 1e-6);
    CHECK_CLOSE(utils::hardsigmoid(-10.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hardsigmoid(10.0f), 1.0f, 1e-6);
    // slope 1/6 inside the linear region, 0 outside
    CHECK_CLOSE(utils::deriv_hardsigmoid(0.0f), 1.0f / 6.0f, 1e-6);
    CHECK_CLOSE(utils::deriv_hardsigmoid(5.0f), 0.0f, 1e-6);
}

TEST(utils, hardtanh) {
    CHECK_CLOSE(utils::hardtanh(0.5f), 0.5f, 1e-6);
    CHECK_CLOSE(utils::hardtanh(2.0f), 1.0f, 1e-6);
    CHECK_CLOSE(utils::hardtanh(-2.0f), -1.0f, 1e-6);
    CHECK_CLOSE(utils::deriv_hardtanh(0.0f), 1.0f, 1e-6);
    CHECK_CLOSE(utils::deriv_hardtanh(2.0f), 0.0f, 1e-6);
}

TEST(utils, hardswish) {
    CHECK_CLOSE(utils::hardswish(0.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hardswish(-3.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hardswish(-10.0f), 0.0f, 1e-6);
    CHECK_CLOSE(utils::hardswish(4.0f), 4.0f, 1e-6);   // x >= 3 -> identity
    // x=3 -> x*(x+3)/6 = 3*6/6 = 3
    CHECK_CLOSE(utils::hardswish(3.0f), 3.0f, 1e-6);
}

TEST(utils, sgn) {
    CHECK_CLOSE(utils::sgn(5.0f), 1.0f, 1e-6);
    CHECK_CLOSE(utils::sgn(-5.0f), -1.0f, 1e-6);
    CHECK_CLOSE(utils::sgn(0.0f), 0.0f, 1e-6);
}

TEST(utils, softmax_uniform_sums_to_one) {
    std::vector<float> v{0.0f, 0.0f, 0.0f, 0.0f};
    utils::Softmax(&v);
    float sum = 0.f;
    for (float x : v) { sum += x; CHECK_CLOSE(x, 0.25f, 1e-6); }
    CHECK_CLOSE(sum, 1.0f, 1e-6);
}

TEST(utils, softmax_monotonic_and_normalised) {
    std::vector<float> v{1.0f, 2.0f, 3.0f};
    utils::Softmax(&v);
    float sum = v[0] + v[1] + v[2];
    CHECK_CLOSE(sum, 1.0f, 1e-5);
    CHECK(v[0] < v[1]);
    CHECK(v[1] < v[2]);
}

TEST(utils, get_id_max_element) {
    std::vector<float> v{0.1f, 0.9f, 0.3f, 0.2f};
    size_t id = 99;
    utils::GetIdMaxElement(v, &id);
    CHECK_EQ(id, (size_t)1);
}

TEST(utils, gen_rand_in_range) {
    std::srand(12345);
    utils::gen_rand<float> g(2.0f);  // range [-1, 1]
    bool all_in_range = true;
    for (int i = 0; i < 1000; ++i) {
        float x = g();
        if (x < -1.0001f || x > 1.0001f) all_in_range = false;
    }
    CHECK(all_in_range);
}
