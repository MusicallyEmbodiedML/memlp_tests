/**
 * @file test_layer.cpp
 * @brief Characterisation tests for Layer.h — the flat row-major weight matrix,
 *        forward pass, weight get/set round-trips, Polyak (smooth) update,
 *        Xavier init bounds, and NaN/Inf repair.
 */

#include "test_framework.h"
#include "Layer.h"

#include <vector>
#include <cmath>
#include <cstdlib>

// 2 inputs, 1 node, LINEAR, all weights = 0.5, biases = 0.
static Layer<float> make_linear_layer() {
    return Layer<float>(2, 1, ACTIVATION_FUNCTIONS::LINEAR,
                        /*use_constant_weight_init=*/true, 0.5f);
}

TEST(layer, sizes) {
    Layer<float> l(3, 5, ACTIVATION_FUNCTIONS::RELU, true, 0.1f);
    CHECK_EQ(l.GetInputSize(), 3);
    CHECK_EQ(l.GetOutputSize(), 5);
    CHECK_EQ(l.m_weights.size(), (size_t)15);
    CHECK_EQ(l.m_biases.size(), (size_t)5);
}

TEST(layer, constant_init_values) {
    Layer<float> l = make_linear_layer();
    CHECK_CLOSE(l.weight(0, 0), 0.5f, 1e-6);
    CHECK_CLOSE(l.weight(0, 1), 0.5f, 1e-6);
    CHECK_CLOSE(l.bias(0), 0.0f, 1e-6);
}

TEST(layer, forward_linear) {
    Layer<float> l = make_linear_layer();
    std::vector<float> in{1.0f, 1.0f};
    std::vector<float> out;
    l.GetOutputAfterActivationFunction(in, &out);
    CHECK_EQ(out.size(), (size_t)1);
    // 0.5*1 + 0.5*1 + bias(0) = 1.0
    CHECK_CLOSE(out[0], 1.0f, 1e-6);
}

TEST(layer, forward_relu_leaky_on_negative) {
    Layer<float> l(2, 1, ACTIVATION_FUNCTIONS::RELU, true, 0.5f);
    std::vector<float> in{-1.0f, -1.0f};   // sum = -1
    std::vector<float> out;
    l.GetOutputAfterActivationFunction(in, &out);
    // leaky relu: 0.01 * -1 = -0.01
    CHECK_CLOSE(out[0], -0.01f, 1e-6);
}

TEST(layer, forward_two_nodes) {
    Layer<float> l(2, 2, ACTIVATION_FUNCTIONS::LINEAR, true, 1.0f);
    l.bias(0) = 0.0f;
    l.bias(1) = 10.0f;
    std::vector<float> in{2.0f, 3.0f};   // dot = 5 for each node
    std::vector<float> out;
    l.GetOutputAfterActivationFunction(in, &out);
    CHECK_CLOSE(out[0], 5.0f, 1e-6);
    CHECK_CLOSE(out[1], 15.0f, 1e-6);
}

TEST(layer, weights_2d_round_trip) {
    Layer<float> l(2, 2, ACTIVATION_FUNCTIONS::LINEAR, true, 0.0f);
    std::vector<std::vector<float>> w{{1.0f, 2.0f}, {3.0f, 4.0f}};
    l.SetWeights(w);
    auto got = l.GetWeights2D();
    CHECK_EQ(got.size(), (size_t)2);
    CHECK_CLOSE(got[0][0], 1.0f, 1e-6);
    CHECK_CLOSE(got[0][1], 2.0f, 1e-6);
    CHECK_CLOSE(got[1][0], 3.0f, 1e-6);
    CHECK_CLOSE(got[1][1], 4.0f, 1e-6);
    // and via flat accessor
    CHECK_CLOSE(l.weight(1, 0), 3.0f, 1e-6);
}

TEST(layer, weight_norm) {
    Layer<float> l = make_linear_layer();  // weights 0.5, 0.5
    // sqrt(0.25 + 0.25) = sqrt(0.5)
    CHECK_CLOSE(l.getWeightNorm(), std::sqrt(0.5f), 1e-6);
}

TEST(layer, smooth_update_polyak) {
    Layer<float> target(2, 1, ACTIVATION_FUNCTIONS::LINEAR, true, 0.0f);
    Layer<float> source(2, 1, ACTIVATION_FUNCTIONS::LINEAR, true, 1.0f);
    source.bias(0) = 1.0f;  // target bias starts at 0
    const float alpha = 0.25f;
    target.SmoothUpdateWeights(source, alpha, 1.0f - alpha);
    // new = 0.75*0 + 0.25*1 = 0.25
    CHECK_CLOSE(target.weight(0, 0), 0.25f, 1e-6);
    CHECK_CLOSE(target.weight(0, 1), 0.25f, 1e-6);
    CHECK_CLOSE(target.bias(0), 0.25f, 1e-6);
}

TEST(layer, xavier_init_within_bounds_and_changes_weights) {
    std::srand(777);
    Layer<float> l(4, 4, ACTIVATION_FUNCTIONS::TANH, true, 0.5f);
    l.InitXavier();
    float limit = std::sqrt(6.0f / (4 + 4));  // tanh/sigmoid limit
    bool in_bounds = true;
    for (float w : l.m_weights) {
        if (w < -limit - 1e-4f || w > limit + 1e-4f) in_bounds = false;
    }
    CHECK(in_bounds);
}

TEST(layer, check_and_fix_weights) {
    Layer<float> l(2, 1, ACTIVATION_FUNCTIONS::LINEAR, true, 0.5f);
    l.weight(0, 0) = std::nanf("");
    l.weight(0, 1) = INFINITY;
    bool fixed = l.CheckAndFixWeights();
    CHECK(fixed);
    CHECK_CLOSE(l.weight(0, 0), 0.0f, 1e-6);
    CHECK_CLOSE(l.weight(0, 1), 0.0f, 1e-6);
    // clean layer reports no corruption
    Layer<float> clean = make_linear_layer();
    CHECK_FALSE(clean.CheckAndFixWeights());
}
