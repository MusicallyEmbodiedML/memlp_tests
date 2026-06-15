/**
 * @file test_loss.cpp
 * @brief Characterisation tests for Loss.h — MSE and Categorical Cross-Entropy,
 *        including the exact derivative values the training code relies on.
 */

#include "test_framework.h"
#include "Loss.h"

#include <vector>
#include <cmath>

TEST(loss, mse_zero_when_equal) {
    std::vector<float> expected{1.0f, 0.0f};
    std::vector<float> actual{1.0f, 0.0f};
    std::vector<float> deriv(2, 999.0f);
    float l = loss::MSE(expected, actual, deriv, 1.0f);
    CHECK_CLOSE(l, 0.0f, 1e-6);
    CHECK_CLOSE(deriv[0], 0.0f, 1e-6);
    CHECK_CLOSE(deriv[1], 0.0f, 1e-6);
}

TEST(loss, mse_single_element) {
    // n_elem = 1, sampleSizeReciprocal = 1
    // loss = (1-0)^2 = 1 ; deriv = -2 * 1 * (1-0) * 1 = -2
    std::vector<float> expected{1.0f};
    std::vector<float> actual{0.0f};
    std::vector<float> deriv(1);
    float l = loss::MSE(expected, actual, deriv, 1.0f);
    CHECK_CLOSE(l, 1.0f, 1e-6);
    CHECK_CLOSE(deriv[0], -2.0f, 1e-6);
}

TEST(loss, mse_two_elements) {
    // n_elem = 2, one_over_n_elem = 0.5, ssr = 1
    // loss = 0.5*(1^2) + 0.5*(3^2) = 0.5 + 4.5 = 5
    // deriv[j] = -2 * 0.5 * diff_j -> -1 and -3
    std::vector<float> expected{1.0f, 3.0f};
    std::vector<float> actual{0.0f, 0.0f};
    std::vector<float> deriv(2);
    float l = loss::MSE(expected, actual, deriv, 1.0f);
    CHECK_CLOSE(l, 5.0f, 1e-5);
    CHECK_CLOSE(deriv[0], -1.0f, 1e-6);
    CHECK_CLOSE(deriv[1], -3.0f, 1e-6);
}

TEST(loss, mse_sample_size_reciprocal_scales) {
    // Same as single_element but ssr = 0.5 -> loss halved, deriv halved.
    std::vector<float> expected{1.0f};
    std::vector<float> actual{0.0f};
    std::vector<float> deriv(1);
    float l = loss::MSE(expected, actual, deriv, 0.5f);
    CHECK_CLOSE(l, 0.5f, 1e-6);
    CHECK_CLOSE(deriv[0], -1.0f, 1e-6);
}

TEST(loss, cross_entropy_uniform_logits) {
    // Two equal logits -> softmax = {0.5, 0.5}; target class 0.
    // loss = -logit_target + logsumexp = -0 + ln(2) = 0.6931
    std::vector<float> expected{1.0f, 0.0f};
    std::vector<float> actual{0.0f, 0.0f};
    std::vector<float> deriv(2);
    float l = loss::CategoricalCrossEntropy(expected, actual, deriv, 1.0f);
    CHECK_CLOSE(l, std::log(2.0f), 1e-5);
    // gradient = softmax - expected
    CHECK_CLOSE(deriv[0], 0.5f - 1.0f, 1e-5);
    CHECK_CLOSE(deriv[1], 0.5f - 0.0f, 1e-5);
}

TEST(loss, cross_entropy_confident_correct_is_low) {
    // Strongly favouring the correct class -> small loss, small gradients.
    std::vector<float> expected{0.0f, 1.0f, 0.0f};
    std::vector<float> actual{0.0f, 10.0f, 0.0f};
    std::vector<float> deriv(3);
    float l = loss::CategoricalCrossEntropy(expected, actual, deriv, 1.0f);
    CHECK(l < 0.01f);
    // softmax of correct class ~ 1 -> its gradient ~ 0
    CHECK_CLOSE(deriv[1], 0.0f, 1e-2);
}

TEST(loss, cross_entropy_gradients_sum_to_zero) {
    // Because grad = softmax - onehot, and both sum to 1, the gradient sums to 0.
    std::vector<float> expected{1.0f, 0.0f, 0.0f};
    std::vector<float> actual{0.3f, -0.7f, 1.2f};
    std::vector<float> deriv(3);
    loss::CategoricalCrossEntropy(expected, actual, deriv, 1.0f);
    CHECK_CLOSE(deriv[0] + deriv[1] + deriv[2], 0.0f, 1e-5);
}
