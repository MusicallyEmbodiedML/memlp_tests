/**
 * @file test_static_mlp_fixed.cpp
 * @brief Validates the fixed-point instantiation of smlp::StaticMLP (T =
 *        FixedPoint::Fixed<...>) against the float dynamic MLP<float> oracle.
 *
 * Coverage:
 *   - Inference cross-validation per activation, fixed vs float, within a
 *     quantization-derived tolerance.
 *   - Numerical stability across several Q-formats (varying fractional bits):
 *     error must shrink as precision grows and never blow up / overflow.
 *   - The 32-bit MAC (no int64) must agree with a double-precision reference for
 *     a wide layer kept inside the documented range budget.
 *   - Training in fixed point (per-sample MSE + mini-batch RMSProp) must reduce
 *     the loss.
 *   - The fixed inference hot path allocates no heap (host tripwire).
 */

#include "test_framework.h"
#include "MLP.h"
#include "StaticMLP.h"
#include "FixedNN.h"

#include <array>
#include <vector>
#include <cmath>
#include <cstdlib>

using smlp::Layout;
using smlp::Activations;
using FixedPoint::Fixed;

// Q-formats under test (int32 storage). FRACTIONAL_BITS drives precision and the
// MAC range budget |sum(w*x)| < 2^(31 - 2*F).
using Q24_7  = Fixed<24, 7,  int32_t>;   // coarse
using Q20_11 = Fixed<20, 11, int32_t>;   // recommended default
using Q17_14 = Fixed<17, 14, int32_t>;   // high precision, tighter range

// ── Global allocation tripwire (host only; Pico SDK supplies its own new) ──
#ifndef MEMLP_TESTS_EMBEDDED
extern volatile bool g_alloc_seen;   // defined in test_static_mlp.cpp
#endif

// Convert a dynamic float network's weights + biases into a fixed-point static one.
template<typename Static>
static void copy_from_dynamic_fixed(MLP<float> & dyn, Static & st) {
    using T = typename Static::value_type;
    auto fw = dyn.GetWeights();                       // vector<vector<vector<float>>>
    typename Static::mlp_weights w(fw.size());
    for (std::size_t l = 0; l < fw.size(); ++l) {
        w[l].resize(fw[l].size());
        for (std::size_t n = 0; n < fw[l].size(); ++n) {
            w[l][n].resize(fw[l][n].size());
            for (std::size_t j = 0; j < fw[l][n].size(); ++j) w[l][n][j] = T(fw[l][n][j]);
        }
    }
    st.SetAllWeights(w);

    typename Static::mlp_biases b(dyn.GetNumLayers());
    for (std::size_t l = 0; l < dyn.GetNumLayers(); ++l) {
        auto & layer = dyn.m_layers[l];
        b[l].resize(layer.GetOutputSize());
        for (int n = 0; n < layer.GetOutputSize(); ++n) b[l][n] = T(layer.bias(n));
    }
    st.SetAllBiases(b);
}

// Run float oracle and fixed net, return max abs difference over the outputs.
template<typename Static>
static double max_output_diff(MLP<float> & dyn, Static & st,
                              const std::vector<float> & in) {
    using T = typename Static::value_type;
    std::vector<float> a;
    dyn.GetOutput(in, &a);
    std::array<T, Static::kNumInputs> fin{};
    for (std::size_t j = 0; j < Static::kNumInputs; ++j) fin[j] = T(in[j]);
    std::array<T, Static::kNumOutputs> b{};
    st.GetOutput(fin.data(), b.data());
    double worst = 0.0;
    for (std::size_t i = 0; i < Static::kNumOutputs; ++i)
        worst = std::max(worst, std::fabs((double)a[i] - (double)b[i].to_float()));
    return worst;
}

// ════════════════════════════════════════════════════════════════════════
//  Inference cross-validation, per activation and per format
// ════════════════════════════════════════════════════════════════════════
template<typename T, ACTIVATION_FUNCTIONS A>
static void cross_validate_fixed(double tol) {
    MLP<float> dyn({3, 6, 4}, {A, A}, loss::LOSS_FUNCTIONS::LOSS_MSE);
    dyn.RandomiseWeightsAndBiasesLin(-0.8f, 0.8f, -0.3f, 0.3f);

    smlp::StaticMLP<T, Layout<3, 6, 4>, Activations<A, A>> st;
    copy_from_dynamic_fixed(dyn, st);

    std::srand(123);
    double worst = 0.0;
    for (int t = 0; t < 12; ++t) {
        std::vector<float> in{
            (float)std::rand() / RAND_MAX * 1.4f - 0.7f,
            (float)std::rand() / RAND_MAX * 1.4f - 0.7f,
            (float)std::rand() / RAND_MAX * 1.4f - 0.7f};
        worst = std::max(worst, max_output_diff(dyn, st, in));
    }
    CHECK_CLOSE(worst, 0.0, tol);
}

// Exact-in-fixed activations: only quantization error (~ a few * 2^-F).
TEST(static_fixed, infer_linear_q20_11)     { cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::LINEAR>(0.02); }
TEST(static_fixed, infer_relu_q20_11)       { cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::RELU>(0.02); }
TEST(static_fixed, infer_hardtanh_q20_11)   { cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::HARDTANH>(0.02); }
TEST(static_fixed, infer_hardsigmoid_q20_11){ cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::HARDSIGMOID>(0.02); }

// Approximated activations: quantization + rational-approx error (~1e-2).
TEST(static_fixed, infer_tanh_q20_11)       { cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::TANH>(0.03); }
TEST(static_fixed, infer_sigmoid_q20_11)    { cross_validate_fixed<Q20_11, ACTIVATION_FUNCTIONS::SIGMOID>(0.03); }

// High-precision format: tighter tolerance for the exact activations.
TEST(static_fixed, infer_relu_q17_14)       { cross_validate_fixed<Q17_14, ACTIVATION_FUNCTIONS::RELU>(0.004); }
TEST(static_fixed, infer_linear_q17_14)     { cross_validate_fixed<Q17_14, ACTIVATION_FUNCTIONS::LINEAR>(0.004); }

// ════════════════════════════════════════════════════════════════════════
//  Stability across formats: error must monotonically shrink with more
//  fractional bits, and stay bounded for the coarse format (no overflow).
// ════════════════════════════════════════════════════════════════════════
template<typename T>
static double format_infer_error() {
    MLP<float> dyn({4, 8, 3},
                   {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR},
                   loss::LOSS_FUNCTIONS::LOSS_MSE);
    dyn.RandomiseWeightsAndBiasesLin(-0.6f, 0.6f, -0.2f, 0.2f);
    smlp::StaticMLP<T, Layout<4, 8, 3>,
                    Activations<ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR>> st;
    copy_from_dynamic_fixed(dyn, st);

    std::srand(2024);
    double worst = 0.0;
    for (int t = 0; t < 20; ++t) {
        std::vector<float> in(4);
        for (auto & x : in) x = (float)std::rand() / RAND_MAX * 1.2f - 0.6f;
        worst = std::max(worst, max_output_diff(dyn, st, in));
    }
    return worst;
}

TEST(static_fixed, stability_precision_ordering) {
    double e7  = format_infer_error<Q24_7>();
    double e11 = format_infer_error<Q20_11>();
    double e14 = format_infer_error<Q17_14>();
    // All finite and bounded (no overflow even for the coarse format).
    CHECK(std::isfinite(e7) && e7 < 0.2);
    CHECK(std::isfinite(e11) && e11 < 0.02);
    CHECK(std::isfinite(e14) && e14 < 0.004);
    // More fractional bits => less error.
    CHECK(e14 < e11);
    CHECK(e11 < e7);
}

// ════════════════════════════════════════════════════════════════════════
//  32-bit MAC vs double reference, wide layer kept within the range budget.
//  Q17_14 has budget 2^(31-28)=8; weights/inputs ~0.5 over width 10 -> |sum|~2.5.
// ════════════════════════════════════════════════════════════════════════
TEST(static_fixed, mac_no_int64_matches_double_reference) {
    constexpr std::size_t W = 10;
    smlp::StaticMLP<Q17_14, Layout<W, 1>, Activations<ACTIVATION_FUNCTIONS::LINEAR>> st;

    std::srand(7);
    std::array<float, W> in{};
    typename decltype(st)::mlp_weights w(1);
    w[0].assign(1, std::vector<Q17_14>(W));
    double ref = 0.0;
    for (std::size_t j = 0; j < W; ++j) {
        float xi = (float)std::rand() / RAND_MAX - 0.5f;     // [-0.5, 0.5]
        float wi = (float)std::rand() / RAND_MAX - 0.5f;
        in[j] = xi;
        w[0][0][j] = Q17_14(wi);
        ref += (double)Q17_14(wi).to_float() * (double)Q17_14(xi).to_float();
    }
    st.SetAllWeights(w);

    std::array<Q17_14, 1> out{};
    std::array<Q17_14, W> fin{};
    for (std::size_t j = 0; j < W; ++j) fin[j] = Q17_14(in[j]);
    st.GetOutput(fin.data(), out.data());
    // Single shift at the end; should match the quantized double sum to ~2^-14.
    CHECK_CLOSE(out[0].to_float(), ref, 1e-3);
}

// ════════════════════════════════════════════════════════════════════════
//  rsqrt primitive (used by fixed-point RMSProp) — 32-bit, no int64.
// ════════════════════════════════════════════════════════════════════════
TEST(static_fixed, rsqrt_accuracy) {
    // Useful range for RMSProp squared-gradient averages; <1% over the mid range.
    for (double v = 0.05; v <= 1000.0; v *= 1.5) {
        double got = smlp::nn::rsqrt(Q17_14(v)).to_float();
        double ref = 1.0 / std::sqrt(v);
        CHECK_CLOSE(got, ref, ref * 0.02);   // within 2%
    }
    CHECK_EQ(smlp::nn::rsqrt(Q17_14(0.0f)).to_float(), 0.0f);   // guarded (no inf-loop)
}

// ════════════════════════════════════════════════════════════════════════
//  Training in fixed point — loss must decrease.
// ════════════════════════════════════════════════════════════════════════
TEST(static_fixed, train_per_sample_reduces_error_mse) {
    // Learn a smooth linear-ish target with a small tanh/linear net.
    MLP<float> seed({2, 5, 1},
                    {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::LINEAR},
                    loss::LOSS_FUNCTIONS::LOSS_MSE);
    seed.RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.2f, 0.2f);

    smlp::StaticMLP<Q17_14, Layout<2, 5, 1>,
                    Activations<ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::LINEAR>> net;
    copy_from_dynamic_fixed(seed, net);

    std::vector<std::vector<Q17_14>> X{
        {Q17_14(0.0f), Q17_14(0.0f)}, {Q17_14(0.2f), Q17_14(0.4f)},
        {Q17_14(0.5f), Q17_14(0.1f)}, {Q17_14(0.3f), Q17_14(0.7f)}};
    std::vector<std::vector<Q17_14>> Y{
        {Q17_14(0.1f)}, {Q17_14(0.4f)}, {Q17_14(0.35f)}, {Q17_14(0.55f)}};

    auto mse = [&]() {
        double s = 0.0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            std::array<Q17_14, 1> o{};
            net.GetOutput(X[i].data(), o.data());
            double d = (double)o[0].to_float() - (double)Y[i][0].to_float();
            s += d * d;
        }
        return s / X.size();
    };

    double before = mse();
    net.Train({X, Y}, 0.05f, 400, -1.0f);
    double after = mse();
    CHECK(std::isfinite(after));
    CHECK(after < before);
}

TEST(static_fixed, train_batch_rmsprop_reduces_error) {
    MLP<float> seed({2, 6, 1},
                    {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::SIGMOID},
                    loss::LOSS_FUNCTIONS::LOSS_MSE);
    seed.RandomiseWeightsAndBiasesLin(-0.8f, 0.8f, -0.3f, 0.3f);

    smlp::StaticMLP<Q17_14, Layout<2, 6, 1>,
                    Activations<ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::SIGMOID>> net;
    copy_from_dynamic_fixed(seed, net);

    std::vector<std::vector<Q17_14>> X{
        {Q17_14(0.0f), Q17_14(0.0f)}, {Q17_14(0.0f), Q17_14(1.0f)},
        {Q17_14(1.0f), Q17_14(0.0f)}, {Q17_14(1.0f), Q17_14(1.0f)}};
    std::vector<std::vector<Q17_14>> Y{
        {Q17_14(0.0f)}, {Q17_14(1.0f)}, {Q17_14(1.0f)}, {Q17_14(0.0f)}};

    float initial = net.TrainBatch({X, Y}, 0.0f, 1, 4, 1e9f).to_float();   // lr 0 baseline
    float final   = net.TrainBatch({X, Y}, 0.05f, 2000, 4, 1e-6f).to_float();
    CHECK(std::isfinite(final));
    CHECK(final < initial);
}

// ════════════════════════════════════════════════════════════════════════
//  No-heap proof for the fixed inference path.
// ════════════════════════════════════════════════════════════════════════
#ifndef MEMLP_TESTS_EMBEDDED
TEST(static_fixed, fixed_inference_does_not_allocate) {
    static smlp::StaticMLP<Q20_11, Layout<4, 8, 3>,
                           Activations<ACTIVATION_FUNCTIONS::RELU,
                                       ACTIVATION_FUNCTIONS::LINEAR>> net;
    net.SetConstantWeights(Q20_11(0.1f));
    std::array<Q20_11, 4> in{Q20_11(0.1f), Q20_11(0.2f), Q20_11(0.3f), Q20_11(0.4f)};
    std::array<Q20_11, 3> out{};
    net.GetOutput(in.data(), out.data());     // warm
    g_alloc_seen = false;
    for (int i = 0; i < 100; ++i) net.GetOutput(in.data(), out.data());
    CHECK_FALSE(g_alloc_seen);
}
#endif
