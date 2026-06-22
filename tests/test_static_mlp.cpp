/**
 * @file test_static_mlp.cpp
 * @brief Cross-validation of the compile-time StaticMLP against the dynamic
 *        MLP<float>: given identical architecture and identical weights/biases,
 *        the two must produce identical inference outputs. This is the primary
 *        correctness guarantee for the static rewrite.
 */

#include "test_framework.h"
#include "MLP.h"
#include "StaticMLP.h"

#include <array>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <new>

using smlp::Layout;
using smlp::Activations;

// ── Global allocation tripwire (proves the hot paths touch no heap) ──
// Host-only: the Pico SDK supplies its own operator new/delete. The no-heap
// guarantee transfers to the embedded build since it is the identical code.
#ifndef MEMLP_TESTS_EMBEDDED
volatile bool g_alloc_seen = false;   // external linkage: shared with test_static_mlp_fixed.cpp
void* operator new(std::size_t n)            { g_alloc_seen = true; return std::malloc(n ? n : 1); }
void* operator new[](std::size_t n)          { g_alloc_seen = true; return std::malloc(n ? n : 1); }
void  operator delete(void* p) noexcept      { std::free(p); }
void  operator delete[](void* p) noexcept    { std::free(p); }
void  operator delete(void* p, std::size_t) noexcept   { std::free(p); }
void  operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif

// Copy a dynamic network's weights AND biases into a static one of matching shape.
template<typename Static>
static void copy_from_dynamic(MLP<float> & dyn, Static & st) {
    st.SetAllWeights(dyn.GetWeights());
    typename Static::mlp_biases biases(dyn.GetNumLayers());
    for (std::size_t l = 0; l < dyn.GetNumLayers(); ++l) {
        auto & layer = dyn.m_layers[l];
        biases[l].resize(layer.GetOutputSize());
        for (int n = 0; n < layer.GetOutputSize(); ++n) biases[l][n] = layer.bias(n);
    }
    st.SetAllBiases(biases);
}

template<typename Static>
static void expect_same_output(MLP<float> & dyn, Static & st,
                               const std::vector<float> & in, double tol = 1e-5) {
    std::vector<float> a;
    dyn.GetOutput(in, &a);
    std::array<float, Static::kNumOutputs> b{};
    st.GetOutput(in.data(), b.data());
    CHECK_EQ(a.size(), (size_t)Static::kNumOutputs);
    for (std::size_t i = 0; i < Static::kNumOutputs; ++i) CHECK_CLOSE(a[i], b[i], tol);
}

TEST(static_mlp, geometry_matches) {
    using Net = smlp::StaticMLP<float, Layout<2, 3, 1>,
                                Activations<ACTIVATION_FUNCTIONS::RELU,
                                            ACTIVATION_FUNCTIONS::LINEAR>>;
    CHECK_EQ(Net::get_num_inputs(), 2);
    CHECK_EQ(Net::get_num_outputs(), 1);
    CHECK_EQ(Net::get_num_hidden_layers(), 1);
    CHECK_EQ(Net::GetNumLayers(), (size_t)2);
    CHECK_EQ(Net::kMaxWidth, (size_t)3);
}

TEST(static_mlp, constant_init_single_layer) {
    MLP<float> dyn({2, 1}, {ACTIVATION_FUNCTIONS::LINEAR},
                   loss::LOSS_FUNCTIONS::LOSS_MSE, true, 0.5f);
    smlp::StaticMLP<float, Layout<2, 1>, Activations<ACTIVATION_FUNCTIONS::LINEAR>> st;
    st.SetConstantWeights(0.5f);
    expect_same_output(dyn, st, {1.0f, 1.0f});
    expect_same_output(dyn, st, {2.0f, 4.0f});
    expect_same_output(dyn, st, {-3.0f, 0.5f});
}

TEST(static_mlp, constant_init_multilayer) {
    MLP<float> dyn({2, 2, 1}, {ACTIVATION_FUNCTIONS::LINEAR, ACTIVATION_FUNCTIONS::LINEAR},
                   loss::LOSS_FUNCTIONS::LOSS_MSE, true, 1.0f);
    smlp::StaticMLP<float, Layout<2, 2, 1>,
                    Activations<ACTIVATION_FUNCTIONS::LINEAR,
                                ACTIVATION_FUNCTIONS::LINEAR>> st;
    st.SetConstantWeights(1.0f);
    expect_same_output(dyn, st, {1.0f, 2.0f});
}

// Exercise each activation function with random weights + biases.
template<ACTIVATION_FUNCTIONS A>
static void cross_validate_activation() {
    MLP<float> dyn({3, 5, 4}, {A, A}, loss::LOSS_FUNCTIONS::LOSS_MSE);
    dyn.RandomiseWeightsAndBiasesLin(-1.0f, 1.0f, -0.5f, 0.5f);

    smlp::StaticMLP<float, Layout<3, 5, 4>, Activations<A, A>> st;
    copy_from_dynamic(dyn, st);

    std::srand(99);
    for (int t = 0; t < 8; ++t) {
        std::vector<float> in{
            (float)std::rand() / RAND_MAX * 2 - 1,
            (float)std::rand() / RAND_MAX * 2 - 1,
            (float)std::rand() / RAND_MAX * 2 - 1};
        expect_same_output(dyn, st, in);
    }
}

TEST(static_mlp, activation_relu)        { cross_validate_activation<ACTIVATION_FUNCTIONS::RELU>(); }
TEST(static_mlp, activation_tanh)        { cross_validate_activation<ACTIVATION_FUNCTIONS::TANH>(); }
TEST(static_mlp, activation_sigmoid)     { cross_validate_activation<ACTIVATION_FUNCTIONS::SIGMOID>(); }
TEST(static_mlp, activation_linear)      { cross_validate_activation<ACTIVATION_FUNCTIONS::LINEAR>(); }
TEST(static_mlp, activation_hardsigmoid) { cross_validate_activation<ACTIVATION_FUNCTIONS::HARDSIGMOID>(); }
TEST(static_mlp, activation_hardswish)   { cross_validate_activation<ACTIVATION_FUNCTIONS::HARDSWISH>(); }
TEST(static_mlp, activation_hardtanh)    { cross_validate_activation<ACTIVATION_FUNCTIONS::HARDTANH>(); }

TEST(static_mlp, deep_mixed_activations) {
    MLP<float> dyn({4, 8, 6, 3},
                   {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::TANH,
                    ACTIVATION_FUNCTIONS::SIGMOID},
                   loss::LOSS_FUNCTIONS::LOSS_MSE);
    dyn.RandomiseWeightsAndBiasesLin(-1.5f, 1.5f, -0.3f, 0.3f);

    smlp::StaticMLP<float, Layout<4, 8, 6, 3>,
                    Activations<ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::SIGMOID>> st;
    copy_from_dynamic(dyn, st);

    std::srand(7);
    for (int t = 0; t < 10; ++t) {
        std::vector<float> in(4);
        for (auto & x : in) x = (float)std::rand() / RAND_MAX * 2 - 1;
        expect_same_output(dyn, st, in);
    }
}

TEST(static_mlp, categorical_crossentropy_softmax) {
    // Both apply softmax at the output during inference when loss is CCE.
    MLP<float> dyn({3, 4}, {ACTIVATION_FUNCTIONS::LINEAR},
                   loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY);
    dyn.RandomiseWeightsAndBiasesLin(-1.0f, 1.0f, -0.5f, 0.5f);

    smlp::StaticMLP<float, Layout<3, 4>, Activations<ACTIVATION_FUNCTIONS::LINEAR>,
                    loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY> st;
    copy_from_dynamic(dyn, st);

    std::vector<float> in{0.2f, -0.4f, 0.8f};
    std::vector<float> a;
    dyn.GetOutput(in, &a);
    std::array<float, 4> b{};
    st.GetOutput(in.data(), b.data());
    float sum = 0.f;
    for (int i = 0; i < 4; ++i) { CHECK_CLOSE(a[i], b[i], 1e-5); sum += b[i]; }
    CHECK_CLOSE(sum, 1.0f, 1e-5);   // softmax normalised
}

TEST(static_mlp, get_output_class) {
    smlp::StaticMLP<float, Layout<2, 3>, Activations<ACTIVATION_FUNCTIONS::LINEAR>> st;
    std::array<float, 3> out{0.1f, 0.9f, 0.3f};
    std::size_t cls = 99;
    st.GetOutputClass(out.data(), &cls);
    CHECK_EQ(cls, (size_t)1);
}

// ── Training: per-sample Train must match the dynamic MLP exactly ──
TEST(static_mlp, train_per_sample_matches_dynamic) {
    MLP<float> dyn({2, 4, 1},
                   {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::SIGMOID},
                   loss::LOSS_FUNCTIONS::LOSS_MSE);
    dyn.RandomiseWeightsAndBiasesLin(-0.8f, 0.8f, -0.3f, 0.3f);

    smlp::StaticMLP<float, Layout<2, 4, 1>,
                    Activations<ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::SIGMOID>> st;
    copy_from_dynamic(dyn, st);

    std::vector<std::vector<float>> X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<float>> Y{{0}, {1}, {1}, {0}};

    // Same data order, same lr/iters, no early stop -> identical trajectories.
    dyn.Train({X, Y}, 0.1f, 150, -1.0f, false);
    st.Train({X, Y}, 0.1f, 150, -1.0f);

    expect_same_output(dyn, st, {0, 0}, 1e-3);
    expect_same_output(dyn, st, {0, 1}, 1e-3);
    expect_same_output(dyn, st, {1, 0}, 1e-3);
    expect_same_output(dyn, st, {1, 1}, 1e-3);
}

TEST(static_mlp, train_batch_converges_on_xor) {
    MLP<float> seed({2, 6, 1},
                    {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::SIGMOID},
                    loss::LOSS_FUNCTIONS::LOSS_MSE);
    seed.RandomiseWeightsAndBiasesLin(-1.0f, 1.0f, -0.3f, 0.3f);

    smlp::StaticMLP<float, Layout<2, 6, 1>,
                    Activations<ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::SIGMOID>> net;
    copy_from_dynamic(seed, net);

    std::vector<std::vector<float>> X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<float>> Y{{0}, {1}, {1}, {0}};

    float initial = net.TrainBatch({X, Y}, 0.0f, 1, 4, 1e9f);   // lr 0 -> baseline
    float final   = net.TrainBatch({X, Y}, 0.05f, 4000, 4, 1e-5f);
    CHECK(std::isfinite(final));
    CHECK(final < initial);
    CHECK(final < 0.5f * initial);
}

// ── Deterministic RMSProp parity at the layer level (no shuffle involved) ──
TEST(static_mlp, rmsprop_apply_matches_dynamic_layer) {
    Layer<float> dl(3, 2, ACTIVATION_FUNCTIONS::RELU, true, 0.5f);
    smlp::StaticLayer<float, 3, 2, ACTIVATION_FUNCTIONS::RELU> sl;
    sl.m_weights.fill(0.5f);

    for (std::size_t k = 0; k < 6; ++k) {
        dl.m_grad_accum[k] = 0.1f * (k + 1);
        sl.m_grad_accum[k] = 0.1f * (k + 1);
    }
    for (std::size_t i = 0; i < 2; ++i) {
        dl.m_bias_grad_accum[i] = 0.2f * (i + 1);
        sl.m_bias_grad_accum[i] = 0.2f * (i + 1);
    }
    dl.ApplyAccumulatedGradients(0.01f, 0.25f);
    sl.ApplyAccumulatedGradients(0.01f, 0.25f);

    for (std::size_t k = 0; k < 6; ++k) CHECK_CLOSE(dl.m_weights[k], sl.m_weights[k], 1e-6);
    for (std::size_t i = 0; i < 2; ++i) CHECK_CLOSE(dl.m_biases[i], sl.m_biases[i], 1e-6);
}

TEST(static_mlp, accumulate_gradients_matches_dynamic_layer) {
    Layer<float> dl(3, 2, ACTIVATION_FUNCTIONS::TANH, true, 0.3f);
    smlp::StaticLayer<float, 3, 2, ACTIVATION_FUNCTIONS::TANH> sl;
    sl.m_weights.fill(0.3f);

    // Identical cached pre-activations so the derivative term matches.
    for (std::size_t i = 0; i < 2; ++i) {
        dl.m_inner_products[i] = 0.4f * (i + 1);
        sl.m_inner_products[i] = 0.4f * (i + 1);
    }
    std::vector<float> input{0.5f, -0.2f, 0.7f};
    std::vector<float> deriv{0.6f, -0.1f};
    std::vector<float> ddelta;
    std::array<float, 3> sdelta{};

    dl.AccumulateGradients(input, deriv, &ddelta);
    sl.AccumulateGradients(input.data(), deriv.data(), sdelta.data());

    for (std::size_t k = 0; k < 6; ++k) CHECK_CLOSE(dl.m_grad_accum[k], sl.m_grad_accum[k], 1e-6);
    for (std::size_t i = 0; i < 2; ++i) CHECK_CLOSE(dl.m_bias_grad_accum[i], sl.m_bias_grad_accum[i], 1e-6);
    for (std::size_t j = 0; j < 3; ++j) CHECK_CLOSE(ddelta[j], sdelta[j], 1e-6);
}

// ── RL surface ──
TEST(static_mlp, smooth_update_matches_dynamic) {
    using Net = smlp::StaticMLP<float, Layout<2, 3, 1>,
                                Activations<ACTIVATION_FUNCTIONS::RELU,
                                            ACTIVATION_FUNCTIONS::LINEAR>>;
    const std::vector<size_t> shape{2, 3, 1};
    const std::vector<ACTIVATION_FUNCTIONS> acts{ACTIVATION_FUNCTIONS::RELU,
                                                 ACTIVATION_FUNCTIONS::LINEAR};
    // Target (plain) and source (shared_ptr, mirroring RL target-network usage).
    MLP<float> dtgt(shape, acts);
    auto dsrc = std::make_shared<MLP<float>>(shape, acts);
    dtgt.RandomiseWeightsAndBiasesLin(-1, 1, -0.5f, 0.5f);
    dsrc->RandomiseWeightsAndBiasesLin(-1, 1, -0.5f, 0.5f);

    Net stgt, ssrc;
    copy_from_dynamic(dtgt, stgt);
    copy_from_dynamic(*dsrc, ssrc);

    const float alpha = 0.1f;
    dtgt.SmoothUpdateWeights(dsrc, alpha);   // dynamic shared_ptr overload
    stgt.SmoothUpdateWeights(ssrc, alpha);   // static reference overload

    expect_same_output(dtgt, stgt, {0.5f, -0.3f}, 1e-5);
    expect_same_output(dtgt, stgt, {-0.7f, 0.9f}, 1e-5);
}

TEST(static_mlp, calc_gradients_matches_dynamic) {
    MLP<float> dyn({3, 4, 2}, {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::LINEAR});
    dyn.RandomiseWeightsAndBiasesLin(-0.8f, 0.8f, -0.2f, 0.2f);

    smlp::StaticMLP<float, Layout<3, 4, 2>,
                    Activations<ACTIVATION_FUNCTIONS::TANH,
                                ACTIVATION_FUNCTIONS::LINEAR>> st;
    copy_from_dynamic(dyn, st);

    std::vector<float> feat{0.3f, -0.5f, 0.1f};
    std::vector<float> derr{0.7f, -0.4f};

    dyn.CalcGradients(feat, derr);
    std::array<float, 3> sg{};
    st.CalcGradients(feat.data(), derr.data(), sg.data());

    // Compare the network-input gradient (dynamic stores it in layer 0's grads).
    auto & dg = dyn.m_layers[0].GetGrads();
    CHECK_EQ(dg.size(), (size_t)3);
    for (std::size_t j = 0; j < 3; ++j) CHECK_CLOSE(dg[j], sg[j], 1e-5);
}

// ── Serialisation interop with the dynamic MLP ──
TEST(static_mlp, serialise_interop_with_dynamic) {
    MLP<float> dyn({3, 5, 2},
                   {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR});
    dyn.RandomiseWeightsAndBiasesLin(-1, 1, 0, 0);  // biases 0 (Serialise is weights-only)

    using Net = smlp::StaticMLP<float, Layout<3, 5, 2>,
                                Activations<ACTIVATION_FUNCTIONS::RELU,
                                            ACTIVATION_FUNCTIONS::LINEAR>>;

    // Dynamic -> buffer -> static : outputs must then match.
    std::vector<uint8_t> buf_dyn;
    std::size_t wrote = dyn.Serialise(0, buf_dyn);
    CHECK_EQ(wrote, buf_dyn.size());

    Net st;
    std::size_t read = st.FromSerialised(0, buf_dyn);
    CHECK_EQ(read, wrote);
    expect_same_output(dyn, st, {0.3f, -0.5f, 0.8f}, 1e-5);

    // Static -> buffer must be byte-identical to the dynamic one.
    std::vector<uint8_t> buf_st;
    st.Serialise(0, buf_st);
    CHECK_EQ(buf_st.size(), buf_dyn.size());
    bool identical = (buf_st == buf_dyn);
    CHECK(identical);

    // And a static buffer loads back into a fresh dynamic net.
    MLP<float> dyn2({3, 5, 2},
                    {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR});
    dyn2.FromSerialised(0, buf_st);
    expect_same_output(dyn2, st, {0.1f, 0.2f, -0.9f}, 1e-5);
}

// ── Initialisation smoke tests ──
TEST(static_mlp, randomise_and_seed_are_deterministic) {
    using Net = smlp::StaticMLP<float, Layout<3, 5, 2>,
                                Activations<ACTIVATION_FUNCTIONS::TANH,
                                            ACTIVATION_FUNCTIONS::LINEAR>>;
    Net a, b;
    a.SetSeed(2024); a.RandomiseWeightsAndBiasesLin(-1, 1, -0.5f, 0.5f);
    b.SetSeed(2024); b.RandomiseWeightsAndBiasesLin(-1, 1, -0.5f, 0.5f);
    std::array<float, 3> in{0.2f, -0.4f, 0.7f};
    std::array<float, 2> oa{}, ob{};
    a.GetOutput(in, oa); b.GetOutput(in, ob);
    CHECK_CLOSE(oa[0], ob[0], 1e-7);   // same seed -> identical weights -> identical output
    CHECK_CLOSE(oa[1], ob[1], 1e-7);

    // Xavier weights stay within the He bound for the ReLU layer.
    smlp::StaticMLP<float, Layout<4, 4>, Activations<ACTIVATION_FUNCTIONS::RELU>> x;
    x.SetSeed(1); x.InitXavier();
    float limit = std::sqrt(6.0f / 4.0f);
    bool in_bounds = true;
    for (float w : x.layer<0>().m_weights)
        if (w < -limit - 1e-4f || w > limit + 1e-4f) in_bounds = false;
    CHECK(in_bounds);
}

// ── Inference-only build (EnableTraining=false) omits optimizer state ──
TEST(static_mlp, inference_only_mode) {
    using Trainable = smlp::StaticMLP<float, Layout<4, 8, 3>,
                                      Activations<ACTIVATION_FUNCTIONS::RELU,
                                                  ACTIVATION_FUNCTIONS::LINEAR>,
                                      loss::LOSS_FUNCTIONS::LOSS_MSE, true>;
    using InferOnly  = smlp::StaticMLP<float, Layout<4, 8, 3>,
                                      Activations<ACTIVATION_FUNCTIONS::RELU,
                                                  ACTIVATION_FUNCTIONS::LINEAR>,
                                      loss::LOSS_FUNCTIONS::LOSS_MSE, false>;
    // Inference-only must be strictly smaller (no grad/optimizer arrays).
    CHECK(sizeof(InferOnly) < sizeof(Trainable));

    // ...and produce identical inference results for the same weights.
    Trainable t; InferOnly inf;
    t.SetSeed(5); t.RandomiseWeightsAndBiasesLin(-1, 1, -0.3f, 0.3f);
    inf.SetAllWeights(t.GetAllWeights());
    inf.SetAllBiases(t.GetAllBiases());
    std::array<float, 4> in{0.5f, -0.5f, 0.25f, 0.75f};
    std::array<float, 3> ot{}, oi{};
    t.GetOutput(in, ot); inf.GetOutput(in, oi);
    for (int i = 0; i < 3; ++i) CHECK_CLOSE(ot[i], oi[i], 1e-6);
}

// ── No-heap proof: the hot paths must not allocate (host-only tripwire) ──
#ifndef MEMLP_TESTS_EMBEDDED
TEST(static_mlp, inference_does_not_allocate) {
    static smlp::StaticMLP<float, Layout<4, 8, 3>,
                           Activations<ACTIVATION_FUNCTIONS::RELU,
                                       ACTIVATION_FUNCTIONS::LINEAR>> net;
    net.SetConstantWeights(0.1f);
    std::array<float, 4> in{0.1f, 0.2f, 0.3f, 0.4f};
    std::array<float, 3> out{};
    net.GetOutput(in, out);          // warm any one-time work first
    g_alloc_seen = false;
    for (int i = 0; i < 100; ++i) net.GetOutput(in, out);
    CHECK_FALSE(g_alloc_seen);
}

TEST(static_mlp, training_step_does_not_allocate) {
    static smlp::StaticMLP<float, Layout<3, 6, 2>,
                           Activations<ACTIVATION_FUNCTIONS::TANH,
                                       ACTIVATION_FUNCTIONS::LINEAR>> net;
    net.SetSeed(11); net.RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
    std::array<float, 3> feat{0.2f, -0.3f, 0.5f};
    std::array<float, 2> err{0.1f, -0.2f};
    net.ApplyLoss(feat.data(), err.data(), 0.01f);   // warm
    g_alloc_seen = false;
    for (int i = 0; i < 100; ++i) net.ApplyLoss(feat.data(), err.data(), 0.01f);
    CHECK_FALSE(g_alloc_seen);
}
#endif // !MEMLP_TESTS_EMBEDDED

TEST(static_mlp, flat_weight_accessor) {
    using Net = smlp::StaticMLP<float, Layout<2, 3, 1>,
                                Activations<ACTIVATION_FUNCTIONS::RELU,
                                            ACTIVATION_FUNCTIONS::LINEAR>>;
    Net net;
    // layer0: 3*2=6 weights, layer1: 1*3=3 -> total 9
    CHECK_EQ(Net::TotalWeights(), (size_t)9);
    // Write each weight via the flat accessor...
    for (std::size_t g = 0; g < Net::TotalWeights(); ++g) {
        float* w = net.WeightPtrAt(g);
        CHECK(w != nullptr);
        *w = (float)g;
    }
    CHECK(net.WeightPtrAt(9) == nullptr);   // out of range
    // ...and confirm layer 0 (first 6) and layer 1 (next 3) see them in order.
    for (std::size_t k = 0; k < 6; ++k) CHECK_CLOSE(net.layer<0>().m_weights[k], (float)k, 1e-6);
    for (std::size_t k = 0; k < 3; ++k) CHECK_CLOSE(net.layer<1>().m_weights[k], (float)(6 + k), 1e-6);
}

TEST(static_mlp, weights_round_trip) {
    smlp::StaticMLP<float, Layout<2, 3, 2>,
                    Activations<ACTIVATION_FUNCTIONS::RELU,
                                ACTIVATION_FUNCTIONS::LINEAR>> st;
    auto w = st.GetAllWeights();
    for (auto & layer : w)
        for (auto & node : layer)
            for (auto & x : node) x = 0.137f;
    st.SetAllWeights(w);
    auto w2 = st.GetAllWeights();
    bool ok = true;
    for (auto & layer : w2)
        for (auto & node : layer)
            for (auto & x : node) if (std::fabs(x - 0.137f) > 1e-7f) ok = false;
    CHECK(ok);
}
