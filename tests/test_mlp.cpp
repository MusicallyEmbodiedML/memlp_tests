/**
 * @file test_mlp.cpp
 * @brief Characterisation tests for the MLP class — construction geometry,
 *        deterministic inference, weight get/set round-trips, in-memory
 *        serialisation, Polyak target-network updates, classification, training
 *        convergence/determinism, and (host only) file save/load.
 *
 * Weight initialisation uses the global rand(), so srand() makes construction
 * deterministic — exploited by the training tests below.
 */

#include "test_framework.h"
#include "MLP.h"

#include <vector>
#include <memory>
#include <cstdlib>
#include <cmath>

using Vec  = std::vector<float>;
using VecV = std::vector<std::vector<float>>;

// Helper: small fixed network with constant weights -> fully deterministic.
static MLP<float> make_constant_mlp(const std::vector<size_t> & shape,
                                    float w) {
    std::vector<ACTIVATION_FUNCTIONS> acts(shape.size() - 1,
                                           ACTIVATION_FUNCTIONS::LINEAR);
    return MLP<float>(shape, acts, loss::LOSS_FUNCTIONS::LOSS_MSE,
                      /*use_constant_weight_init=*/true, w);
}

TEST(mlp, construction_geometry) {
    MLP<float> net = make_constant_mlp({2, 3, 1}, 0.5f);
    CHECK_EQ(net.get_num_inputs(), 2);
    CHECK_EQ(net.get_num_outputs(), 1);
    CHECK_EQ(net.get_num_hidden_layers(), 1);
    CHECK_EQ(net.GetNumLayers(), (size_t)2);
}

TEST(mlp, inference_single_layer_linear) {
    // {2,1} linear, all weights 0.5, biases 0: out = 0.5*x0 + 0.5*x1
    MLP<float> net = make_constant_mlp({2, 1}, 0.5f);
    Vec out;
    net.GetOutput(Vec{1.0f, 1.0f}, &out);
    CHECK_EQ(out.size(), (size_t)1);
    CHECK_CLOSE(out[0], 1.0f, 1e-6);

    net.GetOutput(Vec{2.0f, 4.0f}, &out);
    CHECK_CLOSE(out[0], 3.0f, 1e-6);
}

TEST(mlp, inference_two_layer_linear) {
    // {2,2,1} all weights 1.0, linear, biases 0.
    // hidden = [x0+x1, x0+x1]; output = hidden0 + hidden1 = 2*(x0+x1)
    MLP<float> net = make_constant_mlp({2, 2, 1}, 1.0f);
    Vec out;
    net.GetOutput(Vec{1.0f, 2.0f}, &out);   // x0+x1 = 3 -> 2*3 = 6
    CHECK_CLOSE(out[0], 6.0f, 1e-6);
}

TEST(mlp, output_size_matches_num_outputs) {
    MLP<float> net = make_constant_mlp({3, 5, 4}, 0.1f);
    Vec out;
    net.GetOutput(Vec{0.1f, 0.2f, 0.3f}, &out);
    CHECK_EQ(out.size(), (size_t)4);
}

TEST(mlp, get_set_weights_round_trip) {
    MLP<float> net = make_constant_mlp({2, 2, 1}, 0.0f);
    auto w = net.GetWeights();
    // Mutate every weight and write back.
    for (auto & layer : w)
        for (auto & node : layer)
            for (auto & x : node) x = 0.321f;
    net.SetWeights(w);

    auto w2 = net.GetWeights();
    CHECK_EQ(w2.size(), w.size());
    bool all_match = true;
    for (size_t l = 0; l < w2.size(); ++l)
        for (size_t n = 0; n < w2[l].size(); ++n)
            for (size_t k = 0; k < w2[l][n].size(); ++k)
                if (std::fabs(w2[l][n][k] - 0.321f) > 1e-6f) all_match = false;
    CHECK(all_match);
}

TEST(mlp, serialise_round_trip_in_memory) {
    // Source and destination share shape but differ in weights.
    MLP<float> src = make_constant_mlp({2, 3, 1}, 0.5f);
    MLP<float> dst = make_constant_mlp({2, 3, 1}, 0.1f);

    std::vector<uint8_t> buffer;
    size_t w = src.Serialise(0, buffer);
    CHECK(w > 0);
    CHECK(buffer.size() == w);

    size_t r = dst.FromSerialised(0, buffer);
    CHECK_EQ(r, w);

    // Weights now equal; identical inference confirms it.
    Vec a, b;
    src.GetOutput(Vec{0.7f, -0.3f}, &a);
    dst.GetOutput(Vec{0.7f, -0.3f}, &b);
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) CHECK_CLOSE(a[i], b[i], 1e-6);
}

TEST(mlp, get_output_class) {
    MLP<float> net = make_constant_mlp({2, 3}, 0.1f);
    size_t cls = 99;
    net.GetOutputClass(Vec{0.1f, 0.9f, 0.3f}, &cls);
    CHECK_EQ(cls, (size_t)1);
}

TEST(mlp, smooth_update_moves_toward_source) {
    // target all 0, source all 1; alpha 0.5 -> all 0.5.
    // MLP is non-copyable (holds a std::random_device), so build in place.
    const std::vector<size_t> shape{2, 2, 1};
    const std::vector<ACTIVATION_FUNCTIONS> acts(2, ACTIVATION_FUNCTIONS::LINEAR);
    auto target = std::make_shared<MLP<float>>(
        shape, acts, loss::LOSS_FUNCTIONS::LOSS_MSE, true, 0.0f);
    auto source = std::make_shared<MLP<float>>(
        shape, acts, loss::LOSS_FUNCTIONS::LOSS_MSE, true, 1.0f);
    target->SmoothUpdateWeights(source, 0.5f);
    auto w = target->GetWeights();
    bool all_half = true;
    for (auto & layer : w)
        for (auto & node : layer)
            for (auto & x : node)
                if (std::fabs(x - 0.5f) > 1e-6f) all_half = false;
    CHECK(all_half);
}

TEST(mlp, train_reduces_loss_on_xor) {
    std::srand(42);  // deterministic weight init
    MLP<float> net({2, 6, 1},
                   {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::SIGMOID},
                   loss::LOSS_FUNCTIONS::LOSS_MSE);

    VecV X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    VecV Y{{0},    {1},    {1},    {0}};

    // Baseline loss before training (one iteration, tiny LR ~ measure only).
    float initial = net.Train({X, Y}, 0.0f, 1, /*min_err*/-1.0f, false);
    float final   = net.Train({X, Y}, 0.5f, 4000, /*min_err*/1e-4f, false);

    CHECK(std::isfinite(final));
    CHECK(final < initial);          // training made progress
    CHECK(final < 0.5f * initial);   // and meaningfully so
}

TEST(mlp, train_is_deterministic_with_fixed_seed) {
    auto run_once = []() {
        std::srand(123);
        MLP<float> net({2, 5, 1},
                       {ACTIVATION_FUNCTIONS::TANH, ACTIVATION_FUNCTIONS::SIGMOID},
                       loss::LOSS_FUNCTIONS::LOSS_MSE);
        VecV X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        VecV Y{{0},    {1},    {1},    {0}};
        return net.Train({X, Y}, 0.3f, 500, 1e-6f, false);
    };
    float a = run_once();
    float b = run_once();
    CHECK_CLOSE(a, b, 1e-7);
}

#if defined(ENABLE_SAVE) && ENABLE_SAVE
TEST(mlp, save_load_file_round_trip) {
    const std::string path = "memlp_test_tmp.bin";

    MLP<float> src = make_constant_mlp({2, 4, 3}, 0.42f);
    // Give biases distinct values so the test also covers bias persistence.
    src.GetLayerRef(0).bias(0) = 1.5f;
    src.GetLayerRef(1).bias(2) = -2.5f;

    CHECK(src.SaveMLPNetwork(path));

    MLP<float> dst = make_constant_mlp({1, 1}, 0.0f);  // different shape
    CHECK(dst.LoadMLPNetwork(path));

    CHECK_EQ(dst.get_num_inputs(), 2);
    CHECK_EQ(dst.get_num_outputs(), 3);
    CHECK_EQ(dst.GetNumLayers(), (size_t)2);

    Vec a, b;
    src.GetOutput(Vec{0.3f, -0.6f}, &a);
    dst.GetOutput(Vec{0.3f, -0.6f}, &b);
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) CHECK_CLOSE(a[i], b[i], 1e-6);

    std::remove(path.c_str());
}
#endif
