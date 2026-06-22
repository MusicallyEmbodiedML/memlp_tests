/**
 * @file pico_bench.cpp
 * @brief On-device benchmark: dynamic MLP<float> vs compile-time StaticMLP.
 *
 * Builds for RP2040 / RP2350 (see CMakeLists, -DMEMLP_BENCH=ON). Both networks
 * use the same architecture and the same weights, so the comparison isolates
 * the implementation cost (heap+function-pointer dispatch vs static arrays +
 * inlined compile-time activations). Results stream over USB serial.
 */

#include "pico/stdlib.h"
#include "pico/rand.h"
#include "hardware/clocks.h"

#include <array>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "MLP.h"
#include "StaticMLP.h"

// std::random_device -> getentropy on bare metal; back it with the HW RNG.
extern "C" int getentropy(void * buffer, size_t len) {
    uint8_t * p = static_cast<uint8_t *>(buffer);
    while (len) {
        uint32_t r = get_rand_32();
        size_t n = (len < 4) ? len : 4;
        for (size_t i = 0; i < n; ++i) { *p++ = (uint8_t)(r & 0xFF); r >>= 8; }
        len -= n;
    }
    return 0;
}

// ── Benchmark network: a typical small embedded MLP ──
static constexpr std::size_t NI = 8, NH1 = 16, NH2 = 16, NO = 4;
using Acts = smlp::Activations<ACTIVATION_FUNCTIONS::RELU,
                               ACTIVATION_FUNCTIONS::RELU,
                               ACTIVATION_FUNCTIONS::LINEAR>;
using Layout = smlp::Layout<NI, NH1, NH2, NO>;
using Static = smlp::StaticMLP<float, Layout, Acts>;

// Fixed-point contender: Q20.11 (int32) keeps the dot-product MAC inside the
// no-int64 range budget for this width. RELU/LINEAR are exact in fixed point.
using Fix       = FixedPoint::Fixed<20, 11, int32_t>;
using StaticFix = smlp::StaticMLP<Fix, Layout, Acts>;

// File-scope (static, lives in .bss — zero heap for the networks).
static Static    g_static;
static StaticFix g_fixed;

static void copy_weights_biases(MLP<float> & dyn, Static & st) {
    st.SetAllWeights(dyn.GetWeights());
    Static::mlp_biases b(dyn.GetNumLayers());
    for (std::size_t l = 0; l < dyn.GetNumLayers(); ++l) {
        auto & layer = dyn.m_layers[l];
        b[l].resize(layer.GetOutputSize());
        for (int n = 0; n < layer.GetOutputSize(); ++n) b[l][n] = layer.bias(n);
    }
    st.SetAllBiases(b);
}

// Copy a dynamic float network's weights+biases into the fixed-point net.
static void copy_to_fixed(MLP<float> & dyn, StaticFix & st) {
    auto fw = dyn.GetWeights();
    StaticFix::mlp_weights w(fw.size());
    for (std::size_t l = 0; l < fw.size(); ++l) {
        w[l].resize(fw[l].size());
        for (std::size_t n = 0; n < fw[l].size(); ++n) {
            w[l][n].resize(fw[l][n].size());
            for (std::size_t j = 0; j < fw[l][n].size(); ++j) w[l][n][j] = Fix(fw[l][n][j]);
        }
    }
    st.SetAllWeights(w);
    StaticFix::mlp_biases b(dyn.GetNumLayers());
    for (std::size_t l = 0; l < dyn.GetNumLayers(); ++l) {
        auto & layer = dyn.m_layers[l];
        b[l].resize(layer.GetOutputSize());
        for (int n = 0; n < layer.GetOutputSize(); ++n) b[l][n] = Fix(layer.bias(n));
    }
    st.SetAllBiases(b);
}

static void hb(const char* msg) { printf("[boot] %s\n", msg); fflush(stdout); }

int main() {
    stdio_init_all();
    sleep_ms(3000);   // let USB CDC host attach

    // Boot heartbeat: confirms USB printing works and pinpoints where (if) it dies.
    for (int i = 0; i < 10; ++i) { printf("[boot] alive %d\n", i); fflush(stdout); sleep_ms(200); }

    const uint32_t sys_hz = clock_get_hz(clk_sys);
    hb("clocks read");

    // Dynamic network (heap-backed) with random weights.
    hb("constructing dynamic MLP");
    MLP<float> dyn({NI, NH1, NH2, NO},
                   {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::RELU,
                    ACTIVATION_FUNCTIONS::LINEAR});
    hb("randomising weights");
    dyn.RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
    hb("copying to static");
    copy_weights_biases(dyn, g_static);   // identical weights in the static net
    copy_to_fixed(dyn, g_fixed);          // ...and in the fixed-point net
    hb("entering benchmark loop");

    // I/O buffers (allocated once, reused — fair to both).
    std::vector<float> din(NI, 0.37f), dout;
    std::array<float, NI> sin_{};
    std::array<float, NO> sout{};
    for (std::size_t i = 0; i < NI; ++i) sin_[i] = 0.37f;

    // Fixed-point I/O buffers.
    std::array<Fix, NI> fin{};
    std::array<Fix, NO> fout{};
    for (std::size_t i = 0; i < NI; ++i) fin[i] = Fix(0.37f);

    // Error vector for the training-step benchmark.
    std::vector<float> derr(NO, 0.1f);
    std::array<float, NO> serr{};
    for (std::size_t i = 0; i < NO; ++i) serr[i] = 0.1f;
    std::array<Fix, NO> ferr{};
    for (std::size_t i = 0; i < NO; ++i) ferr[i] = Fix(0.1f);

    const int N = 2000;

    while (true) {
        printf("\n==== MEMLP dynamic vs static benchmark ====\n");
        printf("clk_sys = %lu Hz | arch {%u,%u,%u,%u} RELU,RELU,LINEAR | iters = %d\n",
               (unsigned long)sys_hz, (unsigned)NI, (unsigned)NH1, (unsigned)NH2,
               (unsigned)NO, N);
        printf("sizeof(MLP<float>)=%u B (+heap)   sizeof(StaticMLP)=%u B (all in .bss)\n",
               (unsigned)sizeof(MLP<float>), (unsigned)sizeof(Static));

        // Fresh, sane weights each cycle: the training-step benchmark below runs
        // thousands of identical-gradient ApplyLoss calls and deliberately drives
        // the weights to divergence (float -> inf), so we re-seed here and measure
        // accuracy *before* that, on weights identical across all three networks.
        dyn.RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
        copy_weights_biases(dyn, g_static);
        copy_to_fixed(dyn, g_fixed);

        // ── Inference (warm-up doubles as the accuracy probe on fresh weights) ──
        dyn.GetOutput(din, &dout);          // warm
        g_static.GetOutput(sin_, sout);
        g_fixed.GetOutput(fin.data(), fout.data());
        double max_abs_err = 0.0;
        for (std::size_t o = 0; o < NO; ++o)
            max_abs_err = std::max(max_abs_err, std::fabs((double)sout[o] - (double)fout[o].to_float()));
        printf("[accuracy]   max|float - fixed| over %u outputs = %.5f  (checksum %.4f/%.4f)\n",
               (unsigned)NO, max_abs_err, sout[0], fout[0].to_float());

        uint64_t t0 = time_us_64();
        for (int i = 0; i < N; ++i) dyn.GetOutput(din, &dout);
        uint64_t t1 = time_us_64();
        for (int i = 0; i < N; ++i) g_static.GetOutput(sin_, sout);
        uint64_t t2 = time_us_64();
        for (int i = 0; i < N; ++i) g_fixed.GetOutput(fin.data(), fout.data());
        uint64_t t3 = time_us_64();

        double dyn_us = double(t1 - t0) / N;
        double st_us  = double(t2 - t1) / N;
        double fx_us  = double(t3 - t2) / N;
        printf("\n[inference]  dynamic = %.3f us   static(float) = %.3f us   static(fixed) = %.3f us\n",
               dyn_us, st_us, fx_us);
        printf("             fixed speedup vs dynamic = %.2fx, vs static-float = %.2fx\n",
               dyn_us / fx_us, st_us / fx_us);

        // ── Training step (ApplyLoss: forward + immediate backprop) ──
        dyn.ApplyLoss(din, derr, 0.01f);    // warm
        g_static.ApplyLoss(sin_.data(), serr.data(), 0.01f);
        g_fixed.ApplyLoss(fin.data(), ferr.data(), 0.01f);
        uint64_t t4 = time_us_64();
        for (int i = 0; i < N; ++i) dyn.ApplyLoss(din, derr, 0.01f);
        uint64_t t5 = time_us_64();
        for (int i = 0; i < N; ++i) g_static.ApplyLoss(sin_.data(), serr.data(), 0.01f);
        uint64_t t6 = time_us_64();
        for (int i = 0; i < N; ++i) g_fixed.ApplyLoss(fin.data(), ferr.data(), 0.01f);
        uint64_t t7 = time_us_64();

        double dyn_tr = double(t5 - t4) / N;
        double st_tr  = double(t6 - t5) / N;
        double fx_tr  = double(t7 - t6) / N;
        printf("[train step] dynamic = %.3f us   static(float) = %.3f us   static(fixed) = %.3f us\n",
               dyn_tr, st_tr, fx_tr);
        printf("             fixed speedup vs dynamic = %.2fx, vs static-float = %.2fx\n",
               dyn_tr / fx_tr, st_tr / fx_tr);

        sleep_ms(1500);
    }
}
