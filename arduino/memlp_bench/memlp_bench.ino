/**
 * memlp_bench.ino — on-device benchmark for the MEMLNaut (RP2350, arduino-pico).
 *
 * Compares the dynamic MLP<float> against the compile-time StaticMLP with an
 * identical architecture and identical weights. Built with the MEMLNaut board
 * definition, so it uses the same toolchain as the real firmware — including
 * -DARM_MATH_CM33=1, so BOTH networks take the CMSIS-DSP arm_dot_prod_f32 path
 * on the M33's hardware FPU.
 *
 * Build/flash:
 *   arduino-cli compile --fqbn rp2040:rp2040:memlnaut arduino/memlp_bench
 *   (then upload the .uf2, or arduino-cli upload)
 *
 * Output streams over USB serial at 115200.
 */

#include <Arduino.h>
// arduino-pico defines abs()/round() (and min/max) as C-style macros on the
// RISC-V core, which collide with the STL <random>/<cmath> headers the MLP
// pulls in. Drop the macros before including the library (it uses std:: calls).
#undef abs
#undef round
#undef min
#undef max

#include <array>
#include <vector>

#include "MLP.h"
#include "StaticMLP.h"

// Which dot-product path the layers actually compile to (matches Layer.h guard).
#if defined(ARM_MATH_CM33) && defined(__arm__)
  #define MEMLP_DOTPATH "CMSIS-DSP arm_dot_prod_f32 (FPU)"
#else
  #define MEMLP_DOTPATH "scalar"
#endif

static constexpr size_t NI = 8, NH1 = 16, NH2 = 16, NO = 4;

using Static = smlp::StaticMLP<float,
                               smlp::Layout<NI, NH1, NH2, NO>,
                               smlp::Activations<ACTIVATION_FUNCTIONS::RELU,
                                                 ACTIVATION_FUNCTIONS::RELU,
                                                 ACTIVATION_FUNCTIONS::LINEAR>>;
static Static g_static;                 // lives in .bss, zero heap

static MLP<float>* g_dyn = nullptr;     // heap-backed dynamic network

static const int N = 5000;

static void copy_weights_biases(MLP<float>& dyn, Static& st) {
    st.SetAllWeights(dyn.GetWeights());
    Static::mlp_biases b(dyn.GetNumLayers());
    for (size_t l = 0; l < dyn.GetNumLayers(); ++l) {
        auto& layer = dyn.m_layers[l];
        b[l].resize(layer.GetOutputSize());
        for (int n = 0; n < layer.GetOutputSize(); ++n) b[l][n] = layer.bias(n);
    }
    st.SetAllBiases(b);
}

void setup() {
    Serial.begin(115200);
    unsigned long t0 = millis();
    while (!Serial && (millis() - t0) < 4000) { /* wait for USB host */ }

    g_dyn = new MLP<float>({NI, NH1, NH2, NO},
                           {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::RELU,
                            ACTIVATION_FUNCTIONS::LINEAR});
    g_dyn->RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
    copy_weights_biases(*g_dyn, g_static);
}

void loop() {
    static std::vector<float> din(NI, 0.37f), dout;
    static std::array<float, NI> sin_{};
    static std::array<float, NO> sout{};
    for (size_t i = 0; i < NI; ++i) sin_[i] = 0.37f;

    std::vector<float> derr(NO, 0.1f);
    std::array<float, NO> serr{};
    for (size_t i = 0; i < NO; ++i) serr[i] = 0.1f;

    // Fresh identical weights each pass (the train-step bench below mutates them).
    g_dyn->RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
    copy_weights_biases(*g_dyn, g_static);

    Serial.println();
    Serial.println("==== MEMLNaut (RP2350) dynamic vs static benchmark ====");
    Serial.printf("f_cpu=%lu Hz | arch {%u,%u,%u,%u} RELU,RELU,LINEAR | iters=%d | dot path: %s\n",
                  (unsigned long)F_CPU, (unsigned)NI, (unsigned)NH1, (unsigned)NH2,
                  (unsigned)NO, N, MEMLP_DOTPATH);
    Serial.printf("sizeof(MLP<float>)=%u B (+heap)   sizeof(StaticMLP)=%u B (.bss)\n",
                  (unsigned)sizeof(MLP<float>), (unsigned)sizeof(Static));

    // ── Inference ──
    g_dyn->GetOutput(din, &dout);
    g_static.GetOutput(sin_, sout);
    unsigned long a0 = micros();
    for (int i = 0; i < N; ++i) g_dyn->GetOutput(din, &dout);
    unsigned long a1 = micros();
    for (int i = 0; i < N; ++i) g_static.GetOutput(sin_, sout);
    unsigned long a2 = micros();

    double dyn_us = double(a1 - a0) / N;
    double st_us  = double(a2 - a1) / N;
    Serial.printf("[inference]  dynamic=%.3f us/call   static=%.3f us/call   speedup=%.2fx\n",
                  dyn_us, st_us, dyn_us / st_us);

    // ── Training step (ApplyLoss: forward + immediate backprop) ──
    g_dyn->ApplyLoss(din, derr, 0.01f);
    g_static.ApplyLoss(sin_.data(), serr.data(), 0.01f);
    unsigned long b0 = micros();
    for (int i = 0; i < N; ++i) g_dyn->ApplyLoss(din, derr, 0.01f);
    unsigned long b1 = micros();
    for (int i = 0; i < N; ++i) g_static.ApplyLoss(sin_.data(), serr.data(), 0.01f);
    unsigned long b2 = micros();

    double dyn_tr = double(b1 - b0) / N;
    double st_tr  = double(b2 - b1) / N;
    Serial.printf("[train step] dynamic=%.3f us/call   static=%.3f us/call   speedup=%.2fx\n",
                  dyn_tr, st_tr, dyn_tr / st_tr);

    delay(2000);
}
