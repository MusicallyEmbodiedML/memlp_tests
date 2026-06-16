/**
 * @file pico_main.cpp
 * @brief Bare-metal entry point for RP2040 / RP2350 (Raspberry Pi Pico SDK).
 *
 * Runs the same test suites as the host build, streaming results over USB CDC
 * serial (115200 8N1 by default). Connect with e.g. `minicom -b 115200 -D
 * /dev/ttyACM0` or `screen /dev/ttyACM0 115200` and reset the board.
 *
 * Build: see README.md — configure with -DMEMLP_TESTS_PICO=ON and
 * -DPICO_BOARD=pico (RP2040) or -DPICO_BOARD=pico2 (RP2350).
 */

#include "pico/stdlib.h"
#include "pico/rand.h"
#include "test_framework.h"
#include "StaticMLP.h"

// File-scope static network: demonstrates the whole point of StaticMLP — all of
// its weights/biases/buffers live in .bss (allocated at link time, zeroed at
// boot), with zero heap use. Inspect memlp_tests.elf.map for g_demo_net.
static smlp::StaticMLP<float,
                       smlp::Layout<8, 16, 8, 4>,
                       smlp::Activations<ACTIVATION_FUNCTIONS::RELU,
                                         ACTIVATION_FUNCTIONS::RELU,
                                         ACTIVATION_FUNCTIONS::LINEAR>> g_demo_net;

/**
 * The MLP constructor instantiates std::random_device, whose libstdc++
 * implementation calls getentropy(). Bare-metal newlib doesn't provide one, so
 * we back it with the hardware RNG.
 */
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

int main() {
    stdio_init_all();
    // Give a USB CDC host a few seconds to attach before we start printing.
    sleep_ms(3000);

    printf("\n\n");
    int failed = tf::run_all_tests();

    // Exercise the file-scope static net so the linker keeps it (proving the
    // .bss-resident, heap-free network actually runs on device).
    g_demo_net.SetSeed(get_rand_32());
    g_demo_net.RandomiseWeightsAndBiasesLin(-0.5f, 0.5f, -0.1f, 0.1f);
    std::array<float, 8> demo_in{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::array<float, 4> demo_out{};
    g_demo_net.GetOutput(demo_in, demo_out);
    printf("g_demo_net out: %.3f %.3f %.3f %.3f\n",
           demo_out[0], demo_out[1], demo_out[2], demo_out[3]);

    // Park here, repeating the verdict so it's visible whenever you connect.
    while (true) {
        if (failed == 0) {
            printf("RESULT: ALL TESTS PASSED\n");
        } else {
            printf("RESULT: %d TEST(S) FAILED\n", failed);
        }
        sleep_ms(5000);
    }
}
