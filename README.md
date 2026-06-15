# memlp_tests

Unit / characterisation tests for the [MEMLP](https://github.com/MusicallyEmbodiedML/memlp)
machine-learning library. The library itself is pulled in as a git submodule,
so these tests sit *outside* the library repo and pin its current public
behaviour — handy as a regression net before refactoring.

The same test sources run in two places:

* **Host** — a command-line runner (CI / ctest friendly).
* **Embedded** — firmware for the **RP2040** and **RP2350** built against the
  Raspberry Pi **Pico SDK** (not arduino-pico), streaming results over USB serial.

## Layout

```
memlp_tests/
├── memlp/                 # git submodule -> the library under test
├── tests/
│   ├── test_framework.h   # tiny zero-dependency framework (TEST/CHECK + runner)
│   ├── test_utils.cpp     # activation fns, softmax, argmax, RNG
│   ├── test_loss.cpp      # MSE + categorical cross-entropy (values + gradients)
│   ├── test_layer.cpp     # forward pass, weight get/set, Polyak update, init
│   └── test_mlp.cpp       # geometry, inference, (de)serialise, train, save/load
├── host_main.cpp          # host entry point
├── pico_main.cpp          # RP2040/RP2350 entry point (USB stdio + getentropy)
├── pico_sdk_import.cmake
└── CMakeLists.txt
```

## Getting the code

```bash
git clone <this-repo> memlp_tests
cd memlp_tests
git submodule update --init --recursive
```

## Host build & run

```bash
cmake -S . -B build
cmake --build build
./build/memlp_tests          # exits non-zero if any test fails
# or:
ctest --test-dir build --output-on-failure
```

## Pico SDK build (RP2040 / RP2350)

Requires `PICO_SDK_PATH` set and the `arm-none-eabi` toolchain installed.
The board (and therefore the chip) is selected with `-DPICO_BOARD`:

```bash
# RP2040 (e.g. Raspberry Pi Pico)
cmake -S . -B build-rp2040 -DMEMLP_TESTS_PICO=ON -DPICO_BOARD=pico
cmake --build build-rp2040

# RP2350 (e.g. Raspberry Pi Pico 2)
cmake -S . -B build-rp2350 -DMEMLP_TESTS_PICO=ON -DPICO_BOARD=pico2
cmake --build build-rp2350
```

Flash the resulting `build-*/memlp_tests.uf2` (drag-and-drop in BOOTSEL mode,
or `picotool load -f memlp_tests.uf2`), then connect to the USB serial port:

```bash
minicom -b 115200 -D /dev/ttyACM0      # or: screen /dev/ttyACM0 115200
```

You'll see each test's result followed by a repeating
`RESULT: ALL TESTS PASSED` / `RESULT: N TEST(S) FAILED` line.

> The file-based `SaveMLPNetwork`/`LoadMLPNetwork` test only compiles where
> `ENABLE_SAVE` is defined (host), so it is skipped automatically on bare metal.

## Adding a test

Drop a `TEST(suite, name) { ... }` block into any `tests/*.cpp` (or a new file
added to `MEMLP_TEST_SOURCES` in `CMakeLists.txt`). It self-registers — no need
to touch the entry points.

```cpp
#include "test_framework.h"
#include "MLP.h"

TEST(mlp, my_new_check) {
    MLP<float> net({2, 1}, {ACTIVATION_FUNCTIONS::LINEAR}, loss::LOSS_FUNCTIONS::LOSS_MSE, true, 0.5f);
    std::vector<float> out;
    net.GetOutput({1.0f, 1.0f}, &out);
    CHECK_CLOSE(out[0], 1.0f, 1e-6);
}
```
