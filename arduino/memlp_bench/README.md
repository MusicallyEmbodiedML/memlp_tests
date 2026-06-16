# memlp_bench (arduino-pico)

On-device benchmark comparing the dynamic `MLP<float>` against the compile-time
`StaticMLP`, built with the **arduino-pico** core so it runs on boards that are
programmed that way (e.g. the **MEMLNaut**, RP2350).

Only `memlp_bench.ino` is tracked. The library sources it needs are **copied in**
from the `memlp` submodule by `populate.sh` (and gitignored — the submodule is
the single source of truth).

## Build & flash

```bash
./populate.sh                                              # copy memlp sources in
arduino-cli compile --fqbn rp2040:rp2040:memlnaut .        # or your board's FQBN
# put the board in BOOTSEL, then either:
#   cp build/*.uf2  /media/$USER/RP2350/                   # drag-drop flash
#   arduino-cli upload --fqbn rp2040:rp2040:memlnaut -p /dev/ttyACM0 .
```

Open the serial monitor at 115200. The board defines `ARM_MATH_CM33=1`, so both
networks use the CMSIS-DSP `arm_dot_prod_f32` path on the RP2350's FPU.

## Measured (MEMLNaut RP2350 @150 MHz, arch {8,16,16,4})

| Op | Dynamic | Static | Speedup |
|----|---------|--------|---------|
| inference  | 42.2 µs | 32.7 µs | 1.29× |
| train step | 194 µs  | 94.5 µs | 2.05× |

(For reference, RP2040 @125 MHz soft-float: ~1.03× inference, ~1.05× train — the
static win grows once an FPU makes the float MACs cheap.)
