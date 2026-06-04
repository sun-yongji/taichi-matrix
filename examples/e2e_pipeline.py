"""TaiChi Matrix end-to-end pipeline demonstration.

Runs a 32×128 synthetic tensor through Router→MTP→HexAttention→Quant→Correct.
Works even with partial installs — gracefully skips missing modules.
"""

import sys
import time

import numpy as np

# Add parent to path for local dev
sys.path.insert(0, "..")

from taichi_matrix import TaiChiPipeline


def print_sep(title: str) -> None:
    print(f"\n{'-' * 20} {title} {'-' * 50}")


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("  TaiChi Matrix · End-to-End C6 Pipeline")
    print("  CCF OSS 2026 · Apache 2.0")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0. Status
    # ------------------------------------------------------------------
    pipeline = TaiChiPipeline(rng=rng)
    status = pipeline.status()
    print(f"\nModule status: {pipeline}")
    for name, ok in status.items():
        label = "[OK] installed" if ok else "[  ] not installed"
        print(f"  {name:12s} {label}")

    # ------------------------------------------------------------------
    # 1. Input
    # ------------------------------------------------------------------
    print_sep("INPUT")
    x = rng.normal(0, 1, (32, 128))
    print(f"Shape: {x.shape}    Mean: {x.mean():.3f}    Std: {x.std():.3f}")

    # ------------------------------------------------------------------
    # 2. Run pipeline
    # ------------------------------------------------------------------
    print_sep("PIPELINE")
    t0 = time.perf_counter()
    result = pipeline.run(x)
    elapsed = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # 3. Results
    # ------------------------------------------------------------------
    print_sep("RESULTS")

    # Router
    print(f"\n>> M1 Router")
    print(f"   Mode:       {result.route_mode}")
    print(f"   Weights:    E1={result.route_weights[0]:.3f}  "
          f"E2={result.route_weights[1]:.3f}  "
          f"E3={result.route_weights[2]:.3f}")

    # MTP
    if result.mtp_output is not None:
        print(f"\n>> M2 MTP")
        print(f"   Output:     {result.mtp_output.shape}")
        print(f"   Range:      [{result.mtp_output.min():.3f}, {result.mtp_output.max():.3f}]")
    else:
        print(f"\n>> M2 MTP  (skipped)")

    # HexAttention
    if result.attention_output is not None:
        print(f"\n>> M3 HexAttention")
        print(f"   Output:     {result.attention_output.shape}")
        print(f"   Entropy d:  {result.attention_delta:.2f} (vs standard)")
    else:
        print(f"\n>> M3 HexAttention  (skipped)")

    # Quant
    if result.quant_report:
        qr = result.quant_report
        print(f"\n>> M4 Quant")
        print(f"   Ratio:      {qr.get('compression_ratio', 1):.1f}x")
        print(f"   Fidelity:   {qr.get('fidelity', 1):.1%}")
    else:
        print(f"\n>> M4 Quant  (skipped)")

    # Correct
    if result.corrected is not None:
        print(f"\n>> M5 Correct")
        print(f"   Corrected:  {result.corrected.shape}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Residue -:  {result.residue_reduction:.1%}")
    else:
        print(f"\n>> M5 Correct  (skipped)")

    # ------------------------------------------------------------------
    # 4. Timing
    # ------------------------------------------------------------------
    print_sep("TIMING")
    total_ms = sum(result.timings.values())
    print(f"\n{'Stage':<15s} {'ms':>8s}  {'Bar'}")

    # Find max for scaling
    max_t = max(result.timings.values()) if result.timings else 1
    bar_width = 30

    for stage, ms in result.timings.items():
        n_bars = int(ms / max_t * bar_width) if max_t > 0 else 0
        bar = "#" * n_bars + "-" * (bar_width - n_bars)
        pct = ms / total_ms * 100 if total_ms > 0 else 0
        print(f"  {stage:<15s} {ms:8.2f}  {bar} {pct:5.1f}%")

    print(f"  {'─' * 56}")
    print(f"  {'Total':15s} {total_ms:8.2f} ms")
    print(f"  {'Wall clock':15s} {elapsed:8.2f} ms")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  Pipeline complete · {len(result.modules_available)}/5 modules active")
    print(f"  Total: {total_ms:.2f} ms computational | {elapsed:.2f} ms wall")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = main()
    # Demonstrate PipelineResult is picklable
    import pickle
    data = pickle.dumps(result)
    print(f"\n(Result serialized: {len(data)} bytes)")
