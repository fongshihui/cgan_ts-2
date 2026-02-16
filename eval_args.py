import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="generator.pt")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_classes", type=int, default=3)
    ap.add_argument("--conds", default="0,1,2,0,1")

    ap.add_argument("--desired_volatility", type=float, default=1.0)
    ap.add_argument("--desired_trend", type=float, default=0.0)
    ap.add_argument("--desired_fat_tails", type=float, default=1.0)
    ap.add_argument("--desired_momentum", type=float, default=0.0)

    ap.add_argument("--csv", default=None, help="CSV with Close column for volatility fitting")
    ap.add_argument("--start_price", type=float, default=None, help="Initial price for reconstruction")
    ap.add_argument("--out_npz", default="synthetic_paths.npz", help="Output file for eps/returns/prices")
    ap.add_argument("--out_ohlcv_dir", default="synthetic_ohlcv", help="Directory for per-sample OHLCV CSV files")
    ap.add_argument("--volume_base", type=float, default=1_000_000.0, help="Base synthetic volume level")
    ap.add_argument("--volume_alpha", type=float, default=30.0, help="Volume sensitivity to absolute returns")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for OHLCV synthesis")
    ap.add_argument("--run_quality_eval", action="store_true", help="Run quality metrics after saving npz")
    ap.add_argument("--quality_out_json", default="quality_report.json", help="Quality metrics output JSON path")
    ap.add_argument("--quality_max_lag", type=int, default=10, help="Max lag for ACF quality metrics")
    return ap.parse_args()
