import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Generator
from eval_args import parse_args
from eval_utils import (
    apply_generation_controls,
    garch_sigma_forecast_from_csv,
    reconstruct_returns_and_prices,
    build_synthetic_ohlcv,
    save_ohlcv_frames,
)

def main():
    args = parse_args()
    conds = [int(x.strip()) for x in args.conds.split(",") if x.strip() != ""]
    if len(conds) == 0:
        raise ValueError("--conds must contain at least one class id")

    conds = (conds * ((args.n_samples + len(conds) - 1) // len(conds)))[:args.n_samples]
    c = torch.tensor(conds, dtype=torch.long)

    G = Generator(args.z_dim, args.hidden_dim, args.n_layers, args.n_classes)
    G.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    G.eval()

    z = torch.randn(args.n_samples, args.seq_len, args.z_dim)
    eps = G(z, c).detach().numpy().squeeze(-1)
    eps = apply_generation_controls(
        eps,
        desired_volatility=args.desired_volatility,
        desired_trend=args.desired_trend,
        desired_fat_tails=args.desired_fat_tails,
        desired_momentum=args.desired_momentum,
    )

    for i in range(args.n_samples):
        plt.plot(eps[i], label=f"sample_{i}_cond_{conds[i]}")
    plt.title("Synthetic Standardized Residuals")
    plt.legend()
    plt.show()

    if args.csv is not None:
        sigma, last_close = garch_sigma_forecast_from_csv(args.csv, args.seq_len)
        start_price = args.start_price if args.start_price is not None else last_close
        returns, prices = reconstruct_returns_and_prices(eps, sigma, start_price)

        np.savez(
            args.out_npz,
            eps=eps.astype(np.float32),
            returns=returns.astype(np.float32),
            prices=prices.astype(np.float32),
            sigma=sigma.astype(np.float32),
            cond=np.array(conds, dtype=np.int64),
            start_price=np.array(start_price, dtype=np.float32),
        )
        print(f"Saved reconstructed outputs to {args.out_npz}")

        ohlcv_frames = build_synthetic_ohlcv(
            prices=prices,
            returns=returns,
            volume_base=args.volume_base,
            volume_alpha=args.volume_alpha,
            seed=args.seed,
        )
        ohlcv_paths = save_ohlcv_frames(ohlcv_frames, args.out_ohlcv_dir)
        print(f"Saved OHLCV CSV files to {args.out_ohlcv_dir} ({len(ohlcv_paths)} files)")

        plt.figure()
        for i in range(args.n_samples):
            plt.plot(prices[i], label=f"sample_{i}_cond_{conds[i]}")
        plt.title("Reconstructed Synthetic Price Paths")
        plt.legend()
        plt.show()
    else:
        print("Skipped returns/price reconstruction: pass --csv to fit GARCH and forecast sigma.")

if __name__ == "__main__":
    main()
