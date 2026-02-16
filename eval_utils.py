import numpy as np
import pandas as pd
from arch import arch_model


def apply_generation_controls(
    eps,
    desired_volatility=1.0,
    desired_trend=0.0,
    desired_fat_tails=1.0,
    desired_momentum=0.0,
):
    eps = eps.copy()
    _, seq_len = eps.shape

    eps *= desired_volatility

    drift = desired_trend * np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    eps += drift[None, :]

    eps = np.sign(eps) * (np.abs(eps) + 1e-8) ** desired_fat_tails

    if desired_momentum > 0.0:
        out = np.empty_like(eps)
        out[:, 0] = eps[:, 0]
        for t in range(1, seq_len):
            out[:, t] = desired_momentum * out[:, t - 1] + (1.0 - desired_momentum) * eps[:, t]
        eps = out

    return eps


def garch_sigma_forecast_from_csv(csv_path, seq_len):
    df = pd.read_csv(csv_path)
    close = df["Close"].astype(float).values
    r = np.diff(np.log(close))
    am = arch_model(r * 100, vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    f = res.forecast(horizon=seq_len, reindex=False)
    sigma = np.sqrt(f.variance.values[-1]) / 100.0
    return sigma.astype(np.float32), float(close[-1])


def reconstruct_returns_and_prices(eps, sigma, start_price):
    ret = eps * sigma[None, :]
    log_price_paths = np.log(start_price) + np.cumsum(ret, axis=1)
    prices = np.exp(log_price_paths)
    return ret, prices
