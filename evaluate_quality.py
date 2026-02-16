import argparse
import json
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import StandardScaler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Real data CSV with Close column")
    ap.add_argument("--npz", required=True, help="Synthetic output npz (e.g. synthetic_paths.npz)")
    ap.add_argument("--max_lag", type=int, default=10)
    ap.add_argument("--max_real_windows", type=int, default=2000)
    ap.add_argument("--max_synth_windows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None, help="Optional path to save metrics json")
    return ap.parse_args()


def safe_float(x):
    return float(np.asarray(x))


def summary_stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = x.mean()
    sd = x.std() + 1e-12
    z = (x - mu) / sd
    return {
        "mean": safe_float(mu),
        "std": safe_float(sd),
        "skew": safe_float(np.mean(z ** 3)),
        "kurtosis": safe_float(np.mean(z ** 4)),
        "q01": safe_float(np.quantile(x, 0.01)),
        "q05": safe_float(np.quantile(x, 0.05)),
        "q50": safe_float(np.quantile(x, 0.50)),
        "q95": safe_float(np.quantile(x, 0.95)),
        "q99": safe_float(np.quantile(x, 0.99)),
    }


def var_es(x, alpha):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x <= q]
    es = tail.mean() if len(tail) > 0 else q
    return {"VaR": safe_float(q), "ES": safe_float(es)}


def acf(x, max_lag=10):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    out = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            out.append(0.0)
            continue
        out.append(safe_float(np.dot(x[:-lag], x[lag:]) / denom))
    return out


def fit_real_and_build_windows(close, seq_len):
    r = np.diff(np.log(close))
    am = arch_model(r * 100, vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    sigma = res.conditional_volatility / 100.0
    eps = r / sigma
    eps = StandardScaler().fit_transform(eps.reshape(-1, 1)).flatten()

    X, C = [], []
    for i in range(len(eps) - seq_len):
        X.append(eps[i : i + seq_len])
        C.append(np.mean(sigma[i : i + seq_len]))

    X = np.array(X, dtype=np.float32)
    C = np.array(C, dtype=np.float32)
    C = np.digitize(C, np.quantile(C, [0.33, 0.66])).astype(np.int64)
    return r.astype(np.float32), X, C


def regime_summary(eps_windows, cond, n_classes=3):
    out = {}
    for cls in range(n_classes):
        idx = np.where(cond == cls)[0]
        if len(idx) == 0:
            out[f"class_{cls}"] = {"count": 0, "eps_std": None, "mean_abs_eps": None}
            continue
        v = eps_windows[idx].reshape(-1)
        out[f"class_{cls}"] = {
            "count": int(len(idx)),
            "eps_std": safe_float(np.std(v)),
            "mean_abs_eps": safe_float(np.mean(np.abs(v))),
        }
    return out


def nearest_neighbor_distance(synth_windows, real_windows, max_synth=500, max_real=2000, seed=42):
    rng = np.random.default_rng(seed)
    s = np.asarray(synth_windows, dtype=np.float32)
    r = np.asarray(real_windows, dtype=np.float32)

    if len(s) == 0 or len(r) == 0:
        return {"min": None, "p05": None, "median": None}

    if len(s) > max_synth:
        s = s[rng.choice(len(s), size=max_synth, replace=False)]
    if len(r) > max_real:
        r = r[rng.choice(len(r), size=max_real, replace=False)]

    s = s.reshape(len(s), -1)
    r = r.reshape(len(r), -1)
    r2 = np.sum(r * r, axis=1)
    mins = np.full(len(s), np.inf, dtype=np.float64)

    chunk = 128
    for i in range(0, len(s), chunk):
        sb = s[i : i + chunk]
        sb2 = np.sum(sb * sb, axis=1, keepdims=True)
        d2 = sb2 + r2[None, :] - 2.0 * (sb @ r.T)
        d2 = np.maximum(d2, 0.0)
        mins[i : i + len(sb)] = np.sqrt(np.min(d2, axis=1))

    return {
        "min": safe_float(np.min(mins)),
        "p05": safe_float(np.quantile(mins, 0.05)),
        "median": safe_float(np.median(mins)),
    }


def print_block(title, d):
    print(f"\n{title}")
    for k, v in d.items():
        print(f"  {k}: {v}")


def main():
    args = parse_args()
    syn = np.load(args.npz)
    if "eps" not in syn:
        raise ValueError("Synthetic npz must contain key 'eps'")

    syn_eps = syn["eps"].astype(np.float32)
    seq_len = syn_eps.shape[1]
    syn_cond = syn["cond"].astype(np.int64) if "cond" in syn else np.zeros(len(syn_eps), dtype=np.int64)
    syn_returns = syn["returns"].astype(np.float32) if "returns" in syn else None

    df = pd.read_csv(args.csv)
    close = df["Close"].astype(float).values
    real_returns, real_eps_windows, real_cond = fit_real_and_build_windows(close, seq_len=seq_len)

    # Distribution + tail metrics (returns if available, otherwise eps).
    if syn_returns is not None:
        real_vec = real_returns
        syn_vec = syn_returns.reshape(-1)
        metric_target = "returns"
    else:
        real_vec = real_eps_windows.reshape(-1)
        syn_vec = syn_eps.reshape(-1)
        metric_target = "eps (returns unavailable in npz)"

    metrics = {
        "target": metric_target,
        "summary_real": summary_stats(real_vec),
        "summary_synth": summary_stats(syn_vec),
        "var_es_95_real": var_es(real_vec, 0.95),
        "var_es_95_synth": var_es(syn_vec, 0.95),
        "var_es_99_real": var_es(real_vec, 0.99),
        "var_es_99_synth": var_es(syn_vec, 0.99),
        "acf_real": acf(real_vec, args.max_lag),
        "acf_synth": acf(syn_vec, args.max_lag),
        "acf_abs_real": acf(np.abs(real_vec), args.max_lag),
        "acf_abs_synth": acf(np.abs(syn_vec), args.max_lag),
        "regime_real": regime_summary(real_eps_windows, real_cond),
        "regime_synth": regime_summary(syn_eps, syn_cond),
        "nearest_neighbor_l2": nearest_neighbor_distance(
            syn_eps,
            real_eps_windows,
            max_synth=args.max_synth_windows,
            max_real=args.max_real_windows,
            seed=args.seed,
        ),
    }

    print("Quality Evaluation Summary")
    print(f"real_csv: {args.csv}")
    print(f"synthetic_npz: {args.npz}")
    print(f"seq_len: {seq_len}")
    print_block("summary_real", metrics["summary_real"])
    print_block("summary_synth", metrics["summary_synth"])
    print_block("var_es_95_real", metrics["var_es_95_real"])
    print_block("var_es_95_synth", metrics["var_es_95_synth"])
    print_block("var_es_99_real", metrics["var_es_99_real"])
    print_block("var_es_99_synth", metrics["var_es_99_synth"])
    print(f"\nacf_real: {metrics['acf_real']}")
    print(f"acf_synth: {metrics['acf_synth']}")
    print(f"acf_abs_real: {metrics['acf_abs_real']}")
    print(f"acf_abs_synth: {metrics['acf_abs_synth']}")
    print_block("regime_real", metrics["regime_real"])
    print_block("regime_synth", metrics["regime_synth"])
    print_block("nearest_neighbor_l2", metrics["nearest_neighbor_l2"])

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics JSON to {args.out_json}")


if __name__ == "__main__":
    main()
