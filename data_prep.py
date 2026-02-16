import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import StandardScaler
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq_len', type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    close = df['Close'].astype(float).values
    r = np.diff(np.log(close))

    am = arch_model(r*100, vol='GARCH', p=1, q=1)
    res = am.fit(disp='off')
    sigma = res.conditional_volatility/100
    eps = r/sigma

    scaler = StandardScaler()
    eps = scaler.fit_transform(eps.reshape(-1,1)).flatten()

    X, C = [], []
    for i in range(len(eps)-args.seq_len):
        X.append(eps[i:i+args.seq_len])
        C.append(np.mean(sigma[i:i+args.seq_len]))

    X = np.array(X, dtype=np.float32)
    C = np.digitize(C, np.quantile(C,[0.33,0.66]))

    np.savez(args.out, eps=X, cond=C)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
