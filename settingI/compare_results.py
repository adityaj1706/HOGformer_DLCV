"""Compare two evaluate.py result files with paired statistics.

  python compare_results.py results_baseline.json results_softvote.json [--md table.md]

For every dataset the same images were scored by both models, so we report the mean per-image
difference (proposed - baseline) with a 95% bootstrap confidence interval and the share of
images that improved. A CI that excludes 0 is stronger evidence than two averages alone.
"""
import argparse, json
import numpy as np


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline'); ap.add_argument('proposed'); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    A, B = json.load(open(a.baseline)), json.load(open(a.proposed))
    rows = ['| Dataset | n | PSNR base | PSNR prop | dPSNR (95% CI) | images improved | SSIM base | SSIM prop |',
            '|---|---|---|---|---|---|---|---|']
    for ds in A['datasets']:
        if ds not in B['datasets']:
            continue
        pa, pb = A['datasets'][ds]['per_image'], B['datasets'][ds]['per_image']
        keys = sorted(set(pa) & set(pb))
        assert keys, f'no common images in {ds}'
        x = np.array([pa[k] for k in keys]); y = np.array([pb[k] for k in keys])
        d = y[:, 0] - x[:, 0]
        lo, hi = boot_ci(d)
        rows.append(f"| {ds} | {len(keys)} | {x[:, 0].mean():.3f} | {y[:, 0].mean():.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] "
                    f"| {100 * (d > 0).mean():.0f}% | {x[:, 1].mean():.4f} | {y[:, 1].mean():.4f} |")
    rows.append('')
    rows.append(f"Params: {A['name']} {A['params_M']:.2f} M vs {B['name']} {B['params_M']:.2f} M; "
                f"GMACs@128: {A['GMACs_128x128']} vs {B['GMACs_128x128']} (fvcore does not count every op); "
                f"latency {A.get('latency_ms')} vs {B.get('latency_ms')} ms ({A.get('device')}).")
    out = '\n'.join(rows)
    print(out)
    if a.md:
        open(a.md, 'w').write(out + '\n')


if __name__ == '__main__':
    main()
