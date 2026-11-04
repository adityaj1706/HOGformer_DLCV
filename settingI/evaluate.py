"""Evaluate a trained HOGformer checkpoint on the Setting I test sets.

Run from settingI/ :
  python evaluate.py --config Allweather/Options/Allweather_HOGformer_baseline.yml \
                     --weights path/to/net_g_latest.pth --out results_baseline.json

It follows the same steps as the training-time validation (reflect-pad to a multiple of
window_size, uint8 conversion, PSNR/SSIM on the Y channel, crop_border=0) but, unlike that
code (which stops after the first 60 images), it scores EVERY image of Test1 / RainDrop and a
fixed random subset of Snow100K-S / -L. Per-image scores are saved so two models can be
compared with paired statistics (see compare_results.py).
"""
import argparse, json, os, os.path as osp, random, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F, yaml

sys.path.insert(0, 'basicsr')
from models.archs.hogformer_arch import HOGformer
from utils.img_util import img2tensor, tensor2img
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

DATASETS = {   # name -> (gt folder, input folder), relative to --data_root
    'test1':     ('test/Test1/gt', 'test/Test1/input'),
    'raindrop':  ('test/RainDrop/test_a/gt', 'test/RainDrop/test_a/input'),
    'snow_s':    ('test/Snow100K-Test/Snow100K-S/gt', 'test/Snow100K-Test/Snow100K-S/synthetic'),
    'snow_l':    ('test/Snow100K-Test/Snow100K-L/gt', 'test/Snow100K-Test/Snow100K-L/synthetic'),
}


def build_model(cfg_path, weights, device):
    cfg = yaml.safe_load(open(cfg_path))
    net_opt = dict(cfg['network_g']); net_opt.pop('type')
    net = HOGformer(**net_opt)
    ckpt = torch.load(weights, map_location='cpu')
    sd = ckpt.get('params', ckpt)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), cfg


@torch.no_grad()
def restore(net, lq, window_size, device):
    """lq: float32 BGR image in [0,1] (HxWx3). Returns uint8 BGR output like tensor2img."""
    x = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    _, _, h, w = x.shape
    ph = (window_size - h % window_size) % window_size
    pw = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, pw, 0, ph), 'reflect')
    out = net(x)
    if isinstance(out, list):
        out = out[-1]
    out = out[:, :, :h, :w]
    return tensor2img([out], rgb2bgr=True)


def read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, path
    return img


def count_macs(net, device, size=128):
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            f = FlopCountAnalysis(net, torch.rand(1, 3, size, size, device=device))
            f.unsupported_ops_warnings(False); f.uncalled_modules_warnings(False)
            return f.total() / 1e9
    except Exception as e:  # keep evaluating even if FLOP counting fails
        print('FLOP counting failed:', e); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data_root', default='/content/data/weather')
    ap.add_argument('--datasets', nargs='+', default=['test1', 'raindrop', 'snow_s', 'snow_l'], choices=list(DATASETS))
    ap.add_argument('--snow_n', type=int, default=300, help='fixed random subset size for each Snow100K test set')
    ap.add_argument('--max_images', type=int, default=0, help='debug: cap images per dataset (0 = no cap)')
    ap.add_argument('--seed', type=int, default=0, help='seed of the Snow100K subset selection (keep identical for all models)')
    ap.add_argument('--debug_crop', type=int, default=0, help='debug only: centre-crop images to NxN (for CPU tests)')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, cfg = build_model(args.config, args.weights, device)
    win = cfg.get('val', {}).get('window_size', 16)
    result = {'name': args.name or cfg.get('name'), 'config': args.config, 'weights': args.weights,
              'device': str(device), 'params_M': sum(p.numel() for p in net.parameters()) / 1e6,
              'GMACs_128x128': count_macs(net, device), 'snow_n': args.snow_n, 'seed': args.seed, 'datasets': {}}
    print(f"{result['name']}: {result['params_M']:.2f} M params, {result['GMACs_128x128']} GMACs @128x128 (fvcore; some ops not counted)")

    for ds in args.datasets:
        gt_dir, lq_dir = (osp.join(args.data_root, p) for p in DATASETS[ds])
        names = sorted(os.listdir(gt_dir))
        assert sorted(os.listdir(lq_dir)) == names, f'{ds}: gt/input file names differ'
        if ds.startswith('snow'):
            names = sorted(random.Random(args.seed).sample(names, min(args.snow_n, len(names))))
        if args.max_images:
            names = names[:args.max_images]
        per, t0 = {}, time.time()
        for i, n in enumerate(names):
            lq = read(osp.join(lq_dir, n)).astype(np.float32) / 255.
            gt = read(osp.join(gt_dir, n))
            if args.debug_crop:
                c = args.debug_crop; h0, w0 = (gt.shape[0] - c) // 2, (gt.shape[1] - c) // 2
                lq, gt = lq[h0:h0 + c, w0:w0 + c], gt[h0:h0 + c, w0:w0 + c]
            sr = restore(net, lq, win, device)
            kw = dict(crop_border=0, test_y_channel=True)
            per[n] = [float(calculate_psnr(sr, gt, **kw)), float(calculate_ssim(sr, gt, **kw))]
            if (i + 1) % 50 == 0:
                print(f'  {ds}: {i + 1}/{len(names)}  running PSNR {np.mean([v[0] for v in per.values()]):.3f}', flush=True)
        arr = np.array(list(per.values()))
        result['datasets'][ds] = {'n': len(per), 'psnr': float(arr[:, 0].mean()), 'ssim': float(arr[:, 1].mean()),
                                  'seconds': time.time() - t0, 'per_image': per}
        print(f"{ds:9s} n={len(per):5d}  PSNR {arr[:, 0].mean():.4f}  SSIM {arr[:, 1].mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(result, open(args.out, 'w'))   # save after each dataset so partial results survive a crash
    print('saved', args.out)


if __name__ == '__main__':
    main()
