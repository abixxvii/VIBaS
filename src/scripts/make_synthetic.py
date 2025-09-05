import argparse, os, numpy as np
import imageio.v2 as imageio

def gaussian_2d(h, w, cx, cy, sigma, amp):
    y, x = np.mgrid[0:h, 0:w]
    return amp * np.exp(-(((x-cx)**2 + (y-cy)**2)/(2*sigma**2)))

def make_clip(h=128, w=128, frames=60, ambient=25.0, seed=0):
    rng = np.random.default_rng(seed)
    cx, cy = rng.integers(w//4, 3*w//4), rng.integers(h//4, 3*h//4)
    sigma, amp = 2.0, rng.uniform(5, 20)
    T = []
    for t in range(frames):
        sigma_t = sigma + 0.1*t
        amp_t = amp * (1 + 0.02*t)
        field = ambient + gaussian_2d(h, w, cx, cy, sigma_t, amp_t)
        field += rng.normal(0, 0.1, size=(h,w))  # mild noise
        T.append(field.astype(np.float32))
    return np.stack(T, 0)  # (T,H,W)

def save_clip(clip, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(clip):
        mm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        imageio.imwrite(os.path.join(frames_dir, f"{i:04d}.png"), (mm*65535).astype(np.uint16))
    np.save(os.path.join(outdir, f"{name}.npy"), clip)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/samples")
    args = ap.parse_args()
    for k in range(args.n_clips):
        clip = make_clip(seed=42+k)
        save_clip(clip, args.out, f"clip_{k:03d}")
    print(f"Wrote {args.n_clips} clips to {args.out}")
