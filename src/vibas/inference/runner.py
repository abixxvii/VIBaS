import argparse, os, numpy as np, torch
import imageio.v2 as imageio
from vibas.models.seg_unet import UNetSmall
from vibas.models.physics_step import HeatStep

@torch.no_grad()
def main(args):
    seg = UNetSmall(); heat = HeatStep()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # handle either hybrid or seg-only checkpoint
    if "seg" in ckpt and "heat" in ckpt:
        seg.load_state_dict(ckpt["seg"])
        heat.load_state_dict(ckpt["heat"])
    else:
        seg.load_state_dict(ckpt)

    seg.eval(); heat.eval()
    clip = np.load(args.input)  # (T,H,W)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = imageio.get_writer(args.output, fps=15)

    for t in range(clip.shape[0]-1):
        T0 = torch.from_numpy(clip[t][None,None,...])
        M = seg(T0)
        Tpred = heat(T0, M)  # noqa: Tpred used for future work; we overlay M now

        img = ((clip[t]-clip.min())/(clip.max()-clip.min()+1e-8)*255).astype(np.uint8)
        overlay = np.stack([img, img, img], axis=-1)
        m = (M.squeeze().numpy()>0.5).astype(np.uint8)*255
        overlay[...,0] = np.maximum(overlay[...,0], m)  # red mask
        writer.append_data(overlay)

    writer.close()
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/samples/clip_000.npy")
    ap.add_argument("--ckpt", type=str, default="experiments/hybrid_baseline/model.pt")
    ap.add_argument("--output", type=str, default="experiments/demo/out.mp4")
    args = ap.parse_args()
    main(args)
