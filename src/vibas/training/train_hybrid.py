import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from vibas.models.seg_unet import UNetSmall
from vibas.models.physics_step import HeatStep

class SynthSeq(Dataset):
    def __init__(self, root="data/samples", seq=2):
        self.files = sorted(glob.glob(os.path.join(root, "*.npy")))
        self.seq = seq
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # (T,H,W)
        t = np.random.randint(0, arr.shape[0]-self.seq)
        T0 = arr[t]; T1 = arr[t+1]
        thr = np.quantile(T0, 0.99)
        M0 = (T0 >= thr).astype(np.float32)
        x0 = to_tensor(T0[None,...])
        y1 = to_tensor(T1[None,...])
        m0 = to_tensor(M0[None,...])
        return x0, m0, y1

def train(args):
    ds = SynthSeq(args.data)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    seg = UNetSmall().to(args.device)
    heat = HeatStep(alpha=args.alpha).to(args.device)
    opt = torch.optim.Adam(list(seg.parameters())+list(heat.parameters()), lr=1e-3)
    l2 = torch.nn.MSELoss(); bce = torch.nn.BCELoss()
    for epoch in range(args.epochs):
        seg.train(); heat.train(); loss_sum=0
        for T0, M0, T1 in dl:
            T0, M0, T1 = T0.to(args.device), M0.to(args.device), T1.to(args.device)
            Mhat = seg(T0)
            Tpred = heat(T0, Mhat)
            loss = l2(Tpred, T1) + 0.1*bce(Mhat, M0)  # forecast + weak seg
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*T0.size(0)
        print(f"epoch {epoch+1}: loss={loss_sum/len(ds):.4f}")
    os.makedirs("experiments/hybrid_baseline", exist_ok=True)
    torch.save({"seg": seg.state_dict(), "heat": heat.state_dict()},
               "experiments/hybrid_baseline/model.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/samples")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.5e-5)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train(args)
