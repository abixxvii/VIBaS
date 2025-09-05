import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from vibas.models.seg_unet import UNetSmall

class SynthIR(Dataset):
    def __init__(self, root="data/samples"):
        self.files = sorted(glob.glob(os.path.join(root, "*.npy")))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # (T,H,W)
        t = np.random.randint(0, arr.shape[0])
        frame = arr[t]
        thr = np.quantile(frame, 0.99)
        mask = (frame >= thr).astype(np.float32)
        x = to_tensor(frame[None,...])  # (1,H,W)
        y = to_tensor(mask[None,...])
        return x, y

def train(args):
    ds = SynthIR(args.data)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = UNetSmall().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCELoss()
    for epoch in range(args.epochs):
        model.train(); loss_sum=0
        for x,y in dl:
            x,y = x.to(args.device), y.to(args.device)
            p = model(x)
            loss = bce(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
        print(f"epoch {epoch+1}: loss={loss_sum/len(ds):.4f}")
    os.makedirs("experiments/seg_baseline", exist_ok=True)
    torch.save(model.state_dict(), "experiments/seg_baseline/unet_small.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/samples")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train(args)
