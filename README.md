# VIBaS
Vision-Based Early Detection of Thermal Runaway in Li‑ion Modules (VIBaS)

# Why
Conventional battery safety monitoring (temp/voltage/pressure sensors) can react late. This repo explores a vision-first alternative that fuses thermal image segmentation with a heat‑equation–constrained forecaster to flag risk earlier at the same false‑alarm rate.

# Key Idea
1) Segment incipient hotspots in IR frames. 
2) Forecast near‑future temperature using a differentiable heat equation step with a learnable source term.
3) Compute a risk score from ΔT, growth rate, and propagation to trigger alarms
