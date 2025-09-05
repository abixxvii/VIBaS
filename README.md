# VIBaS
Vision-Based Early Detection of Thermal Runaway in Li‑ion Modules (VIBaS)

# Why
Conventional battery safety monitoring (temp/voltage/pressure sensors) can react late. This repo explores a vision-first alternative that fuses thermal image segmentation with a heat‑equation–constrained forecaster to flag risk earlier at the same false‑alarm rate.

# Key Idea
Recent research has explored multiphysics modeling combined with deep learning to predict thermal runaway in lithium-ion batteries. For example, Goswami et al. (University of Arizona & UIC, 2024) developed a framework coupling detailed electro-thermal simulations with convolutional neural networks to classify and predict thermal runaway events . While effective, these approaches remain largely sensor- or simulation-driven. In contrast, this framework is the first to leverage infrared vision with a physics-guided forecaster based on the heat equation, enabling earlier hotspot detection and quantitative evaluation using engineering metrics such as time-to-detection, ΔT at alarm, and propagation rate.

1) Segment incipient hotspots in IR frames. 
2) Forecast near‑future temperature using a differentiable heat equation step with a learnable source term.
3) Compute a risk score from ΔT, growth rate, and propagation to trigger alarms

# Repository Layout

- src/vibas/ – Python package with data adapters, models, metrics, training & inference.
- src/scripts/ – CLI utilities for dataset prep, synthetic generation, demos.
- notebooks/ – Quickstarts and experiments (segmentation → physics‑guided → evaluation).
- data/ – Your local data workspace (raw/processed are .gitignored; tiny samples tracked).
- experiments/ – Configs, logs, and artifacts per run.
- tests/ – Minimal unit tests for physics step, adapters, and metrics.
