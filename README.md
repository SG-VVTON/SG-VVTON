# 🎭 SG-VVTON: Skeletal-Guided Diffusion for High-Fidelity and Temporally Coherent Video Virtual Try-On

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACM%20%2F%20ArXiv-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*An end-to-end generative framework integrating ViTPose kinematic priors and SAM3 semantic segmentation into a 14B-parameter Video Diffusion Transformer (DiT) for realistic, temporally coherent virtual try-on in unconstrained videos.*

</div>

---

## 🌟 Visual Showcase & Demonstrations

To evaluate the temporal stability and visual authenticity of **SG-VVTON**, we present unified side-by-side video demonstrations. Each clip displays the **Input Frames**, the **Target Garment Image**, and the synthesized **Output Frames** together, highlighting our model's performance across preprocessing extraction, dynamic human motion, severe anatomical occlusions, and complex lighting conditions.

---

### 1. Multimodal Preprocessing & Spatiotemporal Conditioning Signals
Unlike static virtual try-on models, our pipeline explicitly extracts structured representations from unconstrained input video frames before generative denoising. This module isolates the subject's existing clothing boundaries from the video sequence, preserves facial identity, and tracks kinematic motion trajectories without relying on optical flow estimation or auxiliary 3D simulators.

#### Demo 1: Extraction of SAM3 semantic clothing masks, face bounding boxes, and ViTPose skeletal graphs from input videos
> This demonstration visualizes our automated preprocessing pipeline: generating precise binary clothing masks (**`m_garment`**) and facial bounding boxes (**`B_face`**) from input video frames, alongside continuous whole-body kinematic motion signals (**`C_pos`**) across diverse scenes.

<div align="center">
  <video src="https://github.com/user-attachments/assets/f50800b5-83c0-46f3-84b7-9d9d39e57c97" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 2. Fine-Grained Detail Preservation & Photometric Adaptation
Our framework regularizes the latent trajectory using structured negative textual constraints and CLIP-derived visual embeddings (**`c_v`**) extracted from the target reference image, combined with embedded LoRA adaptation layers. This ensures that intricate garment structures (such as high-frequency prints, ribbons, and weaves) are faithfully rendered without inter-frame blurring, while simultaneously adapting fabric shading, wrinkles, and shadows to the complex illumination of the video scene—**all without auxiliary 3D physical simulators.**

#### Demos 2 & 3: High-frequency pattern preservation, realistic ambient shading, and dynamic motion adaptation
> As shown below, complex garment patterns and sharp prints from the reference image remain visually authentic and spatially stable under dynamic locomotion (top), while ambient room lighting seamlessly interacts with the target apparel to generate natural shadows and folds across varying environments (bottom).

<div align="center">
  <video src="https://github.com/user-attachments/assets/511a42c7-58bf-42f6-8cbc-549e85d706b0" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>
<br>
<div align="center">
  <video src="https://github.com/user-attachments/assets/4ae5a8f6-2f11-43c0-a772-3ad0e2a8e578" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 3. Multi-Garment Generalization in Unconstrained Videos
By leveraging the robust generative priors of the 14B-parameter diffusion backbone, **SG-VVTON** generalizes seamlessly across diverse apparel categories and challenging, in-the-wild video footage (including broadcast clips and complex cinematic scenes).

#### Demo 4: In-the-wild generalization across unconstrained broadcast and movie sequences
> Our model accurately transfers reference garments onto diverse body types in unconstrained backgrounds without introducing semantic drift, boundary artifacts, or temporal jitter.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0001c202-8a95-4033-abc2-2ff27dfd8825" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 4. Robustness to Severe Anatomical Occlusions
A primary failure mode of existing video try-on methods is boundary hallucination when body parts or external objects obstruct the torso. By conditioning the diffusion trajectory on continuous skeletal graphs, our architecture maintains structural boundary integrity.

#### Demo 5: Robust spatiotemporal coherence and identity preservation during severe topological occlusions
> Even during complex human-object interactions—such as walking while holding coffee cups or crossing arms over the chest—the synthesized garment boundaries remain temporally stable without artifact accumulation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/de3eb158-6354-4a03-8253-907fa762a134" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

## 📖 Abstract

While recent Video Virtual Try-On (VVTON) methods incorporate temporal attention modules, they frequently struggle to maintain visual fidelity and motion consistency simultaneously, often suffering from temporal instability and detail degradation under complex locomotion. 

In this work, we propose **SG-VVTON**, an end-to-end generative framework that integrates discriminative vision backbones into a large-scale video diffusion architecture. Rather than relying on global feature diffusion or auxiliary 3D physical simulators, our pipeline:
1. **Extracts deterministic skeletal priors** via **ViTPose** whole-body estimation from the input video to parameterize bodily dynamics.
2. **Isolates semantic clothing boundaries** using **SAM3** video segmentation on input frames to remove the original garment without boundary drift.
3. **Directly conditions the denoising trajectory** of a **14B-parameter diffusion backbone (WAN 2.2)** using both video spatial guidance and reference garment visual embeddings, enforcing strict anatomical alignment and spatiotemporal coherence.

Qualitative evaluations across unconstrained broadcast and real-world sequences demonstrate that our approach realistically synthesizes fluid fabric mechanics and ambient photometric shading, significantly mitigating prevalent artifacts such as inter-frame inconsistency, color shift, and the loss of fine-grained details over time.

---

## 🚀 Key Features

- **🔥 Scalable Latent Video Diffusion**: Synthesizes fluid fabric mechanics and complex non-linear deformations directly within the compressed latent space of a **14B-parameter Video DiT (WAN 2.2)** backbone.
- **🦴 Dynamic Kinematic Guidance**: Eliminates inter-frame jitter and anatomical hallucinations by injecting continuous whole-body skeletal graphs (**`C_pos`**) from **ViTPose** directly into the conditioning tensor.
- **🎯 Precise Semantic Mask Propagation**: Utilizes **SAM3** tracking on input video sequences to isolate the original clothing mask (**`m_garment`**), facial identity (**`B_face`**), and background context (**`m_bg`**), ensuring temporal boundary stability without spatial misalignment.
- **💡 Multimodal Visual-Lexical Conditioning**: Employs **CLIP-Vision** texture embeddings (**`c_v`**) extracted directly from the static reference garment image, combined with structured positive/negative lexical prompts (**`c_txt+`**, **`c_txt-`**) via a Multi-Modal CFG formulation to eliminate color shift, anatomical distortion, and undesired transparency.
- **⚡ Zero 3D Simulation Overhead**: Achieves realistic garment draping, wrinkle dynamics, and ambient photometric shading purely through generative diffusion, completely bypassing computationally expensive 3D mesh simulators or optical flow tracking modules.

---

## 🏗️ Methodology & Architecture Overview

We formulate the VVTON task as a latent-space spatiotemporal conditional generation problem. Given an input video sequence **`V`**, a target reference garment image **`I_g`**, and an auxiliary textual prompt **`P`**, our framework approximates the conditional distribution:

$$p_\theta(V \mid V_{\text{mask}}, C_{\text{kin}}, B_{\text{face}}, c_v, c_{\text{txt}})$$

### 1. Preprocessing & Feature Extraction Graph
As illustrated in our pipeline architecture, the input modalities are processed through parallel discriminative branches:
* **Spatiotemporal Masking of Input Video (`V_mask`):** Binary masks from SAM3 applied to input video frames are dilated, smoothed via a Gaussian filter, and inverted to isolate and remove the subject's original clothing while preserving background and facial identity:
  $$V_{\text{mask}, t} = V_t \odot M_{\text{cond}, t}$$
* **Skeletal Rendering (`C_spatiotemporal`):** Skeletal landmarks from ViTPose (extracted from the input video) are rendered as continuous skeleton graphs and concatenated with facial bounding box coordinates and masked video frames:
  $$C_{\text{spatiotemporal}} = [V_{\text{mask}}, C_{\text{pos}}, B_{\text{face}}]$$
* **Visual Texture Embedding of Target Image (`c_v`):** A pre-trained CLIP-Vision encoder extracts high-frequency structural features and surface patterns directly from the target reference garment image:
  $$c_v = E_{\text{CLIP-Vision}}(I_g)$$

### 2. Latent Diffusion & Multi-Modal CFG
The latent conditioning tensor **`Z_cond`** (derived from encoding **`C_spatiotemporal`** via a VAE) is concatenated with the noisy latent state **`z_tau`** along the channel dimension before entering the **14B-parameter DiT backbone (WAN 2.2)**. 

Denoising is executed via a discrete Euler sampler using a uniform noise schedule. At each timestep $\tau$, the predicted noise $\epsilon_\theta$ is modulated using our novel multi-modal Classifier-Free Guidance (CFG) formulation:

$$\tilde{\epsilon}_\theta(z_\tau, \tau) = \epsilon_\theta(z_\tau, \tau, Z_{\text{cond}}, c_{\text{txt}}^-) + \omega \cdot \Big[ \epsilon_\theta(z_\tau, \tau, Z_{\text{cond}}, c_v, c_{\text{txt}}^+) - \epsilon_\theta(z_\tau, \tau, Z_{\text{cond}}, c_{\text{txt}}^-) \Big]$$

where $\omega \ge 1.0$ is the scaling factor controlling guidance intensity toward the target garment distribution (**`c_v`**), while strictly bound by the physical movement defined by **`Z_cond`**.

---

## 🛠️ Code Release & Setup

> **📢 Announcement:** The full source code, pre-trained checkpoints, and detailed inference instructions are currently undergoing clean-up and internal peer-review. **The repository code will be made publicly available immediately after the paper is accepted.**
