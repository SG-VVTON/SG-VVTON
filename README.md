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

To evaluate the temporal stability and visual authenticity of **SG-VVTON**, we present unified side-by-side video demonstrations. Each clip simultaneously displays the **Input Frames**, the **Target Garment Image**, and the synthesized **Output Frames**, highlighting our model's robust performance across feature extraction, dynamic human locomotion, severe anatomical occlusions, and complex ambient lighting conditions.

---

### 1. Multimodal Preprocessing & Spatiotemporal Conditioning Signals
Unlike conventional virtual try-on architectures, our pipeline explicitly extracts structured representations from unconstrained input video sequences prior to generative denoising. This module accurately isolates the subject's existing clothing boundaries, preserves facial identity, and tracks kinematic motion trajectories without relying on optical flow estimation or auxiliary 3D simulators.

#### Demo 1: Extraction of SAM3 semantic clothing masks, face bounding boxes, and ViTPose skeletal graphs from input videos
> This demonstration visualizes our automated preprocessing pipeline: generating precise binary clothing masks (**`m_garment`**) and facial bounding boxes (**`B_face`**) from input video frames, alongside continuous whole-body kinematic motion signals (**`C_pos`**) across diverse real-world scenes.

<div align="center">
  <video src="https://github.com/user-attachments/assets/f50800b5-83c0-46f3-84b7-9d9d39e57c97" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 2. Fine-Grained Detail Preservation & Photometric Adaptation
Our framework regularizes the latent trajectory using structured negative textual constraints and CLIP-derived visual embeddings (**`c_v`**) extracted from the target reference image. This dual-conditioning mechanism ensures that intricate garment structures—such as high-frequency prints, ribbons, and weaves—are faithfully rendered without inter-frame blurring. Simultaneously, the model dynamically adapts fabric shading, wrinkles, and shadows to the complex illumination of the video scene without requiring auxiliary 3D physical simulators.

#### Demos 2 & 3: High-frequency pattern preservation, realistic ambient shading, and dynamic motion adaptation
> As demonstrated below, complex garment patterns and sharp prints from the reference image remain visually authentic and spatially stable under dynamic locomotion (top). Furthermore, ambient scene lighting seamlessly interacts with the target apparel to generate natural shadows and folds across varying environments (bottom).

<div align="center">
  <video src="https://github.com/user-attachments/assets/511a42c7-58bf-42f6-8cbc-549e85d706b0" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>
<br>
<div align="center">
  <video src="https://github.com/user-attachments/assets/4ae5a8f6-2f11-43c0-a772-3ad0e2a8e578" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 3. Multi-Garment Generalization in Unconstrained Videos
By leveraging the robust generative priors of a 14B-parameter diffusion backbone, **SG-VVTON** generalizes seamlessly across diverse apparel categories and challenging, in-the-wild video footage, including broadcast clips and cinematic sequences.

#### Demo 4: In-the-wild generalization across unconstrained broadcast and movie sequences
> Our model accurately transfers reference garments onto diverse body types in unconstrained backgrounds without introducing semantic drift, boundary artifacts, or temporal jitter.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0001c202-8a95-4033-abc2-2ff27dfd8825" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

### 4. Robustness to Severe Anatomical Occlusions
A prevalent failure mode among existing video try-on methods is boundary hallucination and detail degradation when body parts or external objects obstruct the torso. By conditioning the diffusion trajectory on continuous skeletal graphs, our architecture maintains exceptional structural boundary integrity throughout complex interactions.

#### Demo 5: Robust spatiotemporal coherence and identity preservation during severe topological occlusions
> Even during complex human-object interactions—such as walking while holding beverage cups or crossing arms over the chest—the synthesized garment boundaries remain temporally stable without artifact accumulation.

<div align="center">
  <video src="https://github.com/user-attachments/assets/de3eb158-6354-4a03-8253-907fa762a134" controls="controls" autoplay="autoplay" loop="loop" muted="muted" style="max-width: 100%; border-radius: 8px;"></video>
</div>

---

## 📖 Abstract

While recent Video Virtual Try-On (VVTON) methods incorporate temporal attention modules, they frequently struggle to maintain visual fidelity and motion consistency simultaneously, often suffering from temporal instability and detail degradation under complex locomotion. 

In this work, we propose **SG-VVTON**, an end-to-end generative framework that integrates discriminative vision backbones into a large-scale video diffusion architecture. Rather than relying on global feature diffusion or auxiliary 3D physical simulators, our pipeline extracts deterministic skeletal priors via **ViTPose** whole-body estimation and isolates semantic garment boundaries using **SAM3** video segmentation. These structured representations directly condition the denoising trajectory of a 14B-parameter diffusion backbone, enforcing strict anatomical alignment and spatiotemporal coherence. 

Qualitative evaluations across unconstrained sequences demonstrate that our approach realistically synthesizes fluid fabric mechanics and ambient photometric shading, significantly mitigating prevalent artifacts such as temporal inconsistency, color shift, and the loss of fine-grained details over time.

---

## 🏗️ Architecture & Methodology Overview

We formulate the VVTON task as a latent-space spatiotemporal conditional generation problem. Rather than relying on standard temporal attention that diffuses features globally, our computational graph extracts deterministic multimodal representations through parallel processing branches before injecting them directly into the generative backbone.

<div align="center">
  <img width="95%" alt="SG-VVTON Architecture Overview" src="https://github.com/user-attachments/assets/fd6626b3-c510-46cc-b056-d1999fd14571" style="border-radius: 8px; margin-top: 15px; margin-bottom: 10px;" />
  <br>
  <p style="width: 85%; text-align: center; color: #555;">
    <b>Figure 1: Overview of the proposed architecture.</b> Three core modules (<i>Inputs</i>, <i>Preprocessing & Feature Extraction</i>, and <i>Generative AI Synthesis</i>) collaboratively extract multimodal conditioning signals to guide the video diffusion backbone.
  </p>
</div>

### Core Pipeline Components:
1. **Spatiotemporal Mask Propagation:** We utilize **SAM3** to track and isolate three independent semantic targets across frames: the reference garment, the subject's face, and the background context. Selective masking ensures that background features and facial geometry remain invariant during denoising without boundary drift.
2. **Kinematic & Spatial Guidance:** Continuous whole-body skeletal graphs from **ViTPose** are concatenated with facial bounding box coordinates and masked video frames to construct a unified spatiotemporal conditioning tensor.
3. **Multimodal Visual-Lexical Cross-Conditioning:** High-frequency visual texture embeddings (**`c_v`**) extracted via a pre-trained **CLIP-Vision** encoder are combined with structured positive and negative textual descriptors (**`c_txt+`**, **`c_txt-`**). This dual-conditioning scheme constrains the optimization landscape, preventing color shifts, anatomical distortion, and undesired transparency.
4. **Latent Video Diffusion Backbone:** The conditioning tensor is processed along the channel dimension of a 14B-parameter DiT backbone (**WAN 2.2**), utilizing a Multi-Modal Classifier-Free Guidance (CFG) formulation to drive the synthesis trajectory accurately toward the target garment distribution.

---

## 🚀 Key Features

* **🔥 Scalable Latent Video Diffusion**: Synthesizes complex fabric mechanics and fluid deformations directly within the compressed latent space of a 14B-parameter video diffusion backbone.
* **🦴 Dynamic Kinematic Guidance**: Incorporates continuous whole-body skeletal graphs from ViTPose directly into the denoising network. This explicit parameterization models intra-frame anatomical alignment and long-term pose dynamics.
* **🎯 Precise Semantic Mask Propagation**: Utilizes SAM3 video segmentation on input sequences to isolate semantic garment boundaries and remove the original clothing without boundary drift.
* **💡 Multimodal Visual-Lexical Conditioning**: Employs CLIP-derived visual embeddings extracted from the target reference image alongside structured lexical prompts. This dual-conditioning scheme regularizes the latent trajectory to prevent degenerate generation paths, color shift, and detail degradation.
* **⚡ Zero 3D Simulation Overhead**: Achieves realistic garment draping, wrinkle dynamics, and ambient photometric shading purely through generative diffusion. This completely bypasses computationally expensive 3D mesh simulators and external optical flow-tracking modules.

---

## 🛠️ Code Release & Setup

> **📢 Announcement:** The full source code, pre-trained checkpoints, and detailed inference instructions are currently undergoing clean-up and internal peer-review. **The repository code will be made publicly available immediately after paper acceptance.**
