# 🏈 QB-Centric Play Type Classification (Run vs Pass)

## Overview
This project focuses on **early play-type classification (run vs pass)** in American football using **quarterback (QB)–centric video clips**.  
Instead of modeling the entire field, the approach assumes that **QB motion within the first 2–3 seconds** is a strong indicator of play type.

The system uses a **3D CNN (X3D)** to classify short video segments into *run-play* or *pass-play*.

<img width="1536" height="1024" alt="sampling" src="https://github.com/user-attachments/assets/7fabf275-342b-4144-9313-b0936a9c466b" />


---

## Key Idea
> In NFL plays, the quarterback’s early movement largely determines whether a play is a run or a pass.

By tracking the QB and cropping short temporal windows, the model learns **discriminative motion cues** without relying on full-field context.

---

## Dataset
- **Input:** Cropped QB-centric video clips  
- **Duration:** ~2–3 seconds per clip  
- **Classes:** Run-play, Pass-play  
- **Samples:** ~700 clips (≈350 per class)  
- **Frame Sampling:** Uniform temporal sampling (16 or 32 frames per clip)  

> ⚠️ Note: Dataset is proprietary and not publicly shared.

---

## Model
- **Architecture:** X3D (PyTorchVideo)  
- **Input:** 16 or 32-frame RGB video clips  
- **Pretraining:** Kinetics  
- **Loss:** Cross-Entropy  
- **Optimizer:** Adam  

---


