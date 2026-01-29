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
## Results

### Confusion Matrix
![Confusion Matrix] <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/d7bae63c-09b9-4d42-a6c5-5dbc422f3e58" />
 
*Figure: Confusion matrix showing per-class prediction performance.*

### Classification Report
![Classification Report](images/classification_report.png)  
*Figure: Precision, Recall, F1-score, and Accuracy for Run-play and Pass-play.*

**Key Metrics:**

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Run-Play    | 0.76      | 0.93   | 0.83     | 57      |
| Pass-Play   | 0.89      | 0.66   | 0.76     | 50      |
| **Accuracy** | —         | —      | —        | 0.8037  |
| Macro Avg   | 0.82      | 0.79   | 0.80     | 107     |
| Weighted Avg| 0.82      | 0.80   | 0.80     | 107     |

✅ The model shows strong early-stage classification capability, particularly in identifying run plays.

## Key Contributions
- Designed a QB-centric approach for early play prediction.  
- Applied 3D CNN (X3D) for spatiotemporal learning on short video clips.  
- Demonstrated proof-of-concept for automated NFL play analysis, enabling potential real-time decision support.  
- Showcases skills in computer vision, deep learning, and sports analytics, making it suitable for further research under a Graduate Research Assistantship.


## Skills & Tools
- **Deep Learning Frameworks:** PyTorch, PyTorchVideo  
- **Computer Vision:** 3D CNNs, Video Classification, Temporal Modeling  
- **Other Tools:** OpenCV, NumPy, Matplotlib
