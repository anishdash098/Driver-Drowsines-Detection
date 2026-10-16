# 🚗 Real-Time Driver Drowsiness & Yawn Detection System

An Advanced Driver Assistance System (ADAS) that monitors driver alertness in real-time using **Computer Vision** and **Dual MobileNetV2 Deep Learning Architectures**. It detects early signs of fatigue (sustained eye closure and prolonged yawning) and triggers instant acoustic and visual alarms to prevent fatal road accidents.

---

## 🌟 Key Features

* **Dual MobileNetV2 Architecture:**
  * **Eye Closure Model:** Trained on a curated benchmark of **10,000 eye samples** (MRL Eye Dataset & CEW) achieving **100% validation accuracy**.
  * **Yawn Detection Model:** Trained on mouth aperture and yawning samples with **97.2% validation accuracy**.
* **Real-Time Edge Performance:** Runs at **50–60 FPS** on CPU using MobileNetV2's lightweight depthwise separable convolutions.
* **Blink vs. Microsleep Differentiation:** Uses a temporal sliding-window state machine so natural blinks (100–400ms) do not trigger false alarms, while sustained eye closure (>1.5s) sounds the alarm.
* **Smart Yawn Verification:** Combines deep learning classification with dynamic oral cavity aspect ratio analysis to prevent false positives from beards or normal talking.
* **Automotive Heads-Up Display (HUD):** Compact on-screen dashboard displaying real-time state, confidence percentages, and score counters.

---

## 🏗️ System Architecture

```text
                     Webcam Live Feed (720p HD)
                                 │
                     [OpenCV Face Localization]
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
         [Eye Region]                        [Mouth Region]
               │                                   │
      ┌────────────────┐                  ┌────────────────┐
      │ Eye MobileNet  │                  │ Yawn MobileNet │
      │  (Open/Closed) │                  │ (Normal/Yawn)  │
      └────────┬───────┘                  └────────┬───────┘
               │                                   │
        [Eye Score Counter]               [Yawn Score Counter]
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
                     🚨 Alert Decision Engine 🚨
          • Eye Score >= 15 ➔ "DROWSINESS ALERT! WAKE UP!" (Audio Alarm)
          • Yawn Score >= 14 ➔ "FATIGUE WARNING: YAWNING!"
