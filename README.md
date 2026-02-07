# ✈️ AeroGuard Pro - Advanced Flight Analysis Tool

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AeroGuard Pro** is a physics-based flight mission planning and simulation tool developed with Python. It integrates real-time atmospheric data to perform aerodynamic analysis and enforce aviation safety rules.

## 🌟 Key Features

* **Dynamic Route Planning:** Interactive map interface for creating flight paths.
* **Real-Time Weather Integration:** Fetches live temperature and wind data using Open-Meteo API.
* **Physics Engine:** Calculates Stall speeds, Flight Envelopes, and Structural Limits based on aircraft type.
* **Smart Crash Logic:** Simulates critical failures (e.g., Low Altitude Overspeed, Stalling) based on user inputs.
* **Multi-Language Support:** Full UI support for English, Turkish, German, French, Russian, and Japanese.

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/ENG-Goktug/AeroGuard.git](https://github.com/ENG-Goktug/AeroGuard.git)
    cd AeroGuard
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run flight_sim.py
    ```

## 📂 Project Structure

```text
AeroGuard/
├── flight_sim.py        # Main Application Core
├── requirements.txt     # Library dependencies
├── README.md            # Project Documentation
└── images/              # Aircraft images and icons
    ├── b737.jpg
    ├── f16.jpg
    └── ...