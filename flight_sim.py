"""
AeroGuard Flight Analysis & Simulation Tool
-------------------------------------------
A Streamlit-based application for flight planning, aerodynamic analysis,
and real-time flight simulation using Folium maps and Open-Meteo API.

Features:
- Route Planning with interactive maps
- Real-time weather data integration
- Aerodynamic flight envelope analysis
- Dynamic flight simulation with crash logic
- Multi-language support (i18n)

Author: [Your Name Here]
Date: 2026
License: MIT
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import requests
import os
import base64
import time

# --- LIBRARY DEPENDENCY CHECK ---
try:
    from geopy.distance import geodesic
except ImportError:
    st.error("Missing dependencies. Please run: pip install geopy requests")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroGuard Pro", layout="wide")

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #0e1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }

    /* Header Styling */
    h1 { color: #00ffcc; border-bottom: 2px solid #00ffcc; padding-bottom: 10px; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1b1e24; border-radius: 4px; color: #aaa; }
    .stTabs [aria-selected="true"] { background-color: #00ffcc !important; color: #000 !important; font-weight: bold; }

    /* Custom Cards */
    .spec-card { background: #1b1e24; border: 1px solid #444; padding: 20px; border-radius: 8px; margin-top: 10px; }
    .warning-box { border: 1px solid #ffcc00; background: #26220c; color: #ffcc00; padding: 10px; border-radius: 5px; }
    .crash-box { 
        border: 2px solid #ff0000; background: #3d0000; color: #ff4b4b; 
        padding: 20px; border-radius: 10px; text-align: center; 
        font-weight: bold; font-size: 20px; margin-top: 20px; 
        box-shadow: 0 0 20px #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INTERNATIONALIZATION (I18N) DICTIONARY ---
TRANSLATIONS = {
    "TR": {
        "tabs": ["🗺️ Rota Planlama", "✈️ Teknik Özellikler", "📊 Mühendislik Analizi", "🚀 Simülasyon"],
        "cockpit": "Kokpit Paneli", "aircraft": "Uçak Seçimi",
        "params": "Uçuş Parametreleri", "alt": "İrtifa (m)", "spd": "Hız (m/s)",
        "specs_title": "Teknik Veri Kartı", "mass": "Kütle", "span": "Kanat Açıklığı", "len": "Uzunluk", "eng": "Motor",
        "env_title": "Uçuş Zarfı Analizi",
        "env_desc": "Güvenli uçuş sınırlarını gösterir. Çizginin altı Stall bölgesidir.",
        "wind_title": "Kalkış Performansı",
        "wind_desc": "Rüzgar yönünün kalkış hızına etkisi. Karşı rüzgar avantaj sağlar.",
        "start": "UÇUŞU BAŞLAT", "reset": "Rotayı Temizle",
        "no_route": "⚠️ Rota oluşturulmadı! Lütfen harita sekmesinden 2 nokta seçiniz.",
        "click_map": "Başlangıç ve Bitiş noktalarını belirlemek için haritaya tıklayın.",
        "waiting_data": "Veri bekleniyor...", "weather_title": "Atmosferik Veriler",
        "sim_running": "Simülasyon Yürütülüyor...", "sim_done": "Operasyon Başarıyla Tamamlandı",
        "crash_alt_high": "🚨 KRİTİK HATA: İrtifa Limiti Aşıldı (Motorlar Durdu)!",
        "crash_alt_low": "🚨 KRİTİK HATA: Aşırı Alçak İrtifada Yüksek Hız (Yapısal Hasar)!",
        "crash_stall": "🚨 KRİTİK HATA: Stall Hızı! (Tutunma Kaybı)",
        "crash_struct": "🚨 KRİTİK HATA: Yapısal Hız Limiti Aşıldı! (Gövde Parçalandı)",
        "sim_failed": "OPERASYON BAŞARISIZ"
    },
    "EN": {
        "tabs": ["🗺️ Route Planning", "✈️ Tech Specs", "📊 Engineering Analysis", "🚀 Simulation"],
        "cockpit": "Cockpit Panel", "aircraft": "Select Aircraft",
        "params": "Flight Parameters", "alt": "Altitude (m)", "spd": "Speed (m/s)",
        "specs_title": "Technical Data Sheet", "mass": "Mass", "span": "Wingspan", "len": "Length", "eng": "Engine",
        "env_title": "Flight Envelope", "env_desc": "Shows safe flight limits. Below line is Stall zone.",
        "wind_title": "Takeoff Performance", "wind_desc": "Effect of wind on takeoff speed. Headwind is advantageous.",
        "start": "START FLIGHT", "reset": "Clear Route",
        "no_route": "⚠️ No route created! Please select 2 points on the map tab.",
        "click_map": "Click on the map to set Start and End points.",
        "waiting_data": "Waiting for data...", "weather_title": "Atmospheric Data",
        "sim_running": "Simulation Running...", "sim_done": "Operation Complete",
        "crash_alt_high": "🚨 CRITICAL ERROR: Ceiling Exceeded (Flameout)!",
        "crash_alt_low": "🚨 CRITICAL ERROR: Low Altitude Overspeed (Structural Failure)!",
        "crash_stall": "🚨 CRITICAL ERROR: Stall Speed (Lift Lost)!",
        "crash_struct": "🚨 CRITICAL ERROR: Vne Exceeded (Airframe Damage)!",
        "sim_failed": "OPERATION FAILED"
    },
    "DE": {
        "tabs": ["🗺️ Routenplanung", "✈️ Technische Daten", "📊 Analyse", "🚀 Simulation"],
        "cockpit": "Cockpit-Panel", "aircraft": "Flugzeugwahl",
        "params": "Flugparameter", "alt": "Höhe (m)", "spd": "Geschw. (m/s)",
        "specs_title": "Datenblatt", "mass": "Masse", "span": "Spannweite", "len": "Länge", "eng": "Motor",
        "env_title": "Flugbereich", "env_desc": "Zeigt sichere Grenzen. Unter der Linie ist Stall-Bereich.",
        "wind_title": "Startleistung", "wind_desc": "Windeinfluss auf Startgeschw. Gegenwind ist vorteilhaft.",
        "start": "STARTEN", "reset": "Route Löschen",
        "no_route": "⚠️ Keine Route! Bitte wählen Sie 2 Punkte auf der Karte.",
        "click_map": "Klicken Sie auf die Karte, um Start und Ziel festzulegen.",
        "waiting_data": "Warte auf Daten...", "weather_title": "Atmosphärische Daten",
        "sim_running": "Simulation läuft...", "sim_done": "Operation Abgeschlossen",
        "crash_alt_high": "🚨 KRITISCHER FEHLER: Dienstgipfelhöhe überschritten!",
        "crash_alt_low": "🚨 KRITISCHER FEHLER: Zu schnell in Bodennähe!",
        "crash_stall": "🚨 KRITISCHER FEHLER: Strömungsabriss (Stall)!",
        "crash_struct": "🚨 KRITISCHER FEHLER: Geschwindigkeitslimit überschritten!",
        "sim_failed": "OPERATION FEHLGESCHLAGEN"
    },
    "FR": {
        "tabs": ["🗺️ Planification", "✈️ Spécifications", "📊 Analyse", "🚀 Simulation"],
        "cockpit": "Panneau Cockpit", "aircraft": "Choix Avion",
        "params": "Paramètres", "alt": "Altitude (m)", "spd": "Vitesse (m/s)",
        "specs_title": "Fiche Technique", "mass": "Masse", "span": "Envergure", "len": "Longueur", "eng": "Moteur",
        "env_title": "Domaine de Vol", "env_desc": "Limites de sécurité. Zone de décrochage sous la ligne.",
        "wind_title": "Performance Décollage", "wind_desc": "Effet du vent. Le vent de face est avantageux.",
        "start": "DÉMARRER", "reset": "Effacer",
        "no_route": "⚠️ Pas de route! Sélectionnez 2 points sur la carte.",
        "click_map": "Cliquez sur la carte pour définir le départ et l'arrivée.",
        "waiting_data": "En attente...", "weather_title": "Données Atmosphériques",
        "sim_running": "Simulation en cours...", "sim_done": "Opération Terminée",
        "crash_alt_high": "🚨 ERREUR CRITIQUE: Plafond dépassé!",
        "crash_alt_low": "🚨 ERREUR CRITIQUE: Survitesse à basse altitude!",
        "crash_stall": "🚨 ERREUR CRITIQUE: Décrochage!",
        "crash_struct": "🚨 ERREUR CRITIQUE: Vitesse structurelle dépassée!",
        "sim_failed": "ÉCHEC DE L'OPÉRATION"
    },
    "RU": {
        "tabs": ["🗺️ Маршрут", "✈️ Характеристики", "📊 Анализ", "🚀 Симуляция"],
        "cockpit": "Панель Кабины", "aircraft": "Выбор Самолета",
        "params": "Параметры", "alt": "Высота (м)", "spd": "Скорость (м/с)",
        "specs_title": "Тех. Паспорт", "mass": "Масса", "span": "Размах", "len": "Длина", "eng": "Двигатель",
        "env_title": "Огибающая Полета", "env_desc": "Безопасные границы. Ниже линии - сваливание.",
        "wind_title": "Взлетные Хар-ки", "wind_desc": "Влияние ветра. Встречный ветер выгоден.",
        "start": "СТАРТ", "reset": "Сброс",
        "no_route": "⚠️ Нет маршрута! Выберите 2 точки на карте.",
        "click_map": "Нажмите на карту для выбора точек.",
        "waiting_data": "Ожидание данных...", "weather_title": "Атмосферные Данные",
        "sim_running": "Симуляция запущена...", "sim_done": "Операция Завершена",
        "crash_alt_high": "🚨 КРИТИЧЕСКАЯ ОШИБКА: Превышен потолок!",
        "crash_alt_low": "🚨 КРИТИЧЕСКАЯ ОШИБКА: Превышение скорости у земли!",
        "crash_stall": "🚨 КРИТИЧЕСКАЯ ОШИБКА: Сваливание!",
        "crash_struct": "🚨 КРИТИЧЕСКАЯ ОШИБКА: Разрушение конструкции!",
        "sim_failed": "ОПЕРАЦИЯ ПРОВАЛЕНА"
    },
    "JP": {
        "tabs": ["🗺️ ルート計画", "✈️ 機体仕様", "📊 分析", "🚀 シミュレーション"],
        "cockpit": "コックピット", "aircraft": "機体選択",
        "params": "飛行パラメータ", "alt": "高度 (m)", "spd": "速度 (m/s)",
        "specs_title": "技術データ", "mass": "質量", "span": "翼幅", "len": "全長", "eng": "エンジン",
        "env_title": "飛行包絡線", "env_desc": "安全限界を示します。線の下は失速領域です。",
        "wind_title": "離陸性能", "wind_desc": "風の影響。向かい風は離陸に有利です。",
        "start": "開始", "reset": "リセット",
        "no_route": "⚠️ ルートがありません！地図上で2点を選択してください。",
        "click_map": "地図をクリックして始点と終点を設定してください。",
        "waiting_data": "データ待機中...", "weather_title": "気象データ",
        "sim_running": "シミュレーション実行中...", "sim_done": "作戦完了",
        "crash_alt_high": "🚨 致命的エラー: 上昇限度超過!",
        "crash_alt_low": "🚨 致命的エラー: 低高度での速度超過!",
        "crash_stall": "🚨 致命的エラー: 失速 (ストール)!",
        "crash_struct": "🚨 致命的エラー: 構造限界速度超過!",
        "sim_failed": "作戦失敗"
    }
}

# --- AIRCRAFT DATABASE ---
AIRCRAFT_DB = {
    "Boeing 737-800": {
        "mass": 70000, "area": 124.6, "ceiling": 12500, "fuel_rate": 2.8, "speed_limit": 260,
        "low_alt_limit": 170,  # Max safe speed below 1000m
        "img": "b737.jpg", "icon": "plane.png",
        "length": "39.5 m", "span": "35.8 m", "engine": "2x CFM56-7B Turbofan",
        "desc": {
            "TR": "Dünyanın en popüler yolcu uçağı.", "EN": "World's most popular airliner.",
            "DE": "Beliebtestes Verkehrsflugzeug.", "FR": "L'avion de ligne le plus populaire.",
            "RU": "Самый популярный авиалайнер.", "JP": "世界で最も人気のある旅客機。"
        }
    },
    "F-16 Fighting Falcon": {
        "mass": 12000, "area": 27.8, "ceiling": 15000, "fuel_rate": 4.5, "speed_limit": 600,
        "low_alt_limit": 400,
        "img": "f16.jpg", "icon": "jet.png",
        "length": "15.06 m", "span": "9.96 m", "engine": "1x GE F110",
        "desc": {
            "TR": "Yüksek manevra kabiliyetli savaş jeti.", "EN": "High maneuverability fighter jet.",
            "DE": "Hochmanövrierfähiger Kampfjet.", "FR": "Avion de chasse très maniable.",
            "RU": "Высокоманевренный истребитель.", "JP": "高機動戦闘機。"
        }
    },
    "Cessna 172 Skyhawk": {
        "mass": 1100, "area": 16.2, "ceiling": 4100, "fuel_rate": 0.3, "speed_limit": 80,
        "low_alt_limit": 65,
        "img": "cessna.jpg", "icon": "cessna.png",
        "length": "8.28 m", "span": "11.00 m", "engine": "1x Lycoming IO-360",
        "desc": {
            "TR": "Eğitim uçağı.", "EN": "Training aircraft.",
            "DE": "Schulflugzeug.", "FR": "Avion d'entraînement.",
            "RU": "Учебно-тренировочный самолет.", "JP": "練習機。"
        }
    },
    "Custom / Manuel": {
        "mass": 5000, "area": 30.0, "ceiling": 10000, "fuel_rate": 1.5, "speed_limit": 300,
        "low_alt_limit": 200,
        "img": "custom.jpg", "icon": "custom.png",
        "length": "N/A", "span": "N/A", "engine": "Prototype",
        "desc": {
            "TR": "Deneysel.", "EN": "Experimental.",
            "DE": "Experimentell.", "FR": "Expérimental.",
            "RU": "Экспериментальный.", "JP": "実験的。"
        }
    }
}


# --- HELPER FUNCTIONS ---
def image_to_base64(img_path):
    """Converts a local image to Base64 for embedding in Folium maps."""
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None


def get_real_weather(lat, lon):
    """Fetches real-time weather data from Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url).json()
        return r['current_weather']
    except:
        return None


# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🌐 Language / Dil")
lang = st.sidebar.selectbox("", list(TRANSLATIONS.keys()))
T = TRANSLATIONS[lang]

st.sidebar.divider()
st.sidebar.header(f"✈ {T['cockpit']}")

model = st.sidebar.selectbox(T['aircraft'], list(AIRCRAFT_DB.keys()))
ac = AIRCRAFT_DB[model]

# Manual Input Logic
if model == "Custom / Manuel":
    ac["mass"] = st.sidebar.number_input(T['mass'], value=5000)
    ac["area"] = st.sidebar.number_input(T['area'], value=30.0)
    ac["ceiling"] = st.sidebar.number_input("Max Alt (m)", value=10000)

st.sidebar.subheader(T['params'])
target_alt = st.sidebar.slider(T['alt'], 0, 16000, 8000)
velocity = st.sidebar.number_input(T['spd'], value=220)

# Global Physics Calculations
rho = 1.225 * np.exp(-target_alt / 8500)
area_c = ac["area"] if ac["area"] > 0 else 1.0
stall_v = np.sqrt((2 * ac["mass"] * 9.81) / (rho * area_c * 1.6))

# --- MAIN APPLICATION ---
st.title("AEROGUARD PRO")

# Initialize Session State
if 'route' not in st.session_state: st.session_state.route = []

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs(T['tabs'])

# --- TAB 1: ROUTE PLANNING ---
with tab1:
    col_map, col_weather = st.columns([3, 1])
    with col_map:
        m = folium.Map(location=[39.0, 35.0], zoom_start=5, tiles="OpenStreetMap")

        # Draw Existing Markers
        if len(st.session_state.route) > 0:
            folium.Marker(st.session_state.route[0], icon=folium.Icon(color="green", icon="play")).add_to(m)
        if len(st.session_state.route) == 2:
            folium.Marker(st.session_state.route[1], icon=folium.Icon(color="red", icon="flag")).add_to(m)
            folium.PolyLine(st.session_state.route, color="blue", weight=4).add_to(m)

            # Custom Aircraft Icon
            b64 = image_to_base64(os.path.join("images", ac["icon"]))
            if b64:
                icon_obj = folium.CustomIcon(f"data:image/png;base64,{b64}", icon_size=(45, 45))
                folium.Marker(st.session_state.route[0], icon=icon_obj).add_to(m)

        # Map Click Interaction
        map_data = st_folium(m, height=500, width="100%", key="main_map")

        if map_data and map_data['last_clicked']:
            pt = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
            if len(st.session_state.route) < 2:
                # Prevent double-click adding same point
                if not st.session_state.route or st.session_state.route[-1] != pt:
                    st.session_state.route.append(pt)
                    st.rerun()

        if st.button(T['reset']):
            st.session_state.route = []
            st.rerun()

    with col_weather:
        st.subheader(T['weather_title'])
        if len(st.session_state.route) > 0:
            last_pt = st.session_state.route[-1]
            w = get_real_weather(last_pt[0], last_pt[1])
            if w:
                st.markdown(f"""
                <div class="spec-card">
                <b>🌡️ Temp:</b> {w['temperature']} °C<br>
                <b>💨 Wind:</b> {w['windspeed']} km/h<br>
                <b>🧭 Dir:</b> {w['winddirection']}°
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Service Unavailable")
        else:
            st.caption(T['waiting_data'])

# --- TAB 2: SPECIFICATIONS ---
with tab2:
    c_img, c_info = st.columns([1, 2])
    with c_img:
        img_p = os.path.join("images", ac["img"])
        if os.path.exists(img_p): st.image(img_p)
    with c_info:
        st.subheader(f"{model}")
        # Get description based on language, default to English
        desc_text = ac['desc'].get(lang, ac['desc']['EN'])
        st.markdown(f"""
        <div class="spec-card">
        {desc_text}<br><br>
        <b>{T['mass']}:</b> {ac['mass']} kg<br>
        <b>{T['len']}:</b> {ac['length']}<br>
        <b>{T['eng']}:</b> {ac['engine']}
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: ANALYSIS ---
with tab3:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader(T['env_title'])
        # Flight Envelope Plot
        alts = np.linspace(0, 16000, 100)
        stalls = [np.sqrt((2 * ac["mass"] * 9.81) / (1.225 * np.exp(-a / 8500) * area_c * 1.6)) for a in alts]

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#1b1e24');
        ax.set_facecolor('#0e1117')
        ax.plot(alts, stalls, color='#00ffcc', linewidth=2)
        ax.fill_between(alts, stalls, 1000, color='#00ffcc', alpha=0.1)

        pt_c = '#00ff00' if velocity > stall_v else '#ff0000'
        ax.scatter(target_alt, velocity, color=pt_c, s=150, zorder=5)

        ax.set_xlabel(T['alt'], color='white');
        ax.set_ylabel(T['spd'], color='white')
        ax.tick_params(colors='white');
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with col_g2:
        st.subheader(T['wind_title'])
        # Wind vs Ground Speed Plot
        winds = np.linspace(-30, 30, 100)
        req_speed = (stall_v * 1.1) + winds

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#1b1e24');
        ax2.set_facecolor('#0e1117')
        ax2.plot(winds, req_speed, color='#ff00ff', linewidth=2)
        ax2.axvline(0, color='white', linestyle='--')

        ax2.set_xlabel("Wind (m/s)", color='white');
        ax2.set_ylabel("Ground Speed", color='white')
        ax2.tick_params(colors='white');
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

# --- TAB 4: SIMULATION (ADVANCED LOGIC) ---
with tab4:
    if len(st.session_state.route) == 2:
        if st.button(T['start'], type="primary"):
            st.info(T['sim_running'])
            prog = st.progress(0)
            status_text = st.empty()

            c1, c2 = st.columns(2)
            m1 = c1.empty();
            m2 = c2.empty()

            # --- CRASH LOGIC ---
            crash_type = None

            # 1. Ceiling Check
            if target_alt > ac["ceiling"]:
                crash_type = "ALT_HIGH"

            # 2. Stall Check
            elif velocity < stall_v:
                crash_type = "STALL"

            # 3. Structural Speed Limit (Vne) Check
            elif velocity > ac["speed_limit"]:
                crash_type = "STRUCT"

            # 4. Low Altitude Overspeed (Dynamic Pressure) Check
            # If below 1000m and speed is above the low_alt_limit
            elif target_alt < 1000 and velocity > ac.get("low_alt_limit", 200):
                crash_type = "ALT_LOW_SPEED"

            # Simulation Loop
            for i in range(101):
                time.sleep(0.04)
                prog.progress(i)

                # Update Metrics
                curr_alt = int(target_alt * (i / 100))
                m1.metric(T['alt'], f"{curr_alt} m")
                m2.metric("RPM", f"{int(90 + np.random.randn() * 2)} %")

                # Trigger Crash at 60% if condition met
                if crash_type and i == 60:
                    time.sleep(1)  # Dramatic pause

                    if crash_type == "ALT_HIGH":
                        msg = T['crash_alt_high']
                    elif crash_type == "STALL":
                        msg = T['crash_stall']
                    elif crash_type == "STRUCT":
                        msg = T['crash_struct']
                    elif crash_type == "ALT_LOW_SPEED":
                        msg = T['crash_alt_low']

                    status_text.markdown(f"<div class='crash-box'>{msg}<br>{T['sim_failed']}</div>",
                                         unsafe_allow_html=True)
                    st.error(msg)
                    break

            # Success (Only if no crash)
            if not crash_type:
                st.success(f"✅ {T['sim_done']}")
                st.balloons()
    else:
        st.markdown(f"<div class='warning-box'>{T['no_route']}</div>", unsafe_allow_html=True)