

import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Solar Command Center",
    page_icon="☀️",
    layout="wide"
)

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

# ---------------- UI DESIGN ---------------- #

st.markdown("""
<style>

/* Background Image */

.stApp{
    background-image:
    linear-gradient(
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.75)
    ),

    url("https://thumbs.dreamstime.com/z/solar-panels-roof-large-house-solar-panels-roof-modern-house-133375048.jpg");

    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

/* Hide Streamlit */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero */

.hero{

    padding:50px;

    border-radius:35px;

    background:
    rgba(255,255,255,0.05);

    backdrop-filter:blur(16px);

    border:
    1px solid rgba(255,255,255,0.08);

    text-align:center;

    box-shadow:
    0px 0px 50px rgba(0,0,0,0.5);
}

/* Cards */

.card{

    background:
    rgba(255,255,255,0.06);

    padding:25px;

    border-radius:25px;

    backdrop-filter:blur(10px);

    border:
    1px solid rgba(255,255,255,0.08);

    transition:0.4s;
}

.card:hover{

    transform:
    translateY(-8px);

    box-shadow:
    0px 0px 30px rgba(255,255,255,0.1);
}

/* Text */

h1{
    color:white;
    font-size:70px;
    font-weight:900;
}

h2,h3,h4{
    color:#FFD700;
}

p,label,div,li{
    color:white;
}

/* Button */

.stButton>button{

    width:100%;

    height:65px;

    border:none;

    border-radius:20px;

    background:
    linear-gradient(
        to right,
        #ff9800,
        #ff0066
    );

    color:white;

    font-size:20px;

    font-weight:bold;
}

/* Sidebar */

section[data-testid="stSidebar"]{

    background:
    rgba(0,0,0,0.8);
}

/* Metrics */

.metric{

    text-align:center;

    padding:25px;

    border-radius:25px;

    background:
    rgba(255,255,255,0.05);

    border:
    1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ---------------- #

st.markdown("""
<div class="hero">

<h1>☀️ SOLAR COMMAND CENTER</h1>

<h3>
Future AI Solar Monitoring System
</h3>

<p>
Real-Time Solar Analytics,
Weather Intelligence,
Battery Monitoring,
Power Prediction &
Solar Safety Dashboard
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SOLAR COMPONENTS ---------------- #

st.subheader("☀️ Important Solar System Components")

c1,c2,c3,c4,c5 = st.columns(5)

with c1:

    st.markdown("""
    <div class="card">

    <h3>☀️ Solar Panel</h3>

    <p>
    Sunlight collect karta hai
    </p>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="card">

    <h3>🔋 Battery</h3>

    <p>
    Energy storage system
    </p>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="card">

    <h3>⚡ Inverter</h3>

    <p>
    DC ko AC me convert karta hai
    </p>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="card">

    <h3>🌡 Temperature</h3>

    <p>
    Overheating detect karta hai
    </p>

    </div>
    """, unsafe_allow_html=True)

with c5:

    st.markdown("""
    <div class="card">

    <h3>☁️ Weather</h3>

    <p>
    Energy production impact
    </p>

    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- PREPROCESSING ---------------- #

df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

df['Year'] = df['DATE_TIME'].dt.year
df['Month'] = df['DATE_TIME'].dt.month
df['Day'] = df['DATE_TIME'].dt.day
df['Hour'] = df['DATE_TIME'].dt.hour

df.drop(
    ['DATE_TIME','SOURCE_KEY'],
    axis=1,
    inplace=True
)

# ---------------- ADD EXTRA FEATURES ---------------- #

df['Battery_Level'] = np.random.randint(
    50,100,len(df)
)

df['Panel_Efficiency'] = (
    df['IRRADIATION'] /
    (df['MODULE_TEMPERATURE'] + 1)
) * 10

df['Power_Output'] = (
    df['IRRADIATION'] *
    df['Panel_Efficiency']
)

# ---------------- FEATURES ---------------- #

X = df.drop("IRRADIATION", axis=1)

y = df["IRRADIATION"]

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ---------------- #

with st.spinner("🤖 AI Engine Running..."):

    time.sleep(2)

    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = r2_score(y_test, pred)

# ---------------- DASHBOARD ---------------- #

st.subheader("📊 Real-Time Solar AI Dashboard")

m1,m2,m3,m4 = st.columns(4)

with m1:

    st.markdown(f"""
    <div class="metric">

    <h3>AI Accuracy</h3>

    <h1>{accuracy:.3f}</h1>

    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown(f"""
    <div class="metric">

    <h3>Battery Level</h3>

    <h1>{df['Battery_Level'].mean():.0f}%</h1>

    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown(f"""
    <div class="metric">

    <h3>Avg Efficiency</h3>

    <h1>{df['Panel_Efficiency'].mean():.2f}</h1>

    </div>
    """, unsafe_allow_html=True)

with m4:

    st.markdown(f"""
    <div class="metric">

    <h3>Power Output</h3>

    <h1>{df['Power_Output'].mean():.2f}</h1>

    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4 = st.tabs([
    "📂 Dataset",
    "⚡ Energy Analytics",
    "🔮 AI Prediction",
    "🛡 Solar Safety"
])

# ---------------- DATASET ---------------- #

with tab1:

    st.subheader("📂 Solar Dataset Preview")

    st.dataframe(df.head(100))

# ---------------- ANALYTICS ---------------- #

with tab2:

    st.subheader("⚡ Smart Energy Analytics")

    st.metric(
        "Maximum Temperature",
        round(df['MODULE_TEMPERATURE'].max(),2)
    )

    st.metric(
        "Average Irradiation",
        round(df['IRRADIATION'].mean(),2)
    )

    st.metric(
        "Average Power Output",
        round(df['Power_Output'].mean(),2)
    )

    st.success(
        "🌞 High irradiation increases "
        "solar power generation."
    )

# ---------------- PREDICTION ---------------- #

with tab3:

    st.subheader("🔮 Live Solar AI Prediction")

    col1,col2 = st.columns(2)

    with col1:

        ambient = st.slider(
            "Ambient Temperature",
            0.0,60.0,25.0
        )

        module = st.slider(
            "Module Temperature",
            0.0,80.0,22.0
        )

        battery = st.slider(
            "Battery Level",
            0,100,80
        )

    with col2:

        year = st.number_input(
            "Year",
            value=2020
        )

        month = st.slider(
            "Month",
            1,12,5
        )

        hour = st.slider(
            "Hour",
            0,23,12
        )

    if st.button("⚡ Generate Solar Prediction"):

        efficiency = (
            ambient /
            (module + 1)
        ) * 10

        power = efficiency * ambient

        sample = np.array([[
            4135001,
            ambient,
            module,
            year,
            month,
            15,
            hour,
            battery,
            efficiency,
            power
        ]])

        prediction = model.predict(sample)

        st.success(
            f"☀️ Predicted Irradiation = "
            f"{prediction[0]:.4f}"
        )

        if prediction[0] > 0.5:

            st.balloons()

            st.info(
                "🌞 Excellent Solar Production"
            )

        else:

            st.warning(
                "⛅ Low Solar Production"
            )

# ---------------- SAFETY ---------------- #

with tab4:

    st.subheader("🛡 Solar Panel Safety Guide")

    st.markdown("""

### ☀️ Solar Panel Maintenance

- Regular cleaning important hai
- Dust remove kare
- Temperature monitor kare
- Wiring check kare
- Battery health monitor kare

---

### ⚡ Important Components

- Solar Panel
- Battery
- Inverter
- Charge Controller
- Temperature Sensor
- Weather Monitoring

---

### 🌞 Benefits

- Low electricity bill
- Green energy
- Long-term saving
- Pollution free

    """)

# ---------------- FOOTER ---------------- #

st.markdown("""
<hr>

<center>

<h4 style='color:white;'>
☀️ SOLAR COMMAND CENTER AI
</h4>

<p style='color:gray;'>
Next Generation Smart Solar Platform
</p>

</center>
""", unsafe_allow_html=True)






















