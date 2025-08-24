import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

# Custom CSS for styling (unchanged from your Streamlit code)

st.markdown("""
    <style>
    .stApp {
        background-color: #505E05 !important;
        color: #1A202C !important;
    }
    h1 {
        color: #F8E288 !important;
        font-family: 'Roboto', sans-serif;
        font-size: 50px!important;
    }
    h2 {
        color: #F8E288 !important;
        font-family: 'Roboto', sans-serif;
        font-size: 32px !important;
    }
    h3 {
        color: #BEDF71 !important;
        font-family: 'Roboto', sans-serif;
        font-size: 28px !important;
    }
    .stButton>button {
        background-color: #F8E288 !important;
        color: #920C0D !important;
        border-radius: 5px;
        font-size: 18px !important;
        font-family: 'Roboto', sans-serif;
    }
    .stNumberInput input {
        background-color: #F8E288 !important;
        color: #920C0D !important;
        border: 1px solid #2B6CB0;
        border-radius: 5px;
        font-size: 18px !important;
        font-family: 'Roboto', sans-serif;
    }
    .stSelectbox {
        color: #F8E288 !important;
        font-size: 18px !important;
        font-family: 'Roboto', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #E2E8F0 !important;
        color: #F8E288 !important;
        font-size: 18px !important;
        font-family: 'Roboto', sans-serif;
    }
    .stMarkdown p, .stMarkdown li {
        color: #F9FDBB !important;
        font-size: 20px !important;
        font-family: 'Roboto', sans-serif;
    }
    [theme]
primaryColor="#0a3f05"
backgroundColor="#1b3102"
secondaryBackgroundColor="#666909"
textColor="#bae080"
}
    </style>
""", unsafe_allow_html=True)

# Load data with added columns for modeling
@st.cache_data
def load_model_data():
    df = pd.read_csv("TexasTurbine.csv")
    df["Time stamp"] = "2023 " + df["Time stamp"].astype(str)
    df["Time stamp"] = pd.to_datetime(df["Time stamp"], format="%Y %b %d, %I:%M %p", errors='coerce')
    df["hour"] = df["Time stamp"].dt.hour
    df["month"] = df["Time stamp"].dt.month
    df["day"] = df["Time stamp"].dt.day
    df[["hour", "month", "day"]] = df[["hour", "month", "day"]].fillna(0).astype(int)
    return df

df = load_model_data()
@st.cache_resource
def train_models(X_train_scaled, y_train, y_train_scaled, X_test_scaled):
    models_dict = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=500, random_state=42),
        "LSTM": Sequential([
            Input(shape=(X_train_scaled.shape[1], 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
    }

    results = {}
    y_pred_dict = {}
    for name, model in models_dict.items():
        start = time.time()
        try:
            if name == "LSTM":
                model.compile(optimizer="adam", loss="mse")
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    np.expand_dims(X_train_scaled, axis=-1),
                    y_train_scaled,
                    validation_split=0.1,
                    epochs=200,
                    batch_size=32,
                    callbacks=[es],
                    verbose=0
                )
                y_pred = model.predict(np.expand_dims(X_test_scaled, axis=-1), verbose=0).flatten()
                y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()  # تصحيح الخطأ
            else:
                model.fit(X_train_scaled, y_train)
                y_pred_orig = model.predict(X_test_scaled)
            end = time.time()

            y_test_orig = y_test
            y_pred_dict[name] = y_pred_orig
            r2 = r2_score(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            training_time = end - start

            results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "Time (sec)": training_time}
        except Exception as e:
            st.error(f"خطأ في تدريب {name}: {str(e)}")
            results[name] = {"R2": 0, "RMSE": float('inf'), "MAE": float('inf'), "Time (sec)": 0}
            y_pred_dict[name] = np.zeros_like(y_test)

    return models_dict, results, y_pred_dict
# Tabs for pages
tab1, tab2 = st.tabs(["Project Overview", "Models and Predictions"])

with tab1:
    # Project Description (unchanged)
    st.markdown("""
    # 💨 **Wind Turbine Power Prediction**
    Wind turbines are a cornerstone of renewable energy, harnessing wind power to generate clean electricity. This project uses machine learning to predict turbine power output based on environmental factors like wind speed, direction, temperature, and pressure. Accurate predictions help optimize turbine performance, improve energy efficiency, and support sustainable energy solutions.
    """, unsafe_allow_html=True)
    st.image("https://images.pexels.com/photos/31635211/pexels-photo-31635211.jpeg")

    st.markdown("""
    ## <font color="#BEDF71">Overview</font>  
    Predicting the **power output** of a wind turbine using **machine learning**.  
    Dataset = 8760 hourly records with:  
    -  `Power Generated (kW)` → **Target** 
    -  Wind Speed (m/s)  
    -  Wind Direction (°)  
    -  Temperature (°C)  
    -  Pressure (atm)  

    ##  <font color="#BEDF71">Goal</font>  
    - Build ML models to estimate **power output**  
    - Identify key environmental factors  

    ##  <font color="#BEDF71">Methods</font>  
    ·  EDA  
    ·  Evaluation with RMSE, MAE, R²  

    ##  <font color="#BEDF71">Outcome</font>  
    - Accurate prediction of turbine power  
    - Insights into factors affecting generation
    """, unsafe_allow_html=True)    
    st.markdown("""
    ##  <font color="#BEDF71">Dataset Columns</font>
    ` Time stamp:` Date & time of measurement (hourly data, full year)  
    `System power generated (kW):` Target variable → energy produced by the wind turbine  
    `Wind speed (m/s):` Main predictor, directly affects turbine power  
    `Wind direction (deg):`Angle of wind flow, may influence efficiency  
    `Air temperature (°C):` Weather factor affecting air density & power  
    `Pressure (atm):` Atmospheric pressure, can impact wind characteristics  
        """, unsafe_allow_html=True)

    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.subheader("Descriptive Stats")
    st.write(df.describe())
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="greens",
        title="Correlation Matrix for Wind Turbine Data",
        height=800
    )
    fig.update_layout(
        title_font_size=28,
        title_font_color="#BEDF71",
        margin=dict(l=100, r=150, t=150, b=150),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=45, tickfont=dict(size=15, color="#BEDF71")),
        yaxis=dict(tickfont=dict(size=15, color="#BEDF71")), 
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Learn More About Wind Turbines")
    st.video("https://www.youtube.com/watch?v=uzLtx1MCtzE")

with tab2:
    try:
        # Use same features as Python project
        features = ["Wind speed | (m/s)", "month", "hour", "Wind direction | (deg)", "Pressure | (atm)", "day"]
        X = df[features]
        y = df["System power generated | (kW)"]

        # Check for missing values
        if X.isnull().any().any() or y.isnull().any():
            st.error("Error: Missing values detected in the dataset. Please check the data.")
            st.stop()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Scale data with StandardScaler (same as Python project)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


        # Model Training
        models_dict, results, y_pred_dict = train_models(X_train_scaled, y_train, y_train_scaled, X_test_scaled)

        # Bar Chart for all models
        st.subheader("Performance of All Models")
        perf_df = pd.DataFrame.from_dict(results, orient='index')
        metrics = st.multiselect(
            "Select Metrics to Display",
            options=['R2', 'RMSE', 'MAE', 'Time (sec)'],
            default=['R2', 'RMSE', 'MAE', 'Time (sec)']
        )
        if metrics:
            fig = px.bar(
                perf_df.reset_index().melt(id_vars='index', value_vars=metrics, var_name='Metric', value_name='Value'),
                x='index',
                y='Value',
                color='Metric',
                barmode='group',
                title="Performance Metrics for All Models",
                color_discrete_map={
                    'R2': "#139462",
                    'RMSE': "#D9F00B",
                    'MAE': "#D31D78",
                    'Time (sec)': "#261ACC"
                }
            )
            fig.update_layout(
                height=600,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Roboto", size=20, color="#E7DFFA"),
                title_font_size=28,
                xaxis_title="Model",
                yaxis_title="Value",
                xaxis=dict(tickangle=45, tickfont=dict(size=18, color="#E7DFFA")),
                yaxis=dict(tickfont=dict(size=18, color="#E7DFFA")),
                legend_title="Metric"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one metric to display.")

        # Scatter Plots for Actual vs Predicted Power Output
        st.subheader("Actual vs Predicted Power Output")

        # Actual values
        y_true = y_test  # Already in original scale

        # Models info
        models = ["Ridge", "XGBoost", "Neural Network", "LSTM"]
        preds = [y_pred_dict[model] for model in models]
        colors = ["blue", "green", "orange", "purple"]

        # Create 2x2 subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{model}<br>RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_dict[model])):.3f}" for model in models],
            shared_yaxes=True
        )

        # Add scatter plots and diagonal lines
        for i, (model, pred, color) in enumerate(zip(models, preds, colors), 1):
            if len(pred) > 0:  # Ensure predictions exist
                row = 1 if i <= 2 else 2
                col = i if i <= 2 else i - 2
                fig.add_trace(
                    go.Scatter(
                        x=y_true,
                        y=pred,
                        mode='markers',
                        marker=dict(color=color, opacity=0.5, size=8),
                        name=model,
                        text=[f"Actual: {x:.2f}, Predicted: {y:.2f}" for x, y in zip(y_true, pred)],
                        hovertemplate="%{text}<br>x: %{x}<br>y: %{y}",
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=[max(0, y_true.min()), y_true.max()],
                        y=[max(0, y_true.min()), y_true.max()],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Ideal',
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

        # Update layout to mimic Matplotlib
        fig.update_layout(
            height=600,
            width=1200,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=12, color="white"),
            title=dict(
                text="Actual vs Predicted Power Output",
                font=dict(size=14, color="white"),
                x=0.5,
                xanchor="center"
            ),
            showlegend=False
        )
        fig.update_xaxes(
            title_text="Actual Power (kW)",
            title_font=dict(size=10),
            tickfont=dict(size=10, color="white"),
            gridcolor="rgba(200,200,200,0.2)",
            range=[0, max(y_true.max(), max([pred.max() for pred in preds]))]
        )
        fig.update_yaxes(
            title_text="Predicted Power (kW)",
            title_font=dict(size=10),
            tickfont=dict(size=10, color="white"),
            gridcolor="rgba(200,200,200,0.2)",
            range=[0, max(y_true.max(), max([pred.max() for pred in preds]))]
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Table for all metrics
        st.subheader("All Models Metrics")
        st.table(perf_df)

        # Model Selection
        model_choice = st.selectbox("Select Model", list(models_dict.keys()))

        # Prediction
        st.subheader("Make a Prediction")
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
        hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
        wind_direction = st.number_input("Wind Direction (deg)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
        pressure = st.number_input("Pressure (atm)", min_value=0.8, max_value=1.2, value=1.0, step=0.01)
        day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)

        input_data = np.array([[wind_speed, month, hour, wind_direction, pressure, day]])
        input_scaled = scaler_X.transform(input_data)

        model = models_dict[model_choice]
        try:
            if model_choice == "LSTM":
                pred_scaled = model.predict(np.expand_dims(input_scaled, axis=-1), verbose=0)
                pred_power = y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            else:
                pred_power = model.predict(input_scaled)[0]
            st.write(f"**Predicted Power Output**: {pred_power:.2f} kW")
        except Exception as e:
            st.error(f"Error making prediction with {model_choice}: {str(e)}")

    except Exception as e:
        st.error(f"Error in Tab 2: {str(e)}")
        st.stop()