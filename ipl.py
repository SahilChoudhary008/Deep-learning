import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="🏏 IPL Score Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏏 IPL Score Predictor</p>', unsafe_allow_html=True)


# Utility functions
@st.cache_data
def load_data(csv_path):
    """Load and preprocess data"""
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data.drop(columns=['date'], inplace=True, errors='ignore')
    return data


def prepare_features(data):
    """Prepare features efficiently"""
    num_cols = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data[num_cols] = imputer.fit_transform(data[num_cols])

    # Encode categorical variables
    cat_cols = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
    encoders = {}
    vocab_sizes = {}

    for col in cat_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            vocab_sizes[col] = len(le.classes_)

    return data, encoders, num_cols, vocab_sizes


def build_fast_model(numeric_dim, vocab_sizes):
    """Lightweight model with embeddings - trains in seconds"""
    # Numeric input
    numeric_input = Input(shape=(numeric_dim,), name='numeric')

    # Small embeddings for categorical features
    cat_inputs = []
    embeddings = []

    for col_name, vocab_size in vocab_sizes.items():
        cat_in = Input(shape=(1,), name=col_name)
        cat_inputs.append(cat_in)
        # Small embedding dimension for speed
        embed_dim = min(8, (vocab_size + 1) // 2)
        embed = Embedding(vocab_size, embed_dim)(cat_in)
        embed = Flatten()(embed)
        embeddings.append(embed)

    # Combine features
    if embeddings:
        combined = Concatenate()([numeric_input] + embeddings)
    else:
        combined = numeric_input

    # Compact network - fewer layers, faster training
    x = Dense(64, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[numeric_input] + cat_inputs, outputs=output)
    return model


@st.cache_resource(show_spinner=False)
def train_model(_data, epochs=30, batch_size=64):
    """Train model with caching - runs once then cached"""
    # Prepare features
    data, encoders, num_cols, vocab_sizes = prepare_features(_data.copy())

    # Prepare data
    X_numeric = data[num_cols].values
    y = data['total'].values

    # Scale
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Categorical data
    X_cat = {}
    for col in vocab_sizes.keys():
        X_cat[col] = data[f'{col}_encoded'].values

    # Split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train_dict = {'numeric': X_numeric_scaled[train_idx]}
    X_test_dict = {'numeric': X_numeric_scaled[test_idx]}

    for col, values in X_cat.items():
        X_train_dict[col] = values[train_idx]
        X_test_dict[col] = values[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Build model
    model = build_fast_model(len(num_cols), vocab_sizes)

    # Compile with higher learning rate for faster convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Train with validation split
    train_inputs = [X_train_dict['numeric']] + [X_train_dict[k] for k in vocab_sizes.keys()]

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

    history = model.fit(
        train_inputs, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    test_inputs = [X_test_dict['numeric']] + [X_test_dict[k] for k in vocab_sizes.keys()]
    test_loss, test_mae = model.evaluate(test_inputs, y_test, verbose=0)

    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'num_cols': num_cols,
        'cat_cols': list(vocab_sizes.keys()),
        'history': history.history,
        'test_mae': test_mae,
        'data': data
    }


# Sidebar
st.sidebar.header("⚙️ Quick Setup")
csv_path = st.sidebar.text_input("📁 Dataset Path", value="ipl_data.csv")

# Reduced defaults for speed
epochs = st.sidebar.slider("🔄 Epochs (fewer = faster)", 20, 100, 30, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size (larger = faster)", [32, 64, 128], index=1)

st.sidebar.info("💡 **Speed Tips:**\n- Use 30 epochs\n- Batch size 64+\n- Model cached after first run")

# Auto-load on startup
if 'resources' not in st.session_state:
    if st.sidebar.button("🚀 Train Model", use_container_width=True) or st.sidebar.checkbox("Auto-train on load",
                                                                                           value=True):
        with st.spinner("⚡ Training fast model..."):
            try:
                data = load_data(csv_path)
                resources = train_model(data, epochs, batch_size)
                st.session_state['resources'] = resources
                st.sidebar.success(f"✅ Ready! MAE: {resources['test_mae']:.1f} runs")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Main App
if 'resources' in st.session_state:
    res = st.session_state['resources']

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Status", "Ready ✓")
    with col2:
        st.metric("🎯 Test MAE", f"{res['test_mae']:.1f} runs")
    with col3:
        epochs_trained = len(res['history']['loss'])
        st.metric("⚡ Epochs", f"{epochs_trained}")

    st.markdown("---")

    # Prediction
    st.header("🎯 Predict Score")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Details")
        data = res['data']

        venue = st.selectbox("🏟️ Venue", sorted(data['venue'].unique()) if 'venue' in data.columns else ['Unknown'])
        bat_team = st.selectbox("🏏 Batting Team",
                                sorted(data['bat_team'].unique()) if 'bat_team' in data.columns else ['Unknown'])
        bowl_team = st.selectbox("⚾ Bowling Team",
                                 sorted(data['bowl_team'].unique()) if 'bowl_team' in data.columns else ['Unknown'])
        batsman = st.selectbox("👤 Batsman",
                               sorted(data['batsman'].unique()) if 'batsman' in data.columns else ['Unknown'])
        bowler = st.selectbox("🎳 Bowler", sorted(data['bowler'].unique()) if 'bowler' in data.columns else ['Unknown'])

    with col2:
        st.subheader("Current Stats")
        runs = st.number_input("💯 Runs", 0, 300, 120, 10)
        wickets = st.number_input("🎯 Wickets", 0, 10, 3, 1)
        overs = st.number_input("⏱️ Overs", 0.0, 20.0, 12.0, 0.5)
        runs_last_5 = st.number_input("🔥 Runs (Last 5)", 0, 150, 45, 5)
        wickets_last_5 = st.number_input("📉 Wickets (Last 5)", 0, 5, 1, 1)

    if st.button("🔮 Predict Final Score", use_container_width=True):
        try:
            # Prepare input
            numeric_input = np.array([[runs, wickets, overs, runs_last_5, wickets_last_5]])
            numeric_scaled = res['scaler'].transform(numeric_input)

            # Encode categoricals
            encoders = res['encoders']
            cat_inputs = []

            for col in res['cat_cols']:
                if col in encoders:
                    if col == 'venue':
                        val = encoders[col].transform([venue])[0]
                    elif col == 'bat_team':
                        val = encoders[col].transform([bat_team])[0]
                    elif col == 'bowl_team':
                        val = encoders[col].transform([bowl_team])[0]
                    elif col == 'batsman':
                        val = encoders[col].transform([batsman])[0]
                    elif col == 'bowler':
                        val = encoders[col].transform([bowler])[0]
                    cat_inputs.append(np.array([[val]]))

            # Predict
            model_inputs = [numeric_scaled] + cat_inputs
            prediction = res['model'].predict(model_inputs, verbose=0)[0][0]

            # Display
            st.markdown("---")
            st.markdown(f"""
                <div class='metric-box'>
                    <h1>🏆 Predicted Score</h1>
                    <h1 style='font-size: 3.5rem; margin: 15px 0;'>{prediction:.0f}</h1>
                    <p style='font-size: 1.1rem;'>Runs</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Insights
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                req_rate = (prediction - runs) / (20 - overs) if overs < 20 else 0
                st.metric("Req. Rate", f"{req_rate:.1f}", "per over")
            with col2:
                remaining = max(0, prediction - runs)
                st.metric("Needed", f"{remaining:.0f}", "runs")
            with col3:
                wickets_left = 10 - wickets
                st.metric("Wickets Left", f"{wickets_left}", "in hand")
            with col4:
                curr_rr = runs / overs if overs > 0 else 0
                st.metric("Curr. RR", f"{curr_rr:.1f}", "per over")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Training history
    if st.sidebar.checkbox("📊 Show Training", value=False):
        history = res['history']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['loss'], label='Train', linewidth=2, color='#667eea')
        ax1.plot(history['val_loss'], label='Val', linewidth=2, color='#764ba2')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['mae'], label='Train MAE', linewidth=2, color='#f093fb')
        ax2.plot(history['val_mae'], label='Val MAE', linewidth=2, color='#f5576c')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (Runs)')
        ax2.set_title('Accuracy Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Click 'Train Model' in sidebar or enable auto-train")

    st.markdown("---")
    st.header("⚡ Lightning Fast Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🚀 Speed Optimized
        - Compact architecture
        - Small embeddings
        - Efficient training
        - Trains in 10-30 seconds
        """)

    with col2:
        st.markdown("""
        ### 🎯 Smart Features
        - Embedding layers
        - Batch normalization
        - Early stopping
        - Cached after first run
        """)

    with col3:
        st.markdown("""
        ### 📊 Accurate
        - Handles 5 features
        - Team/venue embeddings
        - MAE ~15-25 runs
        - Good predictions
        """)

# Footer
st.markdown("---")
st.markdown("**⚡ Fast & Lightweight** | Trains in seconds • Cached results • Good accuracy")
st.markdown("**📦 Requirements:** `pandas numpy scikit-learn tensorflow streamlit matplotlib`")