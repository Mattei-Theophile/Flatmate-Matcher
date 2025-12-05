import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(page_title="AI Housemate Matcher", page_icon="🏠", layout="wide")


# --- Load Models ---
@st.cache_resource
def load_models():
    if not os.path.exists("kmeans_model.pkl"):
        return None, None, None

    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    weights = joblib.load("weights.pkl")
    return kmeans, scaler, weights


kmeans, scaler, weights = load_models()

# --- Session State for Leaderboard ---
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# --- Sidebar: Leaderboard ---
with st.sidebar:
    st.header("🏆 Leaderboard")
    st.caption("Most recent results")

    if st.session_state.leaderboard:
        # Convert list of dicts to DataFrame
        df_leaderboard = pd.DataFrame(st.session_state.leaderboard)
        # FIX: Replaced use_container_width with width="stretch"
        st.dataframe(df_leaderboard, width="stretch", hide_index=True)
    else:
        st.info("No results yet. Run a prediction to join the board!")

    st.markdown("---")
    st.markdown("**About**\n\nThis app uses K-Means clustering to match housemates based on lifestyle compatibility.")


# --- Helper Functions ---

def get_user_inputs(prefix=""):
    """
    Reusable widget function to collect inputs.
    prefix: string to ensure unique keys for widgets (e.g., 'A_', 'B_')
    """
    st.markdown(f"**Demographics & Basics ({prefix.strip('_') if prefix else 'You'})**")
    col1, col2 = st.columns(2)

    with col1:
        # Only ask for Name/Team if it's the main user (for leaderboard purposes)
        if prefix == "main_":
            name = st.text_input("Name", key=f"{prefix}name")
        else:
            name = st.text_input("Name/Alias", key=f"{prefix}name")

        gender = st.selectbox("Gender", ["Female", "Male"], key=f"{prefix}gender")
        age_cat = st.selectbox("Age Group", ["Less than 20", "21 to 35", "36 to 50", "51 or more"], key=f"{prefix}age")
        income = st.selectbox("Sufficient Income?", ["No", "Yes"], key=f"{prefix}income")

        st.markdown("**Daily Habits**")
        fruits = st.slider("Fruits & Veggies (Servings)", 0, 5, 3, key=f"{prefix}fruits")
        steps = st.number_input("Daily Steps", 0, 30000, 5000, key=f"{prefix}steps")
        sleep = st.slider("Sleep Hours", 3, 12, 7, key=f"{prefix}sleep")

    with col2:
        st.markdown("**Psychology & Social**")
        stress = st.slider("Daily Stress (1-5)", 1, 5, 3, key=f"{prefix}stress")
        social = st.slider("Social Connections (0-10)", 0, 10, 5, key=f"{prefix}social")
        passion = st.slider("Time for Passion (Hrs)", 0, 10, 2, key=f"{prefix}passion")

        with st.expander("Advanced Details"):
            places = st.number_input("New Places Visited", 0, 50, 2, key=f"{prefix}places")
            core_circle = st.number_input("Core Circle Size", 0, 20, 3, key=f"{prefix}core")
            support = st.slider("Supporting Others", 0, 10, 5, key=f"{prefix}support")
            donation = st.slider("Donation Freq", 0, 5, 1, key=f"{prefix}donate")
            shouting = st.slider("Daily Shouting", 0, 10, 1, key=f"{prefix}shout")

    # Return a dictionary of all inputs
    return {
        "name": name, "gender": gender, "age_cat": age_cat,
        "income": income, "fruits": fruits, "steps": steps, "sleep": sleep,
        "stress": stress, "social": social, "passion": passion, "places": places,
        "core_circle": core_circle, "support": support, "donation": donation,
        "shouting": shouting
    }


def process_data(inputs):
    """
    Converts raw input dictionary into the weighted vector expected by the model.
    """
    # 1. Map Inputs
    gender_val = 1 if inputs["gender"] == "Male" else 0
    age_map = {'Less than 20': 16, '21 to 35': 28, '36 to 50': 43, '51 or more': 70}
    age_val = age_map[inputs["age_cat"]]
    income_val = 2 if inputs["income"] == "Yes" else 1

    # 2. Construct Vector (Order must match training data EXACTLY)
    raw_data_array = np.array([[
        inputs["fruits"], inputs["stress"], inputs["places"], inputs["core_circle"],
        inputs["support"], inputs["social"], inputs["donation"],
        inputs["steps"],
        inputs["sleep"], inputs["shouting"], income_val,
        inputs["passion"],
        age_val, gender_val
    ]])

    # FIX: Convert to DataFrame using the scaler's stored feature names
    # This prevents the "X does not have valid feature names" warning
    if hasattr(scaler, "feature_names_in_"):
        raw_data = pd.DataFrame(raw_data_array, columns=scaler.feature_names_in_)
    else:
        raw_data = raw_data_array

    # 3. Scale and Weight
    scaled_data = scaler.transform(raw_data)
    weighted_data = scaled_data * weights

    return weighted_data


# --- Main App Structure ---
st.title("🏠 AI Housemate Classifier & Comparator")

if kmeans is None:
    st.error("❌ Model files not found! Please run training first.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Find My Team", "🤝 Compare Two People"])

# ==========================================
# TAB 1: CLASSIFIER & CLUSTER SIMILARITY
# ==========================================
with tab1:
    st.write("Classify yourself into a housemate personality type.")

    # 1. Collect Inputs
    user_data = get_user_inputs(prefix="main_")

    if st.button("Find My Team"):
        if not user_data["name"]:
            st.warning("Please enter your name to join the leaderboard.")
        else:
            # Process Data
            vector = process_data(user_data)

            # Predict Winner
            cluster = kmeans.predict(vector)[0]

            # Calculate Percentages for ALL clusters
            distances = kmeans.transform(vector)[0]
            similarity = 1 / (distances + 0.1)
            percentages = (similarity / np.sum(similarity)) * 100
            match_score = percentages[cluster]

            # --- Display Winner ---
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.success(f"### You are Cluster {cluster}")
                st.metric("Main Match Confidence", f"{match_score:.1f}%")

            with c2:
                # Basic Chart
                chart_data = pd.DataFrame({
                    "Cluster": [f"Type {i}" for i in range(len(percentages))],
                    "Probability": percentages
                })
                st.bar_chart(chart_data.set_index("Cluster"))

            # --- Similarity System (New Feature) ---
            st.subheader("📊 Similarity with Other Clusters")
            st.caption(
                "While you belong to one team, you may share traits with others. Here is your affinity breakdown:")

            # Create a clean display of similarities
            cols = st.columns(len(percentages))
            for i, p in enumerate(percentages):
                with cols[i]:
                    # Highlight the winner
                    if i == cluster:
                        st.markdown(f"**🏆 Type {i}**")
                    else:
                        st.markdown(f"Type {i}")

                    st.progress(int(p))
                    st.caption(f"{p:.1f}% Match")

            # Update Leaderboard (Session State)
            new_entry = {
                "Name": user_data["name"],
                "Cluster": int(cluster),
                "Confidence": f"{match_score:.1f}%"
            }
            # Insert at the beginning of the list
            st.session_state.leaderboard.insert(0, new_entry)

            # Rerun so the Sidebar updates immediately
            st.rerun()

# ==========================================
# TAB 2: COMPARISON TOOL
# ==========================================
with tab2:
    st.subheader("Compare Compatibility")
    st.write("See how similar two different profiles are based on their weighted habits.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.info("👤 Person A")
        data_a = get_user_inputs(prefix="A_")

    with col_b:
        st.info("👤 Person B")
        data_b = get_user_inputs(prefix="B_")

    if st.button("Calculate Similarity"):
        # 1. Process both to get weighted vectors
        vec_a = process_data(data_a)
        vec_b = process_data(data_b)

        # 2. Calculate Euclidean Distance
        dist = np.linalg.norm(vec_a - vec_b)

        # 3. Convert Distance to Similarity Score (0 to 100%)
        # Logic: 0 distance = 100% match.
        similarity_score = (1 / (1 + dist)) * 100

        st.divider()
        st.markdown(f"<h2 style='text-align: center;'>Compatibility Score: {similarity_score:.1f}%</h2>",
                    unsafe_allow_html=True)

        if similarity_score > 85:
            st.success("🌟 Perfect Match! You have very similar lifestyle patterns.")
        elif similarity_score > 60:
            st.info("✅ Good Match. You share many core traits.")
        else:
            st.warning("⚠️ Low Similarity. You have significantly different habits/lifestyles.")