import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from sklearn.metrics.pairwise import euclidean_distances

# Page Config
st.set_page_config(page_title="Flatmate Matcher", page_icon="🏠", layout="wide")


def main():


    # --- Load Models ---
    @st.cache_resource
    def load_models():
        if not os.path.exists("models/kmeans_model.pkl"):
            return None, None, None

        kmeans = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        weights = joblib.load("models/weights.pkl")
        return kmeans, scaler, weights

    kmeans, scaler, weights = load_models()

    # --- Session State for Leaderboard ---
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = []

    # --- Sidebar: Leaderboard ---
    with st.sidebar:
        st.header("🏠 Flatmate Matcher")
        st.subheader("Find your perfect flatmate!")
        st.caption("The application uses AI to identify your personality type and compare your habits with others.")
        st.caption("The result may contain errors, we aren't responsible for any mismatches.")
        st.markdown("---")

        st.header("🏆 Leaderboard")
        st.caption("Most recent results")

        if st.session_state.leaderboard:
            df_leaderboard = pd.DataFrame(st.session_state.leaderboard)
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
            if prefix == "main_":
                name = st.text_input("Name", key=f"{prefix}name")
            else:
                name = st.text_input("Name/Alias", key=f"{prefix}name")

            gender = st.selectbox("Gender", ["Female", "Male"], key=f"{prefix}gender")
            age_cat = st.selectbox(
                "Age Group",
                ["Less than 20", "21 to 35", "36 to 50", "51 or more"],
                key=f"{prefix}age",
            )
            income = st.selectbox("Sufficient Income?", ["No", "Yes"], key=f"{prefix}income")

            st.markdown("**Daily Habits**")
            fruits = st.slider("Fruits & Veggies (Servings)", 0, 5, 3, key=f"{prefix}fruits")
            steps = st.number_input("Daily Steps (in thousands)", 1, 13, 7, key=f"{prefix}steps")
            sleep = st.slider("Sleep Hours", 1, 10, 5, key=f"{prefix}sleep")

        with col2:
            st.markdown("**Psychology & Social**")
            stress = st.slider("Daily Stress (0-5)", 0, 5, 3, key=f"{prefix}stress")
            social = st.slider("Social Connections (0-10)", 0, 10, 5, key=f"{prefix}social")
            passion = st.slider("Time for Passion (Hrs)", 0, 10, 5, key=f"{prefix}passion")

            with st.expander("Advanced Details"):
                places = st.number_input("New Places Visited", 0, 10, 5, key=f"{prefix}places")
                core_circle = st.number_input("Core Circle Size", 0, 10, 5, key=f"{prefix}core")
                support = st.slider("Supporting Others", 0, 10, 5, key=f"{prefix}support")
                donation = st.slider("Donation Freq", 0, 5, 1, key=f"{prefix}donate")
                shouting = st.slider("Daily Shouting", 0, 10, 1, key=f"{prefix}shout")

        return {
            "name": name,
            "gender": gender,
            "age_cat": age_cat,
            "income": income,
            "fruits": fruits,
            "steps": steps,
            "sleep": sleep,
            "stress": stress,
            "social": social,
            "passion": passion,
            "places": places,
            "core_circle": core_circle,
            "support": support,
            "donation": donation,
            "shouting": shouting,
        }

    def process_data(inputs):
        """
        Converts raw input dictionary into the weighted vector expected by the model.
        """
        gender_val = 3 if inputs["gender"] == "Male" else 0
        age_map = {"Less than 20": 16, "21 to 35": 28, "36 to 50": 43, "51 or more": 70}
        age_val = age_map[inputs["age_cat"]]
        income_val = 2 if inputs["income"] == "Yes" else 1

        raw_data_array = np.array(
            [
                [
                    inputs["fruits"],
                    inputs["stress"],
                    inputs["places"],
                    inputs["core_circle"],
                    inputs["support"],
                    inputs["social"],
                    inputs["donation"],
                    inputs["steps"],
                    inputs["sleep"],
                    inputs["shouting"],
                    income_val,
                    inputs["passion"],
                    age_val,
                    gender_val,
                ]
            ]
        )

        if hasattr(scaler, "feature_names_in_"):
            raw_data = pd.DataFrame(raw_data_array, columns=scaler.feature_names_in_)
        else:
            raw_data = raw_data_array

        scaled_data = scaler.transform(raw_data)
        weighted_data = scaled_data * weights
        return weighted_data

    def find_compatible_matches(user_vector, cluster_data, threshold=80):
        """
        Finds compatible people inside a given candidate set (cluster_data).
        Returns indices and their similarity scores.
        """
        distances = euclidean_distances([user_vector], cluster_data)[0]

        max_dist = distances.max() if distances.max() != 0 else 1.0
        similarities = (1 - distances / max_dist) * 100

        compatible_indices = np.where(similarities >= threshold)[0]
        return compatible_indices, similarities[compatible_indices]

    def calculate_compatibility(person_a, person_b, weights: np.array | None = None) -> float:
        """
        Calculates a compatibility score between two people.
        Accepts either:
          - dicts of numeric features (same order/keys), OR
          - numpy vectors.
        Returns 0..100.
        """
        if isinstance(person_a, dict) and isinstance(person_b, dict):
            vec_a = np.array(list(person_a.values()), dtype=float)
            vec_b = np.array(list(person_b.values()), dtype=float)
        else:
            vec_a = np.asarray(person_a, dtype=float).ravel()
            vec_b = np.asarray(person_b, dtype=float).ravel()

        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            # Apply weights to the absolute diff (same intent as before)
            weighted_diff = np.abs(vec_a - vec_b) * w
            distance = np.sqrt(np.sum(weighted_diff**2))

            # Scale to 0..100; use a safe max distance estimate
            max_possible_distance = np.sqrt(np.sum(w**2)) if np.sum(w**2) > 0 else 1.0
            similarity = (1 - (distance / max_possible_distance)) * 100
        else:
            # If vectors are already weighted (as in process_data), just measure distance
            distance = np.linalg.norm(vec_a - vec_b)
            similarity = (1 / (1 + distance)) * 100

        return float(max(0, min(100, similarity)))

    def is_compatible(score: float, threshold: float = 80.0) -> bool:
        """Returns True if score meets the compatibility threshold."""
        return score >= threshold

    # --- Main App Structure ---
    if kmeans is None:
        st.error("❌ Model files not found! Please run training first.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Find My Team", "🤝 Compare Two People"])

    # ==========================================
    # TAB 1: CLASSIFIER & CLUSTER SIMILARITY
    # ==========================================
    with tab1:
        st.title("Housemate Classifier")
        st.write("Classify yourself into a housemate personality type.")

        user_data = get_user_inputs(prefix="main_")

        if st.button("Find My Team"):
            if not user_data["name"]:
                st.warning("Please enter your name to join the leaderboard.")
            else:
                vector = process_data(user_data)
                cluster = kmeans.predict(vector)[0]

                distances = kmeans.transform(vector)[0]
                similarity = 1 / (distances + 0.1)
                percentages = (similarity / np.sum(similarity)) * 100
                match_score = percentages[cluster]

                st.divider()
                with st.container(border=True):
                    st.markdown(f"### 🎯 Result: You belong to **Team {cluster + 1}**")
                    st.markdown(f"**Confidence Score:** {match_score:.1f}%")
                    st.progress(int(match_score))

                    st.divider()
                    st.markdown("#### 📊 Compatibility with All Teams")

                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        st.caption("Breakdown by Team:")
                        for i, p in enumerate(percentages):
                            label = f"**Team {i + 1} (Winner)**" if i == cluster else f"Team {i + 1}"
                            st.write(f"{label}: {p:.1f}%")
                            st.progress(int(p))

                    with res_col2:
                        st.caption("Distribution Chart:")
                        chart_data = pd.DataFrame(
                            {"Team": [f"Team {i + 1}" for i in range(len(percentages))], "Similarity": percentages}
                        )
                        st.bar_chart(chart_data.set_index("Team"))

                new_entry = {"Name": user_data["name"], "Cluster": int(cluster) + 1, "Confidence": f"{match_score:.1f}%"}
                st.session_state.leaderboard.insert(0, new_entry)

    # ==========================================
    # TAB 2: COMPARISON TOOL
    # ==========================================
    with tab2:
        st.title("Housemate Compatibility Comparator")
        st.write("See how similar two different profiles are based on their weighted habits.")

        col_a, col_b = st.columns(2)

        with col_a:
            st.info("👤 Person A")
            data_a = get_user_inputs(prefix="A_")

        with col_b:
            st.info("👤 Person B")
            data_b = get_user_inputs(prefix="B_")

        if st.button("Calculate Similarity"):
            vec_a = process_data(data_a)
            vec_b = process_data(data_b)

            # Use the shared compatibility function (vectors are already weighted)
            similarity_score = calculate_compatibility(vec_a, vec_b, weights=None)

            st.divider()
            st.markdown(
                f"<h2 style='text-align: center;'>Compatibility Score: {similarity_score:.1f}%</h2>",
                unsafe_allow_html=True,
            )

            if similarity_score > 85:
                st.success("🌟 Perfect Match! You have very similar lifestyle patterns.")
            elif similarity_score > 60:
                st.info("✅ Good Match. You share many core traits.")
            else:
                st.warning("⚠️ Low Similarity. You have significantly different habits/lifestyles.")

            # Also integrate the matching helper on the available candidates (here: just Person B as a demo set)
            candidate_set = np.vstack([vec_b])
            compatible_indices, compatible_scores = find_compatible_matches(
                user_vector=vec_a.ravel(),
                cluster_data=candidate_set,
                threshold=80,
            )

            st.caption("Compatibility threshold check (≥ 80%):")
            if len(compatible_indices) > 0 and is_compatible(float(compatible_scores[0]), 80.0):
                st.success("Person B is a compatible match based on the threshold.")
            else:
                st.info("Person B does not meet the compatibility threshold.")


if __name__ == "__main__":
    main()