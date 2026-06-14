import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Melbourne Housing Predictor", layout="wide")

# --- DATA & MODEL CACHING ---
@st.cache_data
def load_data():
    return pd.read_csv('melb_data.csv')

@st.cache_resource
def train_models(data):
    # Removed Lattitude and Longtitude per your request
    features = ['Rooms', 'Bathroom', 'Landsize']
    
    # Drop missing values
    clean_data = data.dropna(subset=features + ['Price'])
    X = clean_data[features]
    y = clean_data['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    return dt, rf, X_train, X_test, y_train, y_test

# Initialize data and models
try:
    df = load_data()
    dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(df)
except FileNotFoundError:
    st.error("🚨 **File not found!** Please make sure `melb_data.csv` is in the same folder.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎛️ Menu")
page = st.sidebar.radio("Select a Page:", ["Data & Visualisation", "Prediction"])
st.sidebar.divider()

# ==========================================
# PAGE 1: DATA & VISUALISATION
# ==========================================
if page == "Data & Visualisation":
    st.title("📊 Data & Visualisation")
    st.markdown("Explore and visualize the Melbourne housing dataset.")
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filter Properties")
    
    suburbs = sorted(df['Suburb'].dropna().unique().tolist())
    selected_suburbs = st.sidebar.multiselect("Select Suburb(s)", suburbs, default=[])
    
    min_rooms, max_rooms = int(df['Rooms'].min()), int(df['Rooms'].max())
    selected_rooms = st.sidebar.slider("Number of Rooms", min_rooms, max_rooms, (min_rooms, max_rooms))
    
    min_price, max_price = float(df['Price'].min(skipna=True)), float(df['Price'].max(skipna=True))
    selected_price = st.sidebar.slider("Price Range ($)", min_price, max_price, (min_price, max_price), step=50000.0)
    
    # --- APPLY FILTERS ---
    filtered_df = df.copy()
    if selected_suburbs:
        filtered_df = filtered_df[filtered_df['Suburb'].isin(selected_suburbs)]
        
    filtered_df = filtered_df[(filtered_df['Rooms'] >= selected_rooms[0]) & (filtered_df['Rooms'] <= selected_rooms[1])]
    filtered_df = filtered_df[((filtered_df['Price'] >= selected_price[0]) & (filtered_df['Price'] <= selected_price[1])) | filtered_df['Price'].isna()]
    
    # --- MAIN DISPLAY ---
    st.subheader(f"Showing {len(filtered_df)} matching properties")
    st.dataframe(filtered_df, use_container_width=True)
    
    # --- VISUALISATIONS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Summary Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        
    with col2:
        st.subheader("📉 MAE vs Max Leaf Nodes")
        st.markdown("Mean Absolute Error across different `max_leaf_nodes` for the Decision Tree.")
        
        # Dynamically calculate MAE to replicate your Jupyter Notebook graph
        @st.cache_data
        def get_mae_data():
            leaf_nodes = [50, 100, 200, 400, 600, 800, 1000]
            maes = []
            for nodes in leaf_nodes:
                model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                maes.append(mean_absolute_error(y_test, preds))
            return pd.DataFrame({'Max Leaf Nodes': leaf_nodes, 'Mean Absolute Error': maes}).set_index('Max Leaf Nodes')
            
        mae_chart_data = get_mae_data()
        st.line_chart(mae_chart_data)

# ==========================================
# PAGE 2: PREDICTION
# ==========================================
elif page == "Prediction":
    st.title("🏠 Melbourne Housing Price Predictor")
    st.markdown("Compare predictions between Decision Tree and Random Forest models.")
    
    avg_rooms = int(df['Rooms'].mean())
    avg_bath = float(df['Bathroom'].mean())
    avg_land = float(df['Landsize'].mean())

    with st.form("prediction_form"):
        st.subheader("Enter Property Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rooms = st.number_input("Rooms", min_value=1, max_value=20, value=avg_rooms, step=1)
        with col2:
            bathroom = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=avg_bath, step=0.5)
        with col3:
            landsize = st.number_input("Landsize", min_value=1.0, value=avg_land, step=10.0)
            
        predict_button = st.form_submit_button("Predict Price", use_container_width=True)
        
    if predict_button:
        # Match the new features array
        input_data = pd.DataFrame(
            [[rooms, bathroom, landsize]], 
            columns=['Rooms', 'Bathroom', 'Landsize']
        )
        
        dt_price = dt_model.predict(input_data)[0]
        rf_price = rf_model.predict(input_data)[0]
        
        st.success("Models calculated successfully!")
        
        # Results side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Decision Tree Price", f"${dt_price:,.0f}")
        with col2:
            st.metric("Random Forest Price", f"${rf_price:,.0f}")

        # Graph comparison
        st.subheader("Prediction Comparison")
        chart_df = pd.DataFrame({
            "Model": ["Decision Tree", "Random Forest"],
            "Price": [dt_price, rf_price]
        })
        st.bar_chart(chart_df.set_index("Model"))