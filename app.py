import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Restaurant Management", page_icon="🍽️", layout="wide")

BASE=os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE,"decision_tree_restaurant_rating.pkl"),"rb") as f:
    artifact=pickle.load(f)
model=artifact["model"]
y_scaler=artifact["target_scaler"]
features=artifact["feature_columns"]
metrics=artifact["metrics"]

@st.cache_data
def load_data():
    p=os.path.join(BASE,"restaurant_data_clean.csv")
    return pd.read_csv(p)

df=load_data()

st.title("🍽️ Restaurant Management & Rating Predictor")
st.caption("Restaurant analytics frontend powered by a saved Decision Tree Regression model.")

page=st.sidebar.radio("Navigation", ["Dashboard", "Predict Rating", "Restaurant Explorer", "Model Performance"])

if page=="Dashboard":
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Restaurants", f"{len(df):,}")
    c2.metric("Average Rating", f"{df['Aggregate rating'].mean():.2f}")
    c3.metric("Average Votes", f"{df['Votes'].mean():,.0f}")
    c4.metric("Cities", f"{df['City'].nunique():,}")
    st.subheader("Rating Distribution")
    st.bar_chart(df['Aggregate rating'].value_counts().sort_index())
    st.subheader("Top Cities by Restaurant Count")
    st.bar_chart(df['City'].value_counts().head(10))

elif page=="Predict Rating":
    st.header("Predict Restaurant Rating")
    st.write("Enter restaurant details. The saved Decision Tree model predicts the expected aggregate rating.")
    values={}
    cols=st.columns(2)
    for i,col in enumerate(features):
        with cols[i%2]:
            if col in artifact['numeric_features']:
                default=float(df[col].median())
                if col=='Price range': values[col]=st.number_input(col,1,4,int(round(default)))
                elif col=='Votes': values[col]=st.number_input(col,min_value=0,value=int(default),step=1)
                elif col in ('Average Cost for two',): values[col]=st.number_input(col,min_value=0.0,value=default,step=50.0)
                elif col in ('Latitude','Longitude'): values[col]=st.number_input(col,value=default,format='%.6f')
                else: values[col]=st.number_input(col,value=default)
            else:
                opts=sorted(df[col].dropna().astype(str).unique().tolist())
                values[col]=st.selectbox(col,opts,index=0 if opts else None)
    if st.button("Predict Rating",type="primary"):
        row=pd.DataFrame([values],columns=features)
        pred_scaled=model.predict(row)
        pred=float(y_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1,1))[0,0])
        pred=float(np.clip(pred,0,5))
        st.success(f"Predicted Aggregate Rating: {pred:.2f} / 5.00")
        st.progress(pred/5)

elif page=="Restaurant Explorer":
    st.header("Restaurant Explorer")
    city=st.selectbox("Filter by City",["All"]+sorted(df['City'].dropna().unique().tolist()))
    min_rating=st.slider("Minimum Rating",0.0,5.0,0.0,0.1)
    view=df[df['Aggregate rating']>=min_rating].copy()
    if city!="All": view=view[view['City']==city]
    st.dataframe(view.head(200),use_container_width=True)

else:
    st.header("Decision Tree Model Performance")
    st.write("Test-set performance of the saved model:")
    a,b,c,d=st.columns(4)
    a.metric("MAE",f"{metrics['MAE']:.4f}")
    b.metric("MSE",f"{metrics['MSE']:.4f}")
    c.metric("RMSE",f"{metrics['RMSE']:.4f}")
    d.metric("R²",f"{metrics['R2']:.4f}")
    st.info("Predictions are reverse-scaled with StandardScaler.inverse_transform() before the metrics and final rating are displayed.")
