import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

class WaterQualitySystem:
    def __init__(self):
        self.models = self._load_models()
        self.thresholds = {
            'TDS': {'Warning': 500, 'Critical': 1000},
            'Chloride': {'Warning': 250, 'Critical': 400},
            'Sodium': {'Warning': 200, 'Critical': 300},
            'EC': {'Warning': 1500, 'Critical': 2000}
        }
        
    def _load_models(self):
        return {
            'EC_TDS': {
                'model': joblib.load('models/ec_tds_model.pkl'),
                'scaler_X': joblib.load('models/ec_tds_scaler_X.pkl'),
                'scaler_y': joblib.load('models/ec_tds_scaler_y.pkl')
            },
            'TDS_Cl': {
                'model': joblib.load('models/tds_chloride_model.pkl'),
                'scaler_X': joblib.load('models/tds_chloride_scaler_X.pkl'),
                'scaler_y': joblib.load('models/tds_chloride_scaler_y.pkl'),
                'poly': joblib.load('models/tds_chloride_poly.pkl')
            },
            'TDS_Na': {
                'model': joblib.load('models/tds_sodium_model.pkl'),
                'scaler_X': joblib.load('models/tds_sodium_scaler_X.pkl'),
                'scaler_y': joblib.load('models/tds_sodium_scaler_y.pkl'),
                'poly': joblib.load('models/tds_sodium_poly.pkl')
            }
        }

    def predict_and_explain(self, ec_value):
        """Make predictions and provide detailed explanations"""
        results = {
            'predictions': {},
            'explanations': {},
            'alerts': [],
            'relationships': {}
        }
        
        # EC → TDS Prediction
        tds_pred = self.predict_tds(ec_value)
        results['predictions']['TDS'] = tds_pred
        results['explanations']['TDS'] = {
            'process': [
                f"1. Input EC: {ec_value:.1f} µS/cm",
                f"2. Theoretical TDS (EC/1.6): {ec_value/1.6:.1f} mg/L",
                "3. Model Process:",
                "   - Scale EC input using StandardScaler",
                "   - Apply Linear Regression: TDS = m*EC + b",
                f"4. Model adjusted TDS: {tds_pred:.1f} mg/L",
                f"5. Reliability: 94%"
            ],
            'validation': [
                f"EC/TDS ratio: {ec_value/tds_pred:.2f} (normal: 1.4-1.7)",
                f"Prediction within expected range: {'Yes' if 0.8 <= ec_value/tds_pred <= 2.0 else 'No'}"
            ]
        }
        
        # TDS → Chloride Prediction
        cl_pred = self.predict_chloride(tds_pred, ec_value)
        results['predictions']['Chloride'] = cl_pred
        cl_percentage = (cl_pred/tds_pred)*100
        results['explanations']['Chloride'] = {
            'process': [
                f"1. Inputs:",
                f"   - TDS: {tds_pred:.1f} mg/L",
                f"   - EC: {ec_value:.1f} µS/cm",
                "2. Model Process:",
                "   - Scale inputs using StandardScaler",
                "   - Apply Polynomial Features (degree=2):",
                "     * TDS, EC",
                "     * TDS², EC², TDS×EC",
                "   - Apply Linear Regression with polynomial terms",
                f"3. Predicted Chloride: {cl_pred:.1f} mg/L ({cl_percentage:.1f}% of TDS)",
                f"4. Reliability: 84%"
            ],
            'validation': [
                f"Chloride/TDS ratio: {cl_percentage:.1f}% (normal: 10-30%)",
                f"Prediction within expected range: {'Yes' if 10 <= cl_percentage <= 30 else 'No'}"
            ]
        }
        
        # TDS → Sodium Prediction
        na_pred = self.predict_sodium(tds_pred, ec_value, cl_pred)
        results['predictions']['Sodium'] = na_pred
        na_cl_ratio = na_pred/cl_pred
        results['explanations']['Sodium'] = {
            'process': [
                f"1. Inputs:",
                f"   - TDS: {tds_pred:.1f} mg/L",
                f"   - EC: {ec_value:.1f} µS/cm",
                f"   - Chloride: {cl_pred:.1f} mg/L",
                "2. Model Process:",
                "   - Scale inputs using StandardScaler",
                "   - Apply Polynomial Features (degree=2):",
                "     * TDS, EC, Chloride",
                "     * TDS², EC², Chloride²",
                "     * TDS×EC, TDS×Chloride, EC×Chloride",
                "   - Apply Linear Regression with polynomial terms",
                f"3. Predicted Sodium: {na_pred:.1f} mg/L (Na/Cl ratio: {na_cl_ratio:.2f})",
                f"4. Reliability: 90%"
            ],
            'validation': [
                f"Na/Cl ratio: {na_cl_ratio:.2f} (normal: 0.6-0.9)",
                f"Prediction within expected range: {'Yes' if 0.6 <= na_cl_ratio <= 0.9 else 'No'}"
            ]
        }
        
        # Generate alerts
        results['alerts'] = self._generate_alerts(results['predictions'])
        
        # Calculate relationships
        results['relationships'] = {
            'EC_TDS_ratio': ec_value/tds_pred,
            'Cl_TDS_percentage': cl_percentage,
            'Na_Cl_ratio': na_cl_ratio
        }
        
        return results

    def _generate_alerts(self, predictions):
        """Generate alerts based on predictions"""
        alerts = []
        
        for param, value in predictions.items():
            if value > self.thresholds[param]['Critical']:
                alerts.append({
                    'parameter': param,
                    'level': 'Critical',
                    'value': value,
                    'threshold': self.thresholds[param]['Critical'],
                    'message': f"{param} critically high: {value:.1f} (threshold: {self.thresholds[param]['Critical']})"
                })
            elif value > self.thresholds[param]['Warning']:
                alerts.append({
                    'parameter': param,
                    'level': 'Warning',
                    'value': value,
                    'threshold': self.thresholds[param]['Warning'],
                    'message': f"{param} warning: {value:.1f} (threshold: {self.thresholds[param]['Warning']})"
                })
        
        return alerts

    def predict_tds(self, ec_value):
        """Predict TDS from EC value"""
        # Scale input
        ec_scaled = self.models['EC_TDS']['scaler_X'].transform([[ec_value]])
        
        # Predict and inverse transform
        tds_pred = self.models['EC_TDS']['scaler_y'].inverse_transform(
            self.models['EC_TDS']['model'].predict(ec_scaled).reshape(-1, 1)
        )[0][0]
        
        return float(tds_pred)

    def predict_chloride(self, tds, ec):
        """Predict Chloride from TDS and EC values"""
        # Prepare input
        X = np.array([[tds, ec]])
        X_scaled = self.models['TDS_Cl']['scaler_X'].transform(X)
        
        # Apply polynomial transformation
        X_poly = self.models['TDS_Cl']['poly'].transform(X_scaled)
        
        # Predict and inverse transform
        cl_pred = self.models['TDS_Cl']['scaler_y'].inverse_transform(
            self.models['TDS_Cl']['model'].predict(X_poly).reshape(-1, 1)
        )[0][0]
        
        return float(cl_pred)

    def predict_sodium(self, tds, ec, chloride):
        """Predict Sodium from TDS, EC, and Chloride values"""
        # Prepare input
        X = np.array([[tds, ec, chloride]])
        X_scaled = self.models['TDS_Na']['scaler_X'].transform(X)
        
        # Apply polynomial transformation
        X_poly = self.models['TDS_Na']['poly'].transform(X_scaled)
        
        # Predict and inverse transform
        na_pred = self.models['TDS_Na']['scaler_y'].inverse_transform(
            self.models['TDS_Na']['model'].predict(X_poly).reshape(-1, 1)
        )[0][0]
        
        return float(na_pred)

    def get_standard_ranges(self):
        """Return standard ranges for key parameters"""
        return {
            "Parameter": ["EC", "TDS", "Chloride", "Sodium"],
            "Unit": ["µS/cm", "mg/L", "mg/L", "mg/L"],
            "Acceptable Range": ["0 - 1500", "0 - 500", "0 - 250", "0 - 200"],
            "Warning Level": ["1500 - 2000", "500 - 1000", "250 - 400", "200 - 300"],
            "Critical Level": ["> 2000", "> 1000", "> 400", "> 300"],
            "Source": [
                "WHO/EPA Guidelines",
                "WHO/EPA Guidelines",
                "EPA Secondary Standards",
                "WHO Guidelines"
            ]
        }

def main():
    st.set_page_config(
        page_title="Water Quality EWS",
        page_icon="💧",
        layout="wide"
    )
    
    st.title("💧 Water Quality Early Warning System")
    
    system = WaterQualitySystem()
    
    # Input Section
    with st.container():
        st.subheader("📊 Parameter Input")
        ec_value = st.number_input(
            "Enter EC Value (µS/cm)",
            min_value=0.0,
            max_value=5000.0,
            value=750.0,
            step=10.0,
            help="Electrical Conductivity measurement"
        )
    
    if st.button("Analyze Water Quality", type="primary"):
        results = system.predict_and_explain(ec_value)
        
        # Display Predictions and Status
        st.subheader("🔍 Analysis Results")
        cols = st.columns(3)
        
        # Parameter cards
        for i, (param, value) in enumerate(results['predictions'].items()):
            with cols[i]:
                status = "Normal"
                color = "green"
                if value > system.thresholds[param]['Critical']:
                    status = "Critical"
                    color = "red"
                elif value > system.thresholds[param]['Warning']:
                    status = "Warning"
                    color = "orange"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 1px solid {color};">
                    <h3 style="color: {color};">{param}</h3>
                    <h2>{value:.1f}</h2>
                    <p>Status: {status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Alerts Section
        if results['alerts']:
            st.subheader("⚠️ Alerts")
            for alert in results['alerts']:
                if alert['level'] == 'Critical':
                    st.error(alert['message'])
                else:
                    st.warning(alert['message'])
        
        # Standards Section (moved here, after alerts)
        st.subheader("📚 Standard Parameter Ranges")
        
        # Convert to DataFrame and reset index to start from 1
        df_standards = pd.DataFrame(system.get_standard_ranges())
        df_standards.index = range(1, len(df_standards) + 1)
        
        # Style the DataFrame
        styled_df = df_standards.style.set_properties(**{
            'text-align': 'center',
            'font-size': '16px',
            'padding': '10px'
        })
        
        st.table(styled_df)
        
        # Detailed Analysis
        st.subheader("📋 Detailed Analysis")
        tabs = st.tabs(["TDS", "Chloride", "Sodium"])
        
        for tab, param in zip(tabs, results['explanations'].keys()):
            with tab:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Prediction Process")
                    for step in results['explanations'][param]['process']:
                        st.write(step)
                with col2:
                    st.markdown("##### Validation Checks")
                    for check in results['explanations'][param]['validation']:
                        st.write(check)
        
        # Relationships Analysis
        st.subheader("🔗 Parameter Relationships")
        rel_cols = st.columns(3)
        
        with rel_cols[0]:
            st.metric(
                "EC/TDS Ratio",
                f"{results['relationships']['EC_TDS_ratio']:.2f}",
                delta="Normal" if 1.4 <= results['relationships']['EC_TDS_ratio'] <= 1.7 else "Abnormal"
            )
        
        with rel_cols[1]:
            st.metric(
                "Chloride % in TDS",
                f"{results['relationships']['Cl_TDS_percentage']:.1f}%",
                delta="Normal" if 10 <= results['relationships']['Cl_TDS_percentage'] <= 30 else "Abnormal"
            )
        
        with rel_cols[2]:
            st.metric(
                "Na/Cl Ratio",
                f"{results['relationships']['Na_Cl_ratio']:.2f}",
                delta="Normal" if 0.6 <= results['relationships']['Na_Cl_ratio'] <= 0.9 else "Abnormal"
            )

if __name__ == "__main__":
    main() 