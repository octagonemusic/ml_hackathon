import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

class WaterQualityEWS:
    def __init__(self):
        # Load WHO and Indian standards for water quality
        self.quality_thresholds = {
            'TDS (mg/L)': {
                'acceptable': 500,
                'warning': 1000,
                'critical': 2000
            },
            'Chloride (mg/L)': {
                'acceptable': 250,
                'warning': 500,
                'critical': 1000
            },
            'Sodium (mg/L)': {
                'acceptable': 200,
                'warning': 400,
                'critical': 800
            },
            'EC (µS/cm)': {
                'acceptable': 750,
                'warning': 1500,
                'critical': 3000
            }
        }
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load all ML models and their components"""
        try:
            # Load base linear models
            self.models = {
                'EC_TDS': {
                    'model': joblib.load('models/ec_tds_model.pkl'),
                    'scaler_X': joblib.load('models/ec_tds_scaler_X.pkl'),
                    'scaler_y': joblib.load('models/ec_tds_scaler_y.pkl')
                },
                'Cl_Na': {
                    'model': joblib.load('models/cl_na_model.pkl'),
                    'scaler_X': joblib.load('models/cl_na_scaler_X.pkl'),
                    'scaler_y': joblib.load('models/cl_na_scaler_y.pkl')
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
                },
                'Cl_EC': {
                    'model': joblib.load('models/chloride_ec_model.pkl'),
                    'scaler_X': joblib.load('models/chloride_ec_scaler_X.pkl'),
                    'scaler_y': joblib.load('models/chloride_ec_scaler_y.pkl'),
                    'poly': joblib.load('models/chloride_ec_poly.pkl')
                }
            }
            print("All models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please ensure all model files exist in the 'models' directory")
            raise
    
    def assess_parameter(self, value, parameter):
        """Assess a single parameter against thresholds"""
        thresholds = self.quality_thresholds[parameter]
        
        if value <= thresholds['acceptable']:
            return 'Safe', 0
        elif value <= thresholds['warning']:
            return 'Warning', 1
        else:
            return 'Critical', 2
    
    def predict_and_assess(self, measurements):
        """Make predictions and assess water quality"""
        predictions = {}
        alerts = []
        
        # EC → TDS prediction
        if 'EC (µS/cm)' in measurements:
            ec = measurements['EC (µS/cm)']
            ec_scaled = self.models['EC_TDS']['scaler_X'].transform([[ec]])
            tds_pred = self.models['EC_TDS']['scaler_y'].inverse_transform(
                self.models['EC_TDS']['model'].predict(ec_scaled).reshape(-1, 1)
            )[0]
            predictions['TDS (mg/L)'] = tds_pred
        
        # TDS-based predictions
        if 'TDS (mg/L)' in measurements or 'TDS (mg/L)' in predictions:
            tds = measurements.get('TDS (mg/L)', predictions.get('TDS (mg/L)'))
            
            # Predict Chloride
            tds_scaled = self.models['TDS_Cl']['scaler_X'].transform([[tds]])
            tds_poly = self.models['TDS_Cl']['poly'].transform(tds_scaled)
            cl_pred = self.models['TDS_Cl']['scaler_y'].inverse_transform(
                self.models['TDS_Cl']['model'].predict(tds_poly).reshape(-1, 1)
            )[0]
            predictions['Chloride (mg/L)'] = cl_pred
            
            # Predict Sodium
            tds_scaled = self.models['TDS_Na']['scaler_X'].transform([[tds]])
            tds_poly = self.models['TDS_Na']['poly'].transform(tds_scaled)
            na_pred = self.models['TDS_Na']['scaler_y'].inverse_transform(
                self.models['TDS_Na']['model'].predict(tds_poly).reshape(-1, 1)
            )[0]
            predictions['Sodium (mg/L)'] = na_pred
        
        # Assess all parameters
        for param, value in {**measurements, **predictions}.items():
            if param in self.quality_thresholds:
                status, level = self.assess_parameter(value, param)
                if level > 0:
                    alerts.append({
                        'parameter': param,
                        'value': value,
                        'status': status,
                        'level': level,
                        'threshold': self.quality_thresholds[param]['acceptable']
                    })
        
        return predictions, alerts
    
    def generate_report(self, measurements, predictions, alerts):
        """Generate a comprehensive report"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'measurements': measurements,
            'predictions': predictions,
            'alerts': alerts,
            'recommendations': self.get_recommendations(alerts)
        }
        return report
    
    def get_recommendations(self, alerts):
        """Generate recommendations based on alerts"""
        recommendations = []
        
        if not alerts:
            recommendations.append("Water quality parameters within acceptable limits.")
            return recommendations
        
        for alert in alerts:
            if alert['level'] == 1:
                recommendations.append(
                    f"Monitor {alert['parameter']} closely. "
                    f"Current value: {alert['value']:.2f}, "
                    f"Threshold: {alert['threshold']:.2f}"
                )
            elif alert['level'] == 2:
                recommendations.append(
                    f"Immediate action required for {alert['parameter']}. "
                    f"Current value: {alert['value']:.2f}, "
                    f"Threshold: {alert['threshold']:.2f}"
                )
        
        return recommendations

def main():
    # Create reports directory if it doesn't exist
    if not os.path.exists('ews_reports'):
        os.makedirs('ews_reports')
    
    # Initialize EWS
    ews = WaterQualityEWS()
    
    # Example measurements
    measurements = {
        'TDS (mg/L)': 450,
        'EC (µS/cm)': 750
    }
    
    # Get predictions and alerts
    predictions, alerts = ews.predict_and_assess(measurements)
    
    # Generate report
    report = ews.generate_report(measurements, predictions, alerts)
    
    # Save report
    with open('ews_reports/report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Early Warning System Report Generated!")
    print("\nAlerts:", len(alerts))
    for alert in alerts:
        print(f"\nParameter: {alert['parameter']}")
        print(f"Status: {alert['status']}")
        print(f"Value: {alert['value']:.2f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()
