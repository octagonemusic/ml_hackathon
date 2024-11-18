import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import uuid
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

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
        
        try:
            # EC → TDS prediction
            if 'EC (µS/cm)' in measurements:
                print("\nPredicting TDS from EC...")
                ec = float(measurements['EC (µS/cm)'])
                ec_scaled = self.models['EC_TDS']['scaler_X'].transform([[ec]])
                tds_pred = self.models['EC_TDS']['scaler_y'].inverse_transform(
                    self.models['EC_TDS']['model'].predict(ec_scaled).reshape(-1, 1)
                )[0][0]
                predictions['TDS (mg/L)'] = float(tds_pred)
                print(f"Predicted TDS: {float(tds_pred):.2f} mg/L")
            
            # TDS-based predictions
            if 'TDS (mg/L)' in measurements or 'TDS (mg/L)' in predictions:
                print("\nPredicting from TDS...")
                tds = float(measurements.get('TDS (mg/L)', predictions.get('TDS (mg/L)')))
                ec = float(measurements.get('EC (µS/cm)', 0))
                
                # Predict Chloride using TDS and EC
                print("- Predicting Chloride...")
                X_cl = np.array([[tds, ec]], dtype=float)
                X_cl_scaled = self.models['TDS_Cl']['scaler_X'].transform(X_cl)
                X_cl_poly = self.models['TDS_Cl']['poly'].transform(X_cl_scaled)
                cl_pred = self.models['TDS_Cl']['scaler_y'].inverse_transform(
                    self.models['TDS_Cl']['model'].predict(X_cl_poly).reshape(-1, 1)
                )[0][0]
                predictions['Chloride (mg/L)'] = float(cl_pred)
                print(f"  Predicted Chloride: {float(cl_pred):.2f} mg/L")
                
                # Predict Sodium using TDS, EC, and Chloride
                print("- Predicting Sodium...")
                X_na = np.array([[tds, ec, cl_pred]], dtype=float)
                X_na_scaled = self.models['TDS_Na']['scaler_X'].transform(X_na)
                X_na_poly = self.models['TDS_Na']['poly'].transform(X_na_scaled)
                na_pred = self.models['TDS_Na']['scaler_y'].inverse_transform(
                    self.models['TDS_Na']['model'].predict(X_na_poly).reshape(-1, 1)
                )[0][0]
                predictions['Sodium (mg/L)'] = float(na_pred)
                print(f"  Predicted Sodium: {float(na_pred):.2f} mg/L")
            
            # Chloride → EC prediction (if we have Chloride)
            if 'Chloride (mg/L)' in measurements or 'Chloride (mg/L)' in predictions:
                print("\nPredicting EC from Chloride...")
                cl = float(measurements.get('Chloride (mg/L)', predictions.get('Chloride (mg/L)')))
                tds = float(measurements.get('TDS (mg/L)', predictions.get('TDS (mg/L)', 0)))
                na = float(measurements.get('Sodium (mg/L)', predictions.get('Sodium (mg/L)', 0)))
                ph = float(measurements.get('pH', 7.0))
                
                X_ec = np.array([[cl, tds, na, ph]], dtype=float)
                X_ec_scaled = self.models['Cl_EC']['scaler_X'].transform(X_ec)
                X_ec_poly = self.models['Cl_EC']['poly'].transform(X_ec_scaled)
                ec_pred = self.models['Cl_EC']['scaler_y'].inverse_transform(
                    self.models['Cl_EC']['model'].predict(X_ec_poly).reshape(-1, 1)
                )[0][0]
                predictions['EC (µS/cm)'] = float(ec_pred)
                print(f"  Predicted EC: {float(ec_pred):.2f} µS/cm")
            
            # Assess all parameters
            for param, value in {**measurements, **predictions}.items():
                if param in self.quality_thresholds:
                    status, level = self.assess_parameter(float(value), param)
                    if level > 0:
                        alerts.append({
                            'parameter': param,
                            'value': float(value),
                            'status': status,
                            'level': level,
                            'threshold': self.quality_thresholds[param]['acceptable']
                        })
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Current state - measurements: {measurements}")
            print(f"Current state - predictions: {predictions}")
            raise
        
        return predictions, alerts
    
    def generate_report(self, measurements, predictions, alerts):
        """Generate a comprehensive water quality report"""
        
        # Calculate prediction reliability scores (simple version)
        reliability_scores = self._calculate_reliability(measurements, predictions)
        
        # Get trend indicators (↑ → ↓)
        trends = self._get_trend_indicators(predictions)
        
        def get_unit(param):
            """Extract unit from parameter name or return appropriate unit"""
            if '(' in param and ')' in param:
                return param.split('(')[1].strip(')')
            # Handle special cases
            if param == 'pH':
                return 'pH units'
            return 'units'  # Default case
        
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_parameters_monitored': len(set(measurements.keys()) | set(predictions.keys())),
                'alert_count': len(alerts),
                'critical_count': sum(1 for alert in alerts if alert['status'] == 'Critical'),
                'warning_count': sum(1 for alert in alerts if alert['status'] == 'Warning'),
                'overall_status': self._get_overall_status(alerts)
            },
            'measurements': {
                param: {
                    'value': float(value),
                    'unit': get_unit(param),
                    'source': 'measured'
                } for param, value in measurements.items()
            },
            'predictions': {
                param: {
                    'value': float(value),
                    'unit': get_unit(param),
                    'source': 'predicted',
                    'reliability': reliability_scores.get(param, 'N/A'),
                    'trend': trends.get(param, '→')
                } for param, value in predictions.items()
            },
            'alerts': [
                {
                    'parameter': alert['parameter'],
                    'status': alert['status'],
                    'value': float(alert['value']),
                    'threshold': float(alert['threshold']),
                    'exceedance_ratio': float(alert['value']) / float(alert['threshold']),
                    'priority': 'High' if alert['status'] == 'Critical' else 'Medium'
                } for alert in alerts
            ],
            'recommendations': self._generate_detailed_recommendations(alerts, predictions),
            'metadata': {
                'model_version': '1.0',
                'thresholds_source': 'WHO and Indian Standards',
                'report_id': str(uuid.uuid4())
            }
        }
        return report
    
    def _calculate_reliability(self, measurements, predictions):
        """Calculate reliability scores for predictions"""
        reliability_scores = {}
        
        # Base reliability on model R² scores and presence of input parameters
        base_scores = {
            'EC_TDS': 0.94,  # From model performance
            'TDS_Cl': 0.84,
            'TDS_Na': 0.90,
            'Cl_EC': 0.94
        }
        
        for param, value in predictions.items():
            if param == 'TDS (mg/L)' and 'EC (µS/cm)' in measurements:
                reliability_scores[param] = base_scores['EC_TDS']
            elif param == 'Chloride (mg/L)':
                reliability_scores[param] = base_scores['TDS_Cl']
            elif param == 'Sodium (mg/L)':
                reliability_scores[param] = base_scores['TDS_Na']
            elif param == 'EC (µS/cm)':
                reliability_scores[param] = base_scores['Cl_EC']
        
        return reliability_scores
    
    def _get_trend_indicators(self, predictions):
        """Get trend indicators for predictions (placeholder for now)"""
        # This would normally compare with historical data
        return {param: '→' for param in predictions.keys()}
    
    def _get_overall_status(self, alerts):
        """Determine overall water quality status"""
        if any(alert['status'] == 'Critical' for alert in alerts):
            return 'Critical'
        elif any(alert['status'] == 'Warning' for alert in alerts):
            return 'Warning'
        return 'Normal'
    
    def _generate_detailed_recommendations(self, alerts, predictions):
        """Generate detailed recommendations based on alerts and predictions"""
        recommendations = []
        
        if not alerts:
            recommendations.append({
                'type': 'status',
                'priority': 'Low',
                'message': "Water quality parameters within acceptable limits.",
                'action_required': "Continue regular monitoring."
            })
            return recommendations
        
        # Group alerts by status
        critical_alerts = [a for a in alerts if a['status'] == 'Critical']
        warning_alerts = [a for a in alerts if a['status'] == 'Warning']
        
        # Add recommendations for critical alerts
        for alert in critical_alerts:
            recommendations.append({
                'type': 'action',
                'priority': 'High',
                'parameter': alert['parameter'],
                'message': f"Critical level exceeded for {alert['parameter']}",
                'current_value': float(alert['value']),
                'threshold': float(alert['threshold']),
                'action_required': self._get_action_required(alert['parameter'], alert['value'])
            })
        
        # Add recommendations for warning alerts
        for alert in warning_alerts:
            recommendations.append({
                'type': 'monitor',
                'priority': 'Medium',
                'parameter': alert['parameter'],
                'message': f"Warning level exceeded for {alert['parameter']}",
                'current_value': float(alert['value']),
                'threshold': float(alert['threshold']),
                'action_required': "Increase monitoring frequency and check for trends."
            })
        
        return recommendations
    
    def _get_action_required(self, parameter, value):
        """Get specific action required based on parameter and value"""
        actions = {
            'TDS (mg/L)': "Check water source and filtration system. Consider reverse osmosis treatment.",
            'EC (µS/cm)': "Investigate source of dissolved solids. Check for industrial runoff.",
            'Chloride (mg/L)': "Check for saltwater intrusion or industrial contamination.",
            'Sodium (mg/L)': "Evaluate water softening systems and check for saltwater intrusion."
        }
        return actions.get(parameter, "Investigate source of contamination and take corrective action.")

def print_report(report):
    """Print formatted report to console with enhanced visuals"""
    
    # Color codes
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }
    
    # Status colors
    STATUS_COLORS = {
        'Normal': COLORS['GREEN'],
        'Warning': COLORS['YELLOW'],
        'Critical': COLORS['RED']
    }
    
    def colorize(text, color):
        return f"{color}{text}{COLORS['END']}"
    
    # Print header
    print("\n" + "="*60)
    print(colorize(f"Water Quality Report - {report['timestamp']}", COLORS['HEADER'] + COLORS['BOLD']))
    print("="*60)
    
    # Print summary with colored status
    status_color = STATUS_COLORS.get(report['summary']['overall_status'], '')
    print(f"\nOVERALL STATUS: {colorize(report['summary']['overall_status'], status_color + COLORS['BOLD'])}")
    print(f"Total Parameters: {report['summary']['total_parameters_monitored']}")
    print(f"Alerts: {colorize(str(report['summary']['alert_count']), COLORS['BOLD'])} "
          f"(Critical: {colorize(str(report['summary']['critical_count']), COLORS['RED'])}, "
          f"Warning: {colorize(str(report['summary']['warning_count']), COLORS['YELLOW'])})")
    
    # Print measurements
    if report['measurements']:
        print(f"\n{colorize('MEASURED PARAMETERS:', COLORS['BOLD'])}")
        for param, data in report['measurements'].items():
            print(f"{param}: {colorize(f'{data['value']:.2f}', COLORS['BLUE'])} {data['unit']}")
    
    # Print predictions
    if report['predictions']:
        print(f"\n{colorize('PREDICTED PARAMETERS:', COLORS['BOLD'])}")
        for param, data in report['predictions'].items():
            if param not in report['measurements']:
                print(f"{param}: {colorize(f'{data['value']:.2f}', COLORS['BLUE'])} {data['unit']}")
                if data['reliability'] != 'N/A':
                    reliability_color = COLORS['GREEN'] if data['reliability'] > 0.85 else COLORS['YELLOW']
                    print(f"  Reliability: {colorize(f'{data['reliability']:.2f}', reliability_color)}")
                print(f"  Trend: {data['trend']}")
    
    # Print alerts
    if report['alerts']:
        print(f"\n{colorize('ALERTS:', COLORS['BOLD'])}")
        for alert in report['alerts']:
            status_color = COLORS['RED'] if alert['status'] == 'Critical' else COLORS['YELLOW']
            print(f"\n{colorize(f'{alert['status'].upper()}: {alert['parameter']}', status_color + COLORS['BOLD'])}")
            print(f"  Current: {colorize(f'{alert['value']:.2f}', COLORS['BLUE'])}")
            print(f"  Threshold: {alert['threshold']:.2f}")
            exceedance = (alert['exceedance_ratio']-1)*100
            print(f"  Exceedance: {colorize(f'{exceedance:.1f}%', status_color)}")
    
    # Print recommendations
    if report['recommendations']:
        print(f"\n{colorize('RECOMMENDATIONS:', COLORS['BOLD'])}")
        for rec in report['recommendations']:
            priority_color = COLORS['RED'] if rec['priority'] == 'High' else \
                           COLORS['YELLOW'] if rec['priority'] == 'Medium' else \
                           COLORS['GREEN']
            print(f"\n{colorize(f'{rec['priority']} Priority - {rec['type'].title()}:', priority_color + COLORS['BOLD'])}")
            print(f"  {rec['message']}")
            print(f"  {colorize('Action:', COLORS['BOLD'])} {rec['action_required']}")
    
    print("\n" + "="*60 + "\n")

def generate_pdf_report(report, filename):
    """Generate PDF version of the water quality report"""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='Alert',
        parent=styles['Normal'],
        textColor=colors.red,
        spaceAfter=10
    ))
    
    # Title
    title = Paragraph(
        f"Water Quality Report - {report['timestamp']}", 
        styles['Heading1']
    )
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Overall Status
    status_color = colors.red if report['summary']['overall_status'] == 'Critical' else \
                  colors.orange if report['summary']['overall_status'] == 'Warning' else \
                  colors.green
    
    status = Paragraph(
        f"Overall Status: <font color={status_color}>{report['summary']['overall_status']}</font>",
        styles['Heading2']
    )
    elements.append(status)
    elements.append(Spacer(1, 12))
    
    # Summary Table
    summary_data = [
        ['Total Parameters', str(report['summary']['total_parameters_monitored'])],
        ['Total Alerts', str(report['summary']['alert_count'])],
        ['Critical Alerts', str(report['summary']['critical_count'])],
        ['Warning Alerts', str(report['summary']['warning_count'])]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Measurements
    elements.append(Paragraph('Measured Parameters', styles['Heading2']))
    if report['measurements']:
        meas_data = [[
            'Parameter', 'Value', 'Unit'
        ]]
        for param, data in report['measurements'].items():
            meas_data.append([
                param,
                f"{data['value']:.2f}",
                data['unit']
            ])
        
        meas_table = Table(meas_data)
        meas_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(meas_table)
    elements.append(Spacer(1, 20))
    
    # Predictions
    elements.append(Paragraph('Predicted Parameters', styles['Heading2']))
    if report['predictions']:
        pred_data = [[
            'Parameter', 'Value', 'Unit', 'Reliability', 'Trend'
        ]]
        for param, data in report['predictions'].items():
            if param not in report['measurements']:
                pred_data.append([
                    param,
                    f"{data['value']:.2f}",
                    data['unit'],
                    f"{data['reliability']:.2f}" if data['reliability'] != 'N/A' else 'N/A',
                    data['trend']
                ])
        
        pred_table = Table(pred_data)
        pred_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(pred_table)
    elements.append(Spacer(1, 20))
    
    # Alerts
    if report['alerts']:
        elements.append(Paragraph('Alerts', styles['Heading2']))
        for alert in report['alerts']:
            alert_text = (
                f"{alert['status'].upper()}: {alert['parameter']}\n"
                f"Current: {alert['value']:.2f}\n"
                f"Threshold: {alert['threshold']:.2f}\n"
                f"Exceedance: {(alert['exceedance_ratio']-1)*100:.1f}%"
            )
            elements.append(Paragraph(alert_text, styles['Alert']))
    
    # Recommendations
    if report['recommendations']:
        elements.append(Paragraph('Recommendations', styles['Heading2']))
        for rec in report['recommendations']:
            rec_text = (
                f"{rec['priority']} Priority - {rec['type'].title()}:\n"
                f"{rec['message']}\n"
                f"Action: {rec['action_required']}"
            )
            elements.append(Paragraph(rec_text, styles['Normal']))
            elements.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(elements)

def main():
    # Create reports directory if it doesn't exist
    if not os.path.exists('ews_reports'):
        os.makedirs('ews_reports')
    
    # Initialize EWS
    ews = WaterQualityEWS()
    
    # Test cases
    test_cases = {
        "Normal Case": {
            'TDS (mg/L)': 450,
            'EC (µS/cm)': 750,
            'pH': 7.2
        },
        "High Values": {
            'TDS (mg/L)': 1500,
            'EC (µS/cm)': 2200,
            'pH': 7.2
        },
        "Extreme Values": {
            'TDS (mg/L)': 2500,
            'EC (µS/cm)': 3500,
            'pH': 8.5
        },
        "Missing TDS": {
            'EC (µS/cm)': 750,
            'pH': 7.0
        },
        "Borderline Values": {
            'TDS (mg/L)': 990,
            'EC (µS/cm)': 1490,
            'pH': 7.5
        }
    }
    
    # Run all test cases
    for case_name, measurements in test_cases.items():
        print(f"\n\nTEST CASE: {case_name}")
        print("=" * (len(case_name) + 11))
        
        print("\nInput Measurements:")
        for param, value in measurements.items():
            print(f"{param}: {value}")
        
        # Get predictions and alerts
        predictions, alerts = ews.predict_and_assess(measurements)
        
        print("\nPredicted Values:")
        for param, value in predictions.items():
            if param not in measurements:
                print(f"{param}: {value:.2f}")
        
        print(f"\nAlerts: {len(alerts)}")
        for alert in alerts:
            print(f"\nParameter: {alert['parameter']}")
            print(f"Status: {alert['status']}")
            print(f"Value: {alert['value']:.2f}")
            print(f"Threshold: {alert['threshold']:.2f}")
        
        # Generate report
        report = ews.generate_report(measurements, predictions, alerts)
        
        print_report(report)
        
        # Save report
        report_path = f'ews_reports/report_{case_name.lower().replace(" ", "_")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate PDF report
        pdf_path = f'ews_reports/report_{case_name.lower().replace(" ", "_")}.pdf'
        generate_pdf_report(report, pdf_path)

if __name__ == "__main__":
    main()
