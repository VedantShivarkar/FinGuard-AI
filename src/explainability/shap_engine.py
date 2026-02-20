import shap
import pandas as pd
import plotly.express as px

def generate_shap_explanation(model, input_df, feature_names):
    """Generates SHAP values to explain WHY the anomaly was flagged."""
    try:
        # TreeExplainer works perfectly with Isolation Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Create a DataFrame for easy Plotly visualization
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values[0]  # Get values for the single transaction
        })
        
        # Sort by the absolute impact to find the most suspicious features
        shap_df['Abs_Impact'] = shap_df['Impact'].abs()
        shap_df = shap_df.sort_values(by='Abs_Impact', ascending=False)
        
        # Create a beautiful horizontal bar chart
        fig = px.bar(
            shap_df, x='Impact', y='Feature', orientation='h',
            title="🧠 AI Brain: Feature Impact Analysis (SHAP)",
            color='Impact', 
            color_continuous_scale=px.colors.diverging.RdBu
        )
        
        # Create a text summary for the LLM Investigator
        top_features = shap_df.head(3)['Feature'].tolist()
        shap_summary = f"The top 3 anomalies detected were in these features: {', '.join(top_features)}."
        
        return fig, shap_summary
    except Exception as e:
        return None, f"Error generating SHAP explanation: {e}"