import streamlit as st
import pandas as pd 
import numpy as np 
import torch 
from transformers import pipeline as hf_pipeline
from PIL import Image
import joblib
import sys
import os
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.forgery_check import load_model, ensemble_score
models_dir = os.path.join(project_root, 'models')




st.set_page_config(page_title='XYZ Logistics Driver Monitoring', layout='wide')

st.markdown("""
    <style>
        .main { font-size: 1.6rem; }
        .stTextArea textarea { font-size: 1.6rem; }
        .stDataFrame { font-size: 1.1rem; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title('XYZ Logistics - Smart Monitoring Dashboard')


# caching the models
@st.cache_resource 
def load_ratings_model():
    return joblib.load(os.path.join(models_dir, 'ratings_xgbregressor.joblib'))

@st.cache_resource
def load_violation_models():
    return {
        'hard_brake_flag':  joblib.load(os.path.join(models_dir, 'xgb_hard_brake.joblib')),
        'overspeed_flag':   joblib.load(os.path.join(models_dir, 'xgb_overspeed.joblib')),
        'swerving_flag':    joblib.load(os.path.join(models_dir, 'xgb_swerve.joblib'))
    }


@st.cache_resource
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1

    return hf_pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        device=device
    )

@ st.cache_resource
def load_forgery_detection_model():
    return load_model(os.path.join(models_dir, 'forgery_detector.pth'))


ratings_model       = load_ratings_model()
violation_models    = load_violation_models()
sentiment_model     = load_sentiment_model()
forgery_model       = load_forgery_detection_model()


# the app's tabs
tab1, tab2, tab3 = st.tabs(
    ['Telemetry & Violations', 'Feedback Sentiment', 'Forgery Detection']
)


with tab1:
    st.header('🚚Driver Rating & Violation Detection')
    st.write('Upload telemetry data in .csv format to get predicted driver rating and violations.')

    uploaded_csv = st.file_uploader('Upload Driver Telemetry CSV', type=['csv'])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.subheader('Preview')
        st.dataframe(df.head())


        st.subheader('Predicted Driver Ratings')
        rating_feats = [
            'avg_speed_kmh', 'peak_long_g', 'peak_lat_g', 'swerving_count',
            'overspeed_count', 'hard_brake_count'
        ]

        try:
            X_rating = df[rating_feats]
            df['predicted_rating'] = ratings_model.predict(X_rating).round(2)
            st.dataframe(df[['predicted_rating'] + rating_feats])

        except KeyError as e:
            st.error(f'Missing column for rating prediction: {e}')
    




        st.subheader('Violation Flags')
        viol_feats = [
            'max_speed_kmh','avg_speed_kmh','throttle_pct',
            'max_steering_rate_deg_per_s','peak_fwd_x_g','peak_rear_x_g',
            'peak_left_y_g','peak_right_y_g','peak_up_z_g','peak_down_z_g',
            'distance_tr_km','trip_duration_min'
        ]

        try:
            X_viol = df[viol_feats]

            for flag, model in violation_models.items():
                df[flag] = model.predict(X_viol)
            
            st.dataframe(df[['hard_brake_flag', 'overspeed_flag', 'swerving_flag']])

            st.write('**Violation Summary**')

            for flag in ['hard_brake_flag', 'overspeed_flag', 'swerving_flag']:
                count = df[flag].sum()

                label_map = {
                    'hard_brake_flag': 'Hard Braking',
                    'overspeed_flag': 'Overspeeding',
                    'swerving_flag': 'Swerving'
                }

                label = label_map[flag]


                if count > 0:
                    st.warning(f'⚠️{label} : {count} trip(s) flagged.')
                else:
                    st.success(f'✅{label} : No violations detected.')

        except:
            st.error(f'Missing column for violation detection: {e}')


with tab2:
    st.header('🙂Feedback Sentiment Analysis')
    st.write('Enter sample customer/recipient feedback to uncover the underlying sentiment.')

    feedback_txt = st.text_area('Feedback', placeholder='e.g. Delivery was on time. Driver was polite!')

    
    if st.button('Analyze'):
        if feedback_txt.strip() == '':
            st.warning('Feedback text is empty. Please enter valid feedback text.')
        else:
            result = sentiment_model(feedback_txt, truncation=True)[0]
            label = result['label'].lower()
            score = result['score']


            st.subheader('Result')

            if label == 'positive':
                st.success(f'Sentiment: **Positive**😊 (Confidence: {score:.2%})')
            elif label == 'negative':
                st.error(f'Sentiment: **Negative**😤 (Confidence: {score:.2%})')
            else:
                st.info(f'Sentiment: **Neutral**😐 (Confidence: {score:.2%})')


with tab3:
    st.header('🔎ID Forgery Detection')
    st.write('Upload an XYZ delivery driver ID image to determine its authenticity.')

    uploaded_image = st.file_uploader('Upload ID Image', type=['png','jpg','jpeg'])

    if uploaded_image is not None:

        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded ID', width=300)

        temp_path = 'temp_id.png'
        image.save(temp_path)

        if st.button('Run Forgery Check'):
            with st.spinner('Analyzing...'):
                result = ensemble_score(temp_path, forgery_model)

            st.subheader('Result')
            col1, col2, col3 = st.columns(3)
        
            with col1:
                st.metric('CNN Score', f'{result['cnn_score']:.4f}')
            with col2:
                st.metric('OCR Anomaly Score', f'{result['ocr_score']:.4f}')
            with col3:
                st.metric('Ensemble Score', f'{result['ensemble_score']:.4f}')
            
            if result['verdict'] == 'FORGED':
                st.error('Verdict: Likely FORGED⚠️')
            else:
                st.success('Verdict: Likely Genuine✅')
            

            with st.expander('See OCR Raw Output'):
                import pytesseract
                import cv2
                import numpy as np

                img_cv = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                raw_text = pytesseract.image_to_string(img_cv)
                st.text(raw_text)
        
            os.remove(temp_path)