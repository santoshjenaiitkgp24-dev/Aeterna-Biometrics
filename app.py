import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

# --- PIECE 1: CONFIG & NEON BLUE THEME CSS ---
st.set_page_config(page_title="Aeterna Biometrics", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; }

.stApp {
    background-color: #FFFFFF;
    background-image: radial-gradient(circle at 50% 20%, rgba(0,242,255,0.18), rgba(255,255,255,1) 70%);
}

.main-title {
    text-align: center;
    font-size: 90px;
    font-weight: 900;
    color: #0F172A;
    line-height: 1.0;
    margin-bottom: 0px !important;
    padding-bottom: 0px;
    letter-spacing: -2px;
}

.main-description {
    text-align: center;
    font-size: 24px;
    color: #334155;
    max-width: 1100px;
    margin: 0 auto 40px auto;
    font-weight: 500;
    line-height: 1.3;
}
/* Styling the Analyze Button to look high-end */
div.stButton > button {
    background-color: #00F2FF !important; /* Neon Blue */
    color: #0F172A !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 15px !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    box-shadow: 0 4px 15px rgba(0, 242, 255, 0.3) !important;
    transition: 0.3s !important;
}

div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 242, 255, 0.5) !important;
    transform: scale(1.02);
}

/* This makes the question text larger */
.stExpander p {
    font-size: 20px !important;
    font-weight: 600;
    color: #1E293B;
}

/* This makes the clickable header text larger */
.stExpander summary {
    font-size: 30px !important;
    font-weight: 600;
    padding: 15px !important;
}

/* This adds the neon blue glow to the expander boxes */
.stExpander {
    border: 1px solid #E2E8F0 !important;
    border-radius: 15px !important;
    box-shadow: 0 5px 15px rgba(0, 242, 255, 0.1) !important;
    margin-bottom: 10px !important;
}


.info-card {
    background: #FFFFFF !important;
    padding: 25px !important;
    border-radius: 15px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 10px 25px rgba(0,242,255,0.1) !important;
    margin-bottom: 20px !important;
    min-height: 220px !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.card-desc {
    font-style: bold !important;
    font-size: 23px !important;
    color: #1E293B !important;
    line-height: 1.3 !important;
}

/* This targets the titles (h3, h4, h6) inside your info-card */
div.info-card h3, div.info-card h4, div.info-card h6 {
    color: #0F172A !important; /* Force them to be Dark Slate */
    font-size: 28px !important; /* Make them big */
    font-weight: 800 !important; /* Make them thick */
    margin-top: 0px !important;
    margin-bottom: 15px !important;
    display: block !important;
}

div.info-card h6 {
    color: #00D1FF !important; /* Neon Blue */
    font-size: 16px !important;
    letter-spacing: 2px !important;
}

.result-card {
    background: #F8FDFF !important;
    padding: 30px !important;
    border-radius: 25px !important;
    border: 2px solid #00F2FF !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)


# --- PIECE: LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_my_model():
    # Make sure 'best_resnet_competition.keras' is exactly in your PyCharm folder
    return tf.keras.models.load_model('best_resnet_competition.keras')

# This creates the 'model' variable that the error is complaining about
model = load_my_model()

# --- PIECE 2: HEADER ---


st.markdown(
        body="""
        <h1 style="text-align:center; font-size:90px; font-weight:900; margin-bottom:0; color:#0F172A;">
              AETERNA BIOMETRICS
        </h1>
        <p style="text-align:center; font-size:22px; max-width:1000px; margin:auto; margin-top:-10px; color:#334155;">
            Harnessing the power of ResNet50 and advanced CNN architectures, 
            this AI-driven system analyzes facial features to accurately predict age and gender. 
            By combining transfer learning with modern computer vision techniques, 
            it transforms visual data into meaningful demographic insights.
        </p>
        """,
        unsafe_allow_html=True
    )



# 2. THE UI (Place this exactly where you want the uploader to appear)
_, center_col, _ = st.columns([1, 2, 1])

#with center_col:
    # This draws the card
    ##st.markdown("""
    ##<div class="upload-card">
      ##  <div class="icon-circle">↑</div>
       ## <div style="font-size: 45px; font-weight: 800; color: #0F172A; margin-bottom: 5px;">Upload Image</div>
        ##<div style="font-size: 18px; color: #64748B; margin-bottom: 30px;">JPG, PNG, or WebP • Max 5MB</div>
    ##</div>
    ##""", unsafe_allow_html=True)

    # This places the REAL button exactly under the text
st.markdown('<div style="text-align: center; margin-top: -60px;">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"])
st.markdown('</div>', unsafe_allow_html=True)
# --- PIECE: PROCESSING THE UPLOADED FILE ---

if uploaded_file is not None:
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Split the screen into two columns: Left for Image, Right for Result
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Show the image the user just uploaded
        img_pil = Image.open(uploaded_file)
        st.image(img_pil, caption="Target Image", use_container_width=True)

    with col_right:
        st.write("<br><br>", unsafe_allow_html=True)

        # Create the Analyze Button
        if st.button("ANALYZE BIOMETRICS", use_container_width=True, type="primary"):

            # Show progress bar (Professional touch)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Preprocessing for your ResNet50 Model
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (224, 224)) / 255.0
            input_tensor = np.expand_dims(img_resized, axis=0)

            # Generate Predictions
            preds = model.predict(input_tensor)
            age_result = int(preds[0][0][0])
            gender_result = "MALE" if preds[1][0][0] > 0.5 else "FEMALE"

            # Display the result in a Luxury Card
            st.markdown(f"""
                <div style="background:#F8FDFF; padding:40px; border-radius:30px; border:2px solid #00F2FF; text-align:center; box-shadow: 0 10px 30px rgba(0,242,255,0.2);">
                    <h4 style="color:#008D96; margin:0; letter-spacing:1px;">BIOMETRIC MATCH FOUND</h4>
                    <h1 style="font-size:75px; color:#1A1A1B; margin:0;">{age_result} <small style="font-size:25px;">YEARS</small></h1>
                    <h2 style="color:#334155; letter-spacing:3px; margin:0; font-weight:800;">{gender_result}</h2>
                </div>
            """, unsafe_allow_html=True)

# --- PIECE 4: WORKFLOW ---
st.markdown("<br><br><h1 style='text-align: center; color: #0F172A;'>Aeterna's Workflow Analysis</h1>",
            unsafe_allow_html=True)
w1, w2, w3, w4 = st.columns(4)
workflow_steps = [
    ("IMAGE PROCESSING",
     "Input photos are digitally normalized to 224x224. This removes noise and ensures clarity for the ResNet layers."),
    ("FEATURE DETECTION",
     "The analyzer identifies eyes, nose, and jawline density to create a unique spatial representation of the subject."),
    ("AI MODEL ANALYSIS",
     "Our deep-learning ResNet50 model compares features against thousands of samples with high precision."),
    ("FINAL RESULT",
     "Data is pushed through sigmoid heads to provide an accurate biometric estimation of age and gender.")
]

for i, col in enumerate([w1, w2, w3, w4]):
    with col:
        st.markdown(
            f"""<div class="info-card"><h6 style="color: #00F2FF; margin:0;">STEP 0{i + 1}</h6><h4 style="margin: 5px 0;">{workflow_steps[i][0]}</h4><p class="card-desc">{workflow_steps[i][1]}</p></div>""",
            unsafe_allow_html=True)

# PIECE 5: Resnet Advantage Card
st.markdown("<br><br><h1 style='text-align: center; color: #1A1A1B;'>The ResNet Advantage</h1>", unsafe_allow_html=True)

# --- ROW 1 ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="info-card">
            <h4>Precision Through Residual Learning</h4>
            <p class="card-desc">Unlike standard CNNs, our <b>ResNet50 backbone</b> utilizes 'skip connections' to overcome the vanishing gradient problem. This allows the model to learn 50 layers of deep features, resulting in a <b>Mean Absolute Error (MAE)</b> that rivals human perception. </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="info-card">
            <h4>Transfer Learning Mastery</h4>
            <p class="card-desc">By leveraging <b>pre-trained weights</b> from the massive ImageNet database, the model starts with a sophisticated understanding of visual patterns. This <b>knowledge transfer</b> allows the network to identify fundamental shapes and textures before specifically refining its focus on human facial biometrics.</p>
        </div>
    """, unsafe_allow_html=True)

# --- ROW 2 ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
        <div class="info-card">
            <h4>Structural Integrity & Efficiency</h4>
            <p class="card-desc">Unlike older deep networks, the ResNet backbone uses <b>Global Average Pooling</b> to reduce total parameters without losing spatial information. This ensures the system focuses purely on <b>physiological markers</b> of age rather than being distracted by image noise or background inconsistencies.</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="info-card">
            <h4>Dual-Head Architectural Synergy</h4>
            <p class="card-desc">The system utilizes a shared backbone that splits into two <b>specialized 'heads' for age and gender</b>. This multi-task learning approach allows the model to understand how gender-specific traits influence aging patterns, creating a more holistic and <b>statistically accurate</b> biometric profile.</p>
        </div>
    """, unsafe_allow_html=True)

# --- PIECE 5: FACE SHAPES ---
#st.markdown("<br><br><h1 style='text-align: center; color: #0F172A;'>Types of Face Shapes</h1>", unsafe_allow_html=True)
#st.markdown(
 #   "<p style='text-align: center; font-size: 18px; color: #334155;'>There are 6 classifications of face shape around the world. Here is the important characteristics of each face shape type</p>",
 #   unsafe_allow_html=True)
#shapes = [
 #   ("Oval", "Balanced proportions with wider cheekbones and a curved jawline. Most versatile for biometric extraction."),
  #  ("Heart-shaped", "Broad forehead tapering to a narrow chin. Includes prominent cheekbones used as landmarks."),
  #  ("Oblong", "Significantly longer than wide. Features a straight cheek line requiring scaled horizontal analysis."),
  #  ("Square", "Defined by a strong jawline and equal width across forehead and cheeks. High facial symmetry levels."),
  #  ("Round", "Full cheeks and soft jawline with nearly equal width and height. Focuses on soft curve placements."),
   # ("Diamond", "Narrow at forehead and chin, cheekbones are the widest part. Creates unique angles for deep learning.")
#]
#s_cols = st.columns(3)
#for i in range(3):
  #  with s_cols[i]:
 #       st.markdown(
  #          f"""<div class="info-card"><h4 style="color: #008D96; margin:0;">{shapes[i][0]}</h4><p class="card-desc">{shapes[i][1]}</p></div>""",
  #          unsafe_allow_html=True)
#s_cols2 = st.columns(3)
#for i in range(3, 6):
  #  with s_cols2[i - 3]:
  #      st.markdown(
  #          f"""<div class="info-card"><h4 style="color: #008D96; margin:0;">{shapes[i][0]}</h4><p class="card-desc">{shapes[i][1]}</p></div>""",
   #         unsafe_allow_html=True)

#PIECE 6 : INDUSTRIAL APPLICATION CARD

st.markdown("<br><br><h1 style='text-align: center; color: #1A1A1B;'>Industrial Applications</h1>", unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
apps = [
    ("Retail Analytics", "Track customer demographics in real-time to optimize product placement and store layouts without storing personal identities."),
    ("Targeted Marketing", "Deliver personalized content and advertisements based on the estimated age group and gender of the viewer."),
    ("Access Control", "Enhance security systems by adding biometric verification layers that can detect age-restricted access points automatically.")
]
for i, col in enumerate([a1, a2, a3]):
    with col:
        st.markdown(f"""<div class="info-card"><h4>{apps[i][0]}</h4><p class="card-desc">{apps[i][1]}</p></div>""", unsafe_allow_html=True)


# --- PIECE 7: PRIVACY CARDS ---
st.markdown("<br><br><h1 style='text-align: center; color: #1A1A1B;'>Ethical AI & Privacy Charter</h1>", unsafe_allow_html=True)
p1, p2 = st.columns(2)
charter = [
    ("Zero Data Retention", "We do not store, cache, or transmit your uploaded images. Analysis happens entirely in volatile memory and is purged instantly after processing."),
    ("Bias Mitigation", "Our dataset includes a wide spectrum of ethnicities and lighting conditions to ensure the ResNet model remains fair and unbiased across all demographics.")
]
for i, col in enumerate([p1, p2]):
    with col:
        st.markdown(f"""<div class="info-card"><h4>{charter[i][0]}</h4><p class="card-desc">{charter[i][1]}</p></div>""", unsafe_allow_html=True)


 #------PIECE 8 : FAQ SECTION--------

st.markdown("<br><br><h1 style='text-align: center; color: #1A1A1B;'> Frequently Asked Questions</h1>", unsafe_allow_html=True)
#st.markdown('<div class="info-card" style="height: auto;">', unsafe_allow_html=True)
with st.expander("How accurate is the age prediction?"):
    st.write("The model typically predicts within a range of +/- 5 years, depending on image quality and lighting.")
with st.expander("Can it detect multiple faces?"):
    st.write("Currently, the system is optimized for single-subject portrait analysis for maximum precision.")
st.markdown('</div>', unsafe_allow_html=True)
with st.expander("Does the system’s performance change based on lighting or facial angles?"):
    st.write("Aeterna Biometrics is engineered with spatial invariance; however, for the highest biometric precision, we recommend front-facing portraits with even lighting. Extreme shadows or steep angles can occasionally obscure the micro-textures that the ResNet model relies on for accurate chronological age estimation.")
st.markdown('</div>', unsafe_allow_html=True)
with st.expander("To what degree is the predictive accuracy compromised by external variables such as eyewear, headgear, or facial coverings?"):
    st.write(
        "The ResNet architecture utilizes a deep residual hierarchy that allows for occlusion-aware processing. Rather than a total signal failure, the network dynamically weights visible biometric markers more heavily. This ensures the final age and gender estimation is derived from persistent physiological data rather than superficial artifacts or non-biometric noise.")
with st.expander("Can the system differentiate between chronological age and perceived age influenced by makeup?"):
    st.write("While the AI is trained to identify physiological markers like bone structure and skin elasticity, it primarily analyzes perceived biometric data. Consequently, significant aesthetic modifications—such as heavy stage makeup—may slightly influence the numerical result, similar to how they would affect a human observer's estimation.")
st.markdown('</div>', unsafe_allow_html=True)
with st.expander("What impact does image resolution have on the model’s ability to resolve age-related textures?"):
    st.write("To ensure the ResNet backbone can accurately resolve micro-level features—such as skin porosity and fine lines—we recommend a minimum facial resolution of 224x224 pixels. Lower resolutions may trigger feature smoothing, where the AI is forced to rely on macro-level structural geometry rather than the hyper-fine textures required for pinpointing chronological age.")
st.markdown('</div>', unsafe_allow_html=True)






#-----PIECE 9 : FOOTER SECTION ------

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892B0; font-size: 20px;'>Developed by Santosh Jena | Powered by TensorFlow & Streamlit | © 2024 AeternaBiometrics</p>", unsafe_allow_html=True)