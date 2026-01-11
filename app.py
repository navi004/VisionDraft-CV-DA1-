import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import io

# -------------------- UI CONFIG & THEME -------------------- #
st.set_page_config(page_title="GShapeAnalyst | Naveen N", layout="wide", page_icon="🔬")

# CUSTOM CSS: Advanced Gradient Background and Styled Header
st.markdown("""
<style>
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }

    /* Professional Header Card */
    .header-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .student-info {
        color: #38bdf8;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Modernizing Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- CUSTOM HEADER -------------------- #
st.markdown(f"""
<div class="header-box">
    <h1 style='margin:0; padding:0; color:#38bdf8;'>📐 Neural-Shape Contour Engine</h1>
    <p style='margin:0; color:#94a3b8; font-size:1.1rem;'>Advanced Computer Vision Analytics Dashboard</p>
    <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 15px 0;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="student-info">Student: Naveen N</span>
        <span class="student-info">Reg No: 22MIA1049</span>
        <span class="student-info">Course: Computer Vision</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- HELPER FUNCTIONS -------------------- #
def get_image_download(img_array):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_shapes(image, thresh_val, line_color, thickness):
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img_array.copy()
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    data = []
    color_bgr = tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 300: continue 

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        # Simplified Logic for Naveen's Assignment
        if vertices == 3: shape = "Triangle"
        elif vertices == 4:
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif circularity > 0.82: shape = "Circle"
        else: shape = "Polygon"

        cv2.drawContours(output_img, [cnt], -1, color_bgr, thickness)

        # Minimalist Text ID
        label_text = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thick = 0.45, 1
        cv2.putText(output_img, label_text, (x+1, y-7), font, font_scale, (0,0,0), font_thick+1)
        cv2.putText(output_img, label_text, (x, y-8), font, font_scale, color_bgr, font_thick)

        data.append({
            "ID": i + 1, "Shape": shape,"Sides": vertices, "Area (px²)": round(area, 1), 
            "Perimeter (px)": round(perimeter, 1), "Circularity": round(circularity, 2)
        })

    mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return output_img, mask_bgr, data

# -------------------- MAIN DASHBOARD -------------------- #
st.sidebar.markdown("### 🛠️ Lab Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image Target", type=["png", "jpg", "jpeg"])
line_color = st.sidebar.color_picker("Contour Color", "#0000FF")
line_thickness = st.sidebar.slider("Line Width", 1, 5, 1)
threshold = st.sidebar.slider("Binary Sensitivity", 0, 255, 127)

if uploaded_file:
    image = Image.open(uploaded_file)
    processed, mask, results = analyze_shapes(image, threshold, line_color, line_thickness)
    df = pd.DataFrame(results)

    # Sidebar Downloads
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Deliverables")
    st.sidebar.download_button("💾 Download Analysis View", get_image_download(processed), "Naveen_Analysis.png", "image/png", use_container_width=True)
    st.sidebar.download_button("💾 Download Binary Mask", get_image_download(mask), "Naveen_Mask.png", "image/png", use_container_width=True)

    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objects Identified", len(df))
    col2.metric("Unique Geometries", df['Shape'].nunique() if not df.empty else 0)
    col3.metric("Avg Surface Area", f"{df['Area (px²)'].mean():.1f}" if not df.empty else 0)
    col4.metric("Lab Status", "Verified")

    # Tabs for Workflow
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Final Result", "🌑 CV Mask", "📊 Analytics", "📄 Technical Data"])
    
    with tab1:
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Figure 1: Automated Shape Segmentation")
    
    with tab2:
        st.image(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Figure 2: Foreground/Background Extraction")
        
    with tab3:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(df['Shape'].value_counts().reset_index(), x='Shape', y='count', title="Shape Frequency", template="plotly_dark", color_discrete_sequence=['#38bdf8']), use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(df, names='Shape', title="Population Distribution", hole=0.4, template="plotly_dark"), use_container_width=True)
        else:
            st.warning("No data points available.")

    with tab4:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Export Feature Report (CSV)", df.to_csv(index=False).encode('utf-8'), "Shape_Features.csv", "text/csv")
else:
    st.info("👋 Hello ! Please upload an image to begin the analysis.")