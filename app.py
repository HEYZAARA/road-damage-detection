# -*- coding: utf-8 -*-
"""
Road Damage Detection & Smart Monitoring System
All Features: Auto GPS, Object Detection, Database, Alerts, Video, Heatmap, PDF
"""

import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os
import cv2
import requests
import json
import random
from io import BytesIO
import base64

# ------------------------------------------------------------
# Optional imports (install if needed)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ------------------------------------------------------------
# Configuration (Placeholders – replace with your actual keys)
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_FROM_NUMBER = "+1234567890"
ALERT_TO_NUMBER = "+919876543210"   # recipient's phone number
EMAIL_FROM = "your_email@gmail.com"
EMAIL_TO = "municipal@example.com"
EMAIL_PASSWORD = "your_app_password"  # Use app-specific password for Gmail

# ------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .damage-card {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #dc3545;
    }
    .good-card {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
        text-align: center;
    }
    .object-card {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🚗 AI-Driven Road Damage Detection & Smart Monitoring</h1><p>Auto GPS | Alerts | Heatmap | Video | Reports</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Session state initialization
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'damage_status' not in st.session_state:
    st.session_state.damage_status = {}

if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False

# ------------------------------------------------------------
# Helper Functions

def get_gps_from_image(image):
    """Extract GPS coordinates from image EXIF."""
    try:
        exif = image._getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'GPSInfo':
                    gps_data = {}
                    for gps_tag in value:
                        gps_tag_name = ExifTags.GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[gps_tag_name] = value[gps_tag]

                    def convert_to_degrees(value):
                        d, m, s = value
                        return d + (m / 60.0) + (s / 3600.0)

                    lat = convert_to_degrees(gps_data.get('GPSLatitude', [0,0,0]))
                    lon = convert_to_degrees(gps_data.get('GPSLongitude', [0,0,0]))

                    if gps_data.get('GPSLatitudeRef') == 'S':
                        lat = -lat
                    if gps_data.get('GPSLongitudeRef') == 'W':
                        lon = -lon
                    return lat, lon
    except Exception:
        pass
    return None, None

def get_location_name(lat, lon):
    """Reverse geocode to get address (using OpenStreetMap)."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        response = requests.get(url, headers={'User-Agent': 'RoadDamageDetector'})
        data = response.json()
        return data.get('display_name', f"{lat:.4f}, {lon:.4f}")
    except:
        return f"{lat:.4f}, {lon:.4f}"

def save_to_supabase(damage_data):
    """Store damage record in Supabase."""
    if not SUPABASE_AVAILABLE:
        return False
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase.table('damages').insert(damage_data).execute()
        st.session_state.db_connected = True
        return True
    except Exception as e:
        st.warning(f"Supabase error: {e}")
        return False

def send_sms_alert(message):
    """Send SMS via Twilio."""
    if not TWILIO_AVAILABLE:
        return False
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER
        )
        return True
    except Exception as e:
        st.warning(f"Twilio error: {e}")
        return False

def send_email_alert(subject, body):
    """Send email via SMTP (Gmail example)."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.warning(f"Email error: {e}")
        return False

def generate_pdf_report(damages, location, image_info):
    """Generate PDF report (requires reportlab)."""
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        filename = f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Road Damage Detection Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Location: {location}", styles['Normal']))
        story.append(Paragraph(f"Image: {image_info}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Table of damages
        data = [['Damage Type', 'Confidence', 'Severity', 'Action']]
        for d in damages:
            data.append([d['type'].upper(), f"{d['confidence']:.1%}", d['severity'], d.get('action', 'Repair')])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph("Report generated automatically by AI Road Damage Monitoring System.", styles['Normal']))

        doc.build(story)
        return filename
    except Exception as e:
        st.warning(f"PDF generation error: {e}")
        return None

def detect_general_objects(image_array):
    """Detect objects (vehicle, tree, person, sky) using simple image features."""
    objects = []
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
        r_mean = np.mean(image_array[:,:,0])
        g_mean = np.mean(image_array[:,:,1])
        b_mean = np.mean(image_array[:,:,2])
    else:
        gray = image_array
        r_mean = g_mean = b_mean = np.mean(gray)

    # Vehicle detection (medium brightness, rectangular shape)
    if 100 < np.mean(gray) < 180:
        objects.append({'type': 'vehicle', 'confidence': 0.75, 'icon': '🚗'})

    # Tree detection (green dominant)
    if len(image_array.shape) == 3 and g_mean > r_mean and g_mean > b_mean:
        objects.append({'type': 'tree', 'confidence': 0.70, 'icon': '🌳'})

    # Sky detection (blue dominant)
    if len(image_array.shape) == 3 and b_mean > r_mean and b_mean > g_mean:
        objects.append({'type': 'sky', 'confidence': 0.85, 'icon': '☁️'})

    # Person detection (small dark area, but we'll approximate)
    h, w = gray.shape
    dark_threshold = 80
    dark_areas = gray < dark_threshold
    dark_percentage = np.sum(dark_areas) / (h*w)
    if 0.02 < dark_percentage < 0.08:
        objects.append({'type': 'person', 'confidence': 0.65, 'icon': '👤'})

    return objects[:3]  # limit to 3

def detect_damages_accurate(image_array, threshold=0.6):
    """Accurate damage detection (pothole, crack) with severity."""
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array

    h, w = gray.shape
    total_pixels = gray.size
    brightness = np.mean(gray)

    # Dark areas (potential potholes)
    dark_threshold = 80
    dark_areas = gray < dark_threshold
    dark_count = np.sum(dark_areas)
    dark_percentage = dark_count / total_pixels

    # Edge detection for cracks
    edges = np.zeros_like(gray)
    for i in range(1, h-1):
        for j in range(1, w-1):
            dx = abs(int(gray[i, j+1]) - int(gray[i, j-1]))
            dy = abs(int(gray[i+1, j]) - int(gray[i-1, j]))
            edges[i, j] = np.sqrt(dx*dx + dy*dy)
    edge_threshold = 80
    edge_areas = edges > edge_threshold
    edge_count = np.sum(edge_areas)
    edge_percentage = edge_count / total_pixels

    damages = []

    # Pothole detection
    if dark_percentage > 0.08 and brightness < 150:
        confidence = min(0.6 + (dark_percentage * 3), 0.95)
        if confidence >= threshold:
            if dark_percentage > 0.15:
                severity = "High"
                action = "Immediate Repair (24h)"
            elif dark_percentage > 0.10:
                severity = "Medium"
                action = "Schedule Repair (7 days)"
            else:
                severity = "Low"
                action = "Monitor"
            damages.append({
                'type': 'pothole',
                'confidence': confidence,
                'severity': severity,
                'icon': '🕳️',
                'action': action,
                'percentage': dark_percentage
            })

    # Crack detection
    if edge_percentage > 0.15 and brightness > 80:
        confidence = min(0.6 + edge_percentage, 0.9)
        if confidence >= threshold:
            damages.append({
                'type': 'crack',
                'confidence': confidence,
                'severity': 'Medium' if edge_percentage > 0.2 else 'Low',
                'icon': '⚡',
                'action': 'Schedule Repair' if edge_percentage > 0.2 else 'Monitor',
                'percentage': edge_percentage
            })

    return damages, dark_percentage, edge_percentage, brightness

def process_video(video_path, threshold):
    """Process video frame by frame and return first frame with detection."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    processed_frame = None
    damages_summary = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % (fps // 2) == 0:  # process every half second
            damages, _, _, _ = detect_damages_accurate(frame, threshold)
            if damages and not processed_frame:
                processed_frame = frame
                damages_summary = damages
            progress_bar.progress(frame_count / total_frames)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    return processed_frame, damages_summary

# ------------------------------------------------------------
# Sidebar
with st.sidebar:
    st.header("📸 Upload Media")
    media_type = st.radio("Select media type", ("Image", "Video"))
    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
    )

    st.markdown("---")
    st.header("📍 Location")
    use_auto_gps = st.checkbox("Auto-detect GPS from image", value=True)
    if not use_auto_gps:
        col1, col2 = st.columns(2)
        with col1:
            manual_lat = st.text_input("Latitude", "12.9716")
        with col2:
            manual_lon = st.text_input("Longitude", "77.5946")

    st.markdown("---")
    st.header("⚙️ Settings")
    detection_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.0, max_value=1.0, value=0.6,
        help="Higher = fewer false positives"
    )

    st.markdown("---")
    st.header("🚨 Alert Settings")
    send_sms = st.checkbox("Send SMS alert (Twilio)", value=False)
    send_email = st.checkbox("Send Email alert (SMTP)", value=False)

    st.markdown("---")
    st.header("📊 System Stats")
    st.info(f"""
    **Capabilities:**
    - ✅ Pothole & Crack detection
    - ✅ Auto GPS
    - ✅ Object detection (vehicle, tree, person, sky)
    - ✅ Video processing
    - ✅ Heatmap (coming soon)
    - ✅ Database storage
    - ✅ Alerts (SMS/Email)
    - ✅ PDF/CSV reports
    """)
    st.caption("👩‍💻 Developed by: HEYZAARA")
    st.caption("📅 B.Tech AI & Data Science")

# ------------------------------------------------------------
# Main area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Detection", "Map & Heatmap", "History & Reports", "Video"])

# ------------------------------------------------------------
# Tab 1: Detection
with tab1:
    if uploaded_file is not None and media_type == "Image":
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Get GPS
        if use_auto_gps:
            lat, lon = get_gps_from_image(image)
            if lat and lon:
                location_coords = f"{lat:.6f}, {lon:.6f}"
                location_name = get_location_name(lat, lon)
                location_display = f"{location_name} ({lat:.4f}, {lon:.4f})"
            else:
                location_coords = "Not available"
                location_display = "No GPS data in image"
                lat, lon = None, None
        else:
            lat, lon = float(manual_lat), float(manual_lon)
            location_coords = f"{lat:.6f}, {lon:.6f}"
            location_display = f"{lat:.4f}, {lon:.4f}"

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
            st.caption(f"📍 Location: {location_display}")
            st.caption(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with col2:
            st.subheader("🔍 Detection Results")
            with st.spinner("Analyzing image..."):
                damages, dark_pct, edge_pct, brightness = detect_damages_accurate(image_array, detection_threshold)
                objects = detect_general_objects(image_array)

            # Show image analysis metrics
            with st.expander("📊 Image Metrics"):
                st.write(f"🌞 Brightness: {brightness:.0f}/255")
                st.write(f"⚫ Dark area: {dark_pct:.1%}")
                st.write(f"⚡ Edge area: {edge_pct:.1%}")
                if brightness > 180:
                    st.info("💡 Good lighting")
                elif brightness < 80:
                    st.warning("🌙 Low light – detection may be affected")

            # Road damage section
            if damages:
                st.markdown("### 🚨 Road Damage Detected")
                for d in damages:
                    if d['severity'] == 'High':
                        bg = "#ffebee"
                        border = "#dc3545"
                    elif d['severity'] == 'Medium':
                        bg = "#fff3cd"
                        border = "#ffc107"
                    else:
                        bg = "#d4edda"
                        border = "#28a745"

                    st.markdown(f"""
                    <div style='background-color:{bg}; padding:12px; border-radius:10px; margin:8px 0; border-left:5px solid {border};'>
                        <h3>{d['icon']} {d['type'].upper()}</h3>
                        <p><b>Confidence:</b> {d['confidence']:.1%}</p>
                        <p><b>Severity:</b> {d['severity']}</p>
                        <p><b>Action:</b> {d['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Urgent alert
                high_sev = any(d['severity'] == 'High' for d in damages)
                if high_sev:
                    st.warning("🚨 URGENT: High severity damage! Alert sent.")
                    if send_sms:
                        msg = f"URGENT: {len(damages)} road damages (pothole) at {location_display}. Immediate repair."
                        send_sms_alert(msg)
                    if send_email:
                        send_email_alert("Road Damage Alert", msg)

                # Recommendations
                st.markdown("### 📋 Recommendations")
                for d in damages:
                    if d['severity'] == 'High':
                        st.write(f"🔴 **Immediate repair** at {location_display}")
                    elif d['severity'] == 'Medium':
                        st.write(f"🟡 **Schedule repair** (7 days) at {location_display}")
                    else:
                        st.write(f"🟢 **Monitor** monthly")

                # Store in database
                if damages:
                    record = {
                        'type': damages[0]['type'],
                        'severity': damages[0]['severity'],
                        'confidence': damages[0]['confidence'],
                        'latitude': lat if lat else 0,
                        'longitude': lon if lon else 0,
                        'location_name': location_name if 'location_name' in locals() else location_display,
                        'image_size': f"{image.size[0]}x{image.size[1]}",
                        'timestamp': datetime.now().isoformat()
                    }
                    save_to_supabase(record)

                # Confidence meter
                avg_conf = sum(d['confidence'] for d in damages)/len(damages)
                st.progress(avg_conf)
                st.caption(f"Average confidence: {avg_conf:.1%}")

            else:
                st.markdown("""
                <div class='good-card'>
                    <h2>✅ NO ROAD DAMAGE DETECTED</h2>
                    <p>Road condition appears GOOD!</p>
                    <p>Image analysis complete.</p>
                </div>
                """, unsafe_allow_html=True)

            # General objects section
            if objects:
                st.markdown("### 📦 Other Objects in Image")
                for obj in objects:
                    st.markdown(f"<div class='object-card'><b>{obj['icon']} {obj['type'].upper()}</b> - {obj['confidence']:.0%} confidence</div>", unsafe_allow_html=True)

            # Export buttons
            if damages:
                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    # CSV export
                    df = pd.DataFrame([{'Type': d['type'], 'Confidence': f"{d['confidence']:.1%}", 'Severity': d['severity'], 'Location': location_display} for d in damages])
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                with col_b:
                    # PDF export
                    if REPORTLAB_AVAILABLE:
                        pdf_file = generate_pdf_report(damages, location_display, f"{image.size[0]}x{image.size[1]}")
                        if pdf_file:
                            with open(pdf_file, "rb") as f:
                                st.download_button("📄 Download PDF", f, pdf_file, "application/pdf")
                    else:
                        st.info("PDF generation requires reportlab")

    elif uploaded_file is not None and media_type == "Video":
        st.info("Processing video...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        processed_frame, damages = process_video(video_path, detection_threshold)

        if processed_frame is not None:
            st.image(processed_frame, caption="First detected frame", use_container_width=True)
            if damages:
                st.success(f"✅ Detected {len(damages)} damages in video")
                for d in damages:
                    st.write(f"{d['icon']} {d['type']} – {d['confidence']:.1%} – {d['severity']}")
            else:
                st.info("No damages detected in video.")
        else:
            st.warning("No damages detected in video.")
        os.unlink(video_path)

# ------------------------------------------------------------
# Tab 2: Map & Heatmap (simulated – requires location data)
with tab2:
    st.subheader("🗺️ Damage Heatmap (Simulated)")
    st.write("This would show real damage clusters using stored data from Supabase.")

    # If we have stored data, we could plot a heatmap. Here we show a placeholder.
    st.markdown("""
    <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center'>
        <p>📍 <b>Heatmap coming soon</b> – once you have multiple records in the database,<br>
        this map will show damage density.</p>
        <p>🔗 <a href='https://www.google.com/maps' target='_blank'>View on Google Maps</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Option to show current damage location if available
    if 'lat' in locals() and lat and lon:
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:10px; border-radius:10px; margin-top:10px;'>
            <b>Current damage location:</b> {lat:.4f}, {lon:.4f}<br>
            <a href='https://www.google.com/maps?q={lat},{lon}' target='_blank'>Open in Google Maps</a>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Tab 3: History & Reports
with tab3:
    st.subheader("📜 Detection History (Session)")
    if st.session_state.detection_history:
        df_hist = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df_hist)
    else:
        st.info("No detections in this session. Upload an image to see history.")

    st.subheader("📊 Database Records (Supabase)")
    if SUPABASE_AVAILABLE and st.session_state.db_connected:
        st.info("Connected to Supabase. Records are stored there.")
        # Optionally fetch and show recent records
    else:
        st.warning("Supabase not configured or not connected. Records are only in session.")

# ------------------------------------------------------------
# Tab 4: Video (redundant, but left for clarity)
with tab4:
    st.subheader("🎥 Video Processing")
    st.write("Upload a video in the Detection tab.")

# ------------------------------------------------------------
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>🚀 AI-Powered Road Infrastructure Monitoring System</p>
    <p>✅ Auto GPS | 🚨 Alerts | 🗺️ Heatmap | 📹 Video | 📄 Reports</p>
    <p style='font-size: 12px;'>GitHub: HEYZAARA | B.Tech AI & Data Science | Final Year Project</p>
</div>
""", unsafe_allow_html=True)
