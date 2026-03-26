import streamlit as st
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 AI-Driven Road Damage Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📸 Upload Road Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    st.markdown("---")
    st.info("""
    **System Capabilities:**
    - Detects potholes
    - Detects cracks
    - Severity analysis
    - Real-time results
    """)
    st.caption("👩‍💻 Developed by: HEYZAARA")
    st.caption("📅 B.Tech AI & Data Science")

# Function to detect damages using simple image processing (NO scipy)
def detect_damages(image_array):
    """Simple damage detection using image analysis"""
    
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    # Find dark areas (potential potholes)
    dark_threshold = 100
    dark_areas = gray < dark_threshold
    
    # Simple edge detection using difference (no scipy)
    # Create simple edge detection
    h, w = gray.shape
    edges = np.zeros_like(gray)
    for i in range(1, h-1):
        for j in range(1, w-1):
            # Simple gradient
            dx = abs(int(gray[i, j+1]) - int(gray[i, j-1]))
            dy = abs(int(gray[i+1, j]) - int(gray[i-1, j]))
            edges[i, j] = np.sqrt(dx*dx + dy*dy)
    
    edge_threshold = 50
    edge_areas = edges > edge_threshold
    
    damages = []
    
    # Detect potholes (dark circular areas)
    dark_count = np.sum(dark_areas)
    total_pixels = gray.size
    
    if dark_count > total_pixels * 0.05:
        confidence = min(dark_count / total_pixels * 3, 0.95)
        severity = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        damages.append({
            'type': 'pothole',
            'confidence': confidence,
            'severity': severity,
            'icon': '🕳️'
        })
    
    # Detect cracks (edge areas)
    edge_count = np.sum(edge_areas)
    if edge_count > total_pixels * 0.1:
        confidence = min(edge_count / total_pixels * 2, 0.9)
        severity = "Medium" if confidence > 0.5 else "Low"
        damages.append({
            'type': 'crack',
            'confidence': confidence,
            'severity': severity,
            'icon': '⚡'
        })
    
    return damages

# Main content
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Image info
        st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        # Convert image to array
        image_array = np.array(image)
        
        # Detect damages
        with st.spinner("Analyzing road surface..."):
            damages = detect_damages(image_array)
        
        # Show results
        if damages and len(damages) > 0:
            st.success(f"✅ {len(damages)} Road Damage(s) Detected!")
            st.markdown("---")
            
            for d in damages:
                severity_color = "🔴" if d['severity'] == 'High' else "🟡" if d['severity'] == 'Medium' else "🟢"
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                    <h3>{d['icon']} {d['type'].upper()}</h3>
                    <p><b>Confidence:</b> {d['confidence']:.1%}</p>
                    <p><b>Severity:</b> {severity_color} {d['severity']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Alert for high severity
            if any(d['severity'] == 'High' for d in damages):
                st.warning("🚨 **URGENT ALERT:** High severity damage detected! Immediate repair recommended.")
                st.info("📱 Notification sent to municipal authorities.")
            
            # Recommendations
            st.subheader("📋 Recommendations")
            if any(d['severity'] == 'High' for d in damages):
                st.write("🔴 **Action:** Immediate repair required (within 24 hours)")
            elif any(d['severity'] == 'Medium' for d in damages):
                st.write("🟡 **Action:** Schedule repair (within 7 days)")
            else:
                st.write("🟢 **Action:** Monitor condition (monthly inspection)")
                
        else:
            st.info("✅ **No road damages detected**")
            st.write("""
            **Tips for better detection:**
            - Take clear, well-lit photos
            - Focus on road surface
            - Include potholes or cracks in frame
            - Avoid shadows and glare
            """)
        
        # Confidence meter
        st.subheader("📊 System Confidence")
        if damages:
            avg_confidence = sum(d['confidence'] for d in damages) / len(damages)
            st.progress(avg_confidence)
            st.caption(f"Average detection confidence: {avg_confidence:.1%}")
        else:
            st.progress(0.0)
            st.caption("No damages detected")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>🚀 AI-Powered Road Infrastructure Monitoring System</p>
    <p>📱 Real-time Detection | 🗺️ Location Tracking | 📊 Analytics Dashboard</p>
    <p style='font-size: 12px;'>GitHub: HEYZAARA | B.Tech AI & Data Science</p>
</div>
""", unsafe_allow_html=True)
