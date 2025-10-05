import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import base64
import os 
from io import BytesIO
import zipfile

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Filter App",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-15px);
        }
        60% {
            transform: translateY(-8px);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px #667eea;
        }
        50% {
            box-shadow: 0 0 20px #667eea, 0 0 30px #764ba2;
        }
    }
    
    .animated-icon {
        animation: bounce 2s infinite;
        font-size: 2.5rem;
        display: inline-block;
    }
    
    .pulse-icon {
        animation: pulse 1.5s infinite;
        font-size: 2rem;
        display: inline-block;
    }
    
    .rotate-icon {
        animation: rotate 3s linear infinite;
        font-size: 1.5rem;
        display: inline-block;
    }
    
    .glow-effect {
        animation: glow 2s infinite;
    }
    
    .filter-card, .status-card, .info-card, .success-card {
        border-radius: 12px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        padding: 15px;
    }

    .filter-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: transform 0.3s ease;
    }
    
    .filter-card:hover {
        transform: translateY(-3px);
    }
    
    .status-card {
        background: linear-gradient(135deg, #4ac3e4 0%, #36d1dc 100%);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 30px;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f7b733 0%, #fc4a1a 100%);
        text-align: center;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        text-align: center;
    }
    
    .download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    
    .stDownloadButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .capture-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Hands
@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

# Filter Functions
def filter_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def filter_invert(frame):
    return cv2.bitwise_not(frame)

def filter_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def filter_depth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    return colored

def filter_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def filter_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges)
    return cartoon

def filter_emboss(frame):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    emboss = cv2.filter2D(frame, -1, kernel)
    return emboss

def filter_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

# Initialize session state
if 'cam_running' not in st.session_state:
    st.session_state.cam_running = False
if 'last_pinch_time' not in st.session_state:
    st.session_state.last_pinch_time = 0
if 'camera_must_rerun' not in st.session_state:
    st.session_state.camera_must_rerun = False
if 'download_triggered' not in st.session_state:
    st.session_state.download_triggered = False
if 'current_filter' not in st.session_state:
    st.session_state.current_filter = 0
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'latest_capture' not in st.session_state:
    st.session_state.latest_capture = None

def create_zip_download():
    """Create a ZIP file containing all captured images"""
    if not st.session_state.captured_images:
        return None
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_data in enumerate(st.session_state.captured_images):
            zip_file.writestr(img_data['filename'], img_data['data'])
    
    zip_buffer.seek(0)
    return zip_buffer

def process_frame_with_gestures(img, mp_hands, mp_drawing, hands, filters, filter_names):
    """Processes a single frame for hand detection, filter application, and gesture control."""
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    current_filter_index = st.session_state.get('current_filter', 0)

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    left_hand_points = []
    right_hand_points = []
    pinch_detected = False
    pinch_threshold = 60
    
    hands_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 2

    output = img

    if hands_detected:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            
            pinch_dist = np.hypot(x_thumb - x_index, y_thumb - y_index)
            if pinch_dist < pinch_threshold:
                pinch_detected = True
            
            if label == 'Left':
                left_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
            elif label == 'Right':
                right_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
            
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if left_hand_points and right_hand_points:
            left_thumb, left_index = left_hand_points
            right_thumb, right_index = right_hand_points
            
            roi_points = [left_index, right_index, right_thumb, left_thumb]
            pts = np.array(roi_points, np.int32)
            
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            filtered = filters[current_filter_index](img)
            mask3 = cv2.merge([mask, mask, mask]) // 255
            output = filtered * mask3 + img * (1 - mask3)
            output = output.astype(np.uint8)
            
            cv2.putText(output, f"Filter: {filter_names[current_filter_index]}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            cv2.putText(output, f"Filter: {filter_names[current_filter_index]}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if pinch_detected:
                current_time = time.time()
                if current_time - st.session_state.last_pinch_time > 1.0:
                    new_filter_index = (current_filter_index + 1) % len(filters)
                    st.session_state.last_pinch_time = current_time
                    st.session_state.current_filter = new_filter_index
                    
                    # Capture and prepare image for download
                    success, encoded_image = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        timestamp = int(time.time() * 1000)
                        filename = f"gesture_capture_{timestamp}_{filter_names[new_filter_index].replace(' ', '_').replace('&', 'and')}.jpg"
                        
                        # Store the latest capture for download
                        st.session_state.latest_capture = {
                            'filename': filename,
                            'data': encoded_image.tobytes(),
                            'filter': filter_names[new_filter_index],
                            'timestamp': timestamp
                        }
                        
                        # Add to captured images log
                        st.session_state.captured_images.append({
                            'filename': filename,
                            'data': encoded_image.tobytes(),
                            'filter': filter_names[new_filter_index],
                            'timestamp': timestamp
                        })
                        
                        # Trigger download and rerun
                        st.session_state.download_triggered = True
                        st.session_state.camera_must_rerun = True
                    else:
                        cv2.putText(output, "CAPTURE FAILED!", (15, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return output
        
    cv2.putText(output, "Show both hands for gesture control", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, "Pinch thumb & index finger to change filter", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

def main():
    # Header with animated icons
    st.markdown("""
    <div class="main-header">
        <span class="animated-icon">👋</span>
        Hand Gesture Filter App
        <span class="animated-icon">🎨</span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize MediaPipe components
    mp_hands, mp_drawing, hands = initialize_mediapipe()
    filters = [filter_bw, filter_invert, filter_thermal, filter_depth, 
               filter_sepia, filter_cartoon, filter_emboss, filter_blur]
    filter_names = ['Black & White', 'Invert', 'Thermal', 'Depth', 
                    'Sepia', 'Cartoon', 'Emboss', 'Blur']

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <span class="pulse-icon">⚙️</span>
            <h2>Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        filter_names_full = ['🎭 Black & White', '🔄 Invert', '🌡️ Thermal', '🌊 Depth', 
                             '📸 Sepia', '🎨 Cartoon', '✨ Emboss', '🌫️ Blur']
        filter_icons = ['🎭', '🔄', '🌡️', '🌊', '📸', '🎨', '✨', '🌫️']
        
        st.markdown("### 🎨 Available Filters")
        for i, (name, icon) in enumerate(zip(filter_names_full, filter_icons)):
            if i == st.session_state.current_filter:
                st.markdown(f"""
                <div class="filter-card glow-effect">
                    <span class="rotate-icon">{icon}</span> {name} 
                    <br><small>🔥 ACTIVE</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"- {name}")
        
        st.markdown("---")
        
        # Live Stats
        st.markdown(f"""
        <div class="success-card">
            <span class="animated-icon">📊</span>
            <h4>Live Stats</h4>
            <p>📸 Images Captured: {len(st.session_state.captured_images)}</p>
            <p>🎨 Current Filter: {st.session_state.current_filter + 1}/{len(filter_names_full)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download Section
        st.markdown("""
        <div class="download-section">
            <span class="pulse-icon">💾</span>
            <h4>Download Manager</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual download button for latest capture
        if st.session_state.latest_capture:
            capture = st.session_state.latest_capture
            st.download_button(
                label="📥 Download Latest Capture",
                data=capture['data'],
                file_name=capture['filename'],
                mime="image/jpeg",
                use_container_width=True,
                key="download_latest"
            )
        
        # Download all as zip option
        if len(st.session_state.captured_images) > 0:
            zip_buffer = create_zip_download()
            if zip_buffer:
                st.download_button(
                    label=f"📦 Download All Images ({len(st.session_state.captured_images)} files)",
                    data=zip_buffer,
                    file_name=f"all_gesture_captures_{int(time.time())}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="download_all_zip"
                )
            
            # Individual download buttons for each capture
            st.markdown("---")
            st.markdown("### 📁 Individual Downloads")
            st.markdown(f"**Total captures:** {len(st.session_state.captured_images)}")
            
            # Show individual download buttons for recent captures
            for i, img_data in enumerate(reversed(st.session_state.captured_images[-10:])):  # Last 10 captures
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="capture-item">
                            <strong>{img_data['filename']}</strong><br>
                            <small>Filter: {img_data['filter']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.download_button(
                            label="📥",
                            data=img_data['data'],
                            file_name=img_data['filename'],
                            mime="image/jpeg",
                            key=f"download_{i}_{img_data['timestamp']}",
                            use_container_width=True
                        )
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        <div class="info-card">
            <span class="pulse-icon">📋</span>
            <h4>Instructions</h4>
            <p>1. Show both hands to camera</p>
            <p>2. Make pinching gesture to change filter</p>
            <p>3. Images auto-save to your downloads</p>
            <p>4. Use buttons below to download manually</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear button
        if st.button("🗑️ Clear Image Log", type="secondary", use_container_width=True):
            st.session_state.captured_images = []
            st.session_state.latest_capture = None
            st.session_state.last_pinch_time = 0
            st.session_state.download_triggered = False
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera Control
        control_col1, control_col2 = st.columns([1, 2])
        with control_col1:
            if st.session_state.cam_running:
                if st.button("🛑 STOP Camera", type="secondary", use_container_width=True):
                    st.session_state.cam_running = False
                    st.rerun()
            else:
                if st.button("▶️ START Camera", type="primary", use_container_width=True):
                    st.session_state.cam_running = True
                    st.rerun()
        
        video_placeholder = st.empty()
        
        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("🚨 Error: Could not open camera. Ensure your camera is connected and available.")
                st.session_state.cam_running = False
            else:
                # Set camera resolution for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                st.markdown("""
                <div class="success-card">
                    <span class="pulse-icon">📹</span> 
                    Camera active - Show both hands for gesture control
                </div>
                """, unsafe_allow_html=True)

                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.warning("⚠️ Cannot read frame from camera. Stopping feed.")
                        st.session_state.cam_running = False
                        break
                    
                    processed_frame = process_frame_with_gestures(
                        frame, mp_hands, mp_drawing, hands, filters, filter_names
                    )
                    
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    if st.session_state.camera_must_rerun:
                        st.session_state.camera_must_rerun = False
                        cap.release()
                        st.rerun()
                        break
                    
                    time.sleep(0.03)

                cap.release()
                st.session_state.cam_running = False
                st.rerun()
                
        else:
            video_placeholder.info("👆 Click 'START Camera' to begin hand gesture control")
            st.markdown("""
            <div class="info-card">
                <h4>💡 Pro Tip</h4>
                <p>Make sure to allow camera permissions in your browser when prompted.</p>
                <p>The download will automatically start when you change filters using hand gestures.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💾 Download Area")
        
        if st.session_state.latest_capture:
            latest = st.session_state.latest_capture
            st.markdown(f"""
            <div class="success-card">
                <span class="pulse-icon">✅</span>
                <h4>Latest Capture Ready!</h4>
                <p><strong>Filter:</strong> {latest['filter']}</p>
                <p><strong>Time:</strong> {time.strftime('%H:%M:%S', time.localtime(latest['timestamp']/1000))}</p>
                <p><strong>Total Captures:</strong> {len(st.session_state.captured_images)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the captured image
            st.image(latest['data'], caption=f"Latest: {latest['filter']}", use_container_width=True)
            
            # Auto-download using JavaScript (more reliable)
            if st.session_state.download_triggered:
                # Create download link
                b64_data = base64.b64encode(latest['data']).decode()
                href = f'<a href="data:image/jpeg;base64,{b64_data}" download="{latest["filename"]}" id="auto-download"></a>'
                js = """
                <script>
                    var link = document.getElementById('auto-download');
                    if (link) {
                        link.click();
                        // Remove the link after click
                        setTimeout(function() {
                            link.remove();
                        }, 100);
                    }
                </script>
                """
                st.markdown(href + js, unsafe_allow_html=True)
                st.session_state.download_triggered = False
                st.success(f"✅ Download started: {latest['filename']}")
        
        else:
            st.markdown("""
            <div class="info-card">
                <span class="pulse-icon">📸</span>
                <h4>No Captures Yet</h4>
                <p>Start the camera and use hand gestures to capture images!</p>
                <p>Images will appear here and download automatically.</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <span class="pulse-icon">🚀</span>
        <strong>Hand Gesture Filter App</strong> - Powered by Streamlit, MediaPipe & OpenCV
        <span class="pulse-icon">🚀</span>
        <br>
        <small>Made with ❤️ for interactive computer vision</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()