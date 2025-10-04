import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import base64 # Added for automatic download functionality
# Removed: from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# Removed: import av
import os 
# Removed: import threading # We will run the loop directly in the Streamlit thread

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Filter App",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling (kept for aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif';
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
    
    .floating-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 3rem;
        animation: bounce 2s infinite;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Hands
@st.cache_resource
def initialize_mediapipe():
    # Cache the MediaPipe setup to avoid re-initializing on every frame/rerun
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

# --- Filter Functions (Kept as is) ---
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

# Store persistent state for camera logic 
if 'cam_running' not in st.session_state:
    st.session_state.cam_running = False
if 'last_pinch_time' not in st.session_state:
    st.session_state.last_pinch_time = 0
if 'camera_must_rerun' not in st.session_state:
    st.session_state.camera_must_rerun = False
if 'download_data_uri' not in st.session_state:
    st.session_state.download_data_uri = None # Stores {filename, uri} for auto-download

# The 'captured_images' list now stores dictionaries: [{'filename': str, 'data': bytes}]

def process_frame_with_gestures(img, mp_hands, mp_drawing, hands, filters, filter_names):
    """Processes a single frame for hand detection, filter application, and gesture control."""
    img = cv2.flip(img, 1) # Flip the image for a mirror view
    h, w, _ = img.shape
    
    # Safely get current filter index
    current_filter_index = st.session_state.get('current_filter', 0)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    left_hand_points = []
    right_hand_points = []
    pinch_detected = False
    pinch_threshold = 60
    
    hands_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 2

    output = img # Start with the original image
    
    if hands_detected:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            
            # Pinch detection
            pinch_dist = np.hypot(x_thumb - x_index, y_thumb - y_index)
            if pinch_dist < pinch_threshold:
                pinch_detected = True
            
            if label == 'Left':
                left_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
            elif label == 'Right':
                right_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Apply filter only if both hands are detected and points collected
        if left_hand_points and right_hand_points:
            left_thumb, left_index = left_hand_points
            right_thumb, right_index = right_hand_points
            
            roi_points = [left_index, right_index, right_thumb, left_thumb]
            pts = np.array(roi_points, np.int32)
            
            # Draw ROI boundary
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Apply current filter
            filtered = filters[current_filter_index](img)
            mask3 = cv2.merge([mask, mask, mask]) // 255
            output = filtered * mask3 + img * (1 - mask3)
            output = output.astype(np.uint8)
            
            # Show filter name
            cv2.putText(output, f"Filter: {filter_names[current_filter_index]}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            cv2.putText(output, f"Filter: {filter_names[current_filter_index]}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Pinch indicator and filter change/save logic
            if pinch_detected:
                
                current_time = time.time()
                if current_time - st.session_state.last_pinch_time > 1.0: # 1 second debounce
                    # Cycle filter index
                    new_filter_index = (current_filter_index + 1) % len(filters)
                    st.session_state.last_pinch_time = current_time
                    st.session_state.current_filter = new_filter_index
                    
                    # --- AUTO-DOWNLOAD LOGIC ---
                    
                    # Encode frame to JPG bytes
                    success, encoded_image = cv2.imencode('.jpg', output)
                    
                    if success:
                        timestamp = int(time.time() * 1000)
                        filename = f"capture_{timestamp}_{filter_names[new_filter_index].replace(' ', '_').replace('&', 'and')}.jpg"
                        
                        # Convert bytes to base64 string
                        base64_data = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                        
                        # Store image data in the persistent log (for sidebar view)
                        st.session_state.captured_images.append({
                            'filename': filename,
                            'data': encoded_image.tobytes() # Keep bytes for st.download_button fallback/log
                        }) 
                        
                        # Store the data URI for immediate, automatic download on the next rerun
                        st.session_state.download_data_uri = {
                            'filename': filename,
                            'uri': f"data:image/jpeg;base64,{base64_data}"
                        }
                        
                        # Signal the main loop to perform cleanup and rerun
                        st.session_state.camera_must_rerun = True
                        
                    else:
                        cv2.putText(output, "CAPTURE FAILED!", (15, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return output
        
    # If no hands or incomplete detection, return original image with instructions
    cv2.putText(output, "Show both hands for gesture control", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, "Pinch thumb & index finger to change filter", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

# Main app function
def main():
    # Header with animated icons
    st.markdown("""
    <div class="main-header">
        <span class="animated-icon">👋</span>
        Hand Gesture Filter App
        <span class="animated-icon">🎨</span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state (Check done in global scope for robustness)
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = 0
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
    if 'camera_must_rerun' not in st.session_state:
        st.session_state.camera_must_rerun = False
    if 'download_data_uri' not in st.session_state:
        st.session_state.download_data_uri = None

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
        
        # Instructions
        st.markdown("""
        <div class="info-card">
            <span class="pulse-icon">📋</span>
            <h4>Instructions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear button
        if st.button("🗑️ Clear Image Log", type="secondary"):
            st.session_state.captured_images = []
            st.session_state.last_pinch_time = 0 # Reset timer
            st.session_state.download_data_uri = None # Clear any pending download
            st.rerun()
            
        # --- Capture Log ---
        if st.session_state.captured_images:
            st.markdown("---")
            st.markdown("### Capture Log")
            for item in st.session_state.captured_images[-3:]:
                st.markdown(f"📄 `{item['filename']}`")
        
    # --- AUTOMATIC DOWNLOAD TRIGGER BLOCK ---
    if st.session_state.download_data_uri:
        download_info = st.session_state.download_data_uri
        
        # 1. Render a hidden anchor tag with data URI
        download_html = f"""
        <a href="{download_info['uri']}" download="{download_info['filename']}" id="auto-download-link" style="display: none;"></a>
        
        <!-- 2. Use JavaScript to immediately 'click' the link -->
        <script>
            document.getElementById('auto-download-link').click();
        </script>
        """
        st.markdown(download_html, unsafe_allow_html=True)
        
        # 3. Show confirmation and clear flag for next capture
        st.success(f"🎉 **AUTO-DOWNLOADED**: '{download_info['filename']}'")
        st.session_state.download_data_uri = None


    # Main content
    st.markdown("### 📹 Live Camera Feed (OpenCV Capture)")
    
    # Camera Start/Stop Control
    if st.session_state.cam_running:
        stop_button = st.button("🛑 STOP Camera", type="secondary")
        if stop_button:
            st.session_state.cam_running = False
            st.rerun() # Rerun to exit the loop
    else:
        start_button = st.button("▶️ START Camera", type="primary")
        if start_button:
            st.session_state.cam_running = True
            st.rerun() # Rerun to enter the loop
    
    
    video_placeholder = st.empty()
    
    if st.session_state.cam_running:
        # Use 0 for default camera
        cap = cv2.VideoCapture(0) 
        
        if not cap.isOpened():
            st.error("🚨 Error: Could not open camera. Ensure your camera is connected and available to OpenCV (index 0).")
            st.session_state.cam_running = False
            
        else:
            st.markdown("""
            <div class="success-card">
                <span class="rotate-icon">📹</span> 
                Camera is LIVE! Show your hands for gesture control
            </div>
            """, unsafe_allow_html=True)

            while st.session_state.cam_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("⚠️ Cannot read frame from camera. Stopping feed.")
                    st.session_state.cam_running = False
                    break
                
                # Process the frame
                processed_frame = process_frame_with_gestures(
                    frame, mp_hands, mp_drawing, hands, filters, filter_names
                )
                
                # Display the processed frame (BGR to RGB conversion for Streamlit)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # --- RERUN LOGIC: Check for RERUN signal from pinch gesture ---
                if st.session_state.camera_must_rerun:
                    st.session_state.camera_must_rerun = False # Reset flag
                    cap.release() # CRITICAL: Release the camera resource immediately
                    st.rerun() # Stop current script and restart cleanly
                    break # Exit the while loop
                # -----------------------------------------------------------
                
                # Important: Use a small sleep to prevent the loop from consuming 100% CPU
                # and allow the app to be responsive to the STOP button.
                time.sleep(0.01)

            cap.release()
            st.session_state.cam_running = False # Ensure flag is set to False after loop exits
            st.rerun() 
            
    else:
        # Display a placeholder when stopped
        video_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), 
                                 caption="Camera Feed Stopped", 
                                 use_container_width=True) 

    # Removed the entire col2 content block
    # 
    # Floating effect
    st.markdown("""
    <div class="floating-icon">
        ✨
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
