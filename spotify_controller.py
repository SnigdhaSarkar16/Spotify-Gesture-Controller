import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import numpy as np
import math
import time
from io import BytesIO
from PIL import Image
import threading
import json
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

class SpotifyGestureController:
    def __init__(self):
        # Spotify API setup
        self.setup_spotify()
        # ... (other initializations)
        self.spotify_lock = threading.Lock()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Application state
        self.current_track = None
        self.album_cover = None
        self.last_volume = 50
        self.last_progress_update = 0
        self.progress_start_time = 0
        self.cached_progress = 0
        self.is_playing = False
        
        # Volume control parameters
        self.min_distance = 20
        self.max_distance = 200
        self.volume_smoothing = 0.3  # For smooth volume changes
        
        # UI parameters
        self.album_size = 120
        self.overlay_alpha = 0.8
        
        # rotation / vinyl animation
        self.rotation_angle = 0.0
        self.vinyl_diameter = 220  # diameter in pixels for the vinyl disc
        self.last_gesture_time = 0
        
        # Load button images
        self.play_img = self.load_button_image("play.png")
        self.pause_img = self.load_button_image("pause.png")
        self.next_img = self.load_button_image("next.png")
        self.prev_img = self.load_button_image("previous.png")
        self.replay_img = self.load_button_image("replay.png")

    def run_spotify_action_threaded(self, action, *args):
        def thread_target():
            with self.spotify_lock:
                try:
                    if action == "play":
                        self.sp.start_playback()
                    elif action == "pause":
                        self.sp.pause_playback()
                    elif action == "next":
                        self.sp.next_track()
                    elif action == "previous":
                        self.sp.previous_track()
                    elif action == "replay":
                        self.sp.seek_track(0)
                except Exception as e:
                    print(f"❌ Error in threaded Spotify action: {e}")

        # Start a new thread for the action
        thread = threading.Thread(target=thread_target)
        thread.start()

    def load_button_image(self, path, size=(50, 50)):
        """Load and resize an image from a file path."""
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and not img.size == 0:
                img = cv2.resize(img, size)
                return img
            else:
                print(f"Error loading image: {path}")
                return None
        except Exception as e:
            print(f"Exception loading image {path}: {e}")
            return None

    def overlay_image(self, background, overlay, x, y):
        """Overlay a smaller image (with alpha) onto a larger background image."""
        if overlay is None or background is None:
            return

        h, w = overlay.shape[:2]
        
        # Check for transparent channel
        if overlay.shape[2] == 4:
            # Image has alpha channel
            overlay_bgr = overlay[:, :, 0:3]
            overlay_alpha = overlay[:, :, 3] / 255.0
            
            # Ensure coordinates are within bounds
            y1, y2 = y, y + h
            x1, x2 = x, x + w
            
            # Check if ROI is valid
            if y1 >= 0 and y2 <= background.shape[0] and x1 >= 0 and x2 <= background.shape[1]:
                roi = background[y1:y2, x1:x2]
                
                for c in range(0, 3):
                    background[y1:y2, x1:x2, c] = (roi[:, :, c] * (1.0 - overlay_alpha) +
                                                    overlay_bgr[:, :, c] * overlay_alpha)
                    
    def setup_spotify(self):
        """Setup Spotify API authentication"""
        try:
               # Required scopes
            scope = "user-read-currently-playing user-read-playback-state user-modify-playback-state"
            
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=".spotify_cache"
            ))
            
            print("✓ Spotify API connected successfully!")
            
        except Exception as e:
            print(f"✗ Spotify setup failed: {e}")
            print("Please ensure you have valid Spotify credentials and the required scopes.")
            raise
    
    def fetch_current_track(self):
        """Fetch currently playing track from Spotify"""
        try:
            current = self.sp.current_playback()
            if current and current['is_playing']:
                track = current['item']
                if track:
                    self.current_track = {
                        'name': track['name'],
                        'artist': ', '.join([artist['name'] for artist in track['artists']]),
                        'album_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'duration_ms': track['duration_ms'],
                        'progress_ms': current['progress_ms'],
                        'is_playing': current['is_playing']
                    }
                    self.is_playing = current['is_playing']
                    
                    # Update progress tracking
                    if self.is_playing:
                        self.progress_start_time = time.time()
                        self.cached_progress = current['progress_ms']
                    
                    # Download album cover if changed
                    if self.current_track['album_url']:
                        self.download_album_cover(self.current_track['album_url'])
                    
                    return True
            else:
                self.current_track = None
                self.is_playing = False
                return False
                
        except Exception as e:
            print(f"Error fetching current track: {e}")
            return False
    
    def download_album_cover(self, url):
        """Download and process album cover"""
        try:
            if url:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    # Convert to OpenCV format
                    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
                    target = max(self.vinyl_diameter, 320)
                    pil_image = pil_image.resize((target, target))
                    self.album_cover = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error downloading album cover: {e}")
            self.album_cover = None
            
    def make_vinyl_from_cover(self, diameter):
        """Return a circular vinyl numpy image (BGR) sized diameter x diameter from self.album_cover.
        Adds a small black center hole to look like a vinyl."""
        if self.album_cover is None:
            return None

        # Resize album cover to square diameter x diameter
        cover = cv2.resize(self.album_cover, (diameter, diameter))

        # Create a blank image with a circular mask
        vinyl = np.zeros((diameter, diameter, 3), dtype=np.uint8)
        center = (diameter // 2, diameter // 2)
        radius = diameter // 2

        # Draw the album cover onto the blank image, with a circle as a mask
        cv2.circle(vinyl, center, radius, (255, 255, 255), -1)
        vinyl = cv2.bitwise_and(cover, vinyl)

        # Add center hole
        cv2.circle(vinyl, center, int(diameter * 0.06), (0, 0, 0), -1)

        return vinyl

    def rotate_vinyl(self, vinyl_img, angle):
        """Rotate the vinyl (BGR numpy) by angle degrees around center."""
        if vinyl_img is None:
            return None
        h, w = vinyl_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(vinyl_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated

    def draw_cd_disc(self, overlay, center, diameter, angle):
        """Draw a rotating vinyl (circular album art) onto overlay at given center (x,y)."""
        vinyl = self.make_vinyl_from_cover(diameter)
        if vinyl is None:
            return

        rotated = self.rotate_vinyl(vinyl, angle)

        x, y = center
        r = diameter // 2
        y1, y2 = y - r, y + r
        x1, x2 = x - r, x + r

        # Bounds check
        H, W = overlay.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return

        # Prepare mask where rotated circle is non-zero
        mask_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask_bin)

        # Place rotated vinyl onto overlay using masks
        bg_roi = cv2.bitwise_and(overlay[y1:y2, x1:x2], overlay[y1:y2, x1:x2], mask=inv_mask)
        fg_roi = cv2.bitwise_and(rotated, rotated, mask=mask_bin)
        overlay[y1:y2, x1:x2] = cv2.add(bg_roi, fg_roi)

    def draw_controls(self, frame, center_x, top_y):
        """Draw controls using PNG images."""
        button_size = 50
        gap = 70
        
        # Define button positions
        prev_pt = (int(center_x - gap - button_size/2), int(top_y - button_size/2))
        play_pt = (int(center_x - button_size/2), int(top_y - button_size/2))
        next_pt = (int(center_x + gap - button_size/2), int(top_y - button_size/2))
        replay_pt = (int(center_x + gap * 2 - button_size/2), int(top_y - button_size/2))
        
        # Draw buttons on the frame
        if self.prev_img is not None:
            self.overlay_image(frame, self.prev_img, prev_pt[0], prev_pt[1])
        if self.is_playing:
            if self.pause_img is not None:
                self.overlay_image(frame, self.pause_img, play_pt[0], play_pt[1])
        else:
            if self.play_img is not None:
                self.overlay_image(frame, self.play_img, play_pt[0], play_pt[1])
        if self.next_img is not None:
            self.overlay_image(frame, self.next_img, next_pt[0], next_pt[1])
        if self.replay_img is not None:
            self.overlay_image(frame, self.replay_img, replay_pt[0], replay_pt[1])

    def draw_volume_bar(self, overlay, volume):
        """Draw vertical volume bar at right edge of overlay."""
        H, W = overlay.shape[:2]
        bar_w = 24
        bar_h = int(H * 0.6)
        x1 = W - 60
        y1 = (H - bar_h) // 2
        x2 = x1 + bar_w
        y2 = y1 + bar_h
        # background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40,40,40), -1)
        # filled
        filled_h = int((volume / 100) * bar_h)
        cv2.rectangle(overlay, (x1, y2 - filled_h), (x2, y2), (29,185,84), -1)
        # border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (200,200,200), 1)
        # percentage text
        cv2.putText(overlay, f"{int(volume)}%", (x1-50, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    def is_finger_extended(self, hand_landmarks, finger_tip_id, finger_mcp_id):
        """Helper to determine if a finger is extended based on y-coordinates relative to the MCP joint."""
        tip_y = hand_landmarks.landmark[finger_tip_id].y
        mcp_y = hand_landmarks.landmark[finger_mcp_id].y
        return tip_y < mcp_y

    def get_hand_landmarks(self, frame):
        """Detect hand landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def map_distance_to_volume(self, distance, frame_width):
        """Map finger distance to volume (0-100)"""
        # Convert normalized distance to pixel distance
        pixel_distance = distance * frame_width
        
        # Clamp distance to min/max range
        pixel_distance = max(self.min_distance, min(self.max_distance, pixel_distance))
        
        # Map to volume (0-100)
        volume = int(((pixel_distance - self.min_distance) / (self.max_distance - self.min_distance)) * 100)
        return volume
    
    def update_spotify_volume(self, volume):
        """Update Spotify volume with smoothing"""
        try:
            # Apply smoothing to reduce API calls
            smooth_volume = int(self.last_volume * (1 - self.volume_smoothing) + volume * self.volume_smoothing)
            
            # Only update if volume changed significantly
            if abs(smooth_volume - self.last_volume) > 2:
                self.sp.volume(smooth_volume)
                self.last_volume = smooth_volume
                
        except Exception as e:
            print(f"Error updating volume: {e}")

    def draw_progress_bar(self, frame, x, y, width, height):
        """Draw animated progress bar"""
        if not self.current_track:
            return
        
        try:
            # Calculate current progress
            if self.is_playing:
                elapsed_since_update = (time.time() - self.progress_start_time) * 1000
                current_progress = self.cached_progress + elapsed_since_update
            else:
                current_progress = self.cached_progress
            
            duration = self.current_track['duration_ms']
            progress_ratio = min(current_progress / duration, 1.0)
            
            # Draw background bar
            cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
            
            # Draw progress bar
            progress_width = int(width * progress_ratio)
            if progress_width > 0:
                cv2.rectangle(frame, (x, y), (x + progress_width, y + height), (29, 185, 84), -1)
            
            # Draw border
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
            
            # Format and display time
            current_time = self.format_time(int(current_progress / 1000))
            total_time = self.format_time(int(duration / 1000))
            time_text = f"{current_time} / {total_time}"
            
            cv2.putText(frame, time_text, (x, y + height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"Error drawing progress bar: {e}")
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def draw_overlay(self, frame, volume=None, hand_detected=False):
        """Draw the new Spotify-like UI directly onto the frame without a dark overlay."""
        H, W = frame.shape[:2]

        # Update rotation only while playing
        if getattr(self, "rotation_angle", None) is None:
            self.rotation_angle = 0.0
        if self.is_playing:
            # speed depends on playback (constant here)
            self.rotation_angle = (self.rotation_angle + 2.0) % 360

        # Draw the main UI container outline
        container_w = W - 100
        container_h = H - 100
        container_x1 = (W - container_w) // 2
        container_y1 = (H - container_h) // 2
        container_x2 = container_x1 + container_w
        container_y2 = container_y1 + container_h
        
        # Draw small square album art inside the vinyl circle
        vinyl_center = (container_x1 + 100, container_y1 + container_h - 100) # positioned in bottom-left
        self.draw_cd_disc(frame, vinyl_center, self.vinyl_diameter, self.rotation_angle)

        # Draw the progress bar in the center
        progress_y = container_y1 + container_h - 100
        progress_x = container_x1 + 250
        progress_w = container_w - 550
        progress_h = 10
        self.draw_progress_bar(frame, progress_x, progress_y, progress_w, progress_h)

        # Draw the controls
        controls_y = progress_y - 45
        self.draw_controls(frame, W // 2, controls_y)

        # Draw the volume bar on the right side of the container
        if volume is not None:
            bar_w = 24
            bar_h = int(container_h * 0.6)
            x1 = container_x2 - 60
            y1 = container_y1 + (container_h - bar_h) // 2
            x2 = x1 + bar_w
            y2 = y1 + bar_h
            # Draw background and filled part with white borders
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40,40,40), -1)
            filled_h = int((volume / 100) * bar_h)
            cv2.rectangle(frame, (x1, y2 - filled_h), (x2, y2), (29,185,84), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200,200,200), 1)
            # Add percentage text
            cv2.putText(frame, f"{int(volume)}%", (x1-50, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def draw_gesture_guide(self, frame):
        """Draw a more visually appealing cheat sheet of hand gestures on screen."""
        height, width = frame.shape[:2]
        
        # Create a semi-transparent overlay with a smaller size
        overlay = frame.copy()
        
        # New, smaller dimensions for the box (e.g., 30% width, 40% height)
        box_width = int(width * 0.3)
        box_height = int(height * 0.4)
        cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
        
        # Blend overlay with frame to create a semi-transparent background
        alpha = 0.5  # Controls the transparency (0.0 to 1.0)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Title
        cv2.putText(frame, "Gesture Controls:", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (29, 185, 84), 2)
        
        # Gesture instructions
        gestures = [
            "Open Palm      -> Play",
            "Fist           -> Pause",
            "Index + Middle -> Next Track",
            "Index + Pinky  -> Previous Track",
            "Thumbs Down    -> Replay",
            "Thumb + Pinky  -> Volume"
        ]
        
        y_offset = 50
        for g in gestures:
            cv2.putText(frame, g, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
    def process_hand_gestures(self, frame, results):
        """Process hand landmarks for both volume control and playback gestures."""
        hand_detected = False
        current_volume = self.last_volume

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_detected = True

                # Determine extended fingers for gesture recognition
                thumb_extended = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
                index_extended = self.is_finger_extended(hand_landmarks, 8, 5)
                middle_extended = self.is_finger_extended(hand_landmarks, 12, 9)
                ring_extended = self.is_finger_extended(hand_landmarks, 16, 13)
                pinky_extended = self.is_finger_extended(hand_landmarks, 20, 17)

                extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

                # --- GESTURE RECOGNITION (with cooldown) ---
                current_time = time.time()
                cooldown_active = current_time - self.last_gesture_time < 1.0

                gesture_triggered = False

                if not cooldown_active:
                    # PLAY (open palm: all fingers extended)
                    if all(extended):
                        self.run_spotify_action_threaded("play")
                        print("▶️ Play")
                        self.last_gesture_time = current_time
                        gesture_triggered = True
                    
                    # PAUSE (fist: all fingers folded)
                    elif not any(extended):
                        self.run_spotify_action_threaded("pause")
                        print("⏸ Pause")
                        self.last_gesture_time = current_time
                        gesture_triggered = True

                    # NEXT TRACK (index + middle fingers extended)
                    elif index_extended and middle_extended and not thumb_extended and not ring_extended and not pinky_extended:
                        self.run_spotify_action_threaded("next")
                        print("⏭️ Next Track")
                        self.last_gesture_time = current_time
                        gesture_triggered = True
                        
                    # PREVIOUS TRACK (index + pinky fingers extended)
                    elif index_extended and pinky_extended and not thumb_extended and not middle_extended and not ring_extended:
                        self.run_spotify_action_threaded("previous")
                        print("⏮️ Previous Track")
                        self.last_gesture_time = current_time
                        gesture_triggered = True

                    # REPLAY (thumbs down)
                    elif hand_landmarks.landmark[4].y > hand_landmarks.landmark[3].y and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                        self.run_spotify_action_threaded("replay")
                        print("🔁 Replay")
                        self.last_gesture_time = current_time
                        gesture_triggered = True
                
                # --- VOLUME CONTROL ---
                # Check for the specific volume gesture (thumb and pinky extended)
                volume_gesture = thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended
                
                if volume_gesture:
                    thumb_tip = hand_landmarks.landmark[4]
                    pinky_tip = hand_landmarks.landmark[20]

                    # Use distance between thumb and pinky for volume
                    distance = self.calculate_distance(thumb_tip, pinky_tip)
                    volume = self.map_distance_to_volume(distance, frame.shape[1])
                    current_volume = volume
                    self.update_spotify_volume(volume)
                    
                    # Draw visual feedback for volume control
                    h, w = frame.shape[:2]
                    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    pinky_pos = (int(pinky_tip.x * w), int(pinky_tip.y * h))
                    cv2.line(frame, thumb_pos, pinky_pos, (0, 255, 0), 2)
                    cv2.circle(frame, thumb_pos, 8, (255, 0, 0), -1)
                    cv2.circle(frame, pinky_pos, 8, (255, 0, 0), -1)
                else:
                    # If volume gesture is not active, return the last known volume
                    current_volume = self.last_volume

        return hand_detected, current_volume

    def run(self):
        """Main application loop with optimized frame processing."""
        print("🎵 Starting Spotify Gesture Controller...")
        print("👋 Show your hand to control volume with thumb-index distance")
        print("📱 Make sure Spotify is playing music")
        print("🔄 Fetching current track...")
        
        # Initial track fetch
        self.fetch_current_track()
        
        last_track_update = 0
        track_update_interval = 2.0
        frame_skip = 2  # Process only every 2nd frame (adjust as needed)
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Update track info periodically
                current_time = time.time()
                if current_time - last_track_update > track_update_interval:
                    self.fetch_current_track()
                    last_track_update = current_time

                # Process hand gestures only on a subset of frames
                if frame_count % frame_skip == 0:
                    results = self.get_hand_landmarks(frame)
                    hand_detected, current_volume = self.process_hand_gestures(frame, results)
                else:
                    # Continue drawing with last known data
                    hand_detected = False
                    current_volume = self.last_volume

                # Draw overlay
                self.draw_overlay(frame, current_volume, hand_detected)
                self.draw_gesture_guide(frame)
                
                # Show frame
                cv2.imshow('Spotify Gesture Controller', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("✅ Cleanup complete")

def main():
    """Main entry point"""
    print("🎵 Spotify Gesture Controller")
    print("=" * 40)
    
    try:
        controller = SpotifyGestureController()
        controller.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\n📋 Setup checklist:")
        print("1. Install required packages: pip install opencv-python mediapipe spotipy requests numpy pillow")
        print("2. Create Spotify app at https://developer.spotify.com/dashboard")
        print("3. Update SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the code")
        print("4. Make sure Spotify is running and playing music")
        print("5. Ensure your webcam is connected and working")

if __name__ == "__main__":
    main()