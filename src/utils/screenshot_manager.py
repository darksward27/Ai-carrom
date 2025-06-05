"""
Screenshot Manager Utility
Handles screenshot saving, loading, and management
"""

import os
import cv2
import numpy as np
from datetime import datetime
from loguru import logger
from typing import Optional
from PIL import Image


class ScreenshotManager:
    """Manages screenshots for debugging and training data"""
    
    def __init__(self, base_dir: str = "screenshots"):
        """Initialize screenshot manager"""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Create subdirectories
        self.training_dir = os.path.join(base_dir, "training")
        self.debug_dir = os.path.join(base_dir, "debug")
        self.gameplay_dir = os.path.join(base_dir, "gameplay")
        
        for directory in [self.training_dir, self.debug_dir, self.gameplay_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Screenshot Manager initialized: {base_dir}")
    
    def save_screenshot(self, screenshot: np.ndarray, filename: str, 
                       category: str = "debug") -> str:
        """Save screenshot to appropriate directory"""
        try:
            # Determine target directory
            if category == "training":
                target_dir = self.training_dir
            elif category == "gameplay":
                target_dir = self.gameplay_dir
            else:
                target_dir = self.debug_dir
            
            # Ensure filename has extension
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            # Full path
            filepath = os.path.join(target_dir, filename)
            
            # Convert RGB to BGR for OpenCV
            if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
                screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            else:
                screenshot_bgr = screenshot
            
            # Save image
            cv2.imwrite(filepath, screenshot_bgr)
            
            logger.debug(f"Screenshot saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return ""
    
    def save_with_timestamp(self, screenshot: np.ndarray, 
                           prefix: str = "screenshot", 
                           category: str = "debug") -> str:
        """Save screenshot with timestamp filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}.png"
        return self.save_screenshot(screenshot, filename, category)
    
    def load_screenshot(self, filepath: str) -> Optional[np.ndarray]:
        """Load screenshot from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Screenshot file not found: {filepath}")
                return None
            
            # Load image
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Could not load image: {filepath}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Screenshot loaded: {filepath}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error loading screenshot: {e}")
            return None
    
    def save_annotated_screenshot(self, screenshot: np.ndarray, 
                                 annotations: dict, 
                                 filename: str = None) -> str:
        """Save screenshot with annotations (bounding boxes, labels, etc.)"""
        try:
            annotated = screenshot.copy()
            
            # Draw board region
            if 'board_region' in annotations:
                x1, y1, x2, y2 = annotations['board_region']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, "Board", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw pieces
            if 'pieces' in annotations:
                for i, piece in enumerate(annotations['pieces']):
                    pos = piece['position']
                    radius = int(piece.get('radius', 10))
                    piece_type = piece.get('type', 'unknown')
                    
                    # Color based on piece type
                    if piece_type == 'white':
                        color = (255, 255, 255)
                    elif piece_type == 'black':
                        color = (0, 0, 0)
                    elif piece_type == 'queen':
                        color = (255, 0, 0)
                    else:
                        color = (128, 128, 128)
                    
                    cv2.circle(annotated, pos, radius, color, 2)
                    cv2.putText(annotated, piece_type, 
                               (pos[0]-10, pos[1]-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw striker
            if 'striker' in annotations and annotations['striker']:
                pos = annotations['striker']['position']
                radius = int(annotations['striker'].get('radius', 15))
                cv2.circle(annotated, pos, radius, (0, 255, 255), 3)
                cv2.putText(annotated, "Striker", (pos[0]-15, pos[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw pockets
            if 'pockets' in annotations:
                for pocket in annotations['pockets']:
                    pos = pocket['position']
                    radius = pocket['radius']
                    cv2.circle(annotated, pos, radius, (255, 0, 255), 2)
            
            # Save annotated image
            if filename is None:
                filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            return self.save_screenshot(annotated, filename, "debug")
            
        except Exception as e:
            logger.error(f"Error saving annotated screenshot: {e}")
            return ""
    
    def cleanup_old_screenshots(self, days_to_keep: int = 7):
        """Clean up old screenshots to save disk space"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            removed_count = 0
            
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(root, file)
                        
                        # Check file age
                        if os.path.getmtime(filepath) < cutoff_time:
                            os.remove(filepath)
                            removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old screenshots")
            
        except Exception as e:
            logger.error(f"Error during screenshot cleanup: {e}")
    
    def get_screenshot_stats(self) -> dict:
        """Get statistics about stored screenshots"""
        try:
            stats = {
                'total_files': 0,
                'total_size_mb': 0,
                'by_category': {}
            }
            
            for category, directory in [
                ('training', self.training_dir),
                ('debug', self.debug_dir),
                ('gameplay', self.gameplay_dir)
            ]:
                file_count = 0
                total_size = 0
                
                for file in os.listdir(directory):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join(directory, file)
                        file_count += 1
                        total_size += os.path.getsize(filepath)
                
                stats['by_category'][category] = {
                    'files': file_count,
                    'size_mb': total_size / (1024 * 1024)
                }
                
                stats['total_files'] += file_count
                stats['total_size_mb'] += total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting screenshot stats: {e}")
            return {}
    
    def create_training_sequence(self, screenshots: list, 
                                actions: list, filename: str = None) -> str:
        """Create a sequence of screenshots showing action progression"""
        try:
            if len(screenshots) != len(actions):
                raise ValueError("Screenshots and actions lists must have same length")
            
            if not screenshots:
                return ""
            
            # Create a grid layout
            rows = int(np.ceil(len(screenshots) / 3))
            cols = min(3, len(screenshots))
            
            # Get dimensions from first screenshot
            h, w = screenshots[0].shape[:2]
            
            # Create combined image
            combined_h = rows * h
            combined_w = cols * w
            combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            
            for i, (screenshot, action) in enumerate(zip(screenshots, actions)):
                row = i // cols
                col = i % cols
                
                y1 = row * h
                y2 = y1 + h
                x1 = col * w
                x2 = x1 + w
                
                # Resize screenshot if necessary
                if screenshot.shape[:2] != (h, w):
                    screenshot = cv2.resize(screenshot, (w, h))
                
                combined[y1:y2, x1:x2] = screenshot
                
                # Add action label
                action_text = f"Action {i+1}: {action}"
                cv2.putText(combined, action_text, (x1+10, y1+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save sequence
            if filename is None:
                filename = f"training_sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            return self.save_screenshot(combined, filename, "training")
            
        except Exception as e:
            logger.error(f"Error creating training sequence: {e}")
            return "" 