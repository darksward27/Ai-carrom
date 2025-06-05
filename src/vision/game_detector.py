"""
Game Detector using Computer Vision
Detects carrom board, pieces, striker, and game state
"""

import cv2
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional
import os


class GameDetector:
    """Computer vision for Carrom Pool game detection"""
    
    def __init__(self, detection_config: dict):
        """Initialize game detector with configuration"""
        self.config = detection_config
        self.board_template = None
        self.piece_templates = {}
        
        # Load templates if they exist
        self._load_templates()
        
        logger.info("Game Detector initialized")
    
    def _load_templates(self):
        """Load template images for matching"""
        try:
            # Load board template
            board_path = self.config.get('board_template_path')
            if board_path and os.path.exists(board_path):
                self.board_template = cv2.imread(board_path, cv2.IMREAD_COLOR)
                logger.info(f"Board template loaded: {board_path}")
            
            # Load piece templates
            piece_templates = self.config.get('piece_templates', {})
            for piece_type, template_path in piece_templates.items():
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    self.piece_templates[piece_type] = template
                    logger.info(f"Piece template loaded: {piece_type}")
                    
        except Exception as e:
            logger.warning(f"Could not load templates: {e}")
    
    def detect_board_state(self, screenshot: np.ndarray) -> Dict:
        """Detect complete board state from screenshot"""
        try:
            # Convert to working format
            image = screenshot.copy()
            
            # Detect board area
            board_region = self._detect_board_region(image)
            
            if board_region is None:
                logger.warning("Could not detect board region")
                return {}
            
            # Extract board area
            x1, y1, x2, y2 = board_region
            board_image = image[y1:y2, x1:x2]
            
            # Detect pieces
            pieces = self._detect_pieces(board_image, (x1, y1))
            
            # Detect striker
            striker = self._detect_striker(board_image, (x1, y1))
            
            # Detect pockets
            pockets = self._detect_pockets(board_image, (x1, y1))
            
            # Create board state
            board_state = {
                'board_region': board_region,
                'pieces': pieces,
                'striker': striker,
                'pockets': pockets,
                'board_size': (x2 - x1, y2 - y1)
            }
            
            logger.debug(f"Board state detected: {len(pieces)} pieces, striker: {striker is not None}")
            return board_state
            
        except Exception as e:
            logger.error(f"Error detecting board state: {e}")
            return {}
    
    def _detect_board_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the carrom board region in the image"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define brown color range for carrom board
            brown_lower, brown_upper = self.config['color_ranges']['board_brown']
            brown_lower = np.array(brown_lower)
            brown_upper = np.array(brown_upper)
            
            # Create mask for board color
            mask = cv2.inRange(hsv, brown_lower, brown_upper)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (should be the board)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Ensure minimum size
            if w < 200 or h < 200:
                return None
            
            return (x, y, x + w, y + h)
            
        except Exception as e:
            logger.error(f"Error detecting board region: {e}")
            return None
    
    def _detect_pieces(self, board_image: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """Detect carrom pieces on the board"""
        pieces = []
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(board_image, cv2.COLOR_RGB2HSV)
            
            # Detect white pieces
            white_pieces = self._detect_pieces_by_color(
                hsv, 'white_piece', offset
            )
            pieces.extend(white_pieces)
            
            # Detect black pieces
            black_pieces = self._detect_pieces_by_color(
                hsv, 'black_piece', offset
            )
            pieces.extend(black_pieces)
            
            # Detect queen piece
            queen_pieces = self._detect_pieces_by_color(
                hsv, 'queen_piece', offset
            )
            pieces.extend(queen_pieces)
            
            return pieces
            
        except Exception as e:
            logger.error(f"Error detecting pieces: {e}")
            return []
    
    def _detect_pieces_by_color(self, hsv_image: np.ndarray, 
                               piece_type: str, offset: Tuple[int, int]) -> List[Dict]:
        """Detect pieces of a specific color"""
        pieces = []
        
        try:
            # Get color range for piece type
            if piece_type not in self.config['color_ranges']:
                return pieces
            
            lower, upper = self.config['color_ranges'][piece_type]
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Create mask
            mask = cv2.inRange(hsv_image, lower, upper)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if (area < self.config['contour_area_min'] or 
                    area > self.config['contour_area_max']):
                    continue
                
                # Get center point
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"]) + offset[0]
                cy = int(M["m01"] / M["m00"]) + offset[1]
                
                # Calculate radius
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                piece = {
                    'type': piece_type.replace('_piece', ''),
                    'position': (cx, cy),
                    'radius': radius,
                    'area': area,
                    'contour': contour
                }
                
                pieces.append(piece)
            
            return pieces
            
        except Exception as e:
            logger.error(f"Error detecting {piece_type} pieces: {e}")
            return []
    
    def _detect_striker(self, board_image: np.ndarray, offset: Tuple[int, int]) -> Optional[Dict]:
        """Detect the striker on the board"""
        try:
            # Striker is typically larger and has different color characteristics
            # Use template matching if template is available
            if 'striker' in self.piece_templates:
                result = cv2.matchTemplate(
                    board_image, 
                    self.piece_templates['striker'], 
                    cv2.TM_CCOEFF_NORMED
                )
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > self.config['template_match_threshold']:
                    h, w = self.piece_templates['striker'].shape[:2]
                    center_x = max_loc[0] + w // 2 + offset[0]
                    center_y = max_loc[1] + h // 2 + offset[1]
                    
                    return {
                        'position': (center_x, center_y),
                        'confidence': max_val
                    }
            
            # Fallback: detect striker by size and position
            # (typically at the bottom of the board)
            return self._detect_striker_by_position(board_image, offset)
            
        except Exception as e:
            logger.error(f"Error detecting striker: {e}")
            return None
    
    def _detect_striker_by_position(self, board_image: np.ndarray, 
                                   offset: Tuple[int, int]) -> Optional[Dict]:
        """Detect striker by position (usually at bottom)"""
        try:
            h, w = board_image.shape[:2]
            
            # Look in bottom 20% of board
            bottom_region = board_image[int(h * 0.8):, :]
            
            # Convert to grayscale and detect circles
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
            
            # Use HoughCircles to detect circular objects
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                1,
                20,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=30
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Return the first detected circle as striker
                for (x, y, r) in circles:
                    return {
                        'position': (x + offset[0], y + int(h * 0.8) + offset[1]),
                        'radius': r,
                        'confidence': 0.7
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting striker by position: {e}")
            return None
    
    def _detect_pockets(self, board_image: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """Detect corner pockets on the board"""
        pockets = []
        
        try:
            h, w = board_image.shape[:2]
            
            # Pocket positions (corners)
            pocket_positions = [
                (int(w * 0.05), int(h * 0.05)),    # Top-left
                (int(w * 0.95), int(h * 0.05)),    # Top-right
                (int(w * 0.05), int(h * 0.95)),    # Bottom-left
                (int(w * 0.95), int(h * 0.95))     # Bottom-right
            ]
            
            for i, (px, py) in enumerate(pocket_positions):
                pocket = {
                    'id': i,
                    'position': (px + offset[0], py + offset[1]),
                    'radius': 20  # Approximate pocket radius
                }
                pockets.append(pocket)
            
            return pockets
            
        except Exception as e:
            logger.error(f"Error detecting pockets: {e}")
            return []
    
    def is_game_active(self, screenshot: np.ndarray) -> bool:
        """Check if the game is currently active"""
        try:
            # Detect if we can see the carrom board
            board_region = self._detect_board_region(screenshot)
            return board_region is not None
            
        except Exception as e:
            logger.error(f"Error checking if game is active: {e}")
            return False
    
    def is_game_finished(self, screenshot: np.ndarray) -> bool:
        """Check if the game has finished"""
        try:
            # Look for game over indicators
            # This could be specific UI elements or text
            # For now, implement a simple check
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Look for common game over text
            # This would need to be customized for the specific game UI
            return False  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking if game is finished: {e}")
            return False
    
    def is_our_turn(self, screenshot: np.ndarray) -> bool:
        """Check if it's our turn to play"""
        try:
            # Look for turn indicators in the UI
            # This could be highlighting, text, or other visual cues
            # For now, assume it's always our turn (placeholder)
            return True
            
        except Exception as e:
            logger.error(f"Error checking turn: {e}")
            return True
    
    def did_we_win(self, screenshot: np.ndarray) -> bool:
        """Check if we won the game"""
        try:
            # Look for victory indicators
            # This would need to be customized for the specific game UI
            return False  # Placeholder
            
        except Exception as e:
            logger.error(f"Error checking win condition: {e}")
            return False
    
    def detect_ui_elements(self, screenshot: np.ndarray) -> Dict:
        """Detect UI elements like buttons, menus, etc."""
        try:
            ui_elements = {}
            
            # This would detect specific UI elements
            # like play buttons, settings, etc.
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return {}
    
    def save_debug_image(self, image: np.ndarray, board_state: Dict, 
                        filename: str = "debug.png"):
        """Save debug image with detected elements highlighted"""
        try:
            debug_image = image.copy()
            
            # Draw board region
            if 'board_region' in board_state:
                x1, y1, x2, y2 = board_state['board_region']
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw pieces
            if 'pieces' in board_state:
                for piece in board_state['pieces']:
                    pos = piece['position']
                    radius = int(piece.get('radius', 10))
                    
                    # Color based on piece type
                    if piece['type'] == 'white':
                        color = (255, 255, 255)
                    elif piece['type'] == 'black':
                        color = (0, 0, 0)
                    elif piece['type'] == 'queen':
                        color = (255, 0, 0)
                    else:
                        color = (128, 128, 128)
                    
                    cv2.circle(debug_image, pos, radius, color, 2)
            
            # Draw striker
            if 'striker' in board_state and board_state['striker']:
                pos = board_state['striker']['position']
                radius = int(board_state['striker'].get('radius', 15))
                cv2.circle(debug_image, pos, radius, (0, 255, 255), 3)
            
            # Draw pockets
            if 'pockets' in board_state:
                for pocket in board_state['pockets']:
                    pos = pocket['position']
                    radius = pocket['radius']
                    cv2.circle(debug_image, pos, radius, (255, 0, 255), 2)
            
            # Save image
            cv2.imwrite(filename, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            logger.debug(f"Debug image saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving debug image: {e}") 