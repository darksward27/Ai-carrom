"""
Board Classifier using scikit-learn
Classifies board state and identifies piece positions
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from loguru import logger
import cv2
from typing import Dict, List, Tuple
import os


class BoardClassifier:
    """ML-based board state classifier for Carrom Pool"""
    
    def __init__(self, config: dict):
        """Initialize board classifier"""
        self.config = config
        self.input_shape = tuple(config['input_shape'])
        self.num_classes = config['num_classes']
        
        # Build the classifier model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Training state
        self.is_trained = False
        
        logger.info("Board Classifier initialized with Random Forest")
    
    def classify_board_region(self, board_image: np.ndarray, 
                             grid_size: int = 16) -> np.ndarray:
        """Classify board into grid of piece types"""
        try:
            # Resize image to input shape
            resized = cv2.resize(board_image, self.input_shape[:2])
            
            # Split into grid regions and classify each
            grid_predictions = np.zeros((grid_size, grid_size, self.num_classes))
            
            cell_height = self.input_shape[0] // grid_size
            cell_width = self.input_shape[1] // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Extract cell
                    y1 = i * cell_height
                    y2 = (i + 1) * cell_height
                    x1 = j * cell_width
                    x2 = (j + 1) * cell_width
                    
                    cell = resized[y1:y2, x1:x2]
                    
                    # Classify cell
                    class_idx, confidence = self.classify_piece(cell)
                    
                    # Create one-hot encoded prediction
                    prediction = np.zeros(self.num_classes)
                    prediction[class_idx] = confidence
                    grid_predictions[i, j] = prediction
            
            return grid_predictions
            
        except Exception as e:
            logger.error(f"Error classifying board region: {e}")
            return np.zeros((grid_size, grid_size, self.num_classes))
    
    def classify_piece(self, piece_image: np.ndarray) -> Tuple[int, float]:
        """Classify a single piece image"""
        try:
            if not self.is_trained:
                # Use rule-based classification if model not trained
                return self._rule_based_classification(piece_image)
            
            # Extract features from image
            features = self._extract_piece_features(piece_image)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            class_idx = prediction[0]
            confidence = np.max(probabilities[0])
            
            return class_idx, confidence
            
        except Exception as e:
            logger.error(f"Error classifying piece: {e}")
            return 0, 0.0
    
    def _rule_based_classification(self, piece_image: np.ndarray) -> Tuple[int, float]:
        """Rule-based piece classification using color analysis"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(piece_image, cv2.COLOR_RGB2HSV)
            
            # Calculate color statistics
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            
            # Simple color-based classification
            if v_mean > 200:  # Bright -> white piece
                return 1, 0.8
            elif v_mean < 80:  # Dark -> black piece
                return 2, 0.8
            elif s_mean > 100 and (h_mean < 20 or h_mean > 160):  # Red-ish -> queen
                return 3, 0.7
            else:  # Default to empty
                return 0, 0.6
                
        except Exception as e:
            logger.warning(f"Error in rule-based classification: {e}")
            return 0, 0.5
    
    def _extract_piece_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from piece image for classification"""
        features = []
        
        try:
            # Resize to standard size
            resized = cv2.resize(image, (32, 32))
            
            # Color histogram features
            hist_r = cv2.calcHist([resized], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([resized], [1], None, [8], [0, 256])
            hist_b = cv2.calcHist([resized], [2], None, [8], [0, 256])
            
            color_features = np.concatenate([hist_r.flatten(), 
                                           hist_g.flatten(), 
                                           hist_b.flatten()])
            features.extend(color_features)
            
            # HSV statistics
            hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
            h_stats = [np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])]
            s_stats = [np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])]
            v_stats = [np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])]
            features.extend(h_stats + s_stats + v_stats)
            
            # Edge features
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Texture features
            texture = np.std(gray)
            features.append(texture)
            
            # Shape features (basic)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                features.append(circularity)
            else:
                features.append(0)
            
        except Exception as e:
            logger.warning(f"Error extracting piece features: {e}")
            # Return zeros if extraction fails
            features = [0.0] * 33  # 24 color + 6 HSV + 2 texture + 1 shape
        
        return np.array(features)
    
    def extract_board_features(self, board_image: np.ndarray) -> np.ndarray:
        """Extract features from board for ML models"""
        try:
            # Classify board into grid
            grid_predictions = self.classify_board_region(board_image)
            
            # Flatten predictions into feature vector
            grid_features = grid_predictions.flatten()
            
            # Add global board features
            global_features = self._extract_global_features(board_image)
            
            # Combine features
            all_features = np.concatenate([grid_features, global_features])
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting board features: {e}")
            return np.zeros(256)  # Default feature size
    
    def _extract_global_features(self, board_image: np.ndarray) -> np.ndarray:
        """Extract global features from the entire board"""
        features = []
        
        try:
            # Overall color distribution
            hist_r = cv2.calcHist([board_image], [0], None, [4], [0, 256])
            hist_g = cv2.calcHist([board_image], [1], None, [4], [0, 256])
            hist_b = cv2.calcHist([board_image], [2], None, [4], [0, 256])
            
            color_features = np.concatenate([hist_r.flatten(), 
                                           hist_g.flatten(), 
                                           hist_b.flatten()])
            features.extend(color_features)
            
            # Edge density
            gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Texture complexity
            texture = np.std(gray)
            features.append(texture)
            
        except Exception as e:
            logger.warning(f"Error extracting global features: {e}")
            features = [0.0] * 14  # 12 color + 2 global features
        
        return np.array(features)
    
    def train_on_data(self, training_data: List[Tuple[np.ndarray, int]]):
        """Train the classifier on labeled data"""
        if not training_data:
            logger.warning("No training data provided")
            return
        
        try:
            # Prepare data
            X = []
            y = []
            
            for image, label in training_data:
                features = self._extract_piece_features(image)
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate accuracy
            train_score = self.model.score(X_scaled, y)
            
            logger.info(f"Board classifier training completed with accuracy: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Board classifier saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.is_trained = model_data.get('is_trained', False)
                
                logger.info(f"Board classifier loaded from {filepath}")
            else:
                logger.warning(f"Model file not found: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_board_complexity(self, board_image: np.ndarray) -> float:
        """Evaluate how complex the board state is"""
        try:
            # Count number of pieces
            grid_predictions = self.classify_board_region(board_image)
            
            # Count non-empty cells
            non_empty_cells = 0
            for i in range(grid_predictions.shape[0]):
                for j in range(grid_predictions.shape[1]):
                    if np.argmax(grid_predictions[i, j]) != 0:  # 0 = empty
                        non_empty_cells += 1
            
            # Calculate complexity as ratio of occupied cells
            total_cells = grid_predictions.shape[0] * grid_predictions.shape[1]
            complexity = non_empty_cells / total_cells
            
            return complexity
            
        except Exception as e:
            logger.error(f"Error evaluating board complexity: {e}")
            return 0.5  # Default medium complexity
    
    def get_class_names(self) -> List[str]:
        """Get the class names for piece types"""
        return ['empty', 'white_piece', 'black_piece', 'queen_piece'] 