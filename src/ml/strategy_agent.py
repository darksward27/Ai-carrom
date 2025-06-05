"""
Strategy Agent for Carrom Pool using scikit-learn
Learns optimal moves through supervised learning and rule-based strategies
"""

import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
from collections import deque
from loguru import logger
from typing import List, Tuple, Optional


class StrategyAgent:
    """Strategy agent for Carrom Pool using Random Forest"""
    
    def __init__(self, config: dict):
        """Initialize the strategy agent"""
        self.config = config
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 0.1)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)
        self.batch_size = config.get('batch_size', 32)
        
        # Build ML models
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        # Training data
        self.X_train = []
        self.y_train = []
        self.is_trained = False
        
        # Training step counter
        self.step_count = 0
        
        logger.info("Strategy Agent initialized with Random Forest")
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from the agent"""
        if training and np.random.random() <= self.epsilon:
            # Exploration: random action
            return self._get_random_action()
        else:
            # Exploitation: best action from model or rules
            return self._get_best_action(state)
    
    def _get_random_action(self) -> np.ndarray:
        """Generate a random action"""
        # Random action vector [striker_x_norm, angle_norm, power_norm]
        action = np.random.random(3)
        return action
    
    def _get_best_action(self, state: np.ndarray) -> np.ndarray:
        """Get best action from model or rule-based strategy"""
        if self.is_trained and len(self.X_train) > 0:
            return self._get_model_action(state)
        else:
            return self._get_rule_based_action(state)
    
    def _get_model_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from trained model"""
        try:
            # Reshape state for prediction
            state_scaled = self.scaler.transform(state.reshape(1, -1))
            
            # Predict action values
            action_values = self.model.predict(state_scaled)
            
            # Convert to action vector
            action = self._decode_action_values(action_values[0])
            
            return action
            
        except Exception as e:
            logger.warning(f"Error getting model action: {e}")
            return self._get_rule_based_action(state)
    
    def _get_rule_based_action(self, state: np.ndarray) -> np.ndarray:
        """Get action using rule-based strategy"""
        # Extract board information from state
        # For now, use simple heuristics
        
        # Random striker position (can be improved with board analysis)
        striker_x = 0.5 + np.random.normal(0, 0.1)  # Center with some variation
        striker_x = max(0.1, min(0.9, striker_x))  # Keep within bounds
        
        # Aim towards center of board (can be improved)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Medium to high power
        power = np.random.uniform(0.6, 0.9)
        
        action = np.array([striker_x, angle / (2 * np.pi), power])
        return action
    
    def _decode_action_values(self, action_values: np.ndarray) -> np.ndarray:
        """Decode action values to action vector"""
        # Ensure values are in [0, 1] range
        action = np.clip(action_values, 0, 1)
        
        # Ensure we have 3 values
        if len(action) < 3:
            action = np.pad(action, (0, 3 - len(action)), constant_values=0.5)
        elif len(action) > 3:
            action = action[:3]
        
        return action
    
    def train(self):
        """Train the agent using collected data"""
        if len(self.memory) < self.batch_size:
            logger.warning("Not enough data to train")
            return
        
        try:
            # Prepare training data
            self._prepare_training_data()
            
            if len(self.X_train) == 0:
                logger.warning("No training data available")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(self.X_train)
            
            # Train model
            self.model.fit(X_scaled, self.y_train)
            self.is_trained = True
            
            # Update epsilon (exploration rate)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.step_count += 1
            logger.info(f"Training completed with {len(self.X_train)} samples, epsilon: {self.epsilon:.4f}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
    
    def _prepare_training_data(self):
        """Prepare training data from memory"""
        self.X_train = []
        self.y_train = []
        
        for experience in self.memory:
            state, action, reward, next_state, done = experience
            
            # Use state as input features
            self.X_train.append(state)
            
            # Use action as target (can be improved with reward weighting)
            if isinstance(action, np.ndarray) and len(action) >= 3:
                self.y_train.append(action[:3])
            else:
                # Skip invalid actions
                continue
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        logger.debug(f"Prepared training data: {self.X_train.shape}, {self.y_train.shape}")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data.get('is_trained', False)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.step_count = model_data.get('step_count', 0)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities (not applicable for Random Forest)"""
        # Return uniform probabilities for compatibility
        return np.ones(3) / 3
    
    def evaluate_action(self, state: np.ndarray, action: np.ndarray) -> float:
        """Evaluate how good an action is for a given state"""
        if not self.is_trained:
            return 0.5  # Neutral score
        
        try:
            state_scaled = self.scaler.transform(state.reshape(1, -1))
            predicted_action = self.model.predict(state_scaled)[0]
            
            # Calculate similarity between predicted and actual action
            similarity = 1.0 - np.mean(np.abs(predicted_action - action))
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Error evaluating action: {e}")
            return 0.5
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate"""
        return self.epsilon
    
    def set_exploration_rate(self, epsilon: float):
        """Set exploration rate"""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def clear_memory(self):
        """Clear experience memory"""
        self.memory.clear()
        logger.info("Memory cleared")
    
    def get_memory_size(self) -> int:
        """Get current memory size"""
        return len(self.memory)
    
    def get_statistics(self) -> dict:
        """Get agent statistics"""
        return {
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'is_trained': self.is_trained,
            'training_samples': len(self.X_train),
            'step_count': self.step_count
        }


class SimpleReplayMemory:
    """Simple experience replay memory"""
    
    def __init__(self, capacity: int):
        """Initialize replay memory"""
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from memory"""
        if len(self.memory) < batch_size:
            return list(self.memory)
        
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory) 