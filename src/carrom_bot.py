"""
Main Carrom Bot class that orchestrates computer vision, ML models, and ADB control
"""

import time
import numpy as np
import cv2
from loguru import logger
from typing import Dict, Tuple, List, Optional

from .device.adb_controller import ADBController
from .vision.game_detector import GameDetector
from .ml.board_classifier import BoardClassifier
from .ml.strategy_agent import StrategyAgent
from .game.physics_simulator import PhysicsSimulator
from .game.game_state import GameState
from .utils.screenshot_manager import ScreenshotManager


class CarromBot:
    """Main bot class for playing Carrom Pool"""
    
    def __init__(self, config: Dict):
        """Initialize the Carrom Bot with configuration"""
        self.config = config
        self.game_running = False
        
        # Initialize components
        logger.info("Initializing Carrom Bot components...")
        
        self.adb_controller = ADBController(config['device'])
        self.game_detector = GameDetector(config['detection'])
        self.board_classifier = BoardClassifier(config['model']['cnn'])
        self.strategy_agent = StrategyAgent(config['model']['rl'])
        self.physics_simulator = PhysicsSimulator(config['physics'])
        self.screenshot_manager = ScreenshotManager()
        
        self.game_state = GameState()
        
        # Statistics
        self.games_played = 0
        self.games_won = 0
        self.total_shots = 0
        
        logger.info("Carrom Bot initialized successfully")
    
    def train(self):
        """Train the ML models using reinforcement learning"""
        logger.info("Starting training mode")
        episodes = self.config['training']['episodes']
        
        for episode in range(episodes):
            logger.info(f"Training episode {episode + 1}/{episodes}")
            
            try:
                # Start new game
                self._start_new_game()
                
                # Play episode
                self._play_training_episode(episode)
                
                # Save model periodically
                if (episode + 1) % self.config['training']['save_frequency'] == 0:
                    self._save_models()
                    
            except Exception as e:
                logger.error(f"Error in training episode {episode}: {e}")
                continue
        
        logger.info("Training completed")
        self._save_models()
    
    def play(self):
        """Play the game using trained models"""
        logger.info("Starting gameplay mode")
        
        try:
            while True:
                # Wait for game to start
                self._wait_for_game_start()
                
                # Play the game
                self._play_game()
                
                # Wait before next game
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Gameplay stopped by user")
    
    def collect_data(self):
        """Collect training data by observing human gameplay"""
        logger.info("Starting data collection mode")
        
        try:
            while True:
                # Capture screenshot
                screenshot = self.adb_controller.take_screenshot()
                
                # Detect game state
                board_state = self.game_detector.detect_board_state(screenshot)
                
                # Save data
                self._save_training_data(screenshot, board_state)
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
    
    def _start_new_game(self):
        """Start a new game"""
        logger.debug("Starting new game")
        
        # Reset game state
        self.game_state.reset()
        
        # Take screenshot and detect initial state
        screenshot = self.adb_controller.take_screenshot()
        self.game_state.update_from_screenshot(screenshot, self.game_detector)
        
        self.games_played += 1
        logger.info(f"Game {self.games_played} started")
    
    def _wait_for_game_start(self):
        """Wait for a game to start"""
        logger.info("Waiting for game to start...")
        
        while True:
            screenshot = self.adb_controller.take_screenshot()
            
            if self.game_detector.is_game_active(screenshot):
                logger.info("Game detected, starting to play")
                self._start_new_game()
                break
                
            time.sleep(2)
    
    def _play_game(self):
        """Play a complete game"""
        self.game_running = True
        shot_count = 0
        
        while self.game_running:
            try:
                # Take screenshot and update game state
                screenshot = self.adb_controller.take_screenshot()
                self.game_state.update_from_screenshot(screenshot, self.game_detector)
                
                # Check if game is finished
                if self.game_detector.is_game_finished(screenshot):
                    self._handle_game_end(screenshot)
                    break
                
                # Check if it's our turn
                if not self.game_detector.is_our_turn(screenshot):
                    logger.debug("Waiting for our turn...")
                    time.sleep(1)
                    continue
                
                # Make a move
                self._make_move()
                shot_count += 1
                self.total_shots += 1
                
                # Wait for shot to complete
                time.sleep(self.config['gameplay']['shot_delay'])
                
                # Safety check for maximum shots
                if shot_count > self.config['training']['max_steps_per_episode']:
                    logger.warning("Maximum shots reached, ending game")
                    break
                    
            except Exception as e:
                logger.error(f"Error during gameplay: {e}")
                break
        
        self.game_running = False
    
    def _play_training_episode(self, episode: int):
        """Play a single training episode"""
        shot_count = 0
        episode_reward = 0
        
        while shot_count < self.config['training']['max_steps_per_episode']:
            try:
                # Get current state
                state = self.game_state.get_state_vector()
                
                # Get action from agent
                action = self.strategy_agent.get_action(state, training=True)
                
                # Execute action
                reward = self._execute_action(action)
                episode_reward += reward
                
                # Get next state
                screenshot = self.adb_controller.take_screenshot()
                self.game_state.update_from_screenshot(screenshot, self.game_detector)
                next_state = self.game_state.get_state_vector()
                
                # Store experience
                done = self.game_detector.is_game_finished(screenshot)
                self.strategy_agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.strategy_agent.memory) > self.config['model']['rl']['batch_size']:
                    self.strategy_agent.train()
                
                shot_count += 1
                
                if done:
                    break
                    
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                break
        
        logger.info(f"Episode {episode} completed with reward: {episode_reward}")
    
    def _make_move(self):
        """Make a strategic move"""
        try:
            # Get current game state
            state = self.game_state.get_state_vector()
            
            # Get best action from strategy agent
            action = self.strategy_agent.get_action(state, training=False)
            
            # Execute the action
            self._execute_action(action)
            
            logger.debug(f"Move executed: action={action}")
            
        except Exception as e:
            logger.error(f"Error making move: {e}")
    
    def _execute_action(self, action: np.ndarray) -> float:
        """Execute an action and return reward"""
        try:
            # Decode action (striker position, angle, power)
            striker_x, striker_y, angle, power = self._decode_action(action)
            
            # Simulate the shot to predict outcome
            predicted_result = self.physics_simulator.simulate_shot(
                self.game_state, striker_x, striker_y, angle, power
            )
            
            # Execute the shot via ADB
            self.adb_controller.make_shot(striker_x, striker_y, angle, power)
            
            # Calculate reward based on predicted vs actual outcome
            reward = self._calculate_reward(predicted_result)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return -1.0
    
    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        """Decode action vector into shot parameters"""
        # Normalize action values
        action = np.clip(action, 0, 1)
        
        # Get game area bounds
        x1, y1, x2, y2 = self.config['device']['game_area']
        
        # Decode striker position (bottom edge of board)
        striker_x = x1 + action[0] * (x2 - x1)
        striker_y = y2 - 50  # Near bottom edge
        
        # Decode angle (0 to 2π)
        angle = action[1] * 2 * np.pi
        
        # Decode power (within configured range)
        min_power, max_power = self.config['gameplay']['shot_power_range']
        power = min_power + action[2] * (max_power - min_power)
        
        return striker_x, striker_y, angle, power
    
    def _calculate_reward(self, predicted_result: Dict) -> float:
        """Calculate reward based on shot outcome"""
        reward = 0.0
        
        # Positive rewards
        if predicted_result.get('pieces_pocketed', 0) > 0:
            reward += predicted_result['pieces_pocketed'] * 10
        
        if predicted_result.get('queen_pocketed', False):
            reward += 50
        
        if predicted_result.get('game_won', False):
            reward += 100
        
        # Negative rewards
        if predicted_result.get('foul', False):
            reward -= 20
        
        if predicted_result.get('striker_pocketed', False):
            reward -= 15
        
        # Small positive reward for valid shots
        if not predicted_result.get('foul', False):
            reward += 1
        
        return reward
    
    def _handle_game_end(self, screenshot: np.ndarray):
        """Handle end of game"""
        won = self.game_detector.did_we_win(screenshot)
        
        if won:
            self.games_won += 1
            logger.info(f"Game won! Total wins: {self.games_won}/{self.games_played}")
        else:
            logger.info(f"Game lost. Total wins: {self.games_won}/{self.games_played}")
        
        # Print statistics
        win_rate = (self.games_won / self.games_played) * 100 if self.games_played > 0 else 0
        logger.info(f"Win rate: {win_rate:.1f}% ({self.games_won}/{self.games_played})")
        logger.info(f"Total shots fired: {self.total_shots}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            model_path = self.config['training']['model_save_path']
            
            self.board_classifier.save_model(f"{model_path}/board_classifier.h5")
            self.strategy_agent.save_model(f"{model_path}/strategy_agent.h5")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _save_training_data(self, screenshot: np.ndarray, board_state: Dict):
        """Save training data for supervised learning"""
        timestamp = int(time.time())
        
        # Save screenshot
        screenshot_path = f"screenshots/training_{timestamp}.png"
        self.screenshot_manager.save_screenshot(screenshot, screenshot_path)
        
        # Save board state
        state_path = f"screenshots/training_{timestamp}_state.npy"
        np.save(state_path, board_state)
        
        logger.debug(f"Training data saved: {screenshot_path}")
    
    def get_statistics(self) -> Dict:
        """Get bot performance statistics"""
        return {
            'games_played': self.games_played,
            'games_won': self.games_won,
            'win_rate': (self.games_won / self.games_played) * 100 if self.games_played > 0 else 0,
            'total_shots': self.total_shots
        }

    def analyze_expert_video(self, video_path: str):
        """Analyze expert gameplay from video files"""
        logger.info(f"Starting expert video analysis: {video_path}")
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            frame_count = 0
            extracted_moves = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30th frame (1 second intervals for 30fps video)
                if frame_count % 30 == 0:
                    # Detect game state
                    game_state = self.game_detector.detect_game_state(frame)
                    
                    if game_state['status'] == 'active':
                        # Extract board state
                        board_features = self.board_classifier.extract_board_features(frame)
                        
                        # Store this state for learning
                        state_data = {
                            'frame': frame_count,
                            'board_state': board_features,
                            'pieces': game_state.get('pieces', []),
                            'timestamp': frame_count / 30.0  # seconds
                        }
                        
                        extracted_moves.append(state_data)
                        
                        # Save screenshot for analysis
                        screenshot_path = f"screenshots/expert_analysis/frame_{frame_count}.jpg"
                        self.screenshot_manager.save_screenshot(frame, screenshot_path)
                        
                        logger.debug(f"Processed frame {frame_count}: {len(game_state.get('pieces', []))} pieces detected")
            
            cap.release()
            
            # Analyze extracted moves for patterns
            self._analyze_expert_patterns(extracted_moves)
            
            logger.info(f"Expert video analysis complete: {len(extracted_moves)} game states extracted")
            return extracted_moves
            
        except Exception as e:
            logger.error(f"Error analyzing expert video: {e}")
            raise

    def _analyze_expert_patterns(self, moves_data: list):
        """Analyze patterns in expert gameplay"""
        logger.info("Analyzing expert move patterns...")
        
        try:
            # Group moves by game phase
            opening_moves = []
            mid_game_moves = []
            end_game_moves = []
            
            for move in moves_data:
                piece_count = len(move.get('pieces', []))
                
                if piece_count > 15:  # Opening game
                    opening_moves.append(move)
                elif piece_count > 5:  # Mid game
                    mid_game_moves.append(move)
                else:  # End game
                    end_game_moves.append(move)
            
            # Extract strategic insights
            insights = {
                'opening_strategy': self._extract_phase_strategy(opening_moves),
                'mid_game_strategy': self._extract_phase_strategy(mid_game_moves),
                'end_game_strategy': self._extract_phase_strategy(end_game_moves)
            }
            
            # Save insights for training
            import json
            with open('expert_insights.json', 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.info("Expert pattern analysis saved to expert_insights.json")
            
        except Exception as e:
            logger.error(f"Error analyzing expert patterns: {e}")

    def _extract_phase_strategy(self, phase_moves: list):
        """Extract strategy for a specific game phase"""
        if not phase_moves:
            return {}
        
        strategy = {
            'average_pieces': np.mean([len(move.get('pieces', [])) for move in phase_moves]),
            'move_count': len(phase_moves),
            'time_range': [
                min(move['timestamp'] for move in phase_moves),
                max(move['timestamp'] for move in phase_moves)
            ]
        }
        
        return strategy

    def learn_from_expert_data(self, expert_data_path: str):
        """Train the model using expert gameplay data"""
        logger.info("Training model with expert gameplay data...")
        
        try:
            import json
            with open(expert_data_path, 'r') as f:
                expert_data = json.load(f)
            
            # Convert expert insights into training data
            training_states = []
            training_actions = []
            
            # Create synthetic training data based on expert patterns
            for phase, strategy in expert_data.items():
                if isinstance(strategy, dict) and 'move_count' in strategy:
                    # Generate training examples based on expert strategy
                    for _ in range(strategy['move_count']):
                        # Create synthetic state
                        state = np.random.random(self.config['ml']['strategy_agent']['state_size'])
                        
                        # Generate expert-like action based on strategy
                        action = self._generate_expert_action(phase, strategy)
                        
                        training_states.append(state)
                        training_actions.append(action)
            
            # Train the strategy agent with expert data
            for state, action in zip(training_states, training_actions):
                # High reward for expert moves
                self.strategy_agent.remember(state, action, 1.0, state, False)
            
            # Perform training
            self.strategy_agent.train()
            
            logger.info(f"Expert data training complete: {len(training_states)} expert moves learned")
            
        except Exception as e:
            logger.error(f"Error learning from expert data: {e}")

    def _generate_expert_action(self, phase: str, strategy: dict):
        """Generate an expert-like action based on game phase"""
        if 'opening' in phase:
            # Opening: Conservative, aim for easy shots
            striker_x = 0.5 + np.random.normal(0, 0.05)  # Center position
            angle = np.random.uniform(0.3, 0.7)  # Moderate angles
            power = np.random.uniform(0.6, 0.8)  # Medium power
        elif 'mid' in phase:
            # Mid-game: More aggressive, strategic positioning
            striker_x = np.random.uniform(0.2, 0.8)  # Varied positions
            angle = np.random.uniform(0.1, 0.9)  # Wide angle range
            power = np.random.uniform(0.7, 0.9)  # Higher power
        else:  # End game
            # End-game: Precise, calculated shots
            striker_x = 0.5 + np.random.normal(0, 0.03)  # Precise positioning
            angle = np.random.uniform(0.4, 0.6)  # Careful angles
            power = np.random.uniform(0.5, 0.7)  # Controlled power
        
        return np.array([
            max(0.1, min(0.9, striker_x)),
            max(0.0, min(1.0, angle)),
            max(0.3, min(1.0, power))
        ]) 