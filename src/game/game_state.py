"""
Game State Management for Carrom Pool
Tracks current board state, pieces, and game progress
"""

import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class GameState:
    """Represents the current state of a Carrom Pool game"""
    
    # Board state from computer vision
    board_state: Dict = field(default_factory=dict)
    
    # Game metadata
    our_turn: bool = True
    game_active: bool = False
    game_finished: bool = False
    
    # Piece counts
    white_pieces_remaining: int = 9
    black_pieces_remaining: int = 9
    queen_on_board: bool = True
    
    # Player information
    our_color: str = "white"  # or "black"
    opponent_color: str = "black"
    
    # Score tracking
    our_score: int = 0
    opponent_score: int = 0
    
    # Game history
    moves_history: List = field(default_factory=list)
    shots_taken: int = 0
    
    # Timing
    game_start_time: float = 0.0
    last_move_time: float = 0.0
    
    def reset(self):
        """Reset game state for a new game"""
        self.board_state = {}
        self.our_turn = True
        self.game_active = False
        self.game_finished = False
        
        self.white_pieces_remaining = 9
        self.black_pieces_remaining = 9
        self.queen_on_board = True
        
        self.our_score = 0
        self.opponent_score = 0
        
        self.moves_history = []
        self.shots_taken = 0
        
        self.game_start_time = 0.0
        self.last_move_time = 0.0
        
        logger.info("Game state reset")
    
    def update_from_screenshot(self, screenshot: np.ndarray, game_detector):
        """Update game state from a screenshot using game detector"""
        try:
            # Detect board state
            new_board_state = game_detector.detect_board_state(screenshot)
            
            if new_board_state:
                self.board_state = new_board_state
                self._update_piece_counts()
                
                # Update game status
                self.game_active = game_detector.is_game_active(screenshot)
                self.game_finished = game_detector.is_game_finished(screenshot)
                self.our_turn = game_detector.is_our_turn(screenshot)
                
                logger.debug("Game state updated from screenshot")
            
        except Exception as e:
            logger.error(f"Error updating game state: {e}")
    
    def _update_piece_counts(self):
        """Update piece counts based on current board state"""
        if not self.board_state or 'pieces' not in self.board_state:
            return
        
        pieces = self.board_state['pieces']
        
        self.white_pieces_remaining = len([p for p in pieces if p['type'] == 'white'])
        self.black_pieces_remaining = len([p for p in pieces if p['type'] == 'black'])
        self.queen_on_board = any(p['type'] == 'queen' for p in pieces)
        
        logger.debug(f"Piece counts: White={self.white_pieces_remaining}, "
                    f"Black={self.black_pieces_remaining}, Queen={self.queen_on_board}")
    
    def get_our_pieces(self) -> List[Dict]:
        """Get list of our pieces on the board"""
        if not self.board_state or 'pieces' not in self.board_state:
            return []
        
        return [p for p in self.board_state['pieces'] if p['type'] == self.our_color]
    
    def get_opponent_pieces(self) -> List[Dict]:
        """Get list of opponent's pieces on the board"""
        if not self.board_state or 'pieces' not in self.board_state:
            return []
        
        return [p for p in self.board_state['pieces'] if p['type'] == self.opponent_color]
    
    def get_queen_piece(self) -> Optional[Dict]:
        """Get the queen piece if it's on the board"""
        if not self.board_state or 'pieces' not in self.board_state:
            return None
        
        queen_pieces = [p for p in self.board_state['pieces'] if p['type'] == 'queen']
        return queen_pieces[0] if queen_pieces else None
    
    def get_striker_position(self) -> Optional[Tuple[float, float]]:
        """Get current striker position"""
        if (not self.board_state or 
            'striker' not in self.board_state or 
            not self.board_state['striker']):
            return None
        
        return self.board_state['striker']['position']
    
    def get_pockets(self) -> List[Dict]:
        """Get pocket positions"""
        if not self.board_state or 'pockets' not in self.board_state:
            return []
        
        return self.board_state['pockets']
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as a vector for ML model"""
        try:
            # Create a grid representation of the board
            board_size = 16  # 16x16 grid
            state_grid = np.zeros((board_size, board_size, 4))  # 4 channels: white, black, queen, striker
            
            if self.board_state and 'pieces' in self.board_state:
                board_region = self.board_state.get('board_region')
                if board_region:
                    x1, y1, x2, y2 = board_region
                    board_width = x2 - x1
                    board_height = y2 - y1
                    
                    # Map pieces to grid
                    for piece in self.board_state['pieces']:
                        px, py = piece['position']
                        
                        # Convert to grid coordinates
                        grid_x = int((px - x1) / board_width * (board_size - 1))
                        grid_y = int((py - y1) / board_height * (board_size - 1))
                        
                        # Ensure within bounds
                        grid_x = max(0, min(grid_x, board_size - 1))
                        grid_y = max(0, min(grid_y, board_size - 1))
                        
                        # Set appropriate channel
                        if piece['type'] == 'white':
                            state_grid[grid_y, grid_x, 0] = 1.0
                        elif piece['type'] == 'black':
                            state_grid[grid_y, grid_x, 1] = 1.0
                        elif piece['type'] == 'queen':
                            state_grid[grid_y, grid_x, 2] = 1.0
                    
                    # Add striker if present
                    if (self.board_state.get('striker') and 
                        self.board_state['striker'].get('position')):
                        sx, sy = self.board_state['striker']['position']
                        
                        grid_x = int((sx - x1) / board_width * (board_size - 1))
                        grid_y = int((sy - y1) / board_height * (board_size - 1))
                        
                        grid_x = max(0, min(grid_x, board_size - 1))
                        grid_y = max(0, min(grid_y, board_size - 1))
                        
                        state_grid[grid_y, grid_x, 3] = 1.0
            
            # Flatten the grid
            flattened_board = state_grid.flatten()
            
            # Add game metadata
            metadata = np.array([
                1.0 if self.our_turn else 0.0,
                self.white_pieces_remaining / 9.0,  # Normalized
                self.black_pieces_remaining / 9.0,
                1.0 if self.queen_on_board else 0.0,
                self.shots_taken / 50.0,  # Normalized
                1.0 if self.our_color == 'white' else 0.0
            ])
            
            # Combine board state and metadata
            state_vector = np.concatenate([flattened_board, metadata])
            
            # Pad to expected size if necessary
            expected_size = 256  # From config
            if len(state_vector) < expected_size:
                padding = np.zeros(expected_size - len(state_vector))
                state_vector = np.concatenate([state_vector, padding])
            elif len(state_vector) > expected_size:
                state_vector = state_vector[:expected_size]
            
            return state_vector
            
        except Exception as e:
            logger.error(f"Error creating state vector: {e}")
            # Return zero vector as fallback
            return np.zeros(256)
    
    def record_move(self, action: np.ndarray, result: Dict):
        """Record a move in the game history"""
        move = {
            'action': action.copy(),
            'result': result.copy(),
            'board_state_before': self.board_state.copy(),
            'shot_number': self.shots_taken,
            'timestamp': self.last_move_time
        }
        
        self.moves_history.append(move)
        self.shots_taken += 1
        
        logger.debug(f"Move recorded: shot {self.shots_taken}")
    
    def get_game_progress(self) -> float:
        """Get game progress as a percentage (0-1)"""
        if not self.game_active:
            return 0.0
        
        initial_pieces = 18  # 9 white + 9 black
        remaining_pieces = self.white_pieces_remaining + self.black_pieces_remaining
        
        return 1.0 - (remaining_pieces / initial_pieces)
    
    def is_winning(self) -> bool:
        """Check if we are currently winning"""
        our_pieces = self.white_pieces_remaining if self.our_color == 'white' else self.black_pieces_remaining
        opponent_pieces = self.black_pieces_remaining if self.our_color == 'white' else self.white_pieces_remaining
        
        return our_pieces < opponent_pieces
    
    def get_strategic_info(self) -> Dict:
        """Get strategic information for decision making"""
        info = {
            'game_progress': self.get_game_progress(),
            'is_winning': self.is_winning(),
            'queen_available': self.queen_on_board,
            'our_pieces_count': len(self.get_our_pieces()),
            'opponent_pieces_count': len(self.get_opponent_pieces()),
            'shots_taken': self.shots_taken,
            'our_turn': self.our_turn
        }
        
        # Add piece distribution analysis
        our_pieces = self.get_our_pieces()
        if our_pieces:
            positions = [p['position'] for p in our_pieces]
            
            # Calculate center of mass
            center_x = sum(pos[0] for pos in positions) / len(positions)
            center_y = sum(pos[1] for pos in positions) / len(positions)
            info['our_pieces_center'] = (center_x, center_y)
            
            # Calculate spread
            distances = [((pos[0] - center_x)**2 + (pos[1] - center_y)**2)**0.5 for pos in positions]
            info['our_pieces_spread'] = sum(distances) / len(distances)
        
        return info
    
    def validate_state(self) -> bool:
        """Validate the current game state for consistency"""
        try:
            # Check piece counts
            if self.white_pieces_remaining < 0 or self.white_pieces_remaining > 9:
                logger.warning(f"Invalid white piece count: {self.white_pieces_remaining}")
                return False
            
            if self.black_pieces_remaining < 0 or self.black_pieces_remaining > 9:
                logger.warning(f"Invalid black piece count: {self.black_pieces_remaining}")
                return False
            
            # Check board state consistency
            if self.board_state and 'pieces' in self.board_state:
                actual_white = len([p for p in self.board_state['pieces'] if p['type'] == 'white'])
                actual_black = len([p for p in self.board_state['pieces'] if p['type'] == 'black'])
                
                if actual_white != self.white_pieces_remaining:
                    logger.warning(f"White piece count mismatch: {actual_white} vs {self.white_pieces_remaining}")
                    return False
                
                if actual_black != self.black_pieces_remaining:
                    logger.warning(f"Black piece count mismatch: {actual_black} vs {self.black_pieces_remaining}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating game state: {e}")
            return False
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the game state"""
        summary = []
        summary.append(f"Game Active: {self.game_active}")
        summary.append(f"Our Turn: {self.our_turn}")
        summary.append(f"Our Color: {self.our_color}")
        summary.append(f"White Pieces: {self.white_pieces_remaining}")
        summary.append(f"Black Pieces: {self.black_pieces_remaining}")
        summary.append(f"Queen on Board: {self.queen_on_board}")
        summary.append(f"Shots Taken: {self.shots_taken}")
        summary.append(f"Game Progress: {self.get_game_progress():.1%}")
        
        return " | ".join(summary) 