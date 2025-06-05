"""
Physics Simulator for Carrom Pool
Simulates piece movement, collisions, and shot outcomes
"""

import math
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Piece:
    """Represents a carrom piece"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 8.0
    mass: float = 0.8
    type: str = "normal"  # normal, queen, striker
    
    def update_position(self, dt: float):
        """Update piece position based on velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def apply_friction(self, friction: float, dt: float):
        """Apply friction to reduce velocity"""
        friction_force = friction * dt
        speed = math.sqrt(self.vx**2 + self.vy**2)
        
        if speed > 0:
            friction_x = -friction_force * (self.vx / speed)
            friction_y = -friction_force * (self.vy / speed)
            
            # Don't reverse direction
            if abs(friction_x) > abs(self.vx):
                self.vx = 0
            else:
                self.vx += friction_x
                
            if abs(friction_y) > abs(self.vy):
                self.vy = 0
            else:
                self.vy += friction_y


class PhysicsSimulator:
    """Physics simulation for Carrom Pool"""
    
    def __init__(self, config: dict):
        """Initialize physics simulator"""
        self.config = config
        self.friction = config['friction_coefficient']
        self.restitution = config['restitution']
        self.board_size = config['board_size']
        self.pocket_radius = config['pocket_radius']
        self.piece_radius = config['piece_radius']
        self.striker_radius = config['striker_radius']
        
        # Pocket positions (corners)
        self.pockets = [
            (self.pocket_radius, self.pocket_radius),
            (self.board_size - self.pocket_radius, self.pocket_radius),
            (self.pocket_radius, self.board_size - self.pocket_radius),
            (self.board_size - self.pocket_radius, self.board_size - self.pocket_radius)
        ]
        
        logger.info("Physics Simulator initialized")
    
    def simulate_shot(self, game_state, striker_x: float, striker_y: float, 
                     angle: float, power: float) -> Dict:
        """Simulate a complete shot and return outcome"""
        try:
            # Convert real coordinates to simulation coordinates
            sim_pieces = self._convert_to_simulation_coords(game_state)
            
            # Create striker with initial velocity
            striker_velocity = power * 200  # Scale power to velocity
            striker = Piece(
                x=striker_x * (self.board_size / 400),  # Scale to sim coords
                y=striker_y * (self.board_size / 400),
                vx=striker_velocity * math.cos(angle),
                vy=striker_velocity * math.sin(angle),
                radius=self.striker_radius,
                mass=self.config['striker_mass'],
                type="striker"
            )
            
            # Add striker to pieces
            sim_pieces.append(striker)
            
            # Simulate physics
            result = self._simulate_physics(sim_pieces)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in shot simulation: {e}")
            return {'error': str(e)}
    
    def _convert_to_simulation_coords(self, game_state) -> List[Piece]:
        """Convert game state to simulation pieces"""
        sim_pieces = []
        
        board_state = game_state.board_state
        if not board_state or 'pieces' not in board_state:
            return sim_pieces
        
        # Scale factor from real coordinates to simulation
        scale = self.board_size / 400  # Assuming 400px board in real coords
        
        for piece_data in board_state['pieces']:
            piece = Piece(
                x=piece_data['position'][0] * scale,
                y=piece_data['position'][1] * scale,
                radius=self.piece_radius,
                mass=self.config['piece_mass'],
                type=piece_data['type']
            )
            sim_pieces.append(piece)
        
        return sim_pieces
    
    def _simulate_physics(self, pieces: List[Piece], max_time: float = 10.0) -> Dict:
        """Simulate physics until all pieces stop moving"""
        dt = 0.01  # Time step
        time_elapsed = 0.0
        
        pocketed_pieces = []
        collisions = 0
        striker_pocketed = False
        
        while time_elapsed < max_time:
            # Check if any pieces are still moving
            any_moving = any(
                math.sqrt(p.vx**2 + p.vy**2) > 0.1 for p in pieces
            )
            
            if not any_moving:
                break
            
            # Update positions
            for piece in pieces:
                piece.update_position(dt)
            
            # Handle collisions
            collisions += self._handle_collisions(pieces)
            
            # Handle wall bounces
            self._handle_wall_bounces(pieces)
            
            # Check for pocketed pieces
            newly_pocketed = self._check_pocketed_pieces(pieces)
            for piece in newly_pocketed:
                if piece.type == "striker":
                    striker_pocketed = True
                pocketed_pieces.append(piece)
                pieces.remove(piece)
            
            # Apply friction
            for piece in pieces:
                piece.apply_friction(self.friction, dt)
            
            time_elapsed += dt
        
        # Analyze results
        result = self._analyze_simulation_result(
            pocketed_pieces, collisions, striker_pocketed, time_elapsed
        )
        
        return result
    
    def _handle_collisions(self, pieces: List[Piece]) -> int:
        """Handle piece-to-piece collisions"""
        collision_count = 0
        
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                piece1, piece2 = pieces[i], pieces[j]
                
                # Calculate distance
                dx = piece2.x - piece1.x
                dy = piece2.y - piece1.y
                distance = math.sqrt(dx**2 + dy**2)
                
                # Check collision
                min_distance = piece1.radius + piece2.radius
                if distance < min_distance:
                    # Collision detected
                    self._resolve_collision(piece1, piece2, dx, dy, distance)
                    collision_count += 1
        
        return collision_count
    
    def _resolve_collision(self, piece1: Piece, piece2: Piece, 
                          dx: float, dy: float, distance: float):
        """Resolve collision between two pieces"""
        if distance == 0:
            return
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Separate pieces
        overlap = (piece1.radius + piece2.radius) - distance
        piece1.x -= nx * overlap * 0.5
        piece1.y -= ny * overlap * 0.5
        piece2.x += nx * overlap * 0.5
        piece2.y += ny * overlap * 0.5
        
        # Calculate relative velocity
        dvx = piece2.vx - piece1.vx
        dvy = piece2.vy - piece1.vy
        
        # Calculate relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Do not resolve if velocities are separating
        if dvn > 0:
            return
        
        # Calculate restitution
        e = self.restitution
        
        # Calculate impulse scalar
        impulse = -(1 + e) * dvn / (1/piece1.mass + 1/piece2.mass)
        
        # Apply impulse
        impulse_x = impulse * nx
        impulse_y = impulse * ny
        
        piece1.vx -= impulse_x / piece1.mass
        piece1.vy -= impulse_y / piece1.mass
        piece2.vx += impulse_x / piece2.mass
        piece2.vy += impulse_y / piece2.mass
    
    def _handle_wall_bounces(self, pieces: List[Piece]):
        """Handle bounces off board edges"""
        for piece in pieces:
            # Left wall
            if piece.x - piece.radius < 0:
                piece.x = piece.radius
                piece.vx = -piece.vx * self.restitution
            
            # Right wall
            if piece.x + piece.radius > self.board_size:
                piece.x = self.board_size - piece.radius
                piece.vx = -piece.vx * self.restitution
            
            # Top wall
            if piece.y - piece.radius < 0:
                piece.y = piece.radius
                piece.vy = -piece.vy * self.restitution
            
            # Bottom wall
            if piece.y + piece.radius > self.board_size:
                piece.y = self.board_size - piece.radius
                piece.vy = -piece.vy * self.restitution
    
    def _check_pocketed_pieces(self, pieces: List[Piece]) -> List[Piece]:
        """Check which pieces have been pocketed"""
        pocketed = []
        
        for piece in pieces[:]:  # Copy list to avoid modification during iteration
            for pocket_x, pocket_y in self.pockets:
                dx = piece.x - pocket_x
                dy = piece.y - pocket_y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < self.pocket_radius:
                    pocketed.append(piece)
                    break
        
        return pocketed
    
    def _analyze_simulation_result(self, pocketed_pieces: List[Piece], 
                                  collisions: int, striker_pocketed: bool, 
                                  simulation_time: float) -> Dict:
        """Analyze simulation results"""
        result = {
            'pieces_pocketed': len([p for p in pocketed_pieces if p.type != "striker"]),
            'striker_pocketed': striker_pocketed,
            'queen_pocketed': any(p.type == "queen" for p in pocketed_pieces),
            'white_pocketed': len([p for p in pocketed_pieces if p.type == "white"]),
            'black_pocketed': len([p for p in pocketed_pieces if p.type == "black"]),
            'collisions': collisions,
            'simulation_time': simulation_time,
            'foul': self._determine_foul(pocketed_pieces, striker_pocketed),
            'game_won': self._check_game_won(pocketed_pieces)
        }
        
        logger.debug(f"Simulation result: {result}")
        return result
    
    def _determine_foul(self, pocketed_pieces: List[Piece], 
                       striker_pocketed: bool) -> bool:
        """Determine if the shot was a foul"""
        # Striker pocketed is always a foul
        if striker_pocketed:
            return True
        
        # No pieces pocketed might be a foul (depends on game rules)
        if len([p for p in pocketed_pieces if p.type != "striker"]) == 0:
            return False  # Not necessarily a foul in carrom
        
        # Pocketing opponent's pieces first might be a foul
        # This would require knowing which color we're playing
        
        return False
    
    def _check_game_won(self, pocketed_pieces: List[Piece]) -> bool:
        """Check if the game is won with this shot"""
        # This would depend on the specific game rules
        # For now, just check if all pieces of one color are pocketed
        
        white_pocketed = len([p for p in pocketed_pieces if p.type == "white"])
        black_pocketed = len([p for p in pocketed_pieces if p.type == "black"])
        queen_pocketed = any(p.type == "queen" for p in pocketed_pieces)
        
        # Simplified win condition
        return (white_pocketed >= 4 or black_pocketed >= 4) and queen_pocketed
    
    def predict_trajectory(self, start_x: float, start_y: float, 
                          angle: float, power: float, 
                          num_points: int = 50) -> List[Tuple[float, float]]:
        """Predict trajectory of a shot for visualization"""
        trajectory = []
        
        # Initial conditions
        velocity = power * 200
        vx = velocity * math.cos(angle)
        vy = velocity * math.sin(angle)
        x, y = start_x, start_y
        
        dt = 0.1
        
        for i in range(num_points):
            trajectory.append((x, y))
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Apply friction
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 0:
                friction_force = self.friction * dt
                vx -= friction_force * (vx / speed)
                vy -= friction_force * (vy / speed)
            
            # Handle wall bounces
            if x < 0 or x > self.board_size:
                vx = -vx * self.restitution
                x = max(0, min(x, self.board_size))
            
            if y < 0 or y > self.board_size:
                vy = -vy * self.restitution
                y = max(0, min(y, self.board_size))
            
            # Stop if velocity is too low
            if math.sqrt(vx**2 + vy**2) < 1.0:
                break
        
        return trajectory
    
    def calculate_shot_angle(self, start_pos: Tuple[float, float], 
                           target_pos: Tuple[float, float]) -> float:
        """Calculate angle needed to hit a target"""
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        
        return math.atan2(dy, dx)
    
    def estimate_shot_power(self, distance: float) -> float:
        """Estimate power needed for a given distance"""
        # This is a simplified estimation
        # In reality, this would depend on friction and other factors
        base_power = distance / 300.0  # Normalize to 0-1 range
        return max(0.1, min(1.0, base_power))
    
    def find_best_shot_angle(self, striker_pos: Tuple[float, float], 
                            target_piece: Piece, 
                            target_pocket: Tuple[float, float]) -> float:
        """Find optimal angle to hit a piece into a pocket"""
        # Vector from piece to pocket
        piece_to_pocket_x = target_pocket[0] - target_piece.x
        piece_to_pocket_y = target_pocket[1] - target_piece.y
        
        # Normalize
        distance = math.sqrt(piece_to_pocket_x**2 + piece_to_pocket_y**2)
        if distance == 0:
            return 0
        
        piece_to_pocket_x /= distance
        piece_to_pocket_y /= distance
        
        # Point on the piece to hit (opposite to pocket direction)
        hit_point_x = target_piece.x - piece_to_pocket_x * target_piece.radius
        hit_point_y = target_piece.y - piece_to_pocket_y * target_piece.radius
        
        # Calculate angle from striker to hit point
        dx = hit_point_x - striker_pos[0]
        dy = hit_point_y - striker_pos[1]
        
        return math.atan2(dy, dx) 