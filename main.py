#!/usr/bin/env python3
"""
Carrom Pool ML Bot - Main Entry Point
Plays Carrom Pool mobile game using computer vision and machine learning
"""

import os
import sys
import time
import argparse
from pathlib import Path
from loguru import logger

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.carrom_bot import CarromBot
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger


def main():
    """Main function to run the Carrom Pool ML Bot"""
    parser = argparse.ArgumentParser(description="Carrom Pool ML Bot")
    parser.add_argument("--mode", choices=["train", "play", "collect", "expert"], 
                       default="train", help="Bot operation mode")
    parser.add_argument("--config", default="config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--device", help="Specific ADB device ID")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--video", help="Path to expert gameplay video (for expert mode)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(debug=args.debug)
    logger.info("Starting Carrom Pool ML Bot")
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        
        # Override device if specified
        if args.device:
            config['device']['device_id'] = args.device
            
        # Initialize bot
        bot = CarromBot(config)
        
        # Run based on mode
        if args.mode == "train":
            logger.info("Starting training mode")
            for episode in range(args.episodes):
                logger.info(f"Training episode {episode + 1}/{args.episodes}")
                bot.train()
        elif args.mode == "play":
            logger.info("Starting play mode")
            for episode in range(args.episodes):
                logger.info(f"Play episode {episode + 1}/{args.episodes}")
                bot.play_game()
        elif args.mode == "collect":
            logger.info("Starting data collection mode")
            bot.collect_data()
        elif args.mode == "expert":
            if not args.video:
                logger.error("Expert mode requires --video parameter with path to expert gameplay video")
                return
            
            logger.info(f"Starting expert video analysis mode: {args.video}")
            
            # Analyze expert video
            expert_moves = bot.analyze_expert_video(args.video)
            
            # Train with expert data
            if expert_moves:
                bot.learn_from_expert_data('expert_insights.json')
                logger.info("Expert training completed successfully!")
            else:
                logger.warning("No expert moves extracted from video")
        else:
            logger.error(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}")
        raise
    finally:
        logger.info("Carrom Pool ML Bot shutting down")


if __name__ == "__main__":
    main() 