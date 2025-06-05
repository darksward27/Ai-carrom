"""
ADB Controller for device communication and game input
Handles screenshots, touch input, and device control
"""

import subprocess
import time
import math
import numpy as np
from PIL import Image
from io import BytesIO
from loguru import logger
from typing import Tuple, Optional


class ADBController:
    """Controls Android device via ADB for Carrom Pool game"""
    
    def __init__(self, device_config: dict):
        """Initialize ADB controller"""
        self.device_config = device_config
        self.device_id = device_config.get('device_id')
        self.screen_resolution = tuple(device_config['screen_resolution'])
        self.game_area = device_config['game_area']
        
        # Check ADB connection
        self._check_adb_connection()
        
        logger.info(f"ADB Controller initialized for device: {self.device_id or 'default'}")
    
    def _check_adb_connection(self):
        """Check if ADB is working and device is connected"""
        try:
            result = self._run_adb_command(['devices'])
            if 'device' not in result:
                raise Exception("No devices connected via ADB")
            
            logger.info("ADB connection verified")
            
        except Exception as e:
            logger.error(f"ADB connection failed: {e}")
            raise
    
    def _run_adb_command(self, command: list, timeout: int = 30) -> str:
        """Run an ADB command and return output"""
        full_command = ['adb']
        
        if self.device_id:
            full_command.extend(['-s', self.device_id])
        
        full_command.extend(command)
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"ADB command failed: {result.stderr}")
                raise Exception(f"ADB command failed: {result.stderr}")
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {' '.join(full_command)}")
            raise
        except Exception as e:
            logger.error(f"ADB command error: {e}")
            raise
    
    def take_screenshot(self) -> np.ndarray:
        """Take a screenshot and return as numpy array"""
        try:
            # Take screenshot using ADB
            result = self._run_adb_command(['shell', 'screencap', '-p'])
            
            # Convert to image
            image_data = result.encode('latin1')  # Preserve binary data
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB numpy array
            screenshot = np.array(image.convert('RGB'))
            
            logger.debug(f"Screenshot taken: {screenshot.shape}")
            return screenshot
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise
    
    def tap(self, x: float, y: float):
        """Tap at specific coordinates"""
        try:
            # Ensure coordinates are within bounds
            x = max(0, min(x, self.screen_resolution[0]))
            y = max(0, min(y, self.screen_resolution[1]))
            
            self._run_adb_command(['shell', 'input', 'tap', str(int(x)), str(int(y))])
            logger.debug(f"Tapped at ({x}, {y})")
            
        except Exception as e:
            logger.error(f"Failed to tap at ({x}, {y}): {e}")
            raise
    
    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration: int = 500):
        """Swipe from one point to another"""
        try:
            self._run_adb_command([
                'shell', 'input', 'swipe',
                str(int(x1)), str(int(y1)),
                str(int(x2)), str(int(y2)),
                str(duration)
            ])
            
            logger.debug(f"Swiped from ({x1}, {y1}) to ({x2}, {y2})")
            
        except Exception as e:
            logger.error(f"Failed to swipe: {e}")
            raise
    
    def make_shot(self, striker_x: float, striker_y: float, angle: float, power: float):
        """Make a carrom shot with specified parameters"""
        try:
            # Convert angle and power to swipe coordinates
            end_x, end_y, duration = self._calculate_shot_swipe(
                striker_x, striker_y, angle, power
            )
            
            # Perform the shot as a swipe gesture
            self.swipe(striker_x, striker_y, end_x, end_y, duration)
            
            logger.debug(f"Shot made: striker=({striker_x}, {striker_y}), "
                        f"angle={math.degrees(angle):.1f}°, power={power:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to make shot: {e}")
            raise
    
    def _calculate_shot_swipe(self, start_x: float, start_y: float, 
                             angle: float, power: float) -> Tuple[float, float, int]:
        """Calculate swipe end coordinates and duration for shot"""
        
        # Calculate swipe distance based on power (50-300 pixels)
        max_distance = 300
        min_distance = 50
        distance = min_distance + (max_distance - min_distance) * power
        
        # Calculate end coordinates
        end_x = start_x + distance * math.cos(angle)
        end_y = start_y + distance * math.sin(angle)
        
        # Ensure end coordinates are within game area
        x1, y1, x2, y2 = self.game_area
        end_x = max(x1, min(end_x, x2))
        end_y = max(y1, min(end_y, y2))
        
        # Calculate duration based on power (100-800ms)
        min_duration = 100
        max_duration = 800
        duration = int(min_duration + (max_duration - min_duration) * (1 - power))
        
        return end_x, end_y, duration
    
    def press_back(self):
        """Press the back button"""
        try:
            self._run_adb_command(['shell', 'input', 'keyevent', 'KEYCODE_BACK'])
            logger.debug("Back button pressed")
            
        except Exception as e:
            logger.error(f"Failed to press back: {e}")
            raise
    
    def press_home(self):
        """Press the home button"""
        try:
            self._run_adb_command(['shell', 'input', 'keyevent', 'KEYCODE_HOME'])
            logger.debug("Home button pressed")
            
        except Exception as e:
            logger.error(f"Failed to press home: {e}")
            raise
    
    def launch_app(self, package_name: str, activity_name: str = None):
        """Launch an app"""
        try:
            if activity_name:
                intent = f"{package_name}/{activity_name}"
            else:
                intent = package_name
            
            self._run_adb_command(['shell', 'am', 'start', '-n', intent])
            logger.info(f"Launched app: {package_name}")
            
        except Exception as e:
            logger.error(f"Failed to launch app {package_name}: {e}")
            raise
    
    def get_current_activity(self) -> str:
        """Get the current foreground activity"""
        try:
            result = self._run_adb_command([
                'shell', 'dumpsys', 'window', 'windows'
            ])
            
            # Parse output to find current activity
            for line in result.split('\n'):
                if 'mCurrentFocus' in line and 'Window{' in line:
                    # Extract activity name
                    start = line.find('{') + 1
                    end = line.find('}')
                    if start > 0 and end > start:
                        activity_info = line[start:end]
                        return activity_info.split()[-1]
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Failed to get current activity: {e}")
            return "unknown"
    
    def is_app_running(self, package_name: str) -> bool:
        """Check if an app is currently running"""
        try:
            result = self._run_adb_command(['shell', 'ps'])
            return package_name in result
            
        except Exception as e:
            logger.error(f"Failed to check if app is running: {e}")
            return False
    
    def wait_for_device(self, timeout: int = 30):
        """Wait for device to be ready"""
        try:
            self._run_adb_command(['wait-for-device'], timeout=timeout)
            time.sleep(2)  # Additional wait for stability
            
            logger.info("Device is ready")
            
        except Exception as e:
            logger.error(f"Device wait failed: {e}")
            raise
    
    def get_device_info(self) -> dict:
        """Get device information"""
        try:
            info = {}
            
            # Get device model
            model = self._run_adb_command(['shell', 'getprop', 'ro.product.model']).strip()
            info['model'] = model
            
            # Get Android version
            version = self._run_adb_command(['shell', 'getprop', 'ro.build.version.release']).strip()
            info['android_version'] = version
            
            # Get screen size
            size_output = self._run_adb_command(['shell', 'wm', 'size'])
            if 'Physical size:' in size_output:
                size_line = size_output.split('Physical size:')[1].strip()
                width, height = map(int, size_line.split('x'))
                info['screen_size'] = (width, height)
            
            logger.info(f"Device info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def install_apk(self, apk_path: str):
        """Install an APK file"""
        try:
            self._run_adb_command(['install', '-r', apk_path])
            logger.info(f"APK installed: {apk_path}")
            
        except Exception as e:
            logger.error(f"Failed to install APK {apk_path}: {e}")
            raise
    
    def clear_app_data(self, package_name: str):
        """Clear app data"""
        try:
            self._run_adb_command(['shell', 'pm', 'clear', package_name])
            logger.info(f"Cleared data for {package_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear app data for {package_name}: {e}")
            raise 