import os
import time
import base64
import logging
import ipdb
from openai import OpenAI
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

class ChessAssistant:
    def __init__(self):
        logging.info("Initializing ChessAssistant")
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logging.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key
        )
        logging.info("OpenAI client initialized")
        
        # Initialize Chrome driver with larger window
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver.set_window_size(1920, 1080)  # Set a large window size
        self.wait = WebDriverWait(self.driver, 10)
        self.actions = ActionChains(self.driver)
        logging.info("Chrome driver initialized")
        
        # Get the current game URL from the command line or use default
        self.game_url = "https://www.chess.com/play/computer"
        
        # Navigate to the game
        self.driver.get(self.game_url)
        logging.info(f"Navigated to chess game: {self.game_url}")
        
        # Wait for initial load
        time.sleep(2)
        
        # Handle cookie consent if present
        try:
            cookie_button = self.driver.find_element(By.ID, "onetrust-accept-btn-handler")
            if cookie_button.is_displayed():
                cookie_button.click()
                time.sleep(1)
        except:
            logging.info("No cookie consent needed")
        
        # Try to close any initial popups
        try:
            popup_selectors = [
                "[data-cy='modal-first-time-button']",
                ".close-button",
                ".modal-close-button",
                "[aria-label='Close']"
            ]
            for selector in popup_selectors:
                try:
                    popups = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for popup in popups:
                        if popup.is_displayed():
                            popup.click()
                            time.sleep(0.5)
                except:
                    continue
            logging.info("Handled potential popups")
        except Exception as e:
            logging.info(f"Error handling popups: {e}")
        
        # Wait for the board container and pieces
        logging.info("Waiting for board elements...")
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".board-layout-chessboard")))
        pieces = self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".piece")))
        # ipdb.set_trace()  # Debugger will stop here
        # logging.info(f"Pieces located: {pieces}")
        logging.info(f"Found {len(pieces)} pieces")
        
        # Detect which color we're playing as
        try:
            pieces = self.driver.find_elements(By.CSS_SELECTOR, ".piece")
            bottom_pieces = [p for p in pieces if "square-5" in p.get_attribute("class")]
            white_pieces = [p for p in bottom_pieces if "w" in p.get_attribute("class")]
            self.playing_as_white = len(white_pieces) > 0
            logging.info(f"Playing as {'White' if self.playing_as_white else 'Black'}")
        except Exception as e:
            logging.error(f"Error detecting player color: {e}")
            self.playing_as_white = True  # Default to white if detection fails
        
        # Wait for any initial animations
        time.sleep(2)
        
        logging.info("Chess board loaded successfully")


    def capture_board(self):
        """
        Take a screenshot of the chess board using Selenium.
        """
        try:
            logging.info("Capturing screenshot")
            
            # First try to find the board element
            board_selectors = [
                ".board-layout-chessboard",  # Main board container
                ".board",  # Actual board
                ".board-layout-main"  # Fallback to main layout
            ]
            
            board_element = None
            for selector in board_selectors:
                try:
                    board_element = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    logging.info(f"Found board with selector: {selector}")
                    break
                except:
                    continue
            
            if not board_element:
                raise Exception("Could not find chess board element")
            
            # Ensure pieces are loaded
            pieces = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".piece")))
            logging.info(f"Found {len(pieces)} pieces on board")
            
            # Scroll into view and ensure visibility
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", board_element)
            time.sleep(0.5)  # Brief pause for any scroll animations
            
            # Take screenshot
            output_path = "current_board.png"
            
            # Try different screenshot methods
            try:
                # Method 1: Direct element screenshot
                board_element.screenshot(output_path)
            except:
                try:
                    # Method 2: Get element region and crop full page screenshot
                    logging.info("Trying alternate screenshot method...")
                    self.driver.save_screenshot("full_page.png")
                    location = board_element.location
                    size = board_element.size
                    
                    from PIL import Image
                    full_img = Image.open("full_page.png")
                    left = location['x']
                    top = location['y']
                    right = location['x'] + size['width']
                    bottom = location['y'] + size['height']
                    
                    # Account for retina displays
                    scale = 2 if 'Mac' in self.driver.execute_script("return navigator.platform") else 1
                    board_img = full_img.crop((left * scale, top * scale, 
                                              right * scale, bottom * scale))
                    board_img.save(output_path)
                    os.remove("full_page.png")
                except Exception as e:
                    logging.error(f"Both screenshot methods failed: {e}")
                    raise
            
            if not os.path.exists(output_path):
                raise FileNotFoundError("Screenshot was not saved")
                
            logging.info(f"Screenshot saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error capturing screenshot: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise



    def get_move_advice(self, image_path):
        """
        Send the board image to OpenAI GPT-4V API and get move advice
        """
        try:
            # Open image file
            with open(image_path, "rb") as image_file:
                # Create chat completion with the image
                # Read and encode the image
                image_data = base64.b64encode(image_file.read()).decode()
                
                # Get list of available pieces and their positions
                our_color = 'w' if self.playing_as_white else 'b'
                opp_color = 'b' if self.playing_as_white else 'w'
                pieces = self.driver.find_elements(By.CSS_SELECTOR, f".piece")
                our_pieces = []
                our_positions = []
                opp_positions = []
                piece_counts = {'pawn': 0, 'knight': 0, 'bishop': 0, 'rook': 0, 'queen': 0, 'king': 0}
                
                piece_map = {
                    'wp': 'White pawn', 'wn': 'White knight', 'wb': 'White bishop',
                    'wr': 'White rook', 'wq': 'White queen', 'wk': 'White king',
                    'bp': 'Black pawn', 'bn': 'Black knight', 'bb': 'Black bishop',
                    'br': 'Black rook', 'bq': 'Black queen', 'bk': 'Black king'
                }
                
                piece_type_map = {
                    'wp': 'pawn', 'wn': 'knight', 'wb': 'bishop',
                    'wr': 'rook', 'wq': 'queen', 'wk': 'king',
                    'bp': 'pawn', 'bn': 'knight', 'bb': 'bishop',
                    'br': 'rook', 'bq': 'queen', 'bk': 'king'
                }
                
                for piece in pieces:
                    classes = piece.get_attribute('class').split()
                    # Get piece type and position
                    piece_type = next((c for c in classes if len(c) == 2 and (c.startswith('w') or c.startswith('b'))), '')
                    square_class = next((c for c in classes if c.startswith('square-')), '')
                    
                    if piece_type and square_class:
                        # Convert square number to algebraic notation
                        square_num = square_class[7:]
                        file_idx = int(square_num[0]) - 1
                        rank = int(square_num[1])
                        files = 'abcdefgh'
                        position = f"{files[file_idx]}{rank}"
                        
                        # Add to appropriate list
                        piece_info = f"{piece_map.get(piece_type, piece_type)} at {position}"
                        if piece_type.startswith(our_color):
                            our_pieces.append(piece_map.get(piece_type, piece_type))
                            our_positions.append(piece_info)
                            # Count our pieces
                            if piece_type in piece_type_map:
                                piece_counts[piece_type_map[piece_type]] += 1
                        else:
                            opp_positions.append(piece_info)
                
                # Create piece availability summary
                piece_summary = []
                notation_map = {'pawn': '', 'knight': 'N', 'bishop': 'B', 'rook': 'R', 'queen': 'Q', 'king': 'K'}
                for piece_type, count in piece_counts.items():
                    if count > 0:
                        piece_summary.append(f"{piece_type.title()} ({notation_map[piece_type]}): {count}")
                
                our_pieces_str = ', '.join(our_pieces)
                our_positions_str = ', '.join(our_positions)
                opp_positions_str = ', '.join(opp_positions)
                
                logging.info(f"Our pieces: {our_pieces_str}")
                logging.info(f"Our positions: {our_positions_str}")
                logging.info(f"Opponent positions: {opp_positions_str}")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a chess expert analyzing a position where we are playing as {'WHITE' if self.playing_as_white else 'BLACK'}. \n\nAvailable Pieces and Notation:\n{', '.join(piece_summary)}\n\nPiece Positions:\n{our_positions_str}\n\nOpponent pieces for reference:\n{opp_positions_str}\n\nLook at this screenshot of a chess website, analyze the position, and suggest the best LEGAL move for {'WHITE' if self.playing_as_white else 'BLACK'} using ONLY the pieces and notation listed above.\n\nMove Format Rules:\n1. Use standard algebraic notation\n2. For pawn moves, just write the destination square (e.g. 'e4' not 'e2-e4')\n3. For piece moves, write the piece letter followed by destination (e.g. 'Nf3')\n4. For captures, include 'x' (e.g. 'Nxe4' or 'exd5')\n5. DO NOT use explicit source squares with dashes (e.g. NOT 'e2-e4')\n\nPiece Movement Rules:\n1. Pawns can only move forward (up for White, down for Black)\n2. Pawns can only capture diagonally forward\n3. Knights move in L-shapes\n4. Bishops move diagonally\n5. Rooks move horizontally or vertically\n6. Kings move one square in any direction\n\nIMPORTANT: You can ONLY use pieces with non-zero counts in the 'Available Pieces' list above!\n\nYour response must be in this format: 'MOVE: <move in standard algebraic notation> | EXPLANATION: <brief explanation>'. If you cannot find a chess board, respond with 'NO_BOARD_FOUND'."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Find the chess board in this screenshot and suggest the best move:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error getting move advice: {str(e)}")
            return None

    def get_square_position(self, square):
        """
        Convert chess square (e.g., 'e4') to screen coordinates
        """
        # Find the primary monitor
        primary_monitor = None
        for monitor in self.monitors:
            if monitor.x == 0 and monitor.y == 0:
                primary_monitor = monitor
                break
        
        if not primary_monitor:
            raise ValueError("Could not detect primary monitor")
        
        # Take a screenshot to detect the board
        screenshot = pyautogui.screenshot()
        board_width = screenshot.width * 0.3  # Assuming chess board is about 30% of screen width
        square_size = board_width / 8
        
        # Calculate board position (centered on screen)
        board_left = (screenshot.width - board_width) / 2
        board_top = (screenshot.height - board_width) / 2
        
        # Convert algebraic notation to coordinates
        file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        rank_map = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
        
        file_idx = file_map[square[0].lower()]
        rank_idx = rank_map[square[1]]
        
        # Calculate square center position
        x = board_left + (file_idx * square_size) + (square_size / 2)
        y = board_top + (rank_idx * square_size) + (square_size / 2)
        
        return int(x), int(y)
    
    def get_source_square(self, move):
        """
        Get the source square for a move
        """
        # Default source squares for pieces in starting position
        piece_sources = {
            'N': ['b1', 'g1'],  # Knights
            'B': ['c1', 'f1'],  # Bishops
            'R': ['a1', 'h1'],  # Rooks
            'Q': ['d1'],        # Queen
            'K': ['e1']         # King
        }
        
        # For pawn moves
        if move[0].islower():
            file = move[0]
            if len(move) > 1 and move[1].isdigit():
                return f"{file}2"  # White pawns start from rank 2
            
        # For piece moves
        elif move[0].isupper():
            piece = move[0]
            if piece in piece_sources:
                return piece_sources[piece][0]  # Use first possible source
        
        return None

    def get_move_notation(self, move):
        """
        Convert move to source-target notation (e.g., 'e2e4')
        """
        try:
            move = move.strip()
            
            # Get source square
            if move[0].islower():  # Pawn move
                file = move[0]
                if len(move) > 1 and move[1].isdigit():
                    source = f"{file}2"  # White pawns start from rank 2
                    target = move
            elif move[0].isupper():  # Piece move
                piece_starts = {
                    'N': ['b1', 'g1'],  # Knights
                    'B': ['c1', 'f1'],  # Bishops
                    'R': ['a1', 'h1'],  # Rooks
                    'Q': ['d1'],        # Queen
                    'K': ['e1']         # King
                }
                piece = move[0]
                if 'x' in move:
                    target = move[move.index('x')+1:]
                else:
                    target = move[1:]
                source = piece_starts.get(piece, [''])[0]
            
            if source and target:
                return source + target
            return None
            
        except Exception as e:
            logging.error(f"Error converting move: {str(e)}")
            return None

    def find_board_in_screenshot(self):
        """
        Find the chess board in the screenshot using color detection
        """
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for white and dark squares
        # Adjust these ranges based on your chess board colors
        lower_white = np.array([0, 0, 200])  # Very light colors
        upper_white = np.array([180, 30, 255])
        
        lower_dark = np.array([0, 0, 50])   # Dark colors
        upper_dark = np.array([180, 30, 150])
        
        # Create masks for white and dark squares
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, dark_mask)
        
        # Save debug image
        cv2.imwrite('debug_mask.jpg', combined_mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
        
        # Find the largest component that's roughly square
        max_area = 0
        board_rect = None
        
        for i in range(1, num_labels):  # Skip background (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Check if it's roughly square and large enough
            if area > 10000 and 0.8 <= w/h <= 1.2:
                if area > max_area:
                    max_area = area
                    board_rect = (x, y, w, h)
        
        # Draw debug visualization
        debug_img = screenshot_np.copy()
        if board_rect:
            x, y, w, h = board_rect
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw grid
            for i in range(9):
                # Vertical lines
                x_line = x + (w * i // 8)
                cv2.line(debug_img, (x_line, y), (x_line, y+h), (255, 0, 0), 1)
                # Horizontal lines
                y_line = y + (h * i // 8)
                cv2.line(debug_img, (x, y_line), (x+w, y_line), (255, 0, 0), 1)
        cv2.imwrite('debug_board.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        if board_rect:
            logging.info(f"Found chess board at {board_rect}")
            return board_rect
        
        logging.warning("No chess board found in screenshot")
        return None

    def get_square_coordinates(self, square, debug_img=None, is_source=False):
        """
        Get screen coordinates for a chess square
        """
        # Find the board
        board_rect = self.find_board_in_screenshot()
        if not board_rect:
            raise ValueError("Could not find chess board")
            
        x, y, w, h = board_rect
        square_size = w // 8  # Board is 8x8
        
        # Convert algebraic notation to coordinates
        file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        rank_map = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
        
        file_idx = file_map[square[0].lower()]
        rank_idx = rank_map[square[1]]
        
        # Calculate square position (top-left corner)
        square_left = x + (file_idx * square_size)
        square_top = y + (rank_idx * square_size)
        
        # Calculate piece position (adjust offset based on whether it's source or target)
        piece_x = square_left + (square_size // 2)  # Center horizontally
        if is_source:
            piece_y = square_top + int(square_size * 0.7)  # For source square, click further down
        else:
            piece_y = square_top + int(square_size * 0.5)  # For target square, drop at center
        
        # Add safety bounds
        piece_x = max(0, min(piece_x, pyautogui.size().width))
        piece_y = max(0, min(piece_y, pyautogui.size().height))
        
        # Draw debug visualization if image provided
        if debug_img is not None:
            # Draw square outline
            cv2.rectangle(debug_img, 
                         (square_left, square_top), 
                         (square_left + square_size, square_top + square_size), 
                         (0, 255, 0), 1)
            # Draw click point
            cv2.circle(debug_img, (piece_x, piece_y), 3, (0, 0, 255), -1)
            # Add coordinate label
            cv2.putText(debug_img, square, (square_left + 5, square_top + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return piece_x, piece_y

    def execute_move(self, move):
        """Execute a chess move using Selenium WebDriver"""
        try:
            # Clean up move string - remove quotes and extra whitespace
            move = move.strip().strip('\'"')
            move = move.strip()
            logging.info(f"Starting to execute move: {move}")
            
            # Remove move numbers (e.g., "3. Nf3" or "3.Nf3" -> "Nf3")
            if '.' in move:
                # Try to find a number at the start followed by a dot
                import re
                if re.match(r'^\d+\.\s*', move):
                    move = re.sub(r'^\d+\.\s*', '', move)
                    logging.info(f"Removed move number, executing: {move}")
            
            # Convert algebraic notation to chess.com's coordinate system
            def to_chess_com_coords(square):
                if not square or len(square) != 2:
                    logging.warning(f"Invalid square format: {square}")
                    return None
                file_map = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8'}
                rank_map = {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8'}
                file, rank = square[0].lower(), square[1]
                if file in file_map and rank in rank_map:
                    result = f"{file_map[file]}{rank_map[rank]}"
                    logging.info(f"Converted {square} to chess.com coordinates: {result}")
                    return result
                logging.warning(f"Could not convert square {square} to chess.com coordinates")
                return None

            # Handle castling moves
            if move in ['O-O', 'O-O-O']:
                logging.info(f"Detected castling move: {move}")
                # For white castling
                if self.playing_as_white:
                    king_source = 'e1'
                    king_target = 'g1' if move == 'O-O' else 'c1'
                else:  # For black castling
                    king_source = 'e8'
                    king_target = 'g8' if move == 'O-O' else 'c8'
                
                logging.info(f"Castling: Moving king from {king_source} to {king_target}")
                source_square = king_source
                target_square = king_target
                
                # Set up king piece selector
                piece_class = 'w' if self.playing_as_white else 'b'
                piece_type = 'k'  # King piece
                piece_selector = f".piece.{piece_class}{piece_type}"
                logging.info(f"Looking for king with selector: {piece_selector}")
            
            # For pawn moves (e.g., 'e4' or 'dxc5')
            if move[0].islower():
                logging.info(f"Detected pawn move: {move}")
                
                # Handle pawn captures (e.g., 'dxc5')
                if 'x' in move:
                    logging.info("Detected pawn capture")
                    source_file = move[0]
                    target_file = move[2]
                    target_rank = int(move[3])
                    target_square = f"{target_file}{target_rank}"
                    file = source_file  # For finding source pawn
                else:
                    target_square = move
                    file = move[0]
                    target_rank = int(move[1])
                
                # Find the source pawn
                piece_class = 'w' if self.playing_as_white else 'b'
                piece_selector = f".piece.{piece_class}p"
                logging.info(f"Looking for pawns with selector: {piece_selector}")
                
                pawns = self.driver.find_elements(By.CSS_SELECTOR, piece_selector)
                source_square = None
                
                for pawn in pawns:
                    classes = pawn.get_attribute('class').split()
                    for class_name in classes:
                        if class_name.startswith('square-'):
                            current_square = class_name[7:]  # Remove 'square-' prefix
                            # Convert to algebraic notation
                            file_idx = int(current_square[0]) - 1
                            rank = int(current_square[1])
                            current_file = chr(97 + file_idx)
                            
                            # Check if this pawn is in the correct file
                            if current_file == file:
                                source_square = f"{current_file}{rank}"
                                logging.info(f"Found pawn at {source_square}")
                                break
                    if source_square:
                        break
                        
                if not source_square:
                    raise ValueError(f"Could not find pawn in file {file}")
                    
                logging.info(f"Using source square: {source_square}, target square: {target_square}")
            
            # For piece moves (e.g., 'Nf3', 'Nbd2')
            elif move[0].isupper():
                logging.info(f"Detected piece move: {move}")
                piece = move[0]
                
                # Handle piece disambiguation (e.g., 'Nbd2')
                disambiguation = ''
                # Only treat as disambiguation if there's more than 3 characters and second char isn't part of target
                if len(move) > 3 and move[1].islower() and move[2].islower() and not move[1] in 'x':
                    disambiguation = move[1]
                    move = move[0] + move[2:]
                    logging.info(f"Detected disambiguation: {disambiguation}")
                
                if 'x' in move:
                    target_square = move[move.index('x')+1:]
                    logging.info(f"Capture move detected, target square: {target_square}")
                else:
                    target_square = move[1:] if not disambiguation else move[2:]
                    logging.info(f"Regular move detected, target square: {target_square}")
                
                # Remove check/mate symbols from target square
                target_square = target_square.rstrip('+#')
                logging.info(f"Cleaned target square: {target_square}")
                
                # Find the source piece
                piece_class = 'w' if self.playing_as_white else 'b'
                piece_type = piece.lower()
                piece_selector = f".piece.{piece_class}{piece_type}"
                logging.info(f"Looking for piece with selector: {piece_selector}")
                
                pieces = self.driver.find_elements(By.CSS_SELECTOR, piece_selector)
                logging.info(f"Found {len(pieces)} matching pieces")
                if len(pieces) == 0:
                    piece_name = {'q': 'Queen', 'k': 'King', 'b': 'Bishop', 'n': 'Knight', 'r': 'Rook', 'p': 'Pawn'}[piece_type]
                    color = 'White' if self.playing_as_white else 'Black'
                    raise ValueError(f"Cannot find {color} {piece_name} on the board. The piece may have been captured.")
                source_square = None
                
                # Get all possible source squares
                possible_sources = []
                for piece_elem in pieces:
                    classes = piece_elem.get_attribute('class').split()
                    logging.info(f"Checking piece with classes: {classes}")
                    for class_name in classes:
                        if class_name.startswith('square-'):
                            current_square = class_name[7:]  # Remove 'square-' prefix
                            
                            # Convert current_square to file and rank
                            current_file = int(current_square[0])
                            current_rank = int(current_square[1])
                            
                            # Convert target square from algebraic to numeric
                            target_file = ord(target_square[0].lower()) - ord('a') + 1
                            target_rank = int(target_square[1])
                            
                            # For bishop moves, check if it's a valid diagonal move
                            if piece_type == 'b':
                                file_diff = abs(target_file - current_file)
                                rank_diff = abs(target_rank - current_rank)
                                if file_diff == rank_diff:  # Valid diagonal move
                                    possible_sources.append((current_file, current_rank))
                            logging.info(f"Raw square from class: {current_square}")
                            # Convert to algebraic notation
                            file_idx = (int(current_square[0]) - 1)
                            rank = int(current_square[1])
                            # On chess.com, rank 1 is actually 1, not needing 8-rank conversion
                            algebraic_square = f"{chr(97 + file_idx)}{rank}"
                            logging.info(f"Converted coordinates: file_idx={file_idx} ({chr(97 + file_idx)}), rank={rank}")
                            logging.info(f"Found piece at {algebraic_square}")
                            
                            # For knights, check if the move is legal
                            if piece_type == 'n':
                                # Parse target coordinates
                                target_file = ord(target_square[0]) - ord('a')
                                target_rank = int(target_square[1])
                                # Calculate differences
                                file_diff = abs(target_file - file_idx)
                                rank_diff = abs(target_rank - rank)
                                # Knight moves in L-shape: 2 squares in one direction and 1 in other
                                is_valid_knight_move = (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)
                                logging.info(f"Knight move validation: file_diff={file_diff}, rank_diff={rank_diff}, valid={is_valid_knight_move}")
                                if is_valid_knight_move:
                                    source_square = algebraic_square
                            # For bishops, check if the move is legal
                            elif piece_type == 'b':
                                # Parse target coordinates
                                target_file = ord(target_square[0]) - ord('a')
                                target_rank = int(target_square[1])
                                # Calculate differences
                                file_diff = abs(target_file - file_idx)
                                rank_diff = abs(target_rank - rank)
                                # Bishop moves diagonally: equal movement in files and ranks
                                is_valid_bishop_move = file_diff == rank_diff
                                logging.info(f"Bishop move validation: file_diff={file_diff}, rank_diff={rank_diff}, valid={is_valid_bishop_move}")
                                if is_valid_bishop_move:
                                    source_square = algebraic_square
                                    logging.info(f"Found valid bishop move from {algebraic_square}")
                                    break
                            else:
                                # For other pieces, handle disambiguation as before
                                if disambiguation:
                                    if disambiguation == algebraic_square[0]:  # File disambiguation
                                        source_square = algebraic_square
                                        break
                                else:
                                    possible_sources.append(algebraic_square)
                    if source_square:
                        break
                
                # If no valid knight move found or no disambiguation for other pieces, use first source
                if not source_square and possible_sources:
                    source_square = possible_sources[0]
                    logging.info(f"Selected source square: {source_square}")
                        
                if not source_square:
                    raise ValueError(f"Could not find source piece for move {move}")
            
            # Convert squares to chess.com coordinates
            logging.info(f"Converting squares to chess.com coordinates...")
            logging.info(f"Source square: {source_square}, Target square: {target_square}")
            chess_com_source = to_chess_com_coords(source_square)
            chess_com_target = to_chess_com_coords(target_square)
            
            if not chess_com_source or not chess_com_target:
                error_msg = f"Invalid squares: source={source_square}({chess_com_source}), target={target_square}({chess_com_target})"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            logging.info(f"Moving from {source_square}({chess_com_source}) to {target_square}({chess_com_target})")


            try:
                # Find source element
                logging.info("Looking for source element...")
                source_selector = f".piece.square-{chess_com_source}"
                source_element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, source_selector)))
                logging.info(f"Found source element: {source_element.get_attribute('class')}")
                
                # Get board dimensions
                board = source_element.find_element(By.XPATH, "./..")
                board_rect = board.rect
                square_size = board_rect['width'] / 8
                logging.info(f"Board dimensions: {board_rect}, square size: {square_size}")
                
                # Calculate squares to move (target - source)
                source_rank = int(chess_com_source[-1])
                target_rank = int(chess_com_target[-1])
                rank_diff = source_rank - target_rank
                
                source_file = int(chess_com_source[-2])
                target_file = int(chess_com_target[-2])
                file_diff = target_file - source_file  # Changed to target - source for correct direction
                
                logging.info(f"Moving {file_diff} files and {rank_diff} ranks")
                
                # Create action chain for the move
                # Calculate pixel offsets
                x_offset = file_diff * square_size  # Positive because file_diff is now target - source
                y_offset = rank_diff * square_size    # Positive because larger rank number means down
                
                logging.info(f"Moving by offset: x={x_offset}, y={y_offset} pixels")
                
                # Create action chain and perform the move
                actions = ActionChains(self.driver)
                actions.move_to_element(source_element)
                actions.click_and_hold()
                actions.pause(0.2)  # Small pause after clicking
                actions.move_by_offset(x_offset, y_offset)
                actions.pause(0.2)  # Small pause before release
                actions.release()
                
                logging.info("Executing action chain...")
                actions.perform()
                
                # Wait for move to complete
                time.sleep(1)
                
                logging.info("Move execution completed successfully")
                return True
                
            except Exception as e:
                logging.error(f"Error executing move: {str(e)}")
                logging.error(f"Error type: {type(e).__name__}")
                logging.error(f"Board dimensions: {board_rect}")
                logging.error(f"Calculated offsets: x={-file_diff * square_size}, y={rank_diff * square_size}")
                raise
            
            # Get element locations
            source_loc = source_piece.location
            target_loc = target_elem.location
            
            # Execute move with more precise mouse control
            actions = ActionChains(self.driver)
            actions.move_to_element(source_piece)
            actions.click_and_hold()
            time.sleep(0.3)
            
            # Move to target with offset to center
            target_size = target_elem.size
            actions.move_to_element_with_offset(
                target_elem,
                target_size['width'] // 2,
                target_size['height'] // 2
            )
            time.sleep(0.2)
            actions.release()
            actions.perform()
            
            # Wait for move animation
            time.sleep(0.5)
            
            logging.info(f"Executed move {move} ({source_square} → {target_square})")
            print(f"\nExecuted: {move} ({source_square} → {target_square})")
            
        except Exception as e:
            logging.error(f"Error executing move: {str(e)}")
            print(f"\nFailed to execute move: {str(e)}")

    def run(self):
        """Main loop for the chess assistant"""
        try:
            while True:
                try:
                    input("Press Enter for next move...")
                    
                    # Capture the current board state
                    logging.info("Capturing screenshot")
                    board_image = self.capture_board()
                    
                    print("Analyzing...", end="", flush=True)
                    logging.info("Getting move advice from GPT-4 Vision")
                    
                    # Get move advice from GPT-4 Vision
                    move_advice = self.get_move_advice(board_image)
                    if not move_advice:
                        logging.warning("No move advice received")
                        print("Failed to get move advice. Please try again.")
                        continue
                    
                    if move_advice == "NO_BOARD_FOUND":
                        logging.warning("No chess board found in screenshot")
                        print("\nCould not find a chess board in the screenshot. Please make sure the chess website is visible.")
                        continue
                    
                    # Extract just the move from the response
                    move = move_advice.split('|')[0].replace('MOVE:', '').strip()
                    logging.info(f"Move advice received: {move_advice}")
                    print(f"\n> {move}")
                    
                    # Execute the move
                    self.execute_move(move)
                    
                    print("\nPress Enter for next move...")
                    
                except ValueError as ve:
                    logging.error(f"Value error: {str(ve)}")
                    print(f"\nError: {str(ve)}")
                    print("Please try again...")
                    continue
                except Exception as e:
                    logging.error(f"Error during analysis: {str(e)}")
                    print(f"\nAn error occurred: {str(e)}")
                    print("Please try again...")
                    continue
                    
        except KeyboardInterrupt:
            logging.info("Chess assistant stopped by user")
            print("\nChess assistant stopped.")
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            print(f"\nA fatal error occurred: {str(e)}")
            print("Chess assistant stopped.")

if __name__ == "__main__":
    assistant = ChessAssistant()
    assistant.run()
