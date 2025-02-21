# ChessBot - AI Chess Assistant

An intelligent chess assistant that plays chess on chess.com using computer vision and AI techniques. The bot leverages OpenAI's GPT-4 Vision API for move analysis and Selenium WebDriver for browser automation.

## Features

- 🎯 **Automated Chess Play**: Plays chess on chess.com against the computer
- 🧠 **AI-Powered Analysis**: Uses GPT-4 Vision for intelligent move selection
- 👁️ **Computer Vision**: Captures and analyzes the chess board state
- ♟️ **Move Validation**: Comprehensive validation of chess piece movements
- 🔄 **Real-time Adaptation**: Dynamically tracks piece positions and availability
- 🎮 **Browser Automation**: Seamless interaction with chess.com interface

## Prerequisites

- Python 3.11 or higher
- Chrome browser
- OpenAI API key
- Active internet connection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shosseini811/ChessBot.git
cd ChessBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp example.env .env
# Edit .env and add your OpenAI API key
```

## Usage

1. Start the chess assistant:
```bash
python chess_assistant.py
```

2. The bot will:
   - Open chess.com in Chrome
   - Navigate to computer play
   - Start a new game
   - Begin making moves automatically

## Technical Details

### Components

1. **Move Analysis**
   - GPT-4 Vision API analyzes board state
   - Piece availability tracking
   - Legal move validation
   - Standard algebraic notation parsing

2. **Board Interaction**
   - Selenium WebDriver for browser control
   - Dynamic coordinate translation
   - Piece movement validation
   - Error handling and recovery

3. **Move Execution**
   - Support for all piece types
   - Special move handling (castling)
   - Capture move validation
   - Position coordinate mapping

### Move Types Supported

- ♙ Pawn moves and captures
- ♘ Knight L-shaped moves
- ♗ Bishop diagonal moves
- ♖ Rook horizontal/vertical moves
- ♔ King single square moves
- ⚔️ Piece captures
- 🏰 Castling (both kingside and queenside)

## Configuration

The bot can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- Additional configuration options in `.env`

## Development

### Project Structure

```
ChessBot/
├── chess_assistant.py   # Main bot implementation
├── requirements.txt     # Python dependencies
├── example.env         # Environment variable template
├── .env               # Your configuration (git-ignored)
└── README.md          # This documentation
```

### Key Classes

- `ChessAssistant`: Main bot class
  - `execute_move()`: Move execution logic
  - `get_move_advice()`: GPT-4 Vision integration
  - `move_piece()`: Browser interaction

## Recent Improvements

1. **Enhanced Move Validation**
   - Piece counting and availability tracking
   - Legal move enforcement
   - Improved coordinate translation

2. **GPT-4 Vision Integration**
   - Explicit piece availability information
   - Standard algebraic notation enforcement
   - Move format validation

3. **Error Handling**
   - Robust piece selection
   - Move validation checks
   - Recovery mechanisms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 Vision API
- Selenium WebDriver team
- Chess.com for the platform

## Author

Soheil Hosseini (@shosseini811)