Please analyze this codebase and create an AGENT.md file containing:
1. Build/lint/test commands - especially for running a single test
2. Architecture and codebase structure information, including important subprojects, internal APIs, databases, etc.
3. Code style guidelines, including imports, conventions, formatting, types, naming conventions, error handling, etc.

The file you create will be given to agentic coding tools (such as yourself) that operate in this repository. Make it about 20 lines long.

If there are Cursor rules (in .cursor/rules/ or .cursorrules), Claude rules (CLAUDE.md), Windsurf rules (.windsurfrules), Cline rules (.clinerules), Goose rules (.goosehints), or Copilot rules (in .github/copilot-instructions.md), make sure to include them. Also, first check if there is a AGENT.md file, and if so, update it instead of overwriting it.
# Agent Development Guidelines

## Build/Run Commands
- **Main application**: `python main.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Test model**: `python thebot/src/test_model.py`
- **Train YOLO**: `python thebot/scripts/train_yolo.py`

## Architecture
- **Main modules**: thebot/src/ contains core AI targeting system
- **Arduino integration**: Hardware control via serial communication (COM5)
- **YOLO detection**: Real-time object detection using YOLOv8
- **Screen capture**: High-performance capture using bettercam library
- **GUI**: Tkinter-based control interface

## Code Style
- **Imports**: Standard library first, third-party, then local imports
- **Logging**: Use centralized Logger from utils.py
- **Error handling**: Comprehensive try-catch blocks for hardware/model failures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Type hints**: Encouraged for function parameters and returns
- **Documentation**: Docstrings for all public methods and classes

## Critical Dependencies
- bettercam: High-performance screen capture
- ultralytics: YOLOv8 model framework
- opencv-python: Computer vision operations
- pyserial: Arduino communication
- tkinter: GUI framework (built-in)
