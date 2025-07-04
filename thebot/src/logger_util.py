"""
Unified Logging System with Performance Tracing
Purpose: Centralized logging with performance metrics and diagnostic capabilities
"""
import logging
import time
import json
import os
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import threading
from contextlib import contextmanager


class PerformanceTracker:
    """Track performance metrics for operations"""
    
    def __init__(self):
        self._metrics: Dict[str, list] = {}
        self._lock = threading.Lock()
        
    def record(self, operation: str, duration_ms: float) -> None:
        """Record a performance metric"""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append({
                'timestamp': datetime.now().isoformat(),
                'duration_ms': duration_ms
            })
            # Keep only last 1000 entries per operation
            if len(self._metrics[operation]) > 1000:
                self._metrics[operation] = self._metrics[operation][-1000:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        with self._lock:
            if operation not in self._metrics:
                return {}
            
            durations = [m['duration_ms'] for m in self._metrics[operation]]
            if not durations:
                return {}
                
            return {
                'count': len(durations),
                'mean_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'last_ms': durations[-1]
            }


class Logger:
    """Production-grade logger with performance tracking and structured logging"""
    
    _instances: Dict[str, 'Logger'] = {}
    _lock = threading.Lock()
    _performance_tracker = PerformanceTracker()
    
    def __new__(cls, name: str = __name__, **kwargs) -> 'Logger':
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(self, name: str = __name__, 
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 enable_performance: bool = True,
                 structured: bool = True):
        
        # Avoid re-initialization
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.name = name
        self.enable_performance = enable_performance
        self.structured = structured
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with color support
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
        # Performance log file
        if enable_performance:
            perf_file = log_file.replace('.log', '_perf.log') if log_file else 'performance.log'
            self.perf_handler = logging.FileHandler(perf_file)
            self.perf_handler.setFormatter(logging.Formatter('%(message)s'))
    
    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation performance"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            if self.enable_performance:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._performance_tracker.record(operation, duration_ms)
                self.logger.debug(f"[PERF] {operation}: {duration_ms:.2f}ms")
    
    def _log_structured(self, level: str, message: str, **kwargs) -> None:
        """Log in structured format"""
        if self.structured and kwargs:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'logger': self.name,
                'message': message,
                **kwargs
            }
            self.logger.log(
                getattr(logging, level.upper()),
                json.dumps(log_entry, default=str)
            )
        else:
            self.logger.log(getattr(logging, level.upper()), message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._log_structured('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._log_structured('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._log_structured('warning', message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message"""
        if exc_info:
            import traceback
            kwargs['traceback'] = traceback.format_exc()
        self._log_structured('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self._log_structured('critical', message, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all performance statistics"""
        operations = set()
        with self._performance_tracker._lock:
            operations = set(self._performance_tracker._metrics.keys())
        
        return {
            op: self._performance_tracker.get_stats(op)
            for op in operations
        }


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Setup global logging configuration"""
    config = config or {}
    
    # Set root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get('level', logging.INFO))
    
    # Create logs directory
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    # Setup default file handler
    log_file = log_dir / f"ai_assist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    root_logger.addHandler(file_handler)


# Singleton logger instance for backward compatibility
_default_logger: Optional[Logger] = None


def get_logger(name: str = __name__, **kwargs) -> Logger:
    """Get or create a logger instance"""
    return Logger(name, **kwargs)


# Backward compatibility
logger = get_logger(__name__)