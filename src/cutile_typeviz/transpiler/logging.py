import logging
import sys
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "TIME": "\033[90m",  # Gray
        "MODULE": "\033[94m",  # Blue
    }

    def format(self, record):
        # Create colored log level
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_level = f"{self.COLORS[levelname]}{self.COLORS['BOLD']}{levelname}{self.COLORS['RESET']}"
        else:
            colored_level = f"{levelname}"

        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        colored_time = f"{self.COLORS['TIME']}{timestamp}{self.COLORS['RESET']}"

        # Create module name
        module_name = record.module
        colored_module = f"{self.COLORS['MODULE']}{module_name}{self.COLORS['RESET']}"

        # Format the message
        message = record.getMessage()

        # Combine all parts
        formatted = f"{colored_time} [{colored_level} {colored_module}] {message}"

        return formatted


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger with colored output.

    Args:
        name: Logger name (defaults to calling module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "transpiler"

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Create and set formatter
        formatter = ColoredFormatter()
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


# Create a default logger for the transpiler module
transpiler_logger = get_logger("transpiler")
