#!/usr/bin/env python3
"""
ColoredFormatter Module

Provides a custom formatter for logging with colorized output based on log level.
This helps improve log readability in the console.
"""

import logging

# ANSI escape codes for colors
GREY = "\x1b[90m"
YELLOW = "\x1b[33m"
ORANGE = "\x1b[38;5;214m"  # Orange/Amber color
RED = "\x1b[31m"
BOLD_RED = "\x1b[31;1m"
RESET = "\x1b[0m"  # Reset to default color
GREEN = "\x1b[32m" # Added Green for INFO
BOLD = "\x1b[1m"   # Added Bold
CYAN = "\x1b[36m"   # Added Cyan
BRIGHT_WHITE = "\x1b[97m" # Added Bright White
PINK = "\x1b[95m"  # Added Pink
MAGENTA = "\x1b[35m"  # Added Magenta
CORAL_PINK = "\x1b[38;5;204m"  # Coral pink
SPRING_GREEN = "\x1b[38;2;0;250;154m"  # #00fa9a - vibrant spring green

class ColoredFormatter(logging.Formatter):
    """
    A custom formatter class for Python's logging module.

    Applies different colors to log messages based on their severity level,
    making it easier to scan logs visually in a terminal.
    """
    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.INFO: GREEN + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET
    }

    def format(self, record):
        """
        Format the specified record as text.

        Applies the appropriate color formatting based on log level.

        Args:
            record: A LogRecord instance containing all the information
                    pertinent to the event being logged.

        Returns:
            The formatted log message with appropriate colors.
        """
        # Special case for db logger INFO messages - use blue
        if record.name == "db" and record.levelno == logging.INFO:
            log_fmt = CYAN + "%(asctime)s [%(levelname)s] %(name)s: %(message)s" + RESET
        else:
            log_fmt = self.FORMATS.get(record.levelno, logging.BASIC_FORMAT)
        # Use a specific date format
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


