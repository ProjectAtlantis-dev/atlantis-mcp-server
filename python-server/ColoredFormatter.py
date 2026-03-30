#!/usr/bin/env python3
"""
ColoredFormatter Module

Provides a custom formatter for logging with colorized output based on log level.
This helps improve log readability in the console.

Includes ContextFilter which injects the current request ID (short) and shell path
into every log line so you can trace interleaved async operations.
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


class ContextFilter(logging.Filter):
    """Injects request_id (first 4 hex chars) and shell_path from contextvars into log records.

    Produces a compact prefix like [a3f2-4.8] or [----] when no context is set.
    Import is deferred to avoid circular imports with atlantis module.
    """

    def filter(self, record):
        try:
            import atlantis
            req_id = atlantis.get_request_id()
            shell = atlantis.get_shell_path()
        except Exception:
            req_id = None
            shell = None

        short_req = str(req_id)[:4] if req_id else "----"
        shell_str = shell if shell else "-"
        record.ctx = f"[{short_req}-{shell_str}]"
        return True


class ColoredFormatter(logging.Formatter):
    """
    A custom formatter class for Python's logging module.

    Applies different colors to log messages based on their severity level,
    making it easier to scan logs visually in a terminal.

    Includes %(ctx)s placeholder which shows [reqId-shell] context from ContextFilter.
    """
    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET,
        logging.INFO: GREEN + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET
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
        # Ensure ctx attribute exists (in case filter wasn't applied)
        if not hasattr(record, 'ctx'):
            record.ctx = "[----]"

        # Special case for db logger INFO messages - use blue
        if record.name == "db" and record.levelno == logging.INFO:
            log_fmt = CYAN + "%(asctime)s [%(levelname)s] %(name)s %(ctx)s: %(message)s" + RESET
        else:
            log_fmt = self.FORMATS.get(record.levelno, logging.BASIC_FORMAT)
        # Use a specific date format
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


