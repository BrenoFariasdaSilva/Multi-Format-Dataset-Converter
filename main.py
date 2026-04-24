"""
================================================================================
Multi-Format Dataset Converter (main.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-05-31

Short Description:
    Command-line utility that discovers datasets (ARFF, CSV, Parquet, TXT)
    under an input directory, applies lightweight structural cleaning to
    text-based formats, loads them into pandas DataFrames, and writes
    converted outputs (ARFF, CSV, Parquet, TXT) to a mirrored `Output`
    directory structure.

Defaults & Behavior:
    - Default input directory: ./Input
    - Default output directory: ./Output
    - Supported input formats: .arff, .csv, .parquet, .txt
    - Cleaning: minimal whitespace/domain-list normalization for ARFF/CSV/TXT
    - Parquet files are rewritten via `fastparquet` for consistency
    - Conversion preserves directory hierarchy relative to `Input`
    - Optional completion sound (platform-dependent)

Usage:
    - Run interactively:
        python3 main.py
    - Or pass CLI args: `-i/--input`, `-o/--output`, `-f/--formats`, `-v/--verbose`

Dependencies (non-exhaustive):
    - Python 3.8+
    - pandas, fastparquet, scipy, liac-arff (arff), colorama, tqdm

Notes and Caveats:
    - The converter performs pragmatic cleaning only; do not rely on it to
        fully sanitize malformed CSVs.
    - The script uses both `scipy` and `liac-arff` as fallbacks for ARFF.
    - Disk-space verifies are performed before writing outputs.
    - The module expects UTF-8 encoded text files.

TODOs (short):
    - Add unit tests and more robust CSV parsing
    - Add optional parallel conversion mode for large workloads
    - Provide more granular CLI control for cleaning rules
"""


import arff  # Liac-arff, used to save ARFF files
import argparse  # For parsing command-line arguments
import atexit  # For playing a sound when the program finishes
import datetime  # For timestamping
import io  # For in-memory file operations
import numpy as np  # For NaN representation and numeric coercion
import os  # For running commands in the terminal
import pandas as pd  # For handling CSV and TXT file formats
import platform  # For getting the operating system name
import shutil  # For analyzing disk usage
import sys  # For system-specific parameters and functions
import yaml  # For loading configuration from YAML file
from colorama import Style  # For coloring the terminal output
from fastparquet import ParquetFile  # For handling Parquet file format
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from scapy.all import PcapReader  # For memory-efficient PCAP reading using Scapy
from scipy.io import arff as scipy_arff  # Used to read ARFF files
from tqdm import tqdm  # For showing a progress bar
from typing import Optional  # For optional typing hints


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants:
DEFAULTS = None  # Will hold the default configuration loaded from YAML or hardcoded defaults


# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"}  # Sound play command
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # Notification sound path

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,  # Set to True to play a sound when the program finishes
}


# Functions Definitions:


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculate the execution time and return a human-readable string.

    :param start_time: The start time or duration value (datetime, timedelta, or numeric seconds).
    :param finish_time: Optional finish time; if None, start_time is treated as the total duration.
    :return: Human-readable execution time string formatted as days, hours, minutes, and seconds.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
            total_seconds = to_seconds(start_time)  # Try to convert provided value to seconds
            if total_seconds is None:  # Conversion failed
                try:  # Attempt numeric coercion
                    total_seconds = float(start_time)  # Attempt numeric coercion
                except Exception:
                    total_seconds = 0.0  # Fallback to zero
        else:  # Two-argument mode: Compute difference finish_time - start_time
            st = to_seconds(start_time)  # Convert start to seconds if possible
            ft = to_seconds(finish_time)  # Convert finish to seconds if possible
            if st is not None and ft is not None:  # Both converted successfully
                total_seconds = ft - st  # Direct numeric subtraction
            else:  # Fallback to other methods
                try:  # Attempt to subtract (works for datetimes/timedeltas)
                    delta = finish_time - start_time  # Try subtracting (works for datetimes/timedeltas)
                    total_seconds = float(delta.total_seconds())  # Get seconds from the resulting timedelta
                except Exception:  # Subtraction failed
                    try:  # Final attempt: Numeric coercion
                        total_seconds = float(finish_time) - float(start_time)  # Final numeric coercion attempt
                    except Exception:  # Numeric coercion failed
                        total_seconds = 0.0  # Fallback to zero on failure

        if total_seconds is None:  # Ensure a numeric value
            total_seconds = 0.0  # Default to zero
        if total_seconds < 0:  # Normalize negative durations
            total_seconds = abs(total_seconds)  # Use absolute value

        days = int(total_seconds // 86400)  # Compute full days
        hours = int((total_seconds % 86400) // 3600)  # Compute remaining hours
        minutes = int((total_seconds % 3600) // 60)  # Compute remaining minutes
        seconds = int(total_seconds % 60)  # Compute remaining seconds

        if days > 0:  # Include days when present
            return f"{days}d {hours}h {minutes}m {seconds}s"  # Return formatted days+hours+minutes+seconds
        if hours > 0:  # Include hours when present
            return f"{hours}h {minutes}m {seconds}s"  # Return formatted hours+minutes+seconds
        if minutes > 0:  # Include minutes when present
            return f"{minutes}m {seconds}s"  # Return formatted minutes+seconds
        return f"{seconds}s"  # Fallback: only seconds
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def play_sound():
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing

        if verify_filepath_exists(SOUND_FILE):  # If the sound file exists
            if current_os in SOUND_COMMANDS:  # If the platform.system() is in the SOUND_COMMANDS dictionary
                os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")  # Play the sound
            else:  # If the platform.system() is not in the SOUND_COMMANDS dictionary
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
            )
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def main():
    """
    Main function.

    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Multi-Format Dataset Converter{BackgroundColors.GREEN}!{Style.RESET_ALL}\n"
        )  # Output the Welcome message
        start_time = datetime.datetime.now()  # Get the start time of the program
        
        initialize_defaults()  # Initialize DEFAULTS from get_default_config() and config.yaml
        
        args = parse_cli_arguments()  # Parse CLI arguments

        resolved_low_memory = resolve_low_memory(args, DEFAULTS)  # Resolve low_memory setting from CLI arguments and configuration
        
        try:  # Attempt to store resolved low_memory back into DEFAULTS for use in other functions that read from it, but don't fail if this doesn't work (e.g., if DEFAULTS is not a dict or is immutable)
            if isinstance(DEFAULTS, dict):  # Only attempt to set if DEFAULTS is a dict to avoid errors
                DEFAULTS.setdefault("dataset_converter", {})  # Ensure the 'dataset_converter' section exists in DEFAULTS
                DEFAULTS["dataset_converter"]["low_memory"] = resolved_low_memory  # Store the resolved low_memory value in DEFAULTS for use in other functions that read from it
        except Exception:  # Catch any exception that occurs during this storage attempt to avoid crashing the program, since the resolved value is already stored in the local variable and can be used directly by functions that need it without relying on DEFAULTS
            pass  # Silently ignore any errors that occur while trying to store the resolved low_memory value back into DEFAULTS, since this is a best-effort attempt to make it available for functions that read from DEFAULTS, but the program can still function correctly using the local variable without this storage if necessary

        input_paths, output_path = resolve_io_paths(args)  # Resolve and validate paths, returning list of inputs
        if input_paths is None or output_path is None:  # If either resolution failed
            return  # Exit early when inputs/outputs are invalid

        configure_verbose_mode(args)  # Enable verbose mode if requested

        configure_input_output_formats(args)  # Update DEFAULTS with input and output file formats from CLI

        output_formats = args.output_file_formats if args.output_file_formats else (args.formats if args.formats else None)  # Preference: --output-file-formats > --formats > config fallback

        for input_path in input_paths:  # Iterate through each resolved input path
            batch_convert(input_path, output_path, formats=output_formats)  # Perform batch conversion per input path

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message

        (
            atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
        )  # Register the play_sound function to be called when the program exits if RUN_FUNCTIONS["Play Sound"] is True
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


if __name__ == "__main__":
	"""
	This is the standard boilerplate that calls the main() function.

	:return: None
	"""

	try:  # Protect main execution to ensure errors are reported and notified
		main()  # Call the main function
	except Exception as e:  # Catch any unhandled exception from main
		print(str(e))  # Print the exception message to terminal for logs
		raise  # Re-raise to avoid silent failure and preserve original crash behavior
