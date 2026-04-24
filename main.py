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


def load_dataset(input_path):
    """
    Load a dataset from a file in CSV, ARFF, TXT, or Parquet format into a pandas DataFrame.

    :param input_path: Path to the input dataset file.
    :return: pandas DataFrame containing the dataset.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        _, ext = os.path.splitext(input_path)  # Get the file extension of the input file
        ext = ext.lower()  # Convert the file extension to lowercase

        if ext == ".arff":  # If the file is in ARFF format
            df = load_arff_file(input_path)
        elif ext == ".csv":  # If the file is in CSV format
            df = load_csv_file(input_path)
        elif ext == ".parquet":  # If the file is in Parquet format
            df = load_parquet_file(input_path)
        elif ext == ".txt":  # If the file is in TXT format
            df = load_txt_file(input_path)
        elif ext == ".pcap":  # If the file is in PCAP binary format
            df = load_pcap_dataset(input_path)
        elif ext == ".stats":  # If the file is a PCAP statistics text file
            df = load_pcap_stats_file(input_path)
        else:  # Unsupported file format
            raise ValueError(f"Unsupported file format: {ext}")

        return df  # Return the loaded DataFrame
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def convert_to_arff(df, output_path):
    """
    Convert a pandas DataFrame to ARFF format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted ARFF file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to ARFF format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        attributes = [(col, "STRING") for col in df.columns]  # Define all attributes as strings
        df = df.astype(str)  # Ensure all values are strings

        arff_dict = {  # Create a dictionary to hold the ARFF data
            "description": "",  # Description of the dataset (can be left empty)
            "relation": "converted_data",  # Name of the relation (dataset)
            "attributes": attributes,  # List of attributes with their names and types
            "data": df.values.tolist(),  # Convert the DataFrame values to a list of lists for ARFF data
        }

        bytes_needed = estimate_bytes_arff(df, 512, attributes)  # Estimate size needed for ARFF output

        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the ARFF file

        with open(output_path, "w") as f:  # Open the output file for writing
            arff.dump(arff_dict, f)  # Dump the ARFF data into the file using liac-arff
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def convert_to_csv(df, output_path):
    """
    Convert a pandas DataFrame to CSV format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted CSV file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to CSV format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        bytes_needed = estimate_bytes_csv(df, overhead=512)  # Estimate size needed for CSV output
        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the CSV file

        df.to_csv(
            output_path, index=False
        )  # Save the DataFrame to the specified output path in CSV format, without the index
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def convert_to_parquet(df, output_path):
    """
    Convert a pandas DataFrame to PARQUET format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted PARQUET file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to PARQUET format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )

        bytes_needed = estimate_bytes_parquet(df)  # Estimate size needed for PARQUET output
        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the PARQUET file

        df.to_parquet(
            output_path, index=False
        )  # Save the DataFrame to the specified output path in PARQUET format, without the index
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def convert_to_txt(df, output_path):
    """
    Convert a pandas DataFrame to TXT format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted TXT file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to TXT format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        try:  # Try to estimate size by dumping to a string buffer
            buf = io.StringIO()  # Create an in-memory string buffer
            df.to_csv(buf, sep="\t", index=False)  # Dump DataFrame to TXT in the buffer using tab as separator
            lines = buf.getvalue().splitlines()  # Get lines from the buffer
        except Exception:  # Fallback if dumping fails
            lines = None  # Set lines to None to use memory usage estimation

        if lines is not None:  # If lines were successfully obtained
            bytes_needed = estimate_bytes_from_lines(lines, overhead=512)  # Estimate size based on lines
        else:  # Fallback to memory usage estimation
            bytes_needed = estimate_bytes_parquet(df)  # Estimate size based on DataFrame memory usage

        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the TXT file

        df.to_csv(
            output_path, sep="\t", index=False
        )  # Save the DataFrame to the specified output path in TXT format, using tab as the separator and without the index
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def resolve_datasets_cfg(cfg: dict) -> dict:
    """
    Resolve datasets mapping from configuration.

    :param cfg: Configuration dictionary containing dataset mappings.
    :return: Mapping of dataset names to path lists or empty dict.
    """

    try:  # Wrap resolution logic to ensure production-safe monitoring
        datasets_cfg = cfg.get("datasets", {})  # Retrieve the datasets mapping from configuration
        if not isinstance(datasets_cfg, dict):  # Verify datasets_cfg is a mapping of dataset names to path lists
            return {}  # Return an empty mapping when datasets configuration is invalid
        return datasets_cfg  # Return the resolved datasets mapping when valid
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def process_dataset_paths(ds_paths: list, context: dict, cfg: dict) -> None:
    """
    Process configured paths for a single dataset entry.

    :param ds_paths: Iterable of paths configured for the dataset.
    :param context: Processing context dictionary with runtime values.
    :param cfg: Full configuration dictionary for fallback values.
    :return: None
    """

    try:  # Wrap processing logic to ensure production-safe monitoring
        if isinstance(ds_paths, str):  # Verify ds_paths is a single string path
            ds_paths = [ds_paths]  # Wrap single string into a list for uniform processing
        elif not isinstance(ds_paths, (list, tuple)):  # Verify dataset entry is an iterable of paths
            return  # Return early when ds_paths is not a valid iterable

        for ds_path in ds_paths:  # Iterate configured paths for this dataset entry
            effective_input = ds_path  # Set effective input directory for this configured path
            out_dir = context.get("output_directory")  # Retrieve output_directory from context which may be None
            effective_output_base = str(out_dir) if out_dir else os.path.join(str(ds_path), cfg.get("output_directory", "Converted"))  # Determine per-dataset base output directory as a string
            dataset_files = resolve_dataset_files(effective_input)  # Resolve dataset files for this configured path
            if not dataset_files:  # If no files found for this configured path
                print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{effective_input}{Style.RESET_ALL}")  # Print error message when no files found
                continue  # Continue to next configured path when empty

            formats_list = resolve_formats(context.get("formats"))  # Normalize and validate output formats for this path
            len_dataset_files = len(dataset_files)  # Count files to process for progress reporting

            pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting {BackgroundColors.CYAN}{len_dataset_files}{BackgroundColors.GREEN} {'file' if len_dataset_files == 1 else 'files'}{Style.RESET_ALL}", unit="file", colour="green", total=len_dataset_files, leave=False, dynamic_ncols=True)  # Create a single-line progress bar for the conversion process

            for idx, input_path in enumerate(pbar, start=1):  # Iterate files for this configured path with index
                process_dataset_file(idx, len_dataset_files, input_path, effective_input, effective_output_base, formats_list, pbar)  # Delegate per-file processing to function
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def process_dataset_file(idx: int, len_dataset_files: int, input_path: str, effective_input: str, effective_output_base: str, formats_list: list, pbar) -> None:
    """
    Process a single dataset file: clean, load and convert to requested formats.

    :param idx: Index of the file in the current processing batch.
    :param len_dataset_files: Total number of files in the current batch.
    :param input_path: Path to the input file being processed.
    :param effective_input: Effective input directory used for relative calculations.
    :param effective_output_base: Base output directory for converted files.
    :param formats_list: List of output formats to generate for this file.
    :param pbar: Progress bar instance used to display per-file progress.
    :return: None
    """

    try:  # Wrap per-file logic to ensure production-safe monitoring
        file = os.path.basename(str(input_path))  # Extract the file name from the full path
        name, ext = os.path.splitext(file)  # Split file name into base name and extension
        ext = ext.lower()  # Normalize extension to lowercase for matching

        formats_list = resolve_formats(formats_list) if formats_list is not None else []  # Normalize formats_list to a list when possible
        formats_list = resolve_output_file_formats(formats_list)  # Apply configured output_file_formats override when present
        orig_format = ext.lstrip('.')  # Compute original extension without leading dot for native-skip logic

        if orig_format in formats_list:  # Verify whether native format is included in requested targets
            formats_list = [f for f in formats_list if f != orig_format]  # Remove native format to avoid rewriting original file

        if not is_supported_extension(ext):  # Verify the file has an allowed input format extension before any further work
            return  # Return early to the caller when unsupported extension

        dest_dir = resolve_destination_directory(effective_input, input_path, effective_output_base)  # Determine destination directory preserving relative structure
        dest_dir = os.path.abspath(os.path.normpath(str(dest_dir)))  # Normalize and convert destination directory to absolute path to avoid inconsistencies

        needed_targets = []  # Initialize list of formats that actually require conversion for this file
        for fmt in formats_list:  # Iterate requested formats to detect missing outputs
            target_path = os.path.join(dest_dir, f"{name}.{fmt}")  # Compute full target path for candidate output format using normalized dest_dir
            if not verify_filepath_exists(target_path):  # Verify if the target output file is missing on disk using normalized path
                needed_targets.append(fmt)  # Append format when output file does not exist and conversion is required

        if not needed_targets:  # Verify whether any conversion is required after existence verifies
            return  # Return early when the file is already standardized and no conversions are necessary

        size_str = compute_file_size_str(str(input_path))  # Retrieve formatted file size string for current input_path now that conversion is required

        if pbar is not None:  # Verify progress bar instance exists before calling set_description
            try:  # Attempt to compute a relative path for the description to show in pbar
                rel = os.path.relpath(input_path, effective_input) if effective_input and os.path.isdir(effective_input) else os.path.basename(input_path)  # Compute relative path when possible
            except Exception:  # Fallback to basename on error during relpath computation
                rel = os.path.basename(input_path)  # Use basename when relpath fails for pbar description
            pbar.set_description(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{rel}{Style.RESET_ALL}")  # Update the progress bar description with the relative path

        dir_created = create_destination_if_missing(dest_dir, verify_filepath_exists(dest_dir))  # Verify and create destination directory lazily then update flag using normalized path
        cleaned_path = os.path.join(dest_dir, f"{name}{ext}")  # Path for saving the cleaned file prior to conversion using normalized dest_dir
        clean_file(input_path, cleaned_path)  # Clean the file before conversion to normalize content

        df = load_dataset(cleaned_path)  # Load the cleaned dataset into a DataFrame for conversion now that conversion is required
        if "arff" in needed_targets:  # If ARFF format is required for output after verifies
            convert_to_arff(df, os.path.join(str(dest_dir), f"{name}.arff"))  # Convert and save as ARFF format
        if "csv" in needed_targets:  # If CSV format is required for output after verifies
            convert_to_csv(df, os.path.join(str(dest_dir), f"{name}.csv"))  # Convert and save as CSV format
        if "parquet" in needed_targets:  # If Parquet format is required for output after verifies
            convert_to_parquet(df, os.path.join(str(dest_dir), f"{name}.parquet"))  # Convert and save as Parquet format
        if "txt" in needed_targets:  # If TXT format is required for output after verifies
            convert_to_txt(df, os.path.join(str(dest_dir), f"{name}.txt"))  # Convert and save as TXT format

        print()  # Print a newline for better readability between files in terminal output
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs in case of per-file failure when per-file failure occurs
        raise  # Re-raise to preserve original failure semantics for upstream handling


def process_configured_datasets(context: dict) -> None:
    """
    Process datasets defined in configuration mapping.

    :param context: Dictionary with processing context values.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = context.get("cfg", {})  # Retrieve configuration section from context for processing
        datasets_cfg = resolve_datasets_cfg(cfg)  # Resolve and validate datasets mapping from config
        if not datasets_cfg:  # Verify datasets_cfg is a mapping of dataset names to path lists
            return  # Return immediately when configured datasets are not present or invalid

        for ds_name, ds_paths in datasets_cfg.items():  # Iterate each dataset entry in configuration mapping
            process_dataset_paths(ds_paths, context, cfg)  # Delegate per-dataset processing to function
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs in case of top-level failure for top-level failures
        raise  # Re-raise to preserve original failure semantics for callers


def prepare_processing_context(context: dict) -> tuple:
    """
    Prepare common processing context values.

    :param context: Processing context dictionary with runtime values.
    :return: Tuple containing (cfg, input_directory, output_directory).
    """

    try:  # Wrap function logic to ensure production-safe monitoring
        cfg = context.get("cfg", {})  # Retrieve configuration section from context for processing
        input_directory, output_directory = prepare_input_context(context, cfg)  # Prepare input and output directories for processing
        return cfg, input_directory, output_directory  # Return prepared context values
    except Exception as e:  # Catch exceptions inside function
        print(str(e))  # Print function exception to terminal for logs
        raise  # Re-raise to preserve failure semantics


def get_and_verify_dataset_files(input_directory: str, cfg: dict) -> tuple:
    """
    Gather dataset files and verify non-empty, printing message on empty.

    :param input_directory: Path to the input directory to scan for datasets.
    :param cfg: Configuration dictionary used for fallback values.
    :return: Tuple containing (dataset_files_list, len_dataset_files).
    """

    try:  # Wrap function logic to ensure production-safe monitoring
        dataset_files, len_dataset_files = gather_dataset_files(input_directory)  # Gather dataset files and their count for processing
        if not dataset_files:  # If no dataset files were found
            print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{input_directory}{Style.RESET_ALL}")  # Print error message when directory is empty
            return [], 0  # Return empty results to signal caller to exit early
        return dataset_files, len_dataset_files  # Return discovered dataset files and their count
    except Exception as e:  # Catch exceptions inside function
        print(str(e))  # Print function exception to terminal for logs
        raise  # Re-raise to preserve failure semantics


def create_progress_bar(dataset_files: list, len_dataset_files: int):
    """
    Create a progress bar for the conversion process.

    :param dataset_files: List of dataset files to display in the progress bar.
    :param len_dataset_files: Total number of dataset files for progress reporting.
    :return: A tqdm progress bar instance.
    """

    try:  # Wrap function logic to ensure production-safe monitoring
        bar_format_str = BackgroundColors.CYAN + "{l_bar}{bar} " + BackgroundColors.CYAN + "{percentage:3.0f}%" + Style.RESET_ALL + "{r_bar}"  # Compose bar_format with cyan percentage field
        pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting {BackgroundColors.CYAN}{len_dataset_files}{BackgroundColors.GREEN} {'file' if len_dataset_files == 1 else 'files'}{Style.RESET_ALL}", unit="file", colour="green", total=len_dataset_files, leave=False, dynamic_ncols=True, bar_format=bar_format_str)  # Create a single-line progress bar for the conversion process with colored percentage
        return pbar  # Return the created progress bar instance
    except Exception as e:  # Catch exceptions inside function
        print(str(e))  # Print function exception to terminal for logs
        raise  # Re-raise to preserve failure semantics


def iterate_and_process_with_pbar(pbar, input_directory: str, output_directory: str, formats_list: list, len_dataset_files: int) -> None:
    """
    Iterate progress bar and delegate per-file processing to the per-file function.

    :param pbar: Progress bar instance iterating dataset files.
    :param input_directory: Source input directory used for relative calculations.
    :param output_directory: Base output directory for converted files.
    :param formats_list: List of output formats to generate for this run.
    :param len_dataset_files: Total number of files in the current batch.
    :return: None
    """

    try:  # Wrap function logic to ensure production-safe monitoring
        for idx, input_path in enumerate(pbar, start=1):  # Iterate through each dataset file with index
            process_single_input_file(idx, {"input_path": input_path, "input_directory": input_directory, "output_directory": output_directory, "formats_list": formats_list, "len_dataset_files": len_dataset_files, "pbar": pbar})  # Delegate per-file work to function
    except Exception as e:  # Catch exceptions inside function
        print(str(e))  # Print function exception to terminal for logs
        raise  # Re-raise to preserve failure semantics


def process_input_directory(context: dict) -> None:
    """
    Process a single explicit input directory for conversion.

    :param context: Dictionary with processing context values.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg, input_directory, output_directory = prepare_processing_context(context)  # Prepare context and directories for processing

        dataset_files, len_dataset_files = get_and_verify_dataset_files(input_directory, cfg)  # Gather dataset files and verify non-empty
        if not dataset_files:  # If no dataset files were found after verification
            return  # Exit early when function signaled empty discovery

        formats_list = resolve_formats(context.get("formats"))  # Normalize and validate output formats for the run
        formats_list = resolve_output_file_formats(formats_list)  # Apply configured output_file_formats override when present

        pbar = create_progress_bar(dataset_files, len_dataset_files)  # Create a progress bar for the conversion process

        iterate_and_process_with_pbar(pbar, input_directory, output_directory, formats_list, len_dataset_files)  # Iterate and process all files using function
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs when top-level failure occurs for top-level failures
        raise  # Re-raise to preserve original failure semantics


def prepare_input_context(context: dict, cfg: dict) -> tuple:
    """
    Prepare input and output directory values from context and configuration.

    :param context: Processing context dictionary with runtime values.
    :param cfg: Configuration dictionary used for fallback defaults.
    :return: Tuple containing (input_directory, output_directory).
    """

    input_directory = context.get("input_directory")  # Retrieve provided input_directory from context
    output_directory = context.get("output_directory")  # Retrieve provided output_directory from context
    
    if not output_directory:  # If output directory is not provided, get it from defaults
        output_directory = cfg.get("output_directory", "Converted")  # Default to 'Converted' when not specified
    
    return input_directory, output_directory  # Return prepared input and output directories


def gather_dataset_files(input_directory: str) -> tuple:
    """
    Gather dataset files from the input directory and return them with count.

    :param input_directory: Path to the input directory to scan for datasets.
    :return: Tuple containing (dataset_files_list, len_dataset_files).
    """

    dataset_files = resolve_dataset_files(input_directory)  # Get all dataset files from the input directory

    try:  # Attempt to sort files case-insensitively for better user experience on case-insensitive file systems
        dataset_files = sorted(dataset_files, key=lambda p: str(p).lower())  # Sort files case-insensitively by their string representation for consistent ordering across platforms
    except Exception:  # Fallback to regular sorting if case-insensitive sorting fails for any reason
        dataset_files = sorted(dataset_files, key=lambda p: str(p))  # Sort files using regular string representation as a fallback

    len_dataset_files = len(dataset_files)  # Get the number of dataset files found
    
    return dataset_files, len_dataset_files  # Return both the list and its length


def process_single_input_file(idx: int, params: dict) -> None:
    """
    Process a single input file: clean, load and convert to requested formats.

    :param idx: Index of the file in the current processing batch.
    :param params: Dictionary with keys: input_path, input_directory, output_directory, formats_list, len_dataset_files, pbar.
    :return: None
    """

    try:  # Wrap per-file logic to ensure production-safe monitoring
        input_path = params.get("input_path")  # Extract input_path from params for this file
        input_directory = params.get("input_directory")  # Extract input_directory from params for relative pathing
        output_directory = params.get("output_directory")  # Extract output_directory from params for writing outputs
        formats_list = params.get("formats_list")  # Extract formats_list from params to know desired outputs
        len_dataset_files = params.get("len_dataset_files")  # Extract total count for progress messages
        pbar = params.get("pbar")  # Extract progress bar instance for updates

        file = os.path.basename(str(input_path))  # Extract the file name from the full path
        name, ext = os.path.splitext(file)  # Split file name into base name and extension
        ext = ext.lower()  # Normalize extension to lowercase for matching

        formats_list = resolve_formats(formats_list) if formats_list is not None else []  # Normalize formats_list to a list when possible
        formats_list = resolve_output_file_formats(formats_list)  # Apply configured output_file_formats override when present
        orig_format = ext.lstrip('.')  # Compute original extension without leading dot for native-skip logic

        if orig_format in formats_list:  # Verify whether native format is included in requested targets
            formats_list = [f for f in formats_list if f != orig_format]  # Remove native format to avoid rewriting original file

        update_progress_description(pbar, input_path)  # Update progress bar description using function

        if not is_supported_extension(ext):  # Verify the file has an allowed input format extension before further work
            return  # Return early to the caller when unsupported extension

        dest_dir = resolve_destination_directory(input_directory, input_path, output_directory)  # Determine destination directory for converted files
        dest_dir = os.path.abspath(os.path.normpath(str(dest_dir)))  # Normalize and convert destination directory to absolute path to avoid inconsistencies

        needed_targets = []  # Initialize list for target formats that must be created for this file
        for fmt in formats_list:  # Iterate requested formats to verify if outputs already exist
            target_path = os.path.join(dest_dir, f"{name}.{fmt}")  # Build expected target path for each candidate format using normalized dest_dir
            if not verify_filepath_exists(target_path):  # Verify if the specific target file is missing on disk using normalized path
                needed_targets.append(fmt)  # Add format to needed_targets when output file does not exist

        if not needed_targets:  # Verify whether any output formats actually need to be generated
            return  # Return early when file is already standardized and no conversion is required

        dir_created = create_destination_if_missing(dest_dir, verify_filepath_exists(dest_dir))  # Verify and create destination directory lazily then update flag using normalized path
        cleaned_path = os.path.join(dest_dir, f"{name}{ext}")  # Path for saving the cleaned file prior to conversion using normalized dest_dir
        clean_file(input_path, cleaned_path)  # Clean the file before conversion to normalize content

        df = load_dataset(cleaned_path)  # Load the cleaned dataset into a DataFrame for conversion now that conversion is required

        perform_conversions(df, needed_targets, dest_dir, name)  # Perform only the conversions that were detected as necessary
    except Exception as e:  # Catch any exception to ensure logging for per-file failures
        print(str(e))  # Print error to terminal for server logs in case of per-file failure when per-file failure occurs
        raise  # Re-raise to preserve original failure semantics for upstream handling


def compute_file_size_str(path: str) -> str:
    """
    Return formatted file size in GB.

    :param path: Path to the file to measure.
    :return: Formatted size string like "1.23 GB".
    """

    try:  # Wrap size retrieval to avoid raising and ensure safe fallback
        if path and verify_filepath_exists(path):  # Verify the file exists before getting size
            size_bytes = os.path.getsize(path)  # Get file size in bytes from filesystem
            size_gb = size_bytes / (1024 ** 3)  # Convert bytes to gigabytes
            return f"{size_gb:.2f} GB"  # Return formatted size string with two decimal places
        else:  # When file path is empty or does not exist
            return "0.00 GB"  # Default size string for missing file
    except Exception:  # Fallback if os.path operations throw
        return "0.00 GB"  # Use default size string on error


def update_progress_description(pbar, input_path: Optional[str]) -> None:
    """
    Update the progress bar description with the relative path of the current file.

    :param pbar: The tqdm progress bar instance to update.
    :param input_path: The full path of the current input file.
    :return: None
    """

    if pbar is None:  # Verify that a progress bar instance was provided
        return  # Return immediately when no progress bar is available
    try:  # Wrap path and size retrieval to avoid raising inside progress update
        input_path_str = str(input_path) if input_path is not None else ""  # Normalize input_path to string to satisfy os.path expectations
        rel = input_path_str  # Use the full input path string instead of computing a relative path
        try:  # Attempt to retrieve file size in bytes safely
            size_str = compute_file_size_str(input_path_str)  # Retrieve formatted file size string using function
        except Exception:  # Fallback when size retrieval fails for any reason
            size_str = "0.00 GB"  # Use default size string on error
    except Exception:  # Fallback when unexpected errors occur during path normalization
        input_path_str = str(input_path) if input_path is not None else ""  # Normalize input_path again in exception path
        rel = input_path_str  # Use the full input path string in exception path as well
        try:  # Attempt to retrieve file size in exception path
            size_str = compute_file_size_str(input_path_str)  # Retrieve formatted file size string using function in exception path
        except Exception:  # Fallback when size retrieval fails in exception path
            size_str = "0.00 GB"  # Use default size string on error
    pbar.set_description(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{rel}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{size_str}{BackgroundColors.GREEN}){Style.RESET_ALL}")  # Update the progress bar description with the input path and file size


def is_supported_extension(ext: str) -> bool:
    """
    Verify whether the file extension is one of the allowed input dataset formats.

    :param ext: The file extension to evaluate (including leading dot).
    :return: True when extension is supported, otherwise False.
    """

    input_fmts = resolve_input_file_formats(None)  # Resolve allowed input formats from configuration for discovery
    allowed_exts = {"." + str(f).lower().lstrip(".") for f in input_fmts}  # Build allowed extensions set from input formats
    return ext in allowed_exts  # Return True for allowed input format extensions and False otherwise


def create_destination_if_missing(dest_dir: str, dir_created: bool) -> bool:
    """
    Ensure destination directory exists; create lazily before first write.

    :param dest_dir: Destination directory path string.
    :param dir_created: Boolean flag indicating whether directory already exists.
    :return: Updated boolean flag indicating directory existence.
    """

    try:  # Enter guarded execution block to handle exceptions
        if not dir_created:  # Verify destination directory is missing before creation
            create_directories(dest_dir)  # Create destination directory lazily
            dir_created = True  # Mark destination as created to avoid redundant creation attempts
        return dir_created  # Return updated flag indicating directory existence
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def perform_conversions(df, formats_list: Optional[list], dest_dir: Optional[str], name: Optional[str]) -> None:
    """
    Convert a DataFrame into the requested output formats and save to destination directory.

    :param df: The pandas DataFrame to convert.
    :param formats_list: The list of format names to generate (e.g., ['arff','csv']).
    :param dest_dir: The destination directory where outputs will be saved.
    :param name: The base filename (without extension) for output files.
    :return: None
    """

    formats = formats_list or []  # Ensure formats is a list to avoid issues with None
    df = normalize_dataframe_types(df)  # Normalize DataFrame types before any conversion to ensure homogeneous column types
    dest_dir_str = str(dest_dir) if dest_dir is not None else ""  # Ensure os.path.join receives a string
    dest_dir_str = os.path.abspath(os.path.normpath(dest_dir_str))  # Normalize and absolutize dest_dir to ensure consistent writes and space verifies
    name_str = str(name) if name is not None else ""  # Ensure filename component is a string

    dir_created = verify_filepath_exists(dest_dir_str)  # Determine whether destination directory already exists

    if "arff" in formats:  # If ARFF format is requested for output
        dir_created = create_destination_if_missing(dest_dir_str, dir_created)  # Verify and create destination directory lazily then update flag
        convert_to_arff(df, os.path.join(dest_dir_str, f"{name_str}.arff"))  # Convert and save as ARFF format
    if "csv" in formats:  # If CSV format is requested for output
        dir_created = create_destination_if_missing(dest_dir_str, dir_created)  # Verify and create destination directory lazily then update flag
        convert_to_csv(df, os.path.join(dest_dir_str, f"{name_str}.csv"))  # Convert and save as CSV format
    if "parquet" in formats:  # If Parquet format is requested for output
        dir_created = create_destination_if_missing(dest_dir_str, dir_created)  # Verify and create destination directory lazily then update flag
        convert_to_parquet(df, os.path.join(dest_dir_str, f"{name_str}.parquet"))  # Convert and save as Parquet format
    if "txt" in formats:  # If TXT format is requested for output
        dir_created = create_destination_if_missing(dest_dir_str, dir_created)  # Verify and create destination directory lazily then update flag
        convert_to_txt(df, os.path.join(dest_dir_str, f"{name_str}.txt"))  # Convert and save as TXT format


def batch_convert(input_directory=None, output_directory=None, formats=None):
    """
    Batch converts dataset files from the input directory into multiple output formats

    :param input_directory: Path to the input directory containing dataset files.
    :param output_directory: Path to the output directory where converted files will be saved.
    :param formats: List of output formats to generate (e.g., ["arff", "csv"]). If None, all formats are generated.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(f"{BackgroundColors.GREEN}Batch converting dataset files from {BackgroundColors.CYAN}{input_directory}{BackgroundColors.GREEN} to {BackgroundColors.CYAN}{output_directory}{Style.RESET_ALL}")  # Output the verbose message

        cfg = DEFAULTS.get("dataset_converter", {}) if DEFAULTS else {}  # Get default configuration for dataset converter if available

        if not input_directory:  # Verify if no input_directory argument was given
            context = {"cfg": cfg, "output_directory": output_directory, "formats": formats}  # Build context dictionary for configured datasets processing
            process_configured_datasets(context)  # Process datasets defined in configuration
            return  # Completed processing configured datasets, exit early

        context = {"cfg": cfg, "input_directory": input_directory, "output_directory": output_directory, "formats": formats}  # Build context dictionary for input-directory processing
        process_input_directory(context)  # Process the explicit input directory
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring
        if obj is None:  # None can't be converted
            return None  # Signal failure to convert
        if isinstance(obj, (int, float)):  # Already numeric (seconds or timestamp)
            return float(obj)  # Return as float seconds
        if hasattr(obj, "total_seconds"):  # Timedelta-like objects
            try:  # Attempt to call total_seconds()
                return float(obj.total_seconds())  # Use the total_seconds() method
            except Exception:
                pass  # Fallthrough on error
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception:
                pass  # Fallthrough on error
        return None  # Couldn't convert
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


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
