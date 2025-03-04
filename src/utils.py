"""
Utility functions for LongitudinalLLM
"""

import logging
import os
import subprocess
import platform
import json
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


def check_ollama_availability() -> bool:
    """
    Check if Ollama is available and running
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        # Check if ollama command exists
        platform_system = platform.system()
        if platform_system == 'Windows':
            process = subprocess.run(
                ['where', 'ollama'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        else:  # Linux/Mac
            process = subprocess.run(
                ['which', 'ollama'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
        if process.returncode != 0:
            logger.warning("Ollama command not found")
            return False
            
        # Try to ping the Ollama API
        import requests
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            logger.warning("Ollama API not responding")
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Error checking Ollama availability: {str(e)}")
        return False


def check_required_models() -> Dict[str, bool]:
    """
    Check if required Ollama models are available
    
    Returns:
        Dictionary mapping model names to availability status
    """
    required_models = ["llama3", "nomic-embed-text"]
    model_status = {model: False for model in required_models}
    
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags')
        
        if response.status_code == 200:
            available_models = response.json().get('models', [])
            available_model_names = [model.get('name') for model in available_models]
            
            for model in required_models:
                model_status[model] = model in available_model_names
    except Exception as e:
        logger.warning(f"Error checking required models: {str(e)}")
        
    return model_status


def install_ollama() -> str:
    """
    Provide instructions for installing Ollama based on the platform
    
    Returns:
        Installation instructions as a string
    """
    platform_system = platform.system()
    
    if platform_system == 'Darwin':  # macOS
        return """
        To install Ollama on macOS:
        
        1. Run the following command in your terminal:
           ```
           curl -fsSL https://ollama.com/install.sh | sh
           ```
           
        2. After installation, start Ollama from your Applications folder or run:
           ```
           ollama serve
           ```
           
        3. Pull the required models:
           ```
           ollama pull llama3
           ollama pull nomic-embed-text
           ```
        """
    elif platform_system == 'Linux':
        return """
        To install Ollama on Linux:
        
        1. Run the following command in your terminal:
           ```
           curl -fsSL https://ollama.com/install.sh | sh
           ```
           
        2. After installation, start the Ollama service:
           ```
           ollama serve
           ```
           
        3. Pull the required models:
           ```
           ollama pull llama3
           ollama pull nomic-embed-text
           ```
        """
    elif platform_system == 'Windows':
        return """
        To install Ollama on Windows:
        
        1. Download the installer from: https://ollama.com/download/windows
        
        2. Run the installer and follow the instructions
        
        3. After installation, open Command Prompt and pull the required models:
           ```
           ollama pull llama3
           ollama pull nomic-embed-text
           ```
        """
    else:
        return "Please visit https://ollama.com/download for installation instructions for your platform."
        

def format_datetime(dt) -> str:
    """
    Format a datetime object consistently
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted datetime string
    """
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(dt)


def get_date_range_from_year(year: int) -> Dict[str, str]:
    """
    Get a date range for a specific year
    
    Args:
        year: Year as an integer
        
    Returns:
        Dictionary with start and end dates
    """
    return {
        "start": f"{year}-01-01",
        "end": f"{year}-12-31"
    }


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")


def save_query_results(results_df, query_text: str, output_dir: str = "results") -> str:
    """
    Save query results to a CSV file
    
    Args:
        results_df: DataFrame with query results
        query_text: Original query text
        output_dir: Directory to save results to
        
    Returns:
        Path to saved file
    """
    from datetime import datetime
    import pandas as pd
    
    # Create output directory if it doesn't exist
    ensure_directory_exists(output_dir)
    
    # Create a filename based on the query
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_part = query_text.lower().replace(" ", "_")[:30]
    filename = f"{timestamp}_{query_part}.csv"
    file_path = os.path.join(output_dir, filename)
    
    # Save to CSV
    results_df.to_csv(file_path, index=False)
    logger.info(f"Saved query results to {file_path}")
    
    return file_path
