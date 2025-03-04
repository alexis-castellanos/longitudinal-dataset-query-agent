"""
Command-line interface for LongitudinalLLM
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

import pandas as pd

from src.data_manager import DataManager
from src.query_processor import QueryProcessor
from src.utils import check_ollama_availability, check_required_models, install_ollama, save_query_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_ollama() -> bool:
    """
    Check if Ollama is available and properly set up
    
    Returns:
        True if setup is complete, False otherwise
    """
    # Check if Ollama is available
    if not check_ollama_availability():
        print("❌ Ollama is not available.")
        print(install_ollama())
        return False
    
    # Check if required models are available
    model_status = check_required_models()
    missing_models = [model for model, available in model_status.items() if not available]
    
    if missing_models:
        print(f"❌ The following models are required but not available: {', '.join(missing_models)}")
        print("Please pull the missing models:")
        for model in missing_models:
            print(f"  ollama pull {model}")
        return False
    
    print("✅ Ollama is properly set up with all required models.")
    return True


def process_input_query(query: str, data_dir: Optional[str] = None, output: Optional[str] = None) -> None:
    """
    Process a query and display or save results
    
    Args:
        query: The query to process
        data_dir: Optional directory containing datasets
        output: Optional output file path
    """
    # Initialize components
    data_manager = DataManager(data_dir)
    query_processor = QueryProcessor()
    
    # Process query
    print(f"Processing query: {query}")
    result_df, explanation, parsed_query = query_processor.process_query(query, data_manager)
    
    # Display results
    print("\n" + "="*80)
    print("EXPLANATION:")
    print(explanation)
    print("="*80)
    
    if result_df is not None and not result_df.empty:
        print("\nRESULTS:")
        print(result_df)
        
        # Save results if requested
        if output:
            if output.endswith('.csv'):
                result_df.to_csv(output, index=False)
                print(f"\nResults saved to {output}")
            elif output.endswith('.json'):
                result_df.to_json(output, orient='records', indent=2)
                print(f"\nResults saved to {output}")
            else:
                # Use CSV as default format
                output_file = output + '.csv' if not output.endswith('.') else output + 'csv'
                result_df.to_csv(output_file, index=False)
                print(f"\nResults saved to {output_file}")
    else:
        print("\nNo results found.")


def list_available_datasets(data_dir: Optional[str] = None) -> None:
    """
    List available datasets and their information
    
    Args:
        data_dir: Optional directory containing datasets
    """
    # Initialize data manager
    data_manager = DataManager(data_dir)
    
    # Get dataset names
    dataset_names = data_manager.get_dataset_names()
    
    print(f"Found {len(dataset_names)} datasets:")
    for name in dataset_names:
        df = data_manager.get_dataset(name)
        print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {', '.join(df.columns)}")
        print()


def show_dataset_transformations(data_dir: Optional[str] = None) -> None:
    """
    Show transformation information for all datasets
    
    Args:
        data_dir: Optional directory containing datasets
    """
    # Initialize data manager
    data_manager = DataManager(data_dir)
    
    # Get version information
    all_versions = data_manager.get_all_version_info()
    
    print("Dataset Transformation History:")
    for version in all_versions:
        if version["transformations"]:
            print(f"\n{version['dataset_name']} (v{version['version']})")
            
            for transform in version["transformations"]:
                print(f"  Operation: {transform['operation']}")
                print(f"  Rationale: {transform['rationale']}")
                print(f"  Parameters: {transform['parameters']}")
                print(f"  Timestamp: {transform['timestamp']}")
                print("  ---")
    
    # Get column evolution information
    all_columns = data_manager.get_all_column_evolutions()
    
    print("\nColumn Evolution History:")
    for col_info in all_columns:
        if col_info["versions"]:
            print(f"\n{col_info['original_name']} in {col_info['dataset']}")
            
            for version in col_info["versions"]:
                print(f"  Changed to: {version['name']}")
                if version.get("transformation"):
                    print(f"  Transformation: {version['transformation']}")
                if version.get("reason"):
                    print(f"  Reason: {version['reason']}")
                print(f"  Timestamp: {version['timestamp']}")
                print("  ---")


def interactive_mode(data_dir: Optional[str] = None) -> None:
    """
    Run in interactive mode
    
    Args:
        data_dir: Optional directory containing datasets
    """
    print("="*80)
    print("LongitudinalLLM Interactive Mode")
    print("Enter queries in natural language or type 'quit' to exit")
    print("="*80)
    
    # Initialize components
    data_manager = DataManager(data_dir)
    query_processor = QueryProcessor()
    
    while True:
        try:
            query = input("\nEnter query: ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
                
            if query.lower() in ['help', '?']:
                print("\nExample queries:")
                print("  - Show me all patient recovery scores for 2018-2020")
                print("  - What's the average mobility score by gender?")
                print("  - How have columns in the patient demographics dataset changed over time?")
                print("  - Explain how the overall health score was calculated")
                print("  - What transformations were applied to the pain level variable?")
                continue
                
            if query.lower() in ['datasets', 'list']:
                list_available_datasets(data_dir)
                continue
                
            if query.lower() in ['transformations', 'changes']:
                show_dataset_transformations(data_dir)
                continue
                
            # Process normal query
            result_df, explanation, parsed_query = query_processor.process_query(query, data_manager)
            
            # Display results
            print("\n" + "="*80)
            print("EXPLANATION:")
            print(explanation)
            print("="*80)
            
            if result_df is not None and not result_df.empty:
                print("\nRESULTS:")
                print(result_df.head(10))  # Show only first 10 rows
                
                if len(result_df) > 10:
                    print(f"... (showing 10/{len(result_df)} rows)")
                    
                # Ask to save results
                save = input("\nSave results? (y/n): ")
                if save.lower() in ['y', 'yes']:
                    file_path = save_query_results(result_df, query)
                    print(f"Results saved to {file_path}")
            else:
                print("\nNo results found.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="LongitudinalLLM Command Line Interface")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for the 'query' command
    query_parser = subparsers.add_parser("query", help="Process a query")
    query_parser.add_argument("query_text", help="The query text")
    query_parser.add_argument("-d", "--data", help="Directory containing datasets")
    query_parser.add_argument("-o", "--output", help="Output file path")
    
    # Parser for the 'list' command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("-d", "--data", help="Directory containing datasets")
    
    # Parser for the 'transformations' command
    trans_parser = subparsers.add_parser("transformations", help="Show dataset transformations")
    trans_parser.add_argument("-d", "--data", help="Directory containing datasets")
    
    # Parser for the 'interactive' command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    interactive_parser.add_argument("-d", "--data", help="Directory containing datasets")
    
    # Parser for the 'setup' command
    setup_parser = subparsers.add_parser("setup", help="Check and setup Ollama")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute selected command
    if args.command == "query":
        process_input_query(args.query_text, args.data, args.output)
    elif args.command == "list":
        list_available_datasets(args.data)
    elif args.command == "transformations":
        show_dataset_transformations(args.data)
    elif args.command == "interactive":
        interactive_mode(args.data)
    elif args.command == "setup":
        setup_ollama()
    else:
        # Default to interactive mode if no command is specified
        interactive_mode()


if __name__ == "__main__":
    main()
