#!/usr/bin/env python3
"""
LongitudinalLLM - Main Application

A natural language interface for querying longitudinal datasets 
with built-in verification and transformation tracking.
"""

import logging
import os
from datetime import datetime

import streamlit as st
import pandas as pd

from src.data_manager import DataManager
from src.query_processor import QueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application title and configuration
st.set_page_config(
    page_title="LongitudinalLLM",
    page_icon="📊",
    layout="wide",
)

def main():
    """Main application entry point"""
    st.title("📊 LongitudinalLLM")
    st.markdown("""
    Ask questions about longitudinal datasets using natural language. 
    This agent will understand your query, find the right data, and explain 
    what transformations have been applied.
    """)
    
    # Initialize session state
    if "initialized" not in st.session_state:
        # Check if Ollama is available
        try:
            with st.spinner("Initializing LLM connection..."):
                # Initialize components
                data_manager = DataManager()
                query_processor = QueryProcessor()
                
                # Store in session state
                st.session_state.data_manager = data_manager
                st.session_state.query_processor = query_processor
                st.session_state.history = []
                st.session_state.initialized = True
                
                st.success("✅ Connected to Ollama successfully")
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {str(e)}")
            st.info("""
            Please make sure Ollama is installed and running.
            
            Installation instructions:
            - Linux/macOS: `curl -fsSL https://ollama.com/install.sh | sh`
            - Windows: Download from https://ollama.com/download
            
            Then start Ollama and run:
            ```
            ollama pull llama3
            ollama pull nomic-embed-text
            ```
            """)
            return
    
    # Sidebar for dataset exploration
    with st.sidebar:
        st.header("Available Datasets")
        if "data_manager" in st.session_state:
            dataset_to_view = st.selectbox(
                "Select a dataset to preview:",
                st.session_state.data_manager.get_dataset_names()
            )
            
            if dataset_to_view:
                st.dataframe(
                    st.session_state.data_manager.get_dataset_preview(dataset_to_view)
                )
                
        st.markdown("---")
        st.subheader("Example Queries")
        st.markdown("""
        **Data Queries:**
        - Show me all patient recovery scores for 2018-2020
        - What's the average mobility score by gender?
        - Find patients with high recovery scores but low mobility
        
        **Verification Queries:**
        - How have columns in the demographics dataset changed over time?
        - Explain how the overall health score was calculated
        - What transformations were applied to the pain level variable?
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_area("Enter your query:", height=100)
        
        submitted = st.button("Submit Query", type="primary")
        
        if submitted and query and "query_processor" in st.session_state:
            with st.spinner("Processing query..."):
                result_df, explanation, parsed_query = st.session_state.query_processor.process_query(
                    query,
                    st.session_state.data_manager
                )
                
                # Store in history
                st.session_state.history.append({
                    "query": query,
                    "result": result_df,
                    "explanation": explanation,
                    "parsed_query": parsed_query,
                    "timestamp": datetime.now()
                })
    
    with col2:
        st.markdown("### Query History")
        if "history" in st.session_state and st.session_state.history:
            history_items = []
            for i, item in enumerate(st.session_state.history):
                timestamp = item["timestamp"].strftime("%H:%M:%S")
                history_items.append(f"{timestamp}: {item['query'][:50]}...")
            
            selected_history_idx = st.selectbox(
                "Select a previous query:",
                range(len(history_items)),
                format_func=lambda i: history_items[i],
                index=len(history_items) - 1
            )
            
            if st.button("Load Selected Query"):
                # This will repopulate the query text area
                st.session_state.query_to_load = st.session_state.history[selected_history_idx]["query"]
                st.rerun()
        else:
            st.info("No queries yet. Enter a query to get started.")
    
    # Pre-populate query text area if a history item was selected
    if "query_to_load" in st.session_state:
        query = st.session_state.query_to_load
        del st.session_state.query_to_load
    
    # Display results
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        latest = st.session_state.history[-1]
        
        st.markdown("### Query Understanding")
        st.json(latest["parsed_query"])
        
        st.markdown("### Explanation")
        st.write(latest["explanation"])
        
        if latest["result"] is not None:
            st.markdown("### Results")
            st.dataframe(latest["result"])
            
            # Download button
            csv_data = latest["result"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    # Dataset verification section
    with st.expander("Dataset Transformation Details"):
        if "data_manager" in st.session_state:
            st.markdown("### Dataset Versions and Transformations")
            
            for version_info in st.session_state.data_manager.get_all_version_info():
                if version_info["transformations"]:
                    st.subheader(f"{version_info['dataset_name']} (v{version_info['version']})")
                    
                    for transform in version_info["transformations"]:
                        st.markdown(f"**Operation:** {transform['operation']}")
                        st.markdown(f"**Rationale:** {transform['rationale']}")
                        st.markdown(f"**Parameters:** {transform['parameters']}")
                        st.markdown("---")
            
            st.markdown("### Column Evolution")
            
            for col_evo in st.session_state.data_manager.get_all_column_evolutions():
                if col_evo["versions"]:
                    st.subheader(f"{col_evo['original_name']} in {col_evo['dataset']}")
                    
                    for version in col_evo["versions"]:
                        st.markdown(f"**Changed to:** {version['name']}")
                        if version.get("transformation"):
                            st.markdown(f"**Transformation:** {version['transformation']}")
                        if version.get("reason"):
                            st.markdown(f"**Reason:** {version['reason']}")
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()
