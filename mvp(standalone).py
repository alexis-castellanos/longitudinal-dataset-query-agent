#!/usr/bin/env python3
"""
Longitudinal Dataset Query Agent

A demo application that provides a natural language interface for querying
longitudinal datasets with built-in verification and transformation tracking.

Requirements:
- pip install streamlit langchain_community langchain chromadb pandas numpy pydantic langchain-ollama
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import re

import streamlit as st
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class Transformation(BaseModel):
    """Record of a data transformation operation"""
    operation: str = Field(description="The type of transformation applied")
    parameters: Dict[str, Any] = Field(description="Parameters used for the transformation")
    rationale: str = Field(description="Explanation of why this transformation was needed")
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ColumnVersion(BaseModel):
    """Record of a column version"""
    name: str = Field(description="The column name")
    transformation: Optional[str] = Field(description="Transformation applied to create this version")
    reason: Optional[str] = Field(description="Reason for the change")
    timestamp: datetime = Field(default_factory=datetime.now)

class ColumnEvolution(BaseModel):
    """Tracks how a column evolves over time"""
    original_name: str = Field(description="Original column name")
    dataset: str = Field(description="Parent dataset name")
    versions: List[ColumnVersion] = Field(default_factory=list)
    
    def add_version(self, new_name: str, transformation: Optional[str] = None, reason: Optional[str] = None):
        """Add a new version of this column"""
        self.versions.append(ColumnVersion(
            name=new_name,
            transformation=transformation,
            reason=reason
        ))

class DatasetVersion(BaseModel):
    """Metadata about a dataset version including lineage"""
    dataset_name: str = Field(description="Name of the dataset")
    version: str = Field(description="Version identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_versions: List[str] = Field(default_factory=list)
    transformations: List[Transformation] = Field(default_factory=list)
    
    def add_transformation(self, operation: str, params: Dict[str, Any], rationale: str):
        """Record a transformation applied to this dataset"""
        self.transformations.append(Transformation(
            operation=operation,
            parameters=params,
            rationale=rationale
        ))

class ParsedQuery(BaseModel):
    """Structured representation of a natural language query"""
    query_type: str = Field(description="Type of query: data_query or verification_query")
    datasets: List[str] = Field(description="Datasets to query")
    time_range: Optional[Dict[str, str]] = Field(description="Time range specification")
    columns: List[str] = Field(description="Columns or variables requested")
    filters: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    aggregations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    transformations: Optional[Dict[str, List[Dict[str, Any]]]] = Field(default_factory=dict)
    
    # For verification queries
    verification_aspects: Optional[List[str]] = Field(default_factory=list)
    verification_columns: Optional[List[str]] = Field(default_factory=list)
    verification_periods: Optional[List[str]] = Field(default_factory=list)

# ============================================================================
# Mock Data Generation
# ============================================================================

def generate_sample_data():
    """
    Generate sample longitudinal datasets for demonstration
    Returns dictionary of dataframes
    """
    # Patient Demographics
    np.random.seed(42)
    n_patients = 100
    patient_ids = [f"P{i:03d}" for i in range(1, n_patients+1)]
    
    demographics = pd.DataFrame({
        'patient_id': patient_ids,
        'birth_year': np.random.randint(1940, 2000, n_patients),
        'sex': np.random.choice(['M', 'F'], n_patients),
        'enrollment_date': pd.date_range(start='2015-01-01', periods=n_patients, freq='3D')
    })
    
    # Visits - multiple per patient
    visits = []
    for pid in patient_ids:
        n_visits = np.random.randint(1, 6)
        for v in range(n_visits):
            year = np.random.randint(2015, 2023)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 28)
            visits.append({
                'patient_id': pid,
                'visit_id': f"{pid}_V{v+1}",
                'visit_date': f"{year}-{month:02d}-{day:02d}",
                'visit_type': np.random.choice(['routine', 'followup', 'emergency']),
            })
    visits_df = pd.DataFrame(visits)
    
    # Outcomes
    outcomes = []
    for _, visit in visits_df.iterrows():
        outcomes.append({
            'visit_id': visit['visit_id'],
            'patient_id': visit['patient_id'],
            'measurement_date': visit['visit_date'],
            'recovery_score': np.random.randint(1, 11),
            'pain_level': np.random.randint(0, 6),
            'mobility_score': np.random.randint(1, 11)
        })
    outcomes_df = pd.DataFrame(outcomes)
    
    # Create updated datasets with transformations
    # 1. Updated demographics with renamed columns
    demographics_v2 = demographics.copy()
    demographics_v2.rename(columns={
        'birth_year': 'year_of_birth',
        'sex': 'gender'
    }, inplace=True)
    
    # 2. Updated outcomes with derived columns
    outcomes_v2 = outcomes_df.copy()
    outcomes_v2['overall_health'] = outcomes_v2['recovery_score'] * 0.6 + \
                                   outcomes_v2['mobility_score'] * 0.4
    outcomes_v2.rename(columns={
        'pain_level': 'pain_score'
    }, inplace=True)
    
    return {
        'patient_demographics_v1': demographics,
        'patient_demographics_v2': demographics_v2,
        'patient_visits': visits_df,
        'patient_outcomes_v1': outcomes_df,
        'patient_outcomes_v2': outcomes_v2
    }

def create_dataset_metadata():
    """
    Create dataset metadata, including lineage and column evolution
    """
    dataset_versions = {
        'patient_demographics_v1': DatasetVersion(
            dataset_name='patient_demographics',
            version='1.0',
            parent_versions=[]
        ),
        'patient_demographics_v2': DatasetVersion(
            dataset_name='patient_demographics',
            version='2.0',
            parent_versions=['patient_demographics_v1']
        ),
        'patient_visits': DatasetVersion(
            dataset_name='patient_visits',
            version='1.0',
            parent_versions=[]
        ),
        'patient_outcomes_v1': DatasetVersion(
            dataset_name='patient_outcomes',
            version='1.0',
            parent_versions=[]
        ),
        'patient_outcomes_v2': DatasetVersion(
            dataset_name='patient_outcomes',
            version='2.0',
            parent_versions=['patient_outcomes_v1']
        )
    }
    
    # Add transformations
    dataset_versions['patient_demographics_v2'].add_transformation(
        operation='rename_columns',
        params={'birth_year': 'year_of_birth', 'sex': 'gender'},
        rationale='Standardized column names across institutional datasets for consistency'
    )
    
    dataset_versions['patient_outcomes_v2'].add_transformation(
        operation='rename_columns',
        params={'pain_level': 'pain_score'},
        rationale='Renamed for consistency with standard medical terminology'
    )
    
    dataset_versions['patient_outcomes_v2'].add_transformation(
        operation='derive_column',
        params={
            'new_column': 'overall_health', 
            'formula': 'recovery_score * 0.6 + mobility_score * 0.4'
        },
        rationale='Created composite health score as requested by clinical team'
    )
    
    # Column evolution tracking
    column_evolutions = [
        ColumnEvolution(
            original_name='birth_year',
            dataset='patient_demographics',
            versions=[
                ColumnVersion(
                    name='year_of_birth',
                    transformation='rename',
                    reason='Standardized naming conventions across systems'
                )
            ]
        ),
        ColumnEvolution(
            original_name='sex',
            dataset='patient_demographics',
            versions=[
                ColumnVersion(
                    name='gender',
                    transformation='rename',
                    reason='Updated to align with institutional terminology guidelines'
                )
            ]
        ),
        ColumnEvolution(
            original_name='pain_level',
            dataset='patient_outcomes',
            versions=[
                ColumnVersion(
                    name='pain_score',
                    transformation='rename',
                    reason='Aligned with clinical assessment terminology'
                )
            ]
        ),
        ColumnEvolution(
            original_name='NA',
            dataset='patient_outcomes',
            versions=[
                ColumnVersion(
                    name='overall_health',
                    transformation='derived: recovery_score * 0.6 + mobility_score * 0.4',
                    reason='Composite score requested by clinical research team'
                )
            ]
        )
    ]
    
    return dataset_versions, column_evolutions

# ============================================================================
# LLM Query Processing
# ============================================================================

class QueryProcessor:
    """
    Processes natural language queries using LLM and retrieves appropriate data
    """
    
    def __init__(self, model_name="llama3.2:latest"):
        """Initialize with specified Ollama model"""
        self.llm = Ollama(model=model_name)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        # Set up vector store for dataset metadata
        self.setup_vector_db()
    
    def setup_vector_db(self):
        """Set up vector database with dataset schema information"""
        # Schema descriptions for vector search
        schema_descriptions = [
            {"text": "Patient identifier - unique ID for each patient", 
             "metadata": {"dataset": "patient_demographics", "column": "patient_id"}},
            {"text": "Year when patient was born", 
             "metadata": {"dataset": "patient_demographics", "column": "birth_year"}},
            {"text": "Patient's year of birth", 
             "metadata": {"dataset": "patient_demographics", "column": "year_of_birth"}},
            {"text": "Patient's biological sex (M/F)", 
             "metadata": {"dataset": "patient_demographics", "column": "sex"}},
            {"text": "Patient's gender (M/F)", 
             "metadata": {"dataset": "patient_demographics", "column": "gender"}},
            {"text": "Date when patient enrolled in the study", 
             "metadata": {"dataset": "patient_demographics", "column": "enrollment_date"}},
            
            {"text": "Patient's unique identifier", 
             "metadata": {"dataset": "patient_visits", "column": "patient_id"}},
            {"text": "Unique identifier for each clinical visit", 
             "metadata": {"dataset": "patient_visits", "column": "visit_id"}},
            {"text": "Date when patient had a clinical visit", 
             "metadata": {"dataset": "patient_visits", "column": "visit_date"}},
            {"text": "Type of visit (routine, followup, emergency)", 
             "metadata": {"dataset": "patient_visits", "column": "visit_type"}},
            
            {"text": "Unique identifier for clinical visit", 
             "metadata": {"dataset": "patient_outcomes", "column": "visit_id"}},
            {"text": "Patient's unique identifier", 
             "metadata": {"dataset": "patient_outcomes", "column": "patient_id"}},
            {"text": "Date when measurements were taken", 
             "metadata": {"dataset": "patient_outcomes", "column": "measurement_date"}},
            {"text": "Recovery score on scale 1-10", 
             "metadata": {"dataset": "patient_outcomes", "column": "recovery_score"}},
            {"text": "Pain level on scale 0-5", 
             "metadata": {"dataset": "patient_outcomes", "column": "pain_level"}},
            {"text": "Pain score on scale 0-5", 
             "metadata": {"dataset": "patient_outcomes", "column": "pain_score"}},
            {"text": "Mobility score on scale 1-10", 
             "metadata": {"dataset": "patient_outcomes", "column": "mobility_score"}},
            {"text": "Overall health score (derived from recovery and mobility)", 
             "metadata": {"dataset": "patient_outcomes", "column": "overall_health"}},
        ]
        
        # Create temporary directory for Chroma
        self.persist_directory = tempfile.mkdtemp()
        
        # Initialize Chroma DB
        self.vector_db = Chroma.from_texts(
            texts=[item["text"] for item in schema_descriptions],
            metadatas=[item["metadata"] for item in schema_descriptions],
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def parse_query(self, query_text):
        """
        Parse natural language query into structured components
        """
        # Determine if it's a verification query
        verification_prompt = f"""
        Determine if this query is asking about dataset verification, lineage, 
        or transformation explanations: "{query_text}"
        
        If it is a verification request asking about how data was processed, combined or transformed,
        return JSON with "query_type": "verification_query", otherwise "query_type": "data_query".
        
        For verification queries, also extract:
        1. "verification_aspects": What aspects they want verified (combinations, transformations, etc.)
        2. "verification_columns": For which specific variables/columns
        3. "verification_periods": For which time periods
        
        For data queries, extract:
        1. "datasets": List of datasets needed (patient_demographics, patient_visits, patient_outcomes)
        2. "time_range": Any time period specification
        3. "columns": List of columns requested
        4. "filters": Any filtering conditions
        5. "aggregations": Any grouping or summarization requested
        
        Return the structured analysis as a JSON object.
        """
        
        verification_analysis = self.llm.invoke(verification_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', verification_analysis, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = verification_analysis
        
        try:
            parsed_result = json.loads(json_str)
            
            # Ensure the result has the correct structure
            if parsed_result.get("query_type") not in ["verification_query", "data_query"]:
                parsed_result["query_type"] = "data_query"
                
            if "datasets" not in parsed_result:
                parsed_result["datasets"] = []
                
            if "columns" not in parsed_result:
                parsed_result["columns"] = []
                
            return ParsedQuery(**parsed_result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {verification_analysis}")
            
            # Return a default structure
            return ParsedQuery(
                query_type="data_query",
                datasets=["patient_outcomes"],
                columns=["recovery_score"],
                time_range=None
            )
    
    def map_to_schema(self, parsed_query):
        """
        Map natural language column descriptions to actual schema columns
        """
        mapped_columns = {}
        
        # For each requested column
        for column_desc in parsed_query.columns:
            # Find matching columns using vector similarity
            results = self.vector_db.similarity_search(column_desc, k=1)
            
            if results:
                dataset = results[0].metadata["dataset"]
                column = results[0].metadata["column"]
                
                if dataset not in mapped_columns:
                    mapped_columns[dataset] = []
                    
                mapped_columns[dataset].append(column)
        
        return mapped_columns
    
    def generate_query_plan(self, parsed_query, mapped_columns):
        """
        Generate a plan for executing the query
        """
        query_plan = {
            "datasets": [],
            "operations": []
        }
        
        # Add datasets to query plan
        for dataset in mapped_columns.keys():
            # Determine which version to use
            if dataset == "patient_demographics":
                version = "patient_demographics_v2"  # Use latest
            elif dataset == "patient_outcomes":
                version = "patient_outcomes_v2"  # Use latest
            else:
                version = dataset
                
            query_plan["datasets"].append(version)
            
            # Add column selection operation
            query_plan["operations"].append({
                "operation": "select_columns",
                "dataset": version,
                "columns": mapped_columns[dataset]
            })
        
        # Add filter operations if needed
        if parsed_query.filters:
            for filter_op in parsed_query.filters:
                query_plan["operations"].append({
                    "operation": "filter",
                    "dataset": next(iter(mapped_columns.keys())),  # Apply to first dataset for now
                    "condition": filter_op
                })
        
        # Add time filter if specified
        if parsed_query.time_range:
            date_column = None
            if "patient_demographics" in mapped_columns:
                date_column = "enrollment_date"
            elif "patient_visits" in mapped_columns:
                date_column = "visit_date"
            elif "patient_outcomes" in mapped_columns:
                date_column = "measurement_date"
                
            if date_column:
                query_plan["operations"].append({
                    "operation": "filter_dates",
                    "dataset": next(iter(mapped_columns.keys())),
                    "date_column": date_column,
                    "start_date": parsed_query.time_range.get("start"),
                    "end_date": parsed_query.time_range.get("end")
                })
        
        # Add join operations if multiple datasets
        if len(mapped_columns.keys()) > 1:
            datasets = list(mapped_columns.keys())
            for i in range(1, len(datasets)):
                query_plan["operations"].append({
                    "operation": "join",
                    "left_dataset": datasets[0],
                    "right_dataset": datasets[i],
                    "join_column": "patient_id"  # Assume join on patient_id
                })
        
        return query_plan
    
    def execute_query(self, query_plan, datasets_dict):
        """
        Execute query plan against datasets
        """
        # Get required datasets
        working_datasets = {}
        for dataset_name in query_plan["datasets"]:
            working_datasets[dataset_name] = datasets_dict[dataset_name].copy()
        
        result_df = None
        
        # Execute operations
        for operation in query_plan["operations"]:
            op_type = operation["operation"]
            
            if op_type == "select_columns":
                dataset = operation["dataset"]
                columns = operation["columns"]
                
                # Ensure all columns exist
                existing_cols = [col for col in columns if col in working_datasets[dataset].columns]
                if len(existing_cols) > 0:
                    working_datasets[dataset] = working_datasets[dataset][existing_cols]
            
            elif op_type == "filter":
                # Simplified filter implementation
                dataset = operation["dataset"]
                condition = operation["condition"]
                
                if "column" in condition and "value" in condition:
                    col = condition["column"]
                    val = condition["value"]
                    operator = condition.get("operator", "==")
                    
                    if col in working_datasets[dataset].columns:
                        if operator == "==":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] == val
                            ]
                        elif operator == ">":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] > val
                            ]
                        elif operator == "<":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] < val
                            ]
            
            elif op_type == "filter_dates":
                dataset = operation["dataset"]
                date_col = operation["date_column"]
                start_date = operation.get("start_date")
                end_date = operation.get("end_date")
                
                if date_col in working_datasets[dataset].columns:
                    # Ensure date column is datetime type
                    working_datasets[dataset][date_col] = pd.to_datetime(
                        working_datasets[dataset][date_col]
                    )
                    
                    if start_date:
                        working_datasets[dataset] = working_datasets[dataset][
                            working_datasets[dataset][date_col] >= start_date
                        ]
                        
                    if end_date:
                        working_datasets[dataset] = working_datasets[dataset][
                            working_datasets[dataset][date_col] <= end_date
                        ]
            
            elif op_type == "join":
                left = operation["left_dataset"]
                right = operation["right_dataset"]
                join_col = operation["join_column"]
                
                if join_col in working_datasets[left].columns and join_col in working_datasets[right].columns:
                    working_datasets[left] = pd.merge(
                        working_datasets[left],
                        working_datasets[right],
                        on=join_col,
                        how="inner"
                    )
        
        # Return first dataset as result
        if query_plan["datasets"]:
            result_df = working_datasets[query_plan["datasets"][0]]
            
        return result_df
    
    def explain_verification(self, parsed_query, dataset_versions, column_evolutions):
        """
        Generate explanation about dataset transformations for verification queries
        """
        # Construct explanation based on verification needs
        verification_aspects = parsed_query.verification_aspects
        verification_columns = parsed_query.verification_columns
        
        relevant_datasets = []
        if parsed_query.datasets:
            relevant_datasets = parsed_query.datasets
        else:
            # If no specific datasets, look at all
            relevant_datasets = ["patient_demographics", "patient_outcomes"]
        
        relevant_versions = []
        for dataset in relevant_datasets:
            if f"{dataset}_v2" in dataset_versions:
                relevant_versions.append(f"{dataset}_v2")
        
        # Get transformation information
        transformations = []
        for version_key in relevant_versions:
            version = dataset_versions[version_key]
            for transform in version.transformations:
                transformations.append({
                    "dataset": version.dataset_name,
                    "operation": transform.operation,
                    "parameters": transform.parameters,
                    "rationale": transform.rationale
                })
        
        # Get column evolution information
        column_changes = []
        for col_evo in column_evolutions:
            if (not verification_columns or 
                col_evo.original_name in verification_columns or 
                any(v.name in verification_columns for v in col_evo.versions)):
                
                if col_evo.dataset in relevant_datasets:
                    for version in col_evo.versions:
                        column_changes.append({
                            "dataset": col_evo.dataset,
                            "original_name": col_evo.original_name,
                            "new_name": version.name,
                            "transformation": version.transformation,
                            "reason": version.reason
                        })
        
        # Generate explanation with LLM
        explanation_prompt = f"""
        Based on the following information about dataset transformations and column changes,
        generate a clear explanation of how the data was modified and why.
        
        Dataset Transformations:
        {json.dumps(transformations, indent=2)}
        
        Column Changes:
        {json.dumps(column_changes, indent=2)}
        
        Focus on:
        1. What datasets were combined or modified
        2. How variables were renamed or transformed
        3. The rationale behind each change
        4. Any potential impacts on data interpretation
        
        Provide a concise but comprehensive explanation.
        """
        
        explanation = self.llm.invoke(explanation_prompt)
        return explanation
    
    def process_query(self, query_text, datasets, dataset_versions, column_evolutions):
        """
        Process a natural language query and return results with explanations
        """
        # Parse query
        parsed_query = self.parse_query(query_text)
        
        if parsed_query.query_type == "verification_query":
            # Handle verification query
            explanation = self.explain_verification(parsed_query, dataset_versions, column_evolutions)
            return None, explanation, parsed_query
        else:
            # Handle data query
            mapped_columns = self.map_to_schema(parsed_query)
            query_plan = self.generate_query_plan(parsed_query, mapped_columns)
            result_df = self.execute_query(query_plan, datasets)
            
            # Generate explanation for data transformations
            explanation = None
            if result_df is not None:
                explanation_prompt = f"""
                Explain in 2-3 sentences what data was retrieved based on this query:
                "{query_text}"
                
                And this query plan:
                {json.dumps(query_plan, indent=2)}
                """
                explanation = self.llm.invoke(explanation_prompt)
            
            return result_df, explanation, parsed_query

# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main function for Streamlit application"""
    st.set_page_config(
        page_title="Longitudinal Dataset Query Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Longitudinal Dataset Query Agent")
    st.markdown("""
    Ask questions about patient data using natural language. This agent will:
    1. Understand your query
    2. Find the right datasets and columns
    3. Execute the query and show results
    4. Explain what data was retrieved and how it was processed
    """)
    
    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = generate_sample_data()
        st.session_state.dataset_versions, st.session_state.column_evolutions = create_dataset_metadata()
        st.session_state.processor = QueryProcessor()
        st.session_state.history = []
    
    # Sidebar for data exploration
    with st.sidebar:
        st.header("Available Datasets")
        dataset_to_view = st.selectbox(
            "Select a dataset to preview:",
            list(st.session_state.datasets.keys())
        )
        
        if dataset_to_view:
            st.dataframe(st.session_state.datasets[dataset_to_view].head())
            
        st.markdown("---")
        st.subheader("Example Queries")
        st.markdown("""
        - Show me all patient recovery scores for 2018-2020
        - What's the average mobility score by gender?
        - How have columns in the patient demographics dataset changed over time?
        - Explain how the overall health score was calculated
        - What transformations were applied to the pain level variable?
        """)
    
    # Query input
    query = st.text_area("Enter your query:", height=100)
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                result_df, explanation, parsed_query = st.session_state.processor.process_query(
                    query,
                    st.session_state.datasets,
                    st.session_state.dataset_versions,
                    st.session_state.column_evolutions
                )
                
                # Store in history
                st.session_state.history.append({
                    "query": query,
                    "result": result_df,
                    "explanation": explanation,
                    "parsed_query": parsed_query
                })
    
    # Display results
    if st.session_state.history:
        latest = st.session_state.history[-1]
        
        st.markdown("### Query Understanding")
        st.json(latest["parsed_query"].dict())
        
        st.markdown("### Explanation")
        st.write(latest["explanation"])
        
        if latest["result"] is not None:
            st.markdown("### Results")
            st.dataframe(latest["result"])
            
            st.download_button(
                label="Download Results as CSV",
                data=latest["result"].to_csv(index=False).encode('utf-8'),
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )
    
    # Dataset verification section
    with st.expander("Dataset Transformation Details"):
        st.markdown("### Dataset Versions and Transformations")
        
        for version_key, version in st.session_state.dataset_versions.items():
            if version.transformations:
                st.subheader(f"{version.dataset_name} (v{version.version})")
                
                for transform in version.transformations:
                    st.markdown(f"**Operation:** {transform.operation}")
                    st.markdown(f"**Rationale:** {transform.rationale}")
                    st.markdown(f"**Parameters:** {transform.parameters}")
                    st.markdown("---")
        
        st.markdown("### Column Evolution")
        
        for col_evo in st.session_state.column_evolutions:
            if col_evo.versions:
                st.subheader(f"{col_evo.original_name} in {col_evo.dataset}")
                
                for version in col_evo.versions:
                    st.markdown(f"**Changed to:** {version.name}")
                    if version.transformation:
                        st.markdown(f"**Transformation:** {version.transformation}")
                    if version.reason:
                        st.markdown(f"**Reason:** {version.reason}")
                
                st.markdown("---")

if __name__ == "__main__":
    main()