"""
Query Processor for LongitudinalLLM

This module handles processing natural language queries using LLMs.
"""

import logging
import json
import re
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from src.data_models import ParsedQuery, QueryPlan
from src.data_manager import DataManager

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes natural language queries using LLM and retrieves appropriate data
    """
    
    def __init__(self, model_name: str = "llama3.2:latest", embed_model: str = "nomic-embed-text:latest"):
        """
        Initialize with specified Ollama model
        
        Args:
            model_name: Name of the Ollama model to use for query understanding
            embed_model: Name of the Ollama model to use for embeddings
        """
        logger.info(f"Initializing QueryProcessor with model {model_name}")
        
        self.llm = Ollama(model=model_name)
        self.embeddings = OllamaEmbeddings(model=embed_model)
        
        # Set up vector store for dataset metadata
        self.vector_db = None
        self.persist_directory = tempfile.mkdtemp()
        
    def setup_vector_db(self, data_manager: DataManager):
        """
        Set up vector database with dataset schema information
        
        Args:
            data_manager: DataManager instance with dataset information
        """
        logger.info("Setting up vector database for schema mapping")
        
        # Get all datasets from data manager
        schema_descriptions = []
        
        for dataset_name in data_manager.get_dataset_names():
            # Get schema
            schema = data_manager.get_dataset_schema(dataset_name)
            
            # Get description for each column
            for column, dtype in schema.items():
                # Create a generic description based on column name
                words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', column)
                description = " ".join(words).lower()
                
                schema_descriptions.append({
                    "text": f"{description} - {dtype}",
                    "metadata": {
                        "dataset": dataset_name,
                        "column": column
                    }
                })
                
                # Additional descriptions for common column names
                if "id" in column.lower() or "identifier" in column.lower():
                    schema_descriptions.append({
                        "text": f"unique identifier {description}",
                        "metadata": {
                            "dataset": dataset_name,
                            "column": column
                        }
                    })
                elif "date" in column.lower():
                    schema_descriptions.append({
                        "text": f"date or time information for {description}",
                        "metadata": {
                            "dataset": dataset_name,
                            "column": column
                        }
                    })
                elif "name" in column.lower():
                    schema_descriptions.append({
                        "text": f"name or label for {description}",
                        "metadata": {
                            "dataset": dataset_name,
                            "column": column
                        }
                    })
                elif "score" in column.lower() or "level" in column.lower():
                    schema_descriptions.append({
                        "text": f"numerical score or measurement for {description}",
                        "metadata": {
                            "dataset": dataset_name,
                            "column": column
                        }
                    })
        
        # Initialize Chroma DB with schema descriptions
        if schema_descriptions:
            self.vector_db = Chroma.from_texts(
                texts=[item["text"] for item in schema_descriptions],
                metadatas=[item["metadata"] for item in schema_descriptions],
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Initialized vector database with {len(schema_descriptions)} schema descriptions")
    
    def parse_query(self, query_text: str) -> ParsedQuery:
        """
        Parse natural language query into structured components
        
        Args:
            query_text: The natural language query
            
        Returns:
            Structured ParsedQuery object
        """
        logger.info(f"Parsing query: {query_text}")
        
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
        2. "time_range": Any time period specification as {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
        3. "columns": List of columns requested
        4. "filters": Any filtering conditions as [{{"column": "col_name", "operator": "==", "value": val}}]
        5. "aggregations": Any grouping or summarization requested as [{{"type": "avg", "column": "col_name", "groupby": ["col1", "col2"]}}]

        Return the structured analysis as a JSON object.
        """
        
        verification_analysis = self.llm.invoke(verification_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', verification_analysis, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            possible_json = re.search(r'\{.*\}', verification_analysis, re.DOTALL)
            if possible_json:
                json_str = possible_json.group(0)
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
    
    def map_to_schema(self, parsed_query: ParsedQuery) -> Dict[str, List[str]]:
        """
        Map natural language column descriptions to actual schema columns
        
        Args:
            parsed_query: Parsed query structure
            
        Returns:
            Dictionary mapping dataset names to column lists
        """
        logger.info(f"Mapping query to schema: {parsed_query.columns}")
        
        # Initialize vector DB if not already done
        if self.vector_db is None:
            logger.warning("Vector DB not initialized. Schema mapping may not be accurate.")
            return {}
            
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
                    
                if column not in mapped_columns[dataset]:
                    mapped_columns[dataset].append(column)
        
        # If no columns were mapped, try to infer from datasets
        if not mapped_columns and parsed_query.datasets:
            for dataset in parsed_query.datasets:
                mapped_columns[dataset] = []  # Will be filled with default columns later
        
        logger.info(f"Mapped columns: {mapped_columns}")
        return mapped_columns
    
    def generate_query_plan(self, parsed_query: ParsedQuery, mapped_columns: Dict[str, List[str]]) -> QueryPlan:
        """
        Generate a plan for executing the query
        
        Args:
            parsed_query: Parsed query structure
            mapped_columns: Mapped columns by dataset
            
        Returns:
            Query execution plan
        """
        logger.info("Generating query plan")
        
        query_plan = {
            "datasets": [],
            "operations": []
        }
        
        # Add datasets to query plan
        for dataset in mapped_columns.keys():
            # Determine which version to use (latest by default)
            if dataset.startswith("patient_demographics"):
                version = "patient_demographics_v2"  # Use latest
            elif dataset.startswith("patient_outcomes"):
                version = "patient_outcomes_v2"  # Use latest
            else:
                version = dataset
                
            query_plan["datasets"].append(version)
            
            # Add column selection operation
            columns = mapped_columns[dataset]
            if not columns:
                # If no specific columns were requested, select all columns
                columns = ["*"]  # This will be replaced with all columns later
                
            query_plan["operations"].append({
                "operation": "select_columns",
                "dataset": version,
                "columns": columns
            })
        
        # Add filter operations if needed
        if parsed_query.filters:
            for filter_op in parsed_query.filters:
                # Find which dataset contains this column
                target_dataset = None
                for dataset, columns in mapped_columns.items():
                    if filter_op.column in columns:
                        if dataset.startswith("patient_demographics"):
                            target_dataset = "patient_demographics_v2"
                        elif dataset.startswith("patient_outcomes"):
                            target_dataset = "patient_outcomes_v2"
                        else:
                            target_dataset = dataset
                        break
                
                if target_dataset:
                    query_plan["operations"].append({
                        "operation": "filter",
                        "dataset": target_dataset,
                        "condition": {
                            "column": filter_op.column,
                            "operator": filter_op.operator,
                            "value": filter_op.value
                        }
                    })
        
        # Add time filter if specified
        if parsed_query.time_range:
            date_column = None
            target_dataset = None
            
            # Find appropriate date column and dataset
            if any(d.startswith("patient_demographics") for d in mapped_columns.keys()):
                date_column = "enrollment_date"
                target_dataset = "patient_demographics_v2"
            elif any(d.startswith("patient_visits") for d in mapped_columns.keys()):
                date_column = "visit_date"
                target_dataset = "patient_visits"
            elif any(d.startswith("patient_outcomes") for d in mapped_columns.keys()):
                date_column = "measurement_date"
                target_dataset = "patient_outcomes_v2"
                
            if date_column and target_dataset:
                query_plan["operations"].append({
                    "operation": "filter_dates",
                    "dataset": target_dataset,
                    "date_column": date_column,
                    "start_date": parsed_query.time_range.get("start"),
                    "end_date": parsed_query.time_range.get("end")
                })
        
        # Add join operations if multiple datasets
        if len(mapped_columns.keys()) > 1:
            datasets = list(mapped_columns.keys())
            # Replace with actual version names
            dataset_versions = []
            for d in datasets:
                if d.startswith("patient_demographics"):
                    dataset_versions.append("patient_demographics_v2")
                elif d.startswith("patient_outcomes"):
                    dataset_versions.append("patient_outcomes_v2")
                else:
                    dataset_versions.append(d)
            
            # Use the first dataset as the base
            for i in range(1, len(dataset_versions)):
                query_plan["operations"].append({
                    "operation": "join",
                    "left_dataset": dataset_versions[0],
                    "right_dataset": dataset_versions[i],
                    "join_column": "patient_id",  # Assume join on patient_id
                    "join_type": "inner"
                })
        
        # Add aggregation operations if specified
        if parsed_query.aggregations:
            for agg in parsed_query.aggregations:
                # Find which dataset contains this column
                target_dataset = None
                target_column = agg.column
                for dataset, columns in mapped_columns.items():
                    if target_column in columns:
                        if dataset.startswith("patient_demographics"):
                            target_dataset = "patient_demographics_v2"
                        elif dataset.startswith("patient_outcomes"):
                            target_dataset = "patient_outcomes_v2"
                        else:
                            target_dataset = dataset
                        break
                
                if target_dataset:
                    groupby_columns = []
                    if agg.groupby:
                        for groupby_col in agg.groupby:
                            # Map groupby columns to actual schema
                            for dataset, columns in mapped_columns.items():
                                if groupby_col in columns:
                                    groupby_columns.append(groupby_col)
                                    break
                    
                    query_plan["operations"].append({
                        "operation": "aggregate",
                        "dataset": target_dataset,
                        "aggregation_type": agg.type,
                        "column": target_column,
                        "groupby": groupby_columns
                    })
                    
        # Return query plan
        return QueryPlan(**query_plan)
        
    def execute_query(self, query_plan: QueryPlan, data_manager: DataManager) -> pd.DataFrame:
        """
        Execute query plan against datasets
        
        Args:
            query_plan: Query plan with datasets and operations
            data_manager: DataManager instance
            
        Returns:
            DataFrame with query results
        """
        logger.info(f"Executing query plan with {len(query_plan.operations)} operations")
        
        # Execute the query plan using the data manager
        result_df = data_manager.execute_query_plan(query_plan.dict())
        
        return result_df
    
    def explain_verification(self, parsed_query: ParsedQuery, data_manager: DataManager) -> str:
        """
        Generate explanation about dataset transformations for verification queries
        
        Args:
            parsed_query: Parsed query structure
            data_manager: DataManager instance
            
        Returns:
            Explanation text
        """
        logger.info("Generating verification explanation")
        
        # Get relevant datasets
        relevant_datasets = []
        if parsed_query.datasets:
            relevant_datasets = parsed_query.datasets
        else:
            # If no specific datasets, look at all major ones
            relevant_datasets = ["patient_demographics", "patient_outcomes"]
        
        relevant_versions = []
        for dataset in relevant_datasets:
            if dataset.startswith("patient_demographics"):
                relevant_versions.append("patient_demographics_v2")
            elif dataset.startswith("patient_outcomes"):
                relevant_versions.append("patient_outcomes_v2")
            else:
                relevant_versions.append(dataset)
        
        # Get transformation information
        transformations = []
        for version_name in relevant_versions:
            version_info = data_manager.get_version_info(version_name)
            if version_info and "transformations" in version_info:
                for transform in version_info["transformations"]:
                    transformations.append({
                        "dataset": version_info["dataset_name"],
                        "operation": transform["operation"],
                        "parameters": transform["parameters"],
                        "rationale": transform["rationale"]
                    })
        
        # Get column evolution information
        column_changes = []
        for column_name in parsed_query.verification_columns or []:
            # Try to find in each dataset
            for dataset in relevant_datasets:
                column_info = data_manager.get_column_evolution(column_name, dataset)
                if column_info:
                    for version in column_info["versions"]:
                        column_changes.append({
                            "dataset": dataset,
                            "original_name": column_name,
                            "new_name": version["name"],
                            "transformation": version.get("transformation"),
                            "reason": version.get("reason")
                        })
        
        # If no specific columns were requested, get all column changes
        if not column_changes:
            for col_info in data_manager.get_all_column_evolutions():
                if col_info["dataset"] in relevant_datasets:
                    for version in col_info["versions"]:
                        column_changes.append({
                            "dataset": col_info["dataset"],
                            "original_name": col_info["original_name"],
                            "new_name": version["name"],
                            "transformation": version.get("transformation"),
                            "reason": version.get("reason")
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
    
    def process_query(self, query_text: str, data_manager: DataManager) -> Tuple[Optional[pd.DataFrame], str, Dict]:
        """
        Process a natural language query and return results with explanations
        
        Args:
            query_text: Natural language query
            data_manager: DataManager instance
            
        Returns:
            Tuple of (result DataFrame or None, explanation text, parsed query dict)
        """
        # Initialize vector DB if not already done
        if self.vector_db is None:
            self.setup_vector_db(data_manager)
        
        # Parse query
        parsed_query = self.parse_query(query_text)
        
        if parsed_query.query_type == "verification_query":
            # Handle verification query
            explanation = self.explain_verification(parsed_query, data_manager)
            return None, explanation, parsed_query.dict()
        else:
            # Handle data query
            mapped_columns = self.map_to_schema(parsed_query)
            query_plan = self.generate_query_plan(parsed_query, mapped_columns)
            result_df = self.execute_query(query_plan, data_manager)
            
            # Generate explanation for data transformations
            explanation = None
            if result_df is not None and not result_df.empty:
                explanation_prompt = f"""
                Explain in 2-3 sentences what data was retrieved and transformed based on this query:
                "{query_text}"
                
                And this query plan:
                {json.dumps(query_plan.dict(), indent=2)}
                
                Result preview (first few rows):
                {result_df.head(3).to_string()}
                
                Focus on what data was selected, what transformations were applied, and what the result means.
                """
                explanation = self.llm.invoke(explanation_prompt)
            else:
                explanation = f"No data found matching the query: '{query_text}'. This may be because the requested columns or datasets don't exist, or because the filter conditions didn't match any records."
            
            return result_df, explanation, parsed_query.dict()
