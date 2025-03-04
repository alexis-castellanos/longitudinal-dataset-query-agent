"""
Data Manager for LongitudinalLLM

This module handles dataset loading, storage, and transformation tracking.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

import pandas as pd
import numpy as np

from src.data_models import DatasetVersion, ColumnEvolution, ColumnVersion, Transformation

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages longitudinal datasets and tracks transformations
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data manager
        
        Args:
            data_dir: Directory containing datasets (if None, uses sample data)
        """
        self.datasets = {}
        self.dataset_versions = {}
        self.column_evolutions = []
        
        # Load either user-provided data or generate sample data
        if data_dir and os.path.exists(data_dir):
            self._load_datasets_from_dir(data_dir)
        else:
            self._generate_sample_data()
    
    def _load_datasets_from_dir(self, data_dir: str):
        """
        Load datasets from a directory
        
        Args:
            data_dir: Directory containing datasets
        """
        logger.info(f"Loading datasets from {data_dir}")
        
        # Look for dataset files
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                name = os.path.splitext(file)[0]
                file_path = os.path.join(data_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    self.datasets[name] = df
                    
                    # Create basic version info
                    self.dataset_versions[name] = DatasetVersion(
                        dataset_name=name,
                        version="1.0",
                        parent_versions=[]
                    )
                    
                    logger.info(f"Loaded dataset {name} with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    logger.error(f"Failed to load dataset {name}: {str(e)}")
        
        # Look for metadata files
        metadata_file = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load dataset versions
                if "dataset_versions" in metadata:
                    for version_data in metadata["dataset_versions"]:
                        name = version_data.get("dataset_name")
                        version = version_data.get("version")
                        if name and version:
                            # Convert dict to DatasetVersion
                            self.dataset_versions[f"{name}_v{version}"] = DatasetVersion(**version_data)
                
                # Load column evolutions
                if "column_evolutions" in metadata:
                    for col_data in metadata["column_evolutions"]:
                        self.column_evolutions.append(ColumnEvolution(**col_data))
                        
                logger.info(f"Loaded metadata for {len(self.dataset_versions)} dataset versions and "
                           f"{len(self.column_evolutions)} column evolutions")
            except Exception as e:
                logger.error(f"Failed to load metadata: {str(e)}")
    
    def _generate_sample_data(self):
        """
        Generate sample longitudinal datasets for demonstration
        """
        logger.info("Generating sample data")
        
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
        
        # Store datasets
        self.datasets = {
            'patient_demographics_v1': demographics,
            'patient_demographics_v2': demographics_v2,
            'patient_visits': visits_df,
            'patient_outcomes_v1': outcomes_df,
            'patient_outcomes_v2': outcomes_v2
        }
        
        # Create dataset versions
        self.dataset_versions = {
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
        self.dataset_versions['patient_demographics_v2'].add_transformation(
            operation='rename_columns',
            params={'birth_year': 'year_of_birth', 'sex': 'gender'},
            rationale='Standardized column names across institutional datasets for consistency'
        )
        
        self.dataset_versions['patient_outcomes_v2'].add_transformation(
            operation='rename_columns',
            params={'pain_level': 'pain_score'},
            rationale='Renamed for consistency with standard medical terminology'
        )
        
        self.dataset_versions['patient_outcomes_v2'].add_transformation(
            operation='derive_column',
            params={
                'new_column': 'overall_health', 
                'formula': 'recovery_score * 0.6 + mobility_score * 0.4'
            },
            rationale='Created composite health score as requested by clinical team'
        )
        
        # Column evolution tracking
        self.column_evolutions = [
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
        
        logger.info(f"Generated {len(self.datasets)} sample datasets")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a dataset by name
        
        Args:
            name: Dataset name
            
        Returns:
            DataFrame or None if not found
        """
        return self.datasets.get(name)
    
    def get_dataset_names(self) -> List[str]:
        """
        Get all available dataset names
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())
    
    def get_dataset_preview(self, name: str, rows: int = 5) -> pd.DataFrame:
        """
        Get a preview of a dataset
        
        Args:
            name: Dataset name
            rows: Number of rows to preview
            
        Returns:
            DataFrame preview
        """
        df = self.get_dataset(name)
        if df is not None:
            return df.head(rows)
        return pd.DataFrame()
    
    def get_dataset_schema(self, name: str) -> Dict[str, str]:
        """
        Get schema (column names and types) for a dataset
        
        Args:
            name: Dataset name
            
        Returns:
            Dictionary of column names and types
        """
        df = self.get_dataset(name)
        if df is not None:
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {}
    
    def get_version_info(self, dataset_name: str) -> Optional[Dict]:
        """
        Get version information for a dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Version information or None if not found
        """
        if dataset_name in self.dataset_versions:
            return self.dataset_versions[dataset_name].model_dump()
        return None
    
    def get_all_version_info(self) -> List[Dict]:
        """
        Get version information for all datasets
        
        Returns:
            List of version information dictionaries
        """
        return [version.model_dump() for version in self.dataset_versions.values()]
    
    def get_column_evolution(self, original_name: str, dataset: str) -> Optional[Dict]:
        """
        Get evolution information for a column
        
        Args:
            original_name: Original column name
            dataset: Dataset name
            
        Returns:
            Column evolution info or None if not found
        """
        for col_evo in self.column_evolutions:
            if col_evo.original_name == original_name and col_evo.dataset == dataset:
                return col_evo.model_dump()
        return None
    
    def get_all_column_evolutions(self) -> List[Dict]:
        """
        Get evolution information for all columns
        
        Returns:
            List of column evolution dictionaries
        """
        return [col_evo.model_dump() for col_evo in self.column_evolutions]
    
    def execute_query_plan(self, query_plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute a query plan and return results
        
        Args:
            query_plan: Query plan with datasets and operations
            
        Returns:
            DataFrame with query results
        """
        # Get required datasets
        working_datasets = {}
        for dataset_name in query_plan["datasets"]:
            if dataset_name in self.datasets:
                working_datasets[dataset_name] = self.datasets[dataset_name].copy()
        
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
                # Filter implementation
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
                        elif operator == ">=":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] >= val
                            ]
                        elif operator == "<=":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] <= val
                            ]
                        elif operator == "!=":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col] != val
                            ]
                        elif operator == "in":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col].isin(val)
                            ]
                        elif operator == "not in":
                            working_datasets[dataset] = working_datasets[dataset][
                                ~working_datasets[dataset][col].isin(val)
                            ]
                        elif operator == "contains":
                            working_datasets[dataset] = working_datasets[dataset][
                                working_datasets[dataset][col].astype(str).str.contains(str(val))
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
                join_type = operation.get("join_type", "inner")
                
                if (left in working_datasets and right in working_datasets and
                    join_col in working_datasets[left].columns and 
                    join_col in working_datasets[right].columns):
                    
                    working_datasets[left] = pd.merge(
                        working_datasets[left],
                        working_datasets[right],
                        on=join_col,
                        how=join_type
                    )
            
            elif op_type == "aggregate":
                dataset = operation["dataset"]
                agg_type = operation["aggregation_type"]
                agg_col = operation["column"]
                groupby_cols = operation.get("groupby", [])
                
                if dataset in working_datasets and agg_col in working_datasets[dataset].columns:
                    if groupby_cols:
                        # Ensure all groupby columns exist
                        valid_groupby = [col for col in groupby_cols if col in working_datasets[dataset].columns]
                        
                        if valid_groupby:
                            grouped = working_datasets[dataset].groupby(valid_groupby)
                            
                            if agg_type == "sum":
                                working_datasets[dataset] = grouped[agg_col].sum().reset_index()
                            elif agg_type == "avg" or agg_type == "mean":
                                working_datasets[dataset] = grouped[agg_col].mean().reset_index()
                            elif agg_type == "min":
                                working_datasets[dataset] = grouped[agg_col].min().reset_index()
                            elif agg_type == "max":
                                working_datasets[dataset] = grouped[agg_col].max().reset_index()
                            elif agg_type == "count":
                                working_datasets[dataset] = grouped[agg_col].count().reset_index()
        
        # Return first dataset as result
        if query_plan["datasets"]:
            first_dataset = query_plan["datasets"][0]
            if first_dataset in working_datasets:
                result_df = working_datasets[first_dataset]
            
        return result_df if result_df is not None else pd.DataFrame()
    
    def save_metadata(self, output_dir: str):
        """
        Save metadata to a file
        
        Args:
            output_dir: Directory to save metadata to
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        metadata = {
            "dataset_versions": [v.model_dump() for v in self.dataset_versions.values()],
            "column_evolutions": [c.model_dump() for c in self.column_evolutions]
        }
        
        output_file = os.path.join(output_dir, "metadata.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Saved metadata to {output_file}")
