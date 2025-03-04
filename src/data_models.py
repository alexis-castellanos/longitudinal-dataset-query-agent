"""
Data models for LongitudinalLLM

This module defines the Pydantic models used to represent dataset metadata,
transformations, and query structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field

class Transformation(BaseModel):
    """Record of a data transformation operation"""
    operation: str = Field(description="The type of transformation applied")
    parameters: Dict[str, Any] = Field(description="Parameters used for the transformation")
    rationale: str = Field(description="Explanation of why this transformation was needed")
    timestamp: datetime = Field(default_factory=datetime.now)


class ColumnVersion(BaseModel):
    """Record of a column version"""
    name: str = Field(description="The column name")
    transformation: Optional[str] = Field(None, description="Transformation applied to create this version")
    reason: Optional[str] = Field(None, description="Reason for the change")
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


class Filter(BaseModel):
    """Filter condition for queries"""
    column: str
    operator: str  # "==", ">", "<", ">=", "<=", "!=", "in", "not in", "contains"
    value: Any


class Aggregation(BaseModel):
    """Aggregation operation for queries"""
    type: str  # "sum", "avg", "min", "max", "count"
    column: str
    groupby: Optional[List[str]] = None


class ParsedQuery(BaseModel):
    """Structured representation of a natural language query"""
    query_type: str = Field(description="Type of query: data_query or verification_query")
    datasets: List[str] = Field(description="Datasets to query")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range specification")
    columns: List[str] = Field(description="Columns or variables requested")
    filters: Optional[List[Filter]] = Field(default_factory=list)
    aggregations: Optional[List[Aggregation]] = Field(default_factory=list)
    transformations: Optional[Dict[str, List[Dict[str, Any]]]] = Field(default_factory=dict)
    
    # For verification queries
    verification_aspects: Optional[List[str]] = Field(default_factory=list)
    verification_columns: Optional[List[str]] = Field(default_factory=list)
    verification_periods: Optional[List[str]] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Plan for executing a query"""
    datasets: List[str] = Field(description="Datasets to use")
    operations: List[Dict[str, Any]] = Field(description="Operations to perform")
