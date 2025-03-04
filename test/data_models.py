"""
Tests for data models
"""

import unittest
from datetime import datetime

from src.data_models import (
    Transformation, 
    ColumnVersion, 
    ColumnEvolution, 
    DatasetVersion, 
    ParsedQuery
)


class TestDataModels(unittest.TestCase):
    """Tests for data model classes"""
    
    def test_transformation(self):
        """Test Transformation model"""
        transform = Transformation(
            operation="rename_columns",
            parameters={"old_name": "new_name"},
            rationale="For consistency"
        )
        
        self.assertEqual(transform.operation, "rename_columns")
        self.assertEqual(transform.parameters, {"old_name": "new_name"})
        self.assertEqual(transform.rationale, "For consistency")
        self.assertIsInstance(transform.timestamp, datetime)
        
    def test_column_version(self):
        """Test ColumnVersion model"""
        col_version = ColumnVersion(
            name="new_column_name",
            transformation="renamed",
            reason="For clarity"
        )
        
        self.assertEqual(col_version.name, "new_column_name")
        self.assertEqual(col_version.transformation, "renamed")
        self.assertEqual(col_version.reason, "For clarity")
        self.assertIsInstance(col_version.timestamp, datetime)
        
    def test_column_evolution(self):
        """Test ColumnEvolution model with add_version method"""
        col_evo = ColumnEvolution(
            original_name="old_name",
            dataset="test_dataset"
        )
        
        self.assertEqual(col_evo.original_name, "old_name")
        self.assertEqual(col_evo.dataset, "test_dataset")
        self.assertEqual(len(col_evo.versions), 0)
        
        # Add a version
        col_evo.add_version(
            new_name="first_change",
            transformation="renamed",
            reason="First change"
        )
        
        self.assertEqual(len(col_evo.versions), 1)
        self.assertEqual(col_evo.versions[0].name, "first_change")
        
        # Add another version
        col_evo.add_version(
            new_name="second_change",
            transformation="transformed",
            reason="Second change"
        )
        
        self.assertEqual(len(col_evo.versions), 2)
        self.assertEqual(col_evo.versions[1].name, "second_change")
        
    def test_dataset_version(self):
        """Test DatasetVersion model with add_transformation method"""
        version = DatasetVersion(
            dataset_name="test_dataset",
            version="1.0",
            parent_versions=["parent_dataset_v1"]
        )
        
        self.assertEqual(version.dataset_name, "test_dataset")
        self.assertEqual(version.version, "1.0")
        self.assertEqual(version.parent_versions, ["parent_dataset_v1"])
        self.assertEqual(len(version.transformations), 0)
        
        # Add a transformation
        version.add_transformation(
            operation="rename_columns",
            params={"old_col": "new_col"},
            rationale="Testing transformation"
        )
        
        self.assertEqual(len(version.transformations), 1)
        self.assertEqual(version.transformations[0].operation, "rename_columns")
        self.assertEqual(version.transformations[0].parameters, {"old_col": "new_col"})
        
    def test_parsed_query(self):
        """Test ParsedQuery model"""
        query = ParsedQuery(
            query_type="data_query",
            datasets=["dataset1", "dataset2"],
            columns=["col1", "col2"],
            time_range={"start": "2020-01-01", "end": "2020-12-31"},
            filters=[],
            aggregations=[]
        )
        
        self.assertEqual(query.query_type, "data_query")
        self.assertEqual(query.datasets, ["dataset1", "dataset2"])
        self.assertEqual(query.columns, ["col1", "col2"])
        self.assertEqual(query.time_range, {"start": "2020-01-01", "end": "2020-12-31"})
        self.assertEqual(query.filters, [])
        self.assertEqual(query.aggregations, [])
        

if __name__ == "__main__":
    unittest.main()
