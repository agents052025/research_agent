"""
Data Analysis Tool for the Research Agent.
Provides data analysis capabilities using pandas, numpy, and other libraries.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import io
from smolagents import Tool


class DataAnalysisTool(Tool):
    # Атрибути для smolagents.Tool
    name = "data_analysis"
    description = """
    Provides data analysis capabilities for research data.
    Supports statistical analysis, data cleaning, and transformation.
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Analysis action to perform (analyze_text, analyze_numerical, clean_data, extract_insights, compare_datasets)",
        },
        "data": {
            "type": "object",
            "description": "Data to analyze or process",
        },
        "options": {
            "type": "object",
            "description": "Optional parameters for the analysis",
            "nullable": True
        }
    }
    output_type = "object"
    
    def forward(self, action: str, data: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forward method required by smolagents.Tool.
        Dispatches to appropriate analysis methods based on the action.
        
        Args:
            action: Analysis action to perform
            data: Data to analyze or process
            options: Optional parameters for the analysis
            
        Returns:
            Result of the analysis operation
        """
        self.logger.info("Executing analysis action: %s", action)
        
        # Перевіряємо та модифікуємо опції для підвищення "depth" до "deep"
        if options is not None:
            # Якщо в опціях є параметр depth і він встановлений як "medium", змінимо його на "deep"
            if "depth" in options and options["depth"] == "medium":
                options["depth"] = "deep"
                self.logger.info("Automatically upgraded analysis depth from 'medium' to 'deep'")
            elif "detail_level" in options and options["detail_level"] == "medium":
                options["detail_level"] = "deep"
                self.logger.info("Automatically upgraded detail_level from 'medium' to 'deep'")
        
        if action == "analyze_text":
            return self.analyze_text_data(data)
        elif action == "analyze_numerical":
            return self.analyze_numerical_data(data)
        elif action == "clean_data":
            return self.clean_data(data, options)
        elif action == "extract_insights":
            topic = options.get("topic") if options else None
            return self.extract_insights(data, topic)
        elif action == "compare_datasets":
            dataset2 = options.get("dataset2") if options else None
            labels = options.get("labels") if options else None
            
            if dataset2 is None:
                return {"error": "Missing second dataset for comparison"}
                
            return self.compare_datasets(data, dataset2, labels)
        else:
            self.logger.error("Invalid analysis action: %s", action)
            return {"error": "Invalid analysis action: %s" % action}
    
    def __init__(self, analysis_packages: List[str] = None):
        """
        Initialize the Data Analysis Tool.
        
        Args:
            analysis_packages: List of analysis packages to use
        """
        self.logger = logging.getLogger(__name__)
        self.packages = analysis_packages or ["pandas", "numpy"]
        
        # Validate packages
        self.available_packages = {
            "pandas": pd,
            "numpy": np
        }
        
        for package in self.packages:
            if package not in self.available_packages:
                self.logger.warning("Unsupported analysis package: %s", package)
                
        # Додаємо атрибут is_initialized для сумісності з smolagents 1.15.0
        self.is_initialized = True
                
    def analyze_text_data(self, text_data: Any) -> Dict[str, Any]:
        """
        Analyze text data to extract basic statistics and insights.
        
        Args:
            text_data: Text data in various formats (string, list, dict, etc.)
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing text data")
        
        # Convert various data types to text for analysis
        if isinstance(text_data, str):
            text_data = [text_data]
        elif isinstance(text_data, dict):
            # Extract values from dictionary or convert whole dict to string
            try:
                if all(isinstance(v, str) for v in text_data.values()):
                    # If all values are strings, use them
                    text_data = list(text_data.values())
                else:
                    # Convert the whole dictionary to a JSON string
                    text_data = [json.dumps(text_data)]
            except Exception as e:
                self.logger.warning(f"Error converting dictionary to text: {e}")
                text_data = [str(text_data)]
        elif isinstance(text_data, list):
            # Handle list of various types
            if all(isinstance(item, str) for item in text_data):
                # List of strings - keep as is
                pass
            elif all(isinstance(item, dict) for item in text_data):
                # For lists of dictionaries, extract text from each dict
                extracted_texts = []
                for item in text_data:
                    if 'snippet' in item:
                        extracted_texts.append(item['snippet'])
                    elif 'text' in item:
                        extracted_texts.append(item['text'])
                    elif 'title' in item:
                        extracted_texts.append(item['title'])
                    else:
                        # Fallback to JSON string representation
                        extracted_texts.append(json.dumps(item))
                text_data = extracted_texts
            else:
                # Mixed list - convert each item to string
                text_data = [str(item) for item in text_data]
        else:
            # For other types, convert to string
            text_data = [str(text_data)]
            
        # Basic text statistics
        word_counts = [len(text.split()) for text in text_data]
        char_counts = [len(text) for text in text_data]
        
        # Word frequency analysis (across all texts)
        all_words = []
        for text in text_data:
            words = text.lower().split()
            all_words.extend(words)
            
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Calculate readability metrics for each text
        readability_scores = []
        for text in text_data:
            try:
                score = self._calculate_readability(text)
                readability_scores.append(score)
            except Exception as e:
                self.logger.error("Error calculating readability: %s", str(e))
                readability_scores.append(None)
                
        # Prepare result
        result = {
            "document_count": len(text_data),
            "total_words": sum(word_counts),
            "avg_words_per_doc": sum(word_counts) / len(text_data) if text_data else 0,
            "word_counts": word_counts,
            "char_counts": char_counts,
            "top_words": dict(top_words),
            "readability": readability_scores,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability scores
        """
        # Count sentences (simplistic approach)
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Count syllables (simplistic approach)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count = 1
            return count
            
        syllable_count = sum(count_syllables(word) for word in words)
        
        # Calculate scores if there's enough text
        if word_count == 0 or sentence_count == 0:
            return {"error": "Text too short for analysis"}
            
        # Flesch Reading Ease
        flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        
        # SMOG Index (simplified)
        smog = 1.043 * ((syllable_count / word_count * 100) ** 0.5) + 3.1291
        
        return {
            "flesch_reading_ease": round(flesch, 2),
            "flesch_kincaid_grade": round(flesch_kincaid, 2),
            "smog_index": round(smog, 2),
            "sentence_count": sentence_count,
            "word_count": word_count,
            "syllable_count": syllable_count
        }
        
    def analyze_numerical_data(self, data: Union[List[float], List[List[float]], Dict[str, List[float]], pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze numerical data to extract statistics and patterns.
        
        Args:
            data: Numerical data in various formats
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing numerical data")
        
        # Convert to DataFrame if not already
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            if not data:
                return {"error": "Empty data"}
            if isinstance(data[0], list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame({"values": data})
        else:
            return {"error": "Unsupported data format"}
            
        # Basic statistics
        try:
            stats = df.describe().to_dict()
            
            # Add more advanced statistics
            for column in df.select_dtypes(include=np.number).columns:
                stats[column]["median"] = df[column].median()
                stats[column]["skew"] = df[column].skew()
                stats[column]["kurtosis"] = df[column].kurtosis()
                
            # Correlation matrix if multiple columns
            corr_matrix = None
            if len(df.select_dtypes(include=np.number).columns) > 1:
                corr_matrix = df.corr().to_dict()
                
            # Prepare result
            result = {
                "statistics": stats,
                "correlation": corr_matrix,
                "data_shape": df.shape,
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing numerical data: {str(e)}")
            return {"error": str(e)}
            
    def clean_data(self, data: Union[pd.DataFrame, Dict[str, List], List[List]], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Clean and preprocess data.
        
        Args:
            data: Data to clean (DataFrame, dict, or nested list)
            options: Cleaning options
            
        Returns:
            Dictionary with cleaned data and summary of changes
        """
        self.logger.info("Cleaning data")
        
        # Set default options
        default_options = {
            "drop_na": False,
            "fill_na": None,
            "remove_duplicates": False,
            "normalize": False
        }
        
        # Merge with provided options
        opts = {**default_options, **(options or {})}
        
        # Convert to DataFrame if not already
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return {"error": "Unsupported data format"}
            
        # Track changes
        changes = {
            "original_shape": df.shape,
            "operations": []
        }
        
        # Handle missing values
        na_count_before = df.isna().sum().sum()
        
        if opts["drop_na"]:
            df = df.dropna()
            changes["operations"].append(f"Dropped rows with missing values")
        elif opts["fill_na"] is not None:
            if opts["fill_na"] == "mean":
                for col in df.select_dtypes(include=np.number).columns:
                    df[col] = df[col].fillna(df[col].mean())
            elif opts["fill_na"] == "median":
                for col in df.select_dtypes(include=np.number).columns:
                    df[col] = df[col].fillna(df[col].median())
            elif opts["fill_na"] == "mode":
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            else:
                # Fill with specific value
                df = df.fillna(opts["fill_na"])
                
            changes["operations"].append(f"Filled missing values using {opts['fill_na']}")
            
        na_count_after = df.isna().sum().sum()
        changes["missing_values_handled"] = na_count_before - na_count_after
        
        # Remove duplicates
        if opts["remove_duplicates"]:
            dup_count = df.duplicated().sum()
            df = df.drop_duplicates()
            changes["operations"].append(f"Removed {dup_count} duplicate rows")
            
        # Normalize numerical columns
        if opts["normalize"]:
            norm_cols = []
            for col in df.select_dtypes(include=np.number).columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val != max_val:  # Avoid division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    norm_cols.append(col)
                    
            if norm_cols:
                changes["operations"].append(f"Normalized columns: {', '.join(norm_cols)}")
                
        # Finalize result
        changes["final_shape"] = df.shape
        
        # Convert back to original format
        if isinstance(data, dict):
            cleaned_data = df.to_dict('list')
        elif isinstance(data, list) and not isinstance(data[0], dict):
            cleaned_data = df.values.tolist()
        else:
            cleaned_data = df
            
        return {
            "cleaned_data": cleaned_data,
            "changes": changes,
            "timestamp": datetime.now().isoformat()
        }
        
    def extract_insights(self, data: Union[pd.DataFrame, Dict, List], topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract key insights from data based on statistical analysis.
        
        Args:
            data: Data to analyze
            topic: Optional topic context to guide analysis
            
        Returns:
            Dictionary with extracted insights
        """
        self.logger.info(f"Extracting insights from data {f'on {topic}' if topic else ''}")
        
        # Check if data is None or empty
        if data is None or data == []:
            self.logger.warning("No data provided for analysis")
            return {"error": "No data available for analysis", "insights": []}
            
        # Convert to DataFrame based on data type
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            # Handle dictionary data
            try:
                df = pd.DataFrame([data])
            except Exception as e:
                self.logger.warning(f"Could not convert dictionary to DataFrame: {str(e)}")
                # Return formatted insights directly from dict
                insights = []
                for key, value in data.items():
                    insights.append({"insight": key, "description": str(value)})
                return {"insights": insights, "timestamp": datetime.now().isoformat()}
        elif isinstance(data, list):
            # Handle list data
            if len(data) == 0:
                return {"error": "Empty list provided", "insights": []}
            
            if all(isinstance(item, dict) for item in data):
                try:
                    df = pd.DataFrame(data)
                except Exception as e:
                    self.logger.error(f"Error converting list of dicts to DataFrame: {str(e)}")
                    # Process list of dicts directly
                    insights = []
                    for item in data:
                        for key, value in item.items():
                            insights.append({"insight": key, "description": str(value)})
                    return {"insights": insights, "timestamp": datetime.now().isoformat()}
            else:
                # Simple list of values
                try:
                    df = pd.DataFrame({"value": data})
                except Exception as e:
                    self.logger.error(f"Error converting list to DataFrame: {str(e)}")
                    return {"error": f"Could not convert data to analyzable format: {str(e)}", "insights": []}
        elif isinstance(data, str):
            # Handle string data - likely a JSON or text to analyze
            try:
                # Try to parse as JSON first
                json_data = json.loads(data)
                if isinstance(json_data, dict):
                    df = pd.DataFrame([json_data])
                elif isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                else:
                    # Create a simple text analysis DataFrame
                    df = pd.DataFrame({"text": [data]})
            except json.JSONDecodeError:
                # Not JSON, treat as text
                df = pd.DataFrame({"text": [data]})
            except Exception as e:
                self.logger.error(f"Error processing string data: {str(e)}")
                return {"error": f"Could not process text data: {str(e)}", "insights": [{
                    "insight": "Text Analysis", 
                    "description": "Text data provided but could not be processed as structured data."
                }]}
        else:
            # Other types
            try:
                df = pd.DataFrame([{"value": data}])
            except Exception as e:
                self.logger.error(f"Error converting unknown data type to DataFrame: {str(e)}")
                return {"error": f"Unsupported data type: {type(data)}", "insights": []}
            
        insights = []
        numerical_columns = df.select_dtypes(include=np.number).columns
        
        # Check data size
        if len(df) == 0:
            return {"insights": ["Data is empty"], "timestamp": datetime.now().isoformat()}
            
        # Basic statistics insights
        for col in numerical_columns:
            # Check for outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                insights.append({
                    "type": "outliers",
                    "column": col,
                    "description": f"Found {len(outliers)} outliers in column '{col}'",
                    "outlier_count": len(outliers),
                    "percentage": round(len(outliers) / len(df) * 100, 2)
                })
                
            # Check for skewness
            skew = df[col].skew()
            if abs(skew) > 1:
                skew_direction = "positively" if skew > 0 else "negatively"
                insights.append({
                    "type": "distribution",
                    "column": col,
                    "description": f"Column '{col}' is {skew_direction} skewed (skew={round(skew, 2)})",
                    "skew": round(skew, 2)
                })
                
        # Correlation insights
        if len(numerical_columns) > 1:
            corr_matrix = df[numerical_columns].corr()
            
            # Find strong correlations
            strong_corrs = []
            for i in range(len(numerical_columns)):
                for j in range(i + 1, len(numerical_columns)):
                    col1 = numerical_columns[i]
                    col2 = numerical_columns[j]
                    corr = corr_matrix.loc[col1, col2]
                    
                    if abs(corr) > 0.7:
                        strong_corrs.append({
                            "columns": (col1, col2),
                            "correlation": round(corr, 3),
                            "direction": "positive" if corr > 0 else "negative"
                        })
                        
            if strong_corrs:
                for corr in strong_corrs:
                    insights.append({
                        "type": "correlation",
                        "columns": corr["columns"],
                        "description": f"Strong {corr['direction']} correlation ({corr['correlation']}) between '{corr['columns'][0]}' and '{corr['columns'][1]}'",
                        "correlation": corr["correlation"]
                    })
                    
        # Trend detection (basic)
        for col in numerical_columns:
            if 'date' in df.columns or 'time' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'time'
                
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df[date_col] = pd.to_datetime(df[date_col])
                        
                    # Sort by date
                    df_sorted = df.sort_values(by=date_col)
                    
                    # Check for monotonic trend
                    if df_sorted[col].is_monotonic_increasing:
                        insights.append({
                            "type": "trend",
                            "column": col,
                            "description": f"Column '{col}' shows a consistently increasing trend over time",
                            "trend": "increasing"
                        })
                    elif df_sorted[col].is_monotonic_decreasing:
                        insights.append({
                            "type": "trend",
                            "column": col,
                            "description": f"Column '{col}' shows a consistently decreasing trend over time",
                            "trend": "decreasing"
                        })
                except Exception as e:
                    self.logger.error(f"Error in trend detection: {str(e)}")
                    
        # Group differences (if categorical columns exist)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for cat_col in categorical_columns:
            if len(df[cat_col].unique()) <= 10:  # Only for columns with few categories
                for num_col in numerical_columns:
                    group_means = df.groupby(cat_col)[num_col].mean()
                    
                    # Check if there's substantial variation between groups
                    overall_mean = df[num_col].mean()
                    largest_deviation = abs(group_means - overall_mean).max() / overall_mean
                    
                    if largest_deviation > 0.3:  # 30% deviation threshold
                        max_group = group_means.idxmax()
                        min_group = group_means.idxmin()
                        
                        insights.append({
                            "type": "group_difference",
                            "columns": (cat_col, num_col),
                            "description": f"Significant differences in '{num_col}' across '{cat_col}' groups. " +
                                         f"Highest average in '{max_group}', lowest in '{min_group}'",
                            "max_group": max_group,
                            "min_group": min_group,
                            "max_value": round(group_means.max(), 2),
                            "min_value": round(group_means.min(), 2)
                        })
                        
        # Add data summary
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "numerical_columns": len(numerical_columns),
            "categorical_columns": len(categorical_columns),
            "missing_values": df.isna().sum().sum(),
            "duplicates": df.duplicated().sum()
        }
        
        return {
            "insights": insights,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    def compare_datasets(self, dataset1: Union[pd.DataFrame, Dict, List], 
                        dataset2: Union[pd.DataFrame, Dict, List],
                        labels: List[str] = None) -> Dict[str, Any]:
        """
        Compare two datasets and identify key differences.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            labels: Optional labels for the datasets
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing datasets")
        
        # Set default labels
        if not labels or len(labels) < 2:
            labels = ["Dataset 1", "Dataset 2"]
            
        # Convert to DataFrames if not already
        if not isinstance(dataset1, pd.DataFrame):
            try:
                df1 = pd.DataFrame(dataset1)
            except Exception as e:
                return {"error": "Could not convert first dataset to DataFrame: %s" % str(e)}
        else:
            df1 = dataset1.copy()
            
        if not isinstance(dataset2, pd.DataFrame):
            try:
                df2 = pd.DataFrame(dataset2)
            except Exception as e:
                return {"error": "Could not convert second dataset to DataFrame: %s" % str(e)}
        else:
            df2 = dataset2.copy()
            
        # Compare basic properties
        comparison = {
            "sizes": {
                labels[0]: df1.shape,
                labels[1]: df2.shape,
                "difference": {
                    "rows": df1.shape[0] - df2.shape[0],
                    "columns": df1.shape[1] - df2.shape[1]
                }
            },
            "column_comparison": {
                "common": sorted(list(set(df1.columns) & set(df2.columns))),
                "only_in_" + labels[0]: sorted(list(set(df1.columns) - set(df2.columns))),
                "only_in_" + labels[1]: sorted(list(set(df2.columns) - set(df1.columns)))
            }
        }
        
        # Statistical comparison for common numerical columns
        common_cols = comparison["column_comparison"]["common"]
        numerical_comparison = {}
        
        for col in common_cols:
            # Only compare if both are numerical
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                numerical_comparison[col] = {
                    "mean_diff": df1[col].mean() - df2[col].mean(),
                    "std_diff": df1[col].std() - df2[col].std(),
                    "min_diff": df1[col].min() - df2[col].min(),
                    "max_diff": df1[col].max() - df2[col].max(),
                    "median_diff": df1[col].median() - df2[col].median()
                }
                
        comparison["numerical_comparison"] = numerical_comparison
        
        # Identify key differences and similarities
        insights = []
        
        # Size differences
        if abs(comparison["sizes"]["difference"]["rows"]) > 0.1 * max(df1.shape[0], df2.shape[0]):
            insights.append(f"Significant difference in number of rows: {labels[0]} has {df1.shape[0]} rows, {labels[1]} has {df2.shape[0]} rows")
            
        # Column differences
        if len(comparison["column_comparison"]["only_in_" + labels[0]]) > 0:
            insights.append(f"{labels[0]} has {len(comparison['column_comparison']['only_in_' + labels[0]])} unique columns not present in {labels[1]}")
            
        if len(comparison["column_comparison"]["only_in_" + labels[1]]) > 0:
            insights.append(f"{labels[1]} has {len(comparison['column_comparison']['only_in_' + labels[1]])} unique columns not present in {labels[0]}")
            
        # Statistical differences
        for col, diffs in numerical_comparison.items():
            # Check for significant mean difference
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            
            if mean1 == 0 and mean2 == 0:
                continue
                
            rel_diff = abs(diffs["mean_diff"]) / max(abs(mean1), abs(mean2))
            
            if rel_diff > 0.2:  # 20% difference threshold
                insights.append(f"Significant difference in mean of '{col}': {labels[0]} = {round(mean1, 2)}, {labels[1]} = {round(mean2, 2)}")
                
        comparison["insights"] = insights
        comparison["timestamp"] = datetime.now().isoformat()
        
        return comparison
