"""
Visualization Tool for the Research Agent.
Creates terminal-friendly visualizations using ASCII/Unicode characters.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from smolagents import Tool


class VisualizationTool(Tool):
    # Атрибути для smolagents.Tool
    name = "visualization"
    description = """
    Creates terminal-friendly visualizations from data.
    Supports tables, bar charts, histograms, and other visualizations.
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Visualization type to create (table, bar_chart, histogram, line_chart, scatter_plot)",
        },
        "data": {
            "type": "object",
            "description": "Data to visualize",
        },
        "options": {
            "type": "object",
            "description": "Visualization options (title, headers, labels, etc.)",
            "nullable": True
        }
    }
    output_type = "object"
    
    def forward(self, action: str, data: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forward method required by smolagents.Tool.
        Dispatches to appropriate visualization methods based on the action.
        
        Args:
            action: Visualization type to create
            data: Data to visualize
            options: Visualization options
            
        Returns:
            Result of the visualization operation
        """
        self.logger.info("Creating %s visualization", action)
        options = options or {}
        
        if action == "table":
            return self.create_table(
                data=data,
                headers=options.get("headers"),
                title=options.get("title")
            )
        elif action == "bar_chart":
            return self.create_bar_chart(
                data=data,
                labels=options.get("labels"),
                title=options.get("title"),
                sort=options.get("sort", False),
                max_items=options.get("max_items", 20)
            )
        elif action == "histogram":
            return self.create_histogram(
                data=data,
                bins=options.get("bins", 10),
                title=options.get("title")
            )
        elif action == "line_chart":
            return self.create_line_chart(
                y_values=data,
                x_values=options.get("x_values"),
                title=options.get("title"),
                y_label=options.get("y_label"),
                x_label=options.get("x_label"),
                height=options.get("height", 15)
            )
        elif action == "scatter_plot":
            return self.create_scatter_plot(
                x_values=options.get("x_values"),
                y_values=data,
                title=options.get("title"),
                x_label=options.get("x_label"),
                y_label=options.get("y_label"),
                height=options.get("height", 15),
                width=options.get("width", 40)
            )
        else:
            self.logger.error("Invalid visualization type: %s", action)
            return {"error": "Invalid visualization type: %s" % action}
    
    def __init__(self, max_width: int = 80, use_unicode: bool = True):
        """
        Initialize the Visualization Tool.
        
        Args:
            max_width: Maximum width of visualizations in characters
            use_unicode: Whether to use Unicode characters for better visuals
        """
        self.logger = logging.getLogger(__name__)
        self.max_width = max_width
        self.use_unicode = use_unicode
        
        # Додаємо атрибут is_initialized для сумісності з smolagents 1.15.0
        self.is_initialized = True
        
        # Set up Unicode characters for visualizations
        if self.use_unicode:
            self.chars = {
                "bar_h": "█",
                "bar_half": "▌",
                "bar_empty": "░",
                "line_h": "─",
                "line_v": "│",
                "corner_tl": "┌",
                "corner_tr": "┐",
                "corner_bl": "└",
                "corner_br": "┘",
                "t_down": "┬",
                "t_up": "┴",
                "t_left": "┤",
                "t_right": "├",
                "cross": "┼"
            }
        else:
            self.chars = {
                "bar_h": "#",
                "bar_half": "|",
                "bar_empty": ".",
                "line_h": "-",
                "line_v": "|",
                "corner_tl": "+",
                "corner_tr": "+",
                "corner_bl": "+",
                "corner_br": "+",
                "t_down": "+",
                "t_up": "+",
                "t_left": "+",
                "t_right": "+",
                "cross": "+"
            }
            
    def create_table(self, data: Union[pd.DataFrame, List[List[Any]], Dict[str, List[Any]]],
                    headers: Optional[List[str]] = None, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a formatted table visualization.
        
        Args:
            data: Data to visualize
            headers: Column headers (optional if data is DataFrame)
            title: Table title
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating table visualization")
        
        # Convert to DataFrame if not already
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                # Перевірка на випадок, якщо словник містить неоднорідні дані
                try:
                    df = pd.DataFrame(data)
                except ValueError as e:
                    self.logger.warning(f"Error converting dict to DataFrame: {e}")
                    # Спроба перетворити словник у більш простий формат
                    simple_data = {}
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            simple_data[key] = str(value)
                        else:
                            simple_data[key] = value
                    df = pd.DataFrame([simple_data])
            elif isinstance(data, list):
                # Перевірка, чи список містить словники або інші складні типи
                if data and isinstance(data[0], tuple) and len(data[0]) == 2:
                    # Список кортежів (ключ, значення)
                    keys = [item[0] for item in data]
                    values = [item[1] for item in data]
                    df = pd.DataFrame({"Key": keys, "Value": values})
                elif headers:
                    df = pd.DataFrame(data, columns=headers)
                else:
                    df = pd.DataFrame(data)
            else:
                # Для примітивних типів створюємо простий DataFrame
                df = pd.DataFrame({"Value": [str(data)]})
        except Exception as e:
            self.logger.error(f"Failed to convert data to DataFrame: {e}")
            return {"error": f"Failed to create table: {str(e)}", "raw_data": str(data)[:1000]}
            
        # Use DataFrame column names if headers not provided
        if not headers:
            headers = df.columns.tolist()
            
        # Limit to first 1000 rows if larger to avoid excessive output
        if len(df) > 1000:
            df = df.head(1000)
            truncated = True
        else:
            truncated = False
            
        # Convert DataFrame to list of lists
        rows = df.values.tolist()
        
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
                
        # Add padding
        col_widths = [w + 2 for w in col_widths]
        
        # Ensure table doesn't exceed max width
        total_width = sum(col_widths) + len(col_widths) - 1
        if total_width > self.max_width:
            # Scale down columns proportionally
            scale_factor = self.max_width / total_width
            col_widths = [max(4, int(w * scale_factor)) for w in col_widths]
            
        # Create table header
        h_line = self.chars["corner_tl"]
        for i, width in enumerate(col_widths):
            h_line += self.chars["line_h"] * width
            if i < len(col_widths) - 1:
                h_line += self.chars["t_down"]
            else:
                h_line += self.chars["corner_tr"]
                
        # Create header row
        header_row = self.chars["line_v"]
        for i, header in enumerate(headers):
            header_text = str(header)
            if len(header_text) > col_widths[i] - 2:
                header_text = header_text[:col_widths[i] - 3] + "…"
            header_row += " " + header_text.ljust(col_widths[i] - 1) + self.chars["line_v"]
            
        # Create separator line
        sep_line = self.chars["t_right"]
        for i, width in enumerate(col_widths):
            sep_line += self.chars["line_h"] * width
            if i < len(col_widths) - 1:
                sep_line += self.chars["cross"]
            else:
                sep_line += self.chars["t_left"]
                
        # Create table rows
        table_rows = []
        for row in rows:
            table_row = self.chars["line_v"]
            for i, cell in enumerate(row):
                cell_text = str(cell)
                if len(cell_text) > col_widths[i] - 2:
                    cell_text = cell_text[:col_widths[i] - 3] + "…"
                table_row += " " + cell_text.ljust(col_widths[i] - 1) + self.chars["line_v"]
            table_rows.append(table_row)
            
        # Create bottom line
        b_line = self.chars["corner_bl"]
        for i, width in enumerate(col_widths):
            b_line += self.chars["line_h"] * width
            if i < len(col_widths) - 1:
                b_line += self.chars["t_up"]
            else:
                b_line += self.chars["corner_br"]
                
        # Assemble table
        table_lines = []
        if title:
            # Create title centered over the table
            title_line = title.center(len(h_line))
            table_lines.append(title_line)
            
        table_lines.append(h_line)
        table_lines.append(header_row)
        table_lines.append(sep_line)
        table_lines.extend(table_rows)
        table_lines.append(b_line)
        
        # Add truncation note if needed
        if truncated:
            truncate_msg = "Note: Showing first 1000 rows of %d total rows" % len(df)
            truncate_msg = truncate_msg.center(len(h_line))
            table_lines.append(truncate_msg)
            
        # Join lines to create final table
        table_str = "\n".join(table_lines)
        
        return {
            "visualization": table_str,
            "type": "table",
            "rows": len(rows),
            "columns": len(headers),
            "truncated": truncated,
            "timestamp": datetime.now().isoformat()
        }
        
    def create_bar_chart(self, data: Union[Dict[str, Union[int, float]], List[Tuple[str, Union[int, float]]],
                                     pd.Series, List[Union[int, float, dict]], List[Dict[str, Any]]],
                     labels: Optional[List[str]] = None, title: Optional[str] = None,
                     sort: bool = False, max_items: int = 20) -> Dict[str, Any]:
        """
        Create a horizontal bar chart visualization.
        
        Args:
            data: Data to visualize (dict mapping labels to values, list of values, or list of dicts)
            labels: Labels for values (if data is a list of values)
            title: Chart title
            sort: Whether to sort bars by value
            max_items: Maximum number of items to display
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating bar chart visualization")
        
        # Process data into consistent format: list of (label, value) tuples
        if isinstance(data, dict):
            items = list(data.items())
        elif isinstance(data, pd.Series):
            items = list(zip(data.index.astype(str), data.values))
        elif isinstance(data, list):
            # Список словників зі спеціальними ключами Insight/Count
            if all(isinstance(x, dict) and "Insight" in x and "Count" in x for x in data):
                items = [(item["Insight"], item["Count"]) for item in data]
            # Список словників з довільними ключами (беремо перші два ключі)
            elif all(isinstance(x, dict) and len(x) >= 2 for x in data):
                items = []
                for item_dict in data:
                    keys = list(item_dict.keys())
                    if len(keys) >= 2:
                        # Беремо перші два ключі в словнику як мітку та значення
                        items.append((str(item_dict[keys[0]]), item_dict[keys[1]] if isinstance(item_dict[keys[1]], (int, float)) else 1))
                    else:
                        # Якщо тільки один ключ, використовуємо його як мітку та 1 як значення
                        items.append((str(item_dict[keys[0]]), 1))
            # Звичайний список числових значень
            elif all(isinstance(x, (int, float)) for x in data):
                if labels and len(labels) >= len(data):
                    items = list(zip(labels, data))
                else:
                    item_labels = ["Item %d" % (i+1) for i in range(len(data))]
                    items = list(zip(item_labels, data))
            # Список кортежів (мітка, значення)
            elif all(isinstance(x, tuple) and len(x) == 2 for x in data):
                items = data
            else:
                # Для будь-яких інших форматів використовуємо рядкове представлення як мітку
                # і позицію в списку як значення
                items = [(str(x), i+1) for i, x in enumerate(data)]
        else:
            return {"error": "Unsupported data format for bar chart"}
            
        # Sort if requested
        if sort:
            items = sorted(items, key=lambda x: x[1], reverse=True)
            
        # Limit number of items
        if len(items) > max_items:
            items = items[:max_items]
            truncated = True
        else:
            truncated = False
            
        # Find the maximum value and label length
        # Ensure numerical values for max calculation, convert strings to length if needed
        max_value = 0
        for _, val in items:
            if isinstance(val, (int, float)):
                max_value = max(max_value, abs(val))
            elif isinstance(val, str):
                # For string values, use string length or a default value
                max_value = max(max_value, len(val))
            else:
                # For other types, convert to string and use length
                max_value = max(max_value, len(str(val)))
                
        max_label_len = max(len(str(x[0])) for x in items)
        
        # Calculate bar width based on max value and terminal width
        # Reserve space for label and value
        label_space = min(max_label_len, 20) + 2  # Limit label to 20 chars
        value_space = 10  # Space for the value
        bar_space = self.max_width - label_space - value_space
        
        # Create chart
        chart_lines = []
        if title:
            chart_lines.append(title)
            chart_lines.append("")
            
        for label, value in items:
            # Format label
            label_str = str(label)
            if len(label_str) > label_space - 2:
                label_str = label_str[:label_space - 3] + "…"
            label_str = label_str.ljust(label_space)
            
            # Calculate bar length
            bar_length = int((abs(value) / max_value) * bar_space) if max_value > 0 else 0
            
            # Create bar
            if value >= 0:
                # Positive value
                bar = self.chars["bar_h"] * bar_length
            else:
                # Negative value
                bar = "-" + self.chars["bar_h"] * (bar_length - 1) if bar_length > 0 else ""
                
            # Format value
            if abs(value) >= 1000000:
                value_str = "%.1fM" % (value/1000000)
            elif abs(value) >= 1000:
                value_str = "%.1fK" % (value/1000)
            else:
                value_str = "%.1f" % value if isinstance(value, float) else str(value)
                
            # Combine components
            chart_lines.append("%s %s %s" % (label_str, bar, value_str))
            
        # Add truncation note if needed
        if truncated:
            chart_lines.append("")
            chart_lines.append("Note: Showing %d of %d items" % (max_items, len(data)))
            
        # Join lines to create final chart
        chart_str = "\n".join(chart_lines)
        
        return {
            "visualization": chart_str,
            "type": "bar_chart",
            "items": len(items),
            "truncated": truncated,
            "timestamp": datetime.now().isoformat()
        }
        
    def create_histogram(self, data: Union[List[Union[int, float]], np.ndarray, pd.Series],
                         bins: int = 10, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a histogram visualization.
        
        Args:
            data: Numerical data to visualize
            bins: Number of bins
            title: Histogram title
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating histogram visualization")
        
        # Convert data to numpy array
        if isinstance(data, pd.Series):
            values = data.values
        elif isinstance(data, list):
            values = np.array(data)
        elif isinstance(data, np.ndarray):
            values = data
        else:
            return {"error": "Unsupported data format for histogram"}
            
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            self.logger.error("No valid data points for histogram")
            return {"error": "No valid data points for histogram"}
            
        # Calculate histogram
        try:
            hist, bin_edges = np.histogram(values, bins=bins)
        except Exception as e:
            self.logger.error("Error calculating histogram: %s", str(e))
            return {"error": "Failed to calculate histogram: %s" % str(e)}
            
        # Find the maximum count for scaling
        max_count = max(hist)
        
        # Calculate bin labels
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if abs(start) >= 1000000 or abs(end) >= 1000000:
                label = "%.1fM-%.1fM" % (start/1000000, end/1000000)
            elif abs(start) >= 1000 or abs(end) >= 1000:
                label = "%.1fK-%.1fK" % (start/1000, end/1000)
            else:
                label = "%.1f-%.1f" % (start, end)
            bin_labels.append(label)
            
        # Calculate bar width based on terminal width
        max_label_len = max(len(label) for label in bin_labels)
        label_space = max_label_len + 2
        count_space = 7  # Space for count display
        bar_space = self.max_width - label_space - count_space
        
        # Create histogram
        hist_lines = []
        if title:
            hist_lines.append(title)
            hist_lines.append("")
            
        for i, count in enumerate(hist):
            # Format label
            label = bin_labels[i].ljust(label_space)
            
            # Calculate bar length
            bar_length = int((count / max_count) * bar_space) if max_count > 0 else 0
            
            # Create bar
            bar = self.chars["bar_h"] * bar_length
            
            # Combine components
            hist_lines.append("%s %s %d" % (label, bar, count))
            
        # Add summary statistics
        hist_lines.append("")
        hist_lines.append("Total values: %d" % len(values))
        hist_lines.append("Min: %.2f" % np.min(values))
        hist_lines.append("Max: %.2f" % np.max(values))
        hist_lines.append("Mean: %.2f" % np.mean(values))
        hist_lines.append("Median: %.2f" % np.median(values))
        hist_lines.append("Std Dev: %.2f" % np.std(values))
        
        # Join lines to create final histogram
        hist_str = "\n".join(hist_lines)
        
        return {
            "visualization": hist_str,
            "type": "histogram",
            "bins": bins,
            "data_points": len(values),
            "timestamp": datetime.now().isoformat()
        }
        
    def create_line_chart(self, y_values: Union[List[Union[int, float]], np.ndarray, pd.Series],
                         x_values: Optional[Union[List[Any], np.ndarray, pd.Series]] = None,
                         title: Optional[str] = None, y_label: Optional[str] = None,
                         x_label: Optional[str] = None, height: int = 15) -> Dict[str, Any]:
        """
        Create a line chart visualization.
        
        Args:
            y_values: Y-axis values
            x_values: X-axis values or labels (optional)
            title: Chart title
            y_label: Y-axis label
            x_label: X-axis label
            height: Chart height in lines
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating line chart visualization")
        
        # Convert y_values to list
        if isinstance(y_values, pd.Series):
            y_data = y_values.values.tolist()
            if x_values is None:
                x_data = y_values.index.tolist()
            else:
                x_data = x_values
        elif isinstance(y_values, np.ndarray):
            y_data = y_values.tolist()
            if x_values is None:
                x_data = list(range(len(y_data)))
            else:
                x_data = x_values
        elif isinstance(y_values, list):
            y_data = y_values
            if x_values is None:
                x_data = list(range(len(y_data)))
            else:
                x_data = x_values
        else:
            return {"error": "Unsupported data format for line chart"}
            
        # Convert x_values to list if necessary
        if isinstance(x_data, (np.ndarray, pd.Series)):
            x_data = x_data.tolist()
            
        # Ensure x_data and y_data have the same length
        if len(x_data) != len(y_data):
            return {"error": "X and Y data must have the same length"}
            
        # Find min and max y values for scaling
        y_min = min(y_data)
        y_max = max(y_data)
        
        if y_min == y_max:
            # Avoid division by zero if all values are the same
            y_min -= 1
            y_max += 1
            
        # Width available for the chart
        chart_width = self.max_width
        if len(x_data) > chart_width:
            # If too many points, sample to fit width
            step = len(x_data) // chart_width + 1
            x_data = x_data[::step]
            y_data = y_data[::step]
            sampled = True
        else:
            sampled = False
            
        # Create canvas
        canvas = [[' ' for _ in range(chart_width)] for _ in range(height)]
        
        # Plot points on canvas
        for i, y in enumerate(y_data):
            if i >= chart_width:
                break
                
            # Calculate y position on canvas
            y_pos = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
            y_pos = max(0, min(height - 1, y_pos))
            
            # Plot point
            canvas[y_pos][i] = '●' if self.use_unicode else 'o'
            
            # Connect to previous point if not first point
            if i > 0:
                prev_y = y_data[i - 1]
                prev_y_pos = height - 1 - int((prev_y - y_min) / (y_max - y_min) * (height - 1))
                prev_y_pos = max(0, min(height - 1, prev_y_pos))
                
                # Draw line between points
                if prev_y_pos < y_pos:
                    for j in range(prev_y_pos + 1, y_pos):
                        canvas[j][i - 1] = '│' if self.use_unicode else '|'
                elif prev_y_pos > y_pos:
                    for j in range(y_pos + 1, prev_y_pos):
                        canvas[j][i - 1] = '│' if self.use_unicode else '|'
                
        # Create chart
        chart_lines = []
        if title:
            chart_lines.append(title.center(chart_width))
            chart_lines.append("")
            
        # Add y-axis labels
        for i, row in enumerate(canvas):
            # Calculate y value for this row
            y_value = y_max - (y_max - y_min) * i / (height - 1)
            
            # Format y value
            if abs(y_value) >= 1000000:
                y_str = f"{y_value/1000000:.1f}M"
            elif abs(y_value) >= 1000:
                y_str = f"{y_value/1000:.1f}K"
            else:
                y_str = f"{y_value:.1f}"
                
            # Add y label and row
            y_label_str = y_str.rjust(8)
            chart_lines.append("%s %s" % (y_label_str, ''.join(row)))
            
        # Add x-axis
        x_axis = "─" * chart_width if self.use_unicode else "-" * chart_width
        chart_lines.append(" " * 8 + x_axis)
        
        # Add x-axis labels (show a subset to avoid overcrowding)
        if len(x_data) > 0:
            max_labels = min(10, len(x_data))
            step = len(x_data) // max_labels
            x_label_positions = [i * step for i in range(max_labels)]
            
            if len(x_data) - 1 not in x_label_positions:
                x_label_positions[-1] = len(x_data) - 1
                
            x_labels = " " * 8
            for pos in x_label_positions:
                label = str(x_data[pos])
                if len(label) > 10:
                    label = label[:7] + "..."
                    
                x_pos = int(pos * chart_width / len(x_data))
                padding = x_pos - len(x_labels)
                if padding > 0:
                    x_labels += " " * padding + label
                
            chart_lines.append(x_labels)
            
        # Add axis labels
        if y_label:
            y_label_line = y_label.center(8)
            chart_lines.append(y_label_line)
            
        if x_label:
            x_label_line = " " * 8 + x_label.center(chart_width)
            chart_lines.append(x_label_line)
            
        # Add sampling note if needed
        if sampled:
            chart_lines.append("")
            chart_lines.append("Note: Data sampled to fit chart width (showing %d of %d points)" % (len(y_data), len(y_values)))
            
        # Join lines to create final chart
        chart_str = "\n".join(chart_lines)
        
        return {
            "visualization": chart_str,
            "type": "line_chart",
            "data_points": len(y_data),
            "original_points": len(y_values),
            "sampled": sampled,
            "y_range": [y_min, y_max],
            "timestamp": datetime.now().isoformat()
        }
        
    def create_scatter_plot(self, x_values: Union[List[Union[int, float]], np.ndarray, pd.Series],
                           y_values: Union[List[Union[int, float]], np.ndarray, pd.Series],
                           title: Optional[str] = None, x_label: Optional[str] = None,
                           y_label: Optional[str] = None, height: int = 15, width: int = 40) -> Dict[str, Any]:
        """
        Create a scatter plot visualization.
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            height: Plot height in lines
            width: Plot width in characters
            
        Returns:
            Dictionary with visualization results
        """
        self.logger.info("Creating scatter plot visualization")
        
        # Convert to lists
        if isinstance(x_values, (np.ndarray, pd.Series)):
            x_data = x_values.tolist()
        elif isinstance(x_values, list):
            x_data = x_values
        else:
            return {"error": "Unsupported data format for x_values"}
            
        if isinstance(y_values, (np.ndarray, pd.Series)):
            y_data = y_values.tolist()
        elif isinstance(y_values, list):
            y_data = y_values
        else:
            return {"error": "Unsupported data format for y_values"}
            
        # Ensure x_data and y_data have the same length
        if len(x_data) != len(y_data):
            return {"error": "X and Y data must have the same length"}
            
        # Limit width to max_width
        width = min(width, self.max_width - 10)  # Reserve space for y-axis labels
        
        # Find min and max values for scaling
        x_min = min(x_data)
        x_max = max(x_data)
        y_min = min(y_data)
        y_max = max(y_data)
        
        # Avoid division by zero if all values are the same
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        if y_min == y_max:
            y_min -= 1
            y_max += 1
            
        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points on canvas
        for i in range(len(x_data)):
            # Calculate position on canvas
            x_pos = int((x_data[i] - x_min) / (x_max - x_min) * (width - 1))
            y_pos = height - 1 - int((y_data[i] - y_min) / (y_max - y_min) * (height - 1))
            
            # Ensure position is within canvas
            x_pos = max(0, min(width - 1, x_pos))
            y_pos = max(0, min(height - 1, y_pos))
            
            # Plot point
            canvas[y_pos][x_pos] = '•' if self.use_unicode else '.'
            
        # Create plot
        plot_lines = []
        if title:
            plot_lines.append(title.center(width + 10))  # Add padding for y-axis labels
            plot_lines.append("")
            
        # Add y-axis labels and plot
        for i, row in enumerate(canvas):
            # Calculate y value for this row
            y_value = y_max - (y_max - y_min) * i / (height - 1)
            
            # Format y value
            if abs(y_value) >= 1000000:
                y_str = f"{y_value/1000000:.1f}M"
            elif abs(y_value) >= 1000:
                y_str = f"{y_value/1000:.1f}K"
            else:
                y_str = f"{y_value:.1f}"
                
            # Add y label and row
            y_label_str = y_str.rjust(8)
            plot_lines.append("%s %s" % (y_label_str, ''.join(row)))
            
        # Add x-axis
        x_axis = "─" * width if self.use_unicode else "-" * width
        plot_lines.append(" " * 8 + x_axis)
        
        # Add x-axis labels
        max_labels = 5
        step = width // max_labels
        x_label_positions = [i * step for i in range(max_labels)]
        x_label_positions.append(width - 1)
        
        x_labels = " " * 8
        for pos in x_label_positions:
            x_value = x_min + (x_max - x_min) * pos / (width - 1)
            
            # Format x value
            if abs(x_value) >= 1000000:
                x_str = "%.1fM" % (x_value/1000000)
            elif abs(x_value) >= 1000:
                x_str = "%.1fK" % (x_value/1000)
            else:
                x_str = "%.1f" % x_value
                
            x_labels += " " * (pos - len(x_labels)) + x_str
            
        plot_lines.append(x_labels)
        
        # Add axis labels
        if y_label:
            y_label_line = y_label.center(8)
            plot_lines.append(y_label_line)
            
        if x_label:
            x_label_line = " " * 8 + x_label.center(width)
            plot_lines.append(x_label_line)
            
        # Add data info
        plot_lines.append("")
        plot_lines.append("Points: %d" % len(x_data))
        
        # Join lines to create final plot
        plot_str = "\n".join(plot_lines)
        
        return {
            "visualization": plot_str,
            "type": "scatter_plot",
            "data_points": len(x_data),
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "timestamp": datetime.now().isoformat()
        }
