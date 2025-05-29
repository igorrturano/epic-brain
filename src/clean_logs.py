import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from config import RAW_DATA_DIR

class LogCleaner:
    def __init__(self, directory: str = RAW_DATA_DIR):
        """
        Initialize the LogCleaner.
        
        Args:
            directory (str): The root directory containing log files
        """
        self.root_dir = Path(directory)
        
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """
        Parse a single log line into structured data.
        
        Args:
            line (str): Raw log line
            
        Returns:
            Dict containing parsed log data
        """
        try:
            # Split the line by comma, but preserve quoted strings
            parts = []
            current_part = []
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(''.join(current_part).strip())
                    current_part = []
                else:
                    current_part.append(char)
            
            if current_part:
                parts.append(''.join(current_part).strip())
            
            # Remove quotes from parts
            parts = [p.strip('"') for p in parts]
            
            # Parse date
            date_str = parts[0]
            try:
                date = datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S')
            except ValueError:
                return None
            
            return {
                'timestamp': date,
                'character': parts[1],
                'race': parts[2],
                'location': parts[3],
                'message': parts[4],
                'audience': parts[5] if len(parts) > 5 else None
            }
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {str(e)}")
            return None

    def clean_log_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Clean and structure a single log file.
        
        Args:
            file_path (Path): Path to the log file
            
        Returns:
            List of structured log entries
        """
        cleaned_entries = []
        seen_messages = set()  # To track duplicates
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Parse the line
                    entry = self.parse_log_line(line)
                    if not entry:
                        continue
                    
                    # Create a unique key for deduplication
                    message_key = f"{entry['timestamp']}_{entry['character']}_{entry['message']}"
                    
                    # Skip if we've seen this message before
                    if message_key in seen_messages:
                        continue
                        
                    seen_messages.add(message_key)
                    cleaned_entries.append(entry)
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            
        return cleaned_entries

    def process_logs(self) -> pd.DataFrame:
        """
        Process all log files in the directory.
        
        Returns:
            DataFrame containing all cleaned log entries
        """
        all_entries = []
        
        # Find all log files
        log_files = list(self.root_dir.rglob('*.log'))
        
        for file_path in log_files:
            print(f"Processing {file_path}...")
            entries = self.clean_log_file(file_path)
            all_entries.extend(entries)
            
        # Convert to DataFrame
        df = pd.DataFrame(all_entries)
        
        # Sort by timestamp
        if not df.empty:
            df = df.sort_values('timestamp')
            
        return df

    def save_cleaned_logs(self, df: pd.DataFrame, output_file: str = "cleaned_logs.csv"):
        """
        Save cleaned logs to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing cleaned logs
            output_file (str): Name of the output file
        """
        output_path = self.root_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned logs to {output_path}")

def clean_log_files(directory: str = RAW_DATA_DIR):
    """
    Clean all log files in the given directory and its subdirectories.
    
    Args:
        directory (str): The root directory to start searching for log files
    """
    cleaner = LogCleaner(directory)
    df = cleaner.process_logs()
    cleaner.save_cleaned_logs(df)

if __name__ == '__main__':
    print(f"Starting to clean log files in {RAW_DATA_DIR}...")
    clean_log_files()
    print("Finished cleaning log files.") 