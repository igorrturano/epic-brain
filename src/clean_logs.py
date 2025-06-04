import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from config import RAW_DATA_DIR, LOG_CONFIG, LOGS_DIR

class LogCleaner:
    def __init__(self, directory: str = RAW_DATA_DIR):
        """
        Initialize the LogCleaner.
        
        Args:
            directory (str): The root directory containing log files
        """
        self.root_dir = Path(directory)
        self.max_days_old = LOG_CONFIG["max_days_old"]
        self.date_format = LOG_CONFIG["date_format"]
        
    def is_file_recent(self, file_path: Path) -> bool:
        """
        Check if the log file is within the configured time window.
        
        Args:
            file_path (Path): Path to the log file
            
        Returns:
            bool: True if the file is recent enough to process
        """
        try:
            # Extract date from path (e.g., raw/players/17-01-2025/player1.log)
            date_str = file_path.parent.name
            file_date = datetime.strptime(date_str, self.date_format)
            
            # Calculate the cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.max_days_old)
            
            return file_date >= cutoff_date
        except ValueError:
            print(f"Could not parse date from path: {file_path}")
            return False

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
        Process all log files in the directory that are within the configured time window.
        
        Returns:
            DataFrame containing all cleaned log entries
        """
        all_entries = []
        
        # Find all log files matching the pattern recursively
        log_files = list(self.root_dir.rglob('*.log'))
        
        if not log_files:
            print(f"No log files found in {self.root_dir} or its subdirectories")
            return pd.DataFrame()
            
        print(f"Found {len(log_files)} log files to process")
        
        for file_path in log_files:
            # Skip files that are too old
            if not self.is_file_recent(file_path):
                print(f"Skipping old file: {file_path}")
                continue
                
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
        
        # Delete old cleaned log files
        try:
            # Find all existing cleaned log files
            old_files = list(self.root_dir.glob("cleaned_logs*.csv"))
            for old_file in old_files:
                try:
                    old_file.unlink()
                    print(f"Deleted old file: {old_file}")
                except Exception as e:
                    print(f"Warning: Could not delete {old_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error while cleaning up old files: {str(e)}")
        
        # Save with proper quoting and escaping
        df.to_csv(
            output_path,
            index=False,
            quoting=1,  # QUOTE_ALL - quote all fields
            escapechar='\\',  # Use backslash as escape character
            encoding='utf-8'
        )
        print(f"Saved cleaned logs to {output_path}")
        
        # Verify the file can be read back correctly
        try:
            test_df = pd.read_csv(output_path, encoding='utf-8')
            if len(test_df) == len(df):
                print("Verification successful: File can be read back correctly")
            else:
                print("Warning: Number of rows in saved file doesn't match original data")
        except Exception as e:
            print(f"Warning: Could not verify saved file: {str(e)}")

def clean_log_files(directory: str = LOGS_DIR):
    """
    Clean all log files in the given directory and its subdirectories.
    
    Args:
        directory (str): The root directory to start searching for log files
    """
    # Create necessary directories
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    cleaner = LogCleaner(directory)
    df = cleaner.process_logs()
    
    if df.empty:
        print("No log files found to process.")
        return
        
    cleaner.save_cleaned_logs(df)

if __name__ == '__main__':
    print(f"Starting to clean log files in {LOGS_DIR}...")
    clean_log_files()
    print("Finished cleaning log files.") 