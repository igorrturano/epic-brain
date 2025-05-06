import os
import re
from pathlib import Path
from config import RAW_DATA_DIR

def clean_markdown_files(directory=RAW_DATA_DIR):
    """
    Remove entire lines containing [Image] from all markdown files in the given directory and its subdirectories.
    
    Args:
        directory (str): The root directory to start searching for markdown files
    """
    # Convert to Path object for better path handling
    root_dir = Path(directory)
    
    # Find all markdown files
    markdown_files = list(root_dir.rglob('*.md'))
    
    # Process each markdown file
    for file_path in markdown_files:
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Filter out lines containing [Image]
            cleaned_lines = [line for line in lines if '[Image]' not in line]
            
            # Write back only if changes were made
            if len(cleaned_lines) != len(lines):
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(cleaned_lines)
                print(f"Cleaned {file_path}")
            else:
                print(f"No [Image] lines found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == '__main__':
    print(f"Starting to clean markdown files in {RAW_DATA_DIR}...")
    clean_markdown_files()
    print("Finished cleaning markdown files.") 