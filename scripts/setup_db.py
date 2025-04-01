"""
Database setup script.
This script sets up the required tables and functions in the Supabase database.
"""
import os
import sys
import argparse

# Make sure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.db import create_supabase_client
from config.settings import get_settings

def read_sql_file(filename):
    """Read SQL from a file"""
    with open(filename, 'r') as f:
        return f.read()

def main():
    """Main entry point for the database setup script"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Set up database tables and functions')
    parser.add_argument('--sql-file', type=str, default='db/site_pages.sql',
                       help='SQL file to execute (default: db/site_pages.sql)')
    parser.add_argument('--ollama', action='store_true',
                       help='Use Ollama-specific SQL file (db/ollama_site_pages.sql)')
    
    args = parser.parse_args()
    
    # Get settings
    settings = get_settings()
    
    # Validate required settings
    missing = settings.validate()
    if missing:
        print(f"Missing required settings: {', '.join(missing)}")
        return
    
    # Create client
    try:
        supabase = create_supabase_client()
    except ValueError as e:
        print(f"Error creating Supabase client: {e}")
        return
    
    # Determine SQL file to use
    if args.ollama:
        sql_file = 'db/ollama_site_pages.sql'
    else:
        sql_file = args.sql_file
    
    # Read SQL file
    try:
        sql = read_sql_file(sql_file)
    except Exception as e:
        print(f"Error reading SQL file {sql_file}: {e}")
        return
    
    # Split into statements
    statements = sql.split(';')
    
    # Execute statements
    print(f"Executing SQL from {sql_file}...")
    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue
        
        try:
            # Add semicolon back for execution
            result = supabase.execute(f"{statement};")
            print(f"Executed statement: {statement[:50]}...")
        except Exception as e:
            print(f"Error executing statement: {statement[:50]}...\nError: {e}")
    
    print("Database setup complete.")

if __name__ == "__main__":
    main() 