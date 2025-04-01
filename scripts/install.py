#!/usr/bin/env python
"""
Installation script for the Archon agent.
This script helps users install the required dependencies and set up the project.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_venv():
    """Create a virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    # Determine the pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:  # Unix/MacOS
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    # Upgrade pip
    subprocess.run([pip_path, 'install', '--upgrade', 'pip'])
    
    return pip_path

def install_requirements(pip_path):
    """Install requirements"""
    print("Installing requirements...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'])

def setup_env_file():
    """Set up the .env file"""
    if not os.path.exists('.env'):
        print("Creating .env file from .env.example...")
        example_path = Path('.env.example')
        if example_path.exists():
            with open(example_path, 'r') as f:
                example_content = f.read()
            
            with open('.env', 'w') as f:
                f.write(example_content)
            
            print(".env file created. Please edit it with your API keys and configuration.")
        else:
            print("Warning: .env.example not found. Please create a .env file manually.")
    else:
        print(".env file already exists.")

def create_workbench():
    """Create the workbench directory"""
    workbench_dir = Path('workbench')
    if not workbench_dir.exists():
        print("Creating workbench directory...")
        workbench_dir.mkdir()
    else:
        print("Workbench directory already exists.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Install Archon Agent")
    parser.add_argument('--no-venv', action='store_true', help="Skip virtual environment creation")
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Create virtual environment
    if args.no_venv:
        print("Skipping virtual environment creation...")
        pip_path = 'pip'
    else:
        pip_path = create_venv()
    
    # Install requirements
    install_requirements(pip_path)
    
    # Set up .env file
    setup_env_file()
    
    # Create workbench directory
    create_workbench()
    
    print("\nInstallation complete!")
    if not args.no_venv:
        if os.name == 'nt':  # Windows
            print("\nTo activate the virtual environment, run:")
            print("venv\\Scripts\\activate")
        else:  # Unix/MacOS
            print("\nTo activate the virtual environment, run:")
            print("source venv/bin/activate")
    
    print("\nNext steps:")
    print("1. Edit the .env file with your API keys and configuration.")
    print("2. Set up the database: python scripts/setup_db.py")
    print("3. Crawl the documentation: python scripts/crawler.py")
    print("4. Run the application: streamlit run main.py")

if __name__ == "__main__":
    main() 