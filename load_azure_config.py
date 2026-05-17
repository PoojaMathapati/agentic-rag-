#!/usr/bin/env python3
"""
Script to load Azure OpenAI configuration from azure_config.env file.
This helps you set environment variables easily.
"""

import os
from pathlib import Path

def load_env_file(env_file_path: str = "azure_config.env"):
    """Load environment variables from a file."""
    env_path = Path(env_file_path)
    
    if not env_path.exists():
        print(f"❌ Environment file {env_file_path} not found!")
        print("Please create it using the template provided.")
        return False
    
    print(f"📁 Loading configuration from {env_file_path}...")
    
    loaded_vars = []
    with open(env_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Don't load placeholder values
                if value and not value.startswith('your-'):
                    os.environ[key] = value
                    loaded_vars.append(key)
                    print(f"✅ {key} = {value[:20]}...")
                else:
                    print(f"⚠️  {key} has placeholder value, skipping")
            else:
                print(f"⚠️  Line {line_num}: Invalid format: {line}")
    
    if loaded_vars:
        print(f"\n🎉 Successfully loaded {len(loaded_vars)} environment variables!")
        return True
    else:
        print("\n❌ No valid environment variables found.")
        print("Please update azure_config.env with your actual Azure OpenAI credentials.")
        return False

def verify_config():
    """Verify that all required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    print("\n🔍 Verifying configuration...")
    all_set = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value and not value.startswith('your-'):
            print(f"✅ {var} = {value[:20]}...")
        else:
            print(f"❌ {var} = Not set or placeholder value")
            all_set = False
    
    return all_set

def main():
    """Main function."""
    print("🔧 Azure OpenAI Configuration Loader")
    print("=" * 50)
    
    # Try to load from environment file
    if load_env_file():
        if verify_config():
            print("\n🚀 Configuration is ready! You can now run:")
            print("   python agentic_RAG.py")
        else:
            print("\n⚠️  Please update azure_config.env with your actual credentials.")
    else:
        print("\n📝 Please create azure_config.env with your Azure OpenAI credentials.")
        print("See azure_config.env for the template.")

if __name__ == "__main__":
    main()
