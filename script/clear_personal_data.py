import os
import shutil
from pathlib import Path
import toml
import json

def clear_personal_data():
    """
    Clears all personalized configuration and data from the repository.
    This includes:
    - Deleting all .csv files in the 'preference/' directory.
    - Emptying the 'keywords.json' file.
    - Resetting 'config.toml' to its default configuration.
    - Deleting all .jsonl files in the 'summarized/' directory.
    """
    
    # Define paths
    current_dir = Path(__file__).parent.parent
    preference_dir = current_dir / "preference"
    keywords_file = current_dir / "keywords.json"
    config_file = current_dir / "config.toml"
    summarized_dir = current_dir / "summarized"

    print("Starting to clear personal data...")

    # 1. Delete all .csv files in the 'preference/' directory
    if preference_dir.exists() and preference_dir.is_dir():
        for file in preference_dir.glob("*.csv"):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")
    else:
        print(f"Preference directory not found: {preference_dir}")

    # 3. Reset 'config.toml' to its default configuration
    # Read existing config to preserve non-personal settings like LLM configurations
    current_config = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                current_config = toml.load(f)
        except toml.TomlDecodeError as e:
            print(f"Warning: Could not decode existing config.toml, creating a new one. Error: {e}")
        except IOError as e:
            print(f"Warning: Could not read existing config.toml, creating a new one. Error: {e}")

    # Ensure 'data' section exists
    if "data" not in current_config:
        current_config["data"] = {}
    
    # Reset 'data' section specific personalized settings to default
    current_config["data"]["categories"] = ["cs.CL", "cs.CV", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.CY"]
    current_config["data"]["sample_rate"] = 0.004
    current_config["data"]["preference_dir"] = "preference"
    current_config["data"]["summarized_dir"] = "summarized"
    current_config["data"]["keywords_file"] = "keywords.json"
    current_config["t_path"] = "./recommendations.parquet" # Ensure t_path is also reset if it's a personalized setting

    # Remove github and openai sections if they exist, as they are managed by env vars/secrets
    if "github" in current_config:
        del current_config["github"]
    if "openai" in current_config:
        del current_config["openai"]

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            toml.dump(current_config, f)
        print(f"Reset: {config_file} personalized data configuration to default.")
    except IOError as e:
        print(f"Error resetting {config_file}: {e}")

    # 4. Delete all .jsonl files in the 'summarized/' directory
    if summarized_dir.exists() and summarized_dir.is_dir():
        for file in summarized_dir.glob("*.jsonl"):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")
    else:
        print(f"Summarized data directory not found: {summarized_dir}")

    print("Personal data clearing process completed.")

if __name__ == "__main__":
    clear_personal_data()