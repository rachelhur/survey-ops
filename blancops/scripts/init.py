import os
import argparse
from pathlib import Path
import importlib.resources as pkg_resources

def main():
    parser = argparse.ArgumentParser(description="Initialize workspace.")
    parser.add_argument(
        '--workspace', 
        type=Path, 
        default=Path(os.getenv("SURVEY_OPS_WORKSPACE", Path.home() / ".survey_ops")),
        help="Target directory to initialize the workspace. Defaults to ~/.survey_ops"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Overwrite existing configuration files if they already exist."
    )
    
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    print(f"Initializing workspace at: {workspace}")

    # 1. Create the necessary directory structure
    directories_to_create = [
        workspace,
        workspace / "configs",
        workspace / "experiments",
        workspace / "models",
        workspace / "data" / "lookups"
    ]
    
    for dir_path in directories_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [+] Created directory: {dir_path}")

    # 2. Copy the default global_config.json out of the package
    config_dest = workspace / "configs" / "global_config.json"
    
    if config_dest.exists() and not args.force:
        print(f"  [!] Config already exists at {config_dest}. Use --force to overwrite.")
    else:
        try:
            # Copy global_config.json from within package to config_dest
            config_text = pkg_resources.files('survey_ops.configs').joinpath('global_config.json').read_text()
            config_dest.write_text(config_text)
            print(f"  [+] Copied default global_config.json to: {config_dest}")
        except Exception as e:
            print(f"  [!] Failed to copy config. Reason: {e}")
    # save workspace pointer file
    pointer_file = Path.home() / ".survey_ops_profile"
    pointer_file.write_text(str(workspace))
    print(f"  [+] Saved workspace pointer to {pointer_file}")

    print("\nInitialization complete!")

if __name__ == "__main__":
    main()