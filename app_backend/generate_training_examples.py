#!/usr/bin/env -S uv run
"""
Script to automatically generate training examples for description_generator.py

This script takes commit hashes and automatically generates the TrainingRunInput data
by fetching commit information and diff stats from git, eliminating the need for manual
creation of training examples.
"""

import argparse
import sys
from typing import List

from app_backend.git_client import GitClient, GitCommit
from app_backend.description_generator import TrainingRunInput


def generate_training_example(commit_hash: str, expected_response: str, base_branch: str = "main") -> str:
    """
    Generate a training example for the given commit hash.
    
    Args:
        commit_hash: The commit hash to generate an example for
        expected_response: The expected LLM response for this example
        base_branch: Base branch to compare against
        
    Returns:
        Python code string for the training example
    """
    git_client = GitClient()
    
    try:
        # Get commit information
        commits = git_client.get_commit_range(commit_hash, base_branch)
        if not commits:
            commits = [git_client.get_commit_range(commit_hash, commit_hash)[0]]  # Single commit case
            
        # Get file stats
        file_stats = git_client.get_range_diff_stats(commit_hash, base_branch)
        
        # Get actual diff
        actual_diff = git_client.get_range_diff(commit_hash, base_branch)
        
        # Format as Python code
        commits_code = "[\n"
        for commit in commits:
            commits_code += f'                GitCommit(\n'
            commits_code += f'                    hash="{commit.hash}",\n'
            commits_code += f'                    message="{commit.message}",\n'
            commits_code += f'                    author="{commit.author}",\n'
            commits_code += f'                    date="{commit.date}",\n'
            commits_code += f'                ),\n'
        commits_code += "            ]"
        
        # Format file stats (escape quotes and format for Python)
        file_stats_escaped = file_stats.replace('"', '\\"').replace('\n', '\\n')
        
        # Truncate diff if too long for readability
        if len(actual_diff) > 500:
            actual_diff = actual_diff[:500] + "... (truncated)"
        actual_diff_escaped = actual_diff.replace('"', '\\"').replace('\n', '\\n')
        
        example_code = f'''    # Example: {commits[0].message} (commit {commit_hash[:8]})
    (
        TrainingRunInput(
            commits={commits_code},
            file_stats="""{file_stats}""",
            actual_diff="""{actual_diff[:200]}{"..." if len(actual_diff) > 200 else ""}""",
        ),
        """{expected_response}""",
    ),'''
        
        return example_code
        
    except Exception as e:
        print(f"Error generating example for {commit_hash}: {e}", file=sys.stderr)
        return f"    # Error: Could not generate example for {commit_hash}: {e}"


def main():
    """Main function to generate training examples."""
    parser = argparse.ArgumentParser(description="Generate training examples from commit hashes")
    parser.add_argument("commit_hashes", nargs="+", help="Commit hashes to generate examples for")
    parser.add_argument("--base-branch", default="main", help="Base branch to compare against")
    parser.add_argument("--responses", nargs="+", help="Expected responses for each commit (same order)")
    
    args = parser.parse_args()
    
    if args.responses and len(args.responses) != len(args.commit_hashes):
        print("Error: Number of responses must match number of commit hashes", file=sys.stderr)
        sys.exit(1)
    
    print("# Generated training examples")
    print("TRAINING_EXAMPLES: List[tuple[TrainingRunInput, str]] = [")
    
    for i, commit_hash in enumerate(args.commit_hashes):
        response = args.responses[i] if args.responses else "TODO: Add expected response"
        example = generate_training_example(commit_hash, response, args.base_branch)
        print(example)
        
    print("]")


if __name__ == "__main__":
    main()