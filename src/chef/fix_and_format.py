import sys
import re
import os
import subprocess
import traceback

# --- Configuration ---
target_filename = "videousfinale.py"

# Patterns for known single-line syntax issues that might break formatters
# Pattern 1: if ... : ... ; if ... ; ... tk_write(...)
pattern1 = r"^(\s*)(if\s+fail_count\s*>\s*0\s*:.*?)(if\s+failed_videos\s*:.*?)(tk_write\(fail_msg,.*?)"
replacement1 = r"""\1if fail_count > 0:
\1    fail_msg = f"{fail_count} video(s) failed analysis or timed out." # Initialize message
\1    if failed_videos: # Check if the list is not empty
\1        # Append the list of failed videos (up to 5)
\1        fail_msg += f"\nFailed: {', '.join(failed_videos[:5])}{('...' if len(failed_videos) > 5 else '')}"
\1    # Call tk_write only if there were failures
\1    tk_write(fail_msg, parent=self, level="warning")"""

# Pattern 2: if ... : try : ... except ... (General pattern for the other errors)
pattern2 = r"^(\s*)(if\s+.*?): try:\s*(.*?)\s*except\s+(.*?):\s*(.*)"
replacement2 = r"""\1\2:
\1    try:
\1        \3
\1    except \4:
\1        \5"""

# Pattern 3: Simplified if ... : try : ... # Handle cases without except on same line
pattern3 = r"^(\s*)(if\s+.*?): try:\s*(.*)"
replacement3 = r"""\1\2:
\1    try:
\1        \3"""


patterns_and_replacements = [
    (pattern1, replacement1),
    (pattern2, replacement2),
    (pattern3, replacement3),
    # Add more patterns here if other specific syntax errors are found
]

# --- End Configuration ---

def fix_specific_syntax_errors(filename):
    """Attempts to fix known single-line syntax errors using regex."""
    if not os.path.exists(filename):
        print(f"Error: Target file '{filename}' not found.")
        return False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            original_content = f.read()

        modified_content = original_content
        changes_made = False

        # Apply each pattern sequentially
        for pattern, replacement in patterns_and_replacements:
            # Use re.MULTILINE to handle patterns across lines if needed, but focus on single lines here
            new_content, num_subs = re.subn(pattern, replacement, modified_content, flags=re.MULTILINE)
            if num_subs > 0:
                print(f"Applied fix for pattern starting with: '{pattern[:30]}...' ({num_subs} occurrence(s))")
                modified_content = new_content
                changes_made = True

        # Only write back if changes were actually made
        if changes_made:
            print(f"Writing syntax-corrected content back to {filename}...")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print("Syntax correction pass complete.")
            return True
        else:
            print("No specific known syntax errors found matching patterns.")
            return True # Return True even if no changes, so formatting still runs

    except Exception as e:
        print(f"An error occurred during syntax fixing: {e}")
        traceback.print_exc()
        return False

def run_code_formatter(filename):
    """Runs the 'black' code formatter on the file."""
    if not os.path.exists(filename):
        print(f"Error: Target file '{filename}' not found for formatting.")
        return False

    try:
        print(f"Running 'black' code formatter on {filename}...")
        # Ensure black is run using the python from the active venv
        python_executable = sys.executable
        command = [python_executable, "-m", "black", filename]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            if "reformatted" in result.stderr or "1 file reformatted" in result.stderr:
                 print("'black' successfully reformatted the file.")
            elif "unchanged" in result.stderr:
                 print("'black' found no changes needed.")
            else:
                 print("'black' ran successfully (check output for details).")
                 print("--- black stdout ---")
                 print(result.stdout)
                 print("--- black stderr ---")
                 print(result.stderr)
            return True
        else:
            print(f"Error: 'black' formatter failed with return code {result.returncode}.")
            print("--- black stdout ---")
            print(result.stdout)
            print("--- black stderr ---")
            print(result.stderr)
            print("\nPotential issues: 'black' might not be installed (`pip install black`) or the file may have severe syntax errors remaining.")
            return False

    except FileNotFoundError:
         print("Error: 'black' command not found. Is it installed in your environment (`pip install black`) and is the environment activated?")
         return False
    except Exception as e:
        print(f"An unexpected error occurred while running 'black': {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"--- Attempting to fix {target_filename} ---")

    # Step 1: Fix known syntax errors first
    syntax_fixed = fix_specific_syntax_errors(target_filename)

    if not syntax_fixed:
        print("\nCould not complete initial syntax fixing. Aborting formatting.")
    else:
        # Step 2: Run black to fix indentation and formatting
        formatting_success = run_code_formatter(target_filename)

        if formatting_success:
            print(f"\nSuccessfully fixed and formatted {target_filename}.")
            print("Please try running your main script again.")
        else:
            print(f"\nFormatting {target_filename} failed. Please check the output above for errors.")