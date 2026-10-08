import subprocess
import re
import os

def apply_custom_patch(patch_file_path):
    # Read the patch file content
    with open(patch_file_path, 'r') as file:
        patch_content = file.read()

    # Extract commit message, author name, and author email
    commit_message_match = re.search(r'Subject: \[PATCH\] (.+)', patch_content)
    author_match = re.search(r'From: (.+) <(.+)>', patch_content)

    commit_message = commit_message_match.group(1) if commit_message_match else "Applied patch"
    author_name = author_match.group(1) if author_match else ""
    author_email = author_match.group(2) if author_match else ""

    # Isolate the diff part
    diff_content_start = patch_content.find('\n\n---\n\n') + 5
    diff_content = patch_content[diff_content_start:] if diff_content_start > 4 else ""

    # Temporarily write the diff content to a file
    diff_file_path = 'temp_diff.patch'
    with open(diff_file_path, 'w') as diff_file:
        diff_file.write(diff_content)

    # Apply the diff using git apply
    try:
        subprocess.run(['git', 'apply', diff_file_path], check=True)
        print("Diffs applied successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to apply diffs:", e)
        return
    finally:
        # Delete the temporary diff file
        os.remove(diff_file_path)
        os.remove(patch_file_path)
        print("Temporary diff file deleted.")

    # Stage changes
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'config', 'user.email', author_email], check=True)
    subprocess.run(['git', 'config', 'user.name', author_name], check=True)

    # Commit changes with extracted metadata
    commit_command = ['git', 'commit', '-m', commit_message]
    try:
        subprocess.run(commit_command, check=True)
        print("Changes committed with extracted metadata.")
    except subprocess.CalledProcessError as e:
        print("Failed to commit changes:", e)

# Path to your custom patch file
patch_file_path = 'custom_patch_with_metadata.patch'
apply_custom_patch(patch_file_path)

