import subprocess
import os

def read_paths_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return []

def generate_custom_patch(paths_to_filter):
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

        # Getting commit metadata
        author_name = subprocess.check_output(['git', 'log', '--format=%an', '-n', '1', commit_hash]).decode('utf-8').strip()
        author_email = subprocess.check_output(['git', 'log', '--format=%ae', '-n', '1', commit_hash]).decode('utf-8').strip()
        commit_date = subprocess.check_output(['git', 'log', '--format=%ad', '--date=iso', '-n', '1', commit_hash]).decode('utf-8').strip()
        commit_message = subprocess.check_output(['git', 'log', '--format=%B', '-n', '1', commit_hash]).decode('utf-8').strip()

        # Get list of modified files in the top commit that match the filter paths
        cmd = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash]
        modified_files = subprocess.check_output(cmd).decode('utf-8').splitlines()
        filtered_files = [file for file in modified_files if any(file.startswith(path) for path in paths_to_filter)]

        # Generate diff for the filtered files and write to a patch file
        if filtered_files:
            patch_file_name = 'custom_patch_with_metadata.patch'
            with open(patch_file_name, 'w') as patch_file:
                patch_file.write(f'From: {author_name} <{author_email}>\n')
                patch_file.write(f'Date: {commit_date}\n')
                patch_file.write(f'Subject: [PATCH] {commit_message}\n')
                patch_content = f'\n\n---\n\n'
                for file in filtered_files:
                    diff_cmd = ['git', 'diff', '--binary', commit_hash + '^', commit_hash, '--', file]
                    diff_output = subprocess.check_output(diff_cmd).decode('utf-8')
                    if diff_output:
                        patch_content += diff_output
                patch_file.write(patch_content)
            print(f'Patch file generated: {patch_file_name}')
        else:
            print('No modified files match the specified paths.')
    except subprocess.CalledProcessError as e:
        print(f'Error generating custom patch: {e}')

# Assuming SYNCLIST is in the root directory of the repository
root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip()
sync_list_path = os.path.join(root_dir, 'SYNCLIST')

# Read paths to filter from SYNCLIST file
paths_to_filter = read_paths_from_file(sync_list_path)

if paths_to_filter:
    generate_custom_patch(paths_to_filter)
else:
    print('No paths to filter. Please check the SYNCLIST file.')

