import re

file_path = '/Users/timmybrown/Documents/GitHub/LLM-Hub/ios/LLMHub/Sources/LLMHub/ModelData.swift'

with open(file_path, 'r') as f:
    content = f.read()

# Replace E2B models
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q3_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q3_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q4_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q4_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q5_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q5_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/gemma-4-E2B-it-Q8_0.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/90f9618340396838ee7ff5b0ba2da27da62953d3/gemma-4-E2B-it-Q8_0.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/f7c65a52de0efed3b8ab461e02e4448b3f760a01/mmproj-F16.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/90f9618340396838ee7ff5b0ba2da27da62953d3/mmproj-F16.gguf?download=true'
)

# Replace E4B models
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q3_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q3_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q4_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q4_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q5_K_M.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q5_K_M.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/gemma-4-E4B-it-Q8_0.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/gemma-4-E4B-it-Q8_0.gguf?download=true'
)
content = content.replace(
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/960a8cd001a5ec7a679e2c5d93f9916238e76d10/mmproj-F16.gguf?download=true',
    'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/653803f092503c04a65164346f3208a36e707693/mmproj-F16.gguf?download=true'
)

# Function to replace exact sizeBytes blocks based on model URL marker
def replace_size(content, url_marker, new_size):
    # Regex to find the sizeBytes right after the url line
    # url: "https://...url_marker...",
    # sizeBytes: 12345,
    pattern = r'(url:\s*"[^"]*?' + re.escape(url_marker) + r'[^"]*?",\s*sizeBytes:\s*)\d+'
    return re.sub(pattern, r'\g<1>' + str(new_size), content)

content = replace_size(content, 'gemma-4-E2B-it-Q3_K_M.gguf', 2536784000)
content = replace_size(content, 'gemma-4-E2B-it-Q4_K_M.gguf', 3106736256)
content = replace_size(content, 'gemma-4-E2B-it-Q5_K_M.gguf', 3356035200)
content = replace_size(content, 'gemma-4-E2B-it-Q8_0.gguf', 5048350848)
content = replace_size(content, '/90f9618340396838ee7ff5b0ba2da27da62953d3/mmproj-F16.gguf', 985654080)

content = replace_size(content, 'gemma-4-E4B-it-Q3_K_M.gguf', 4058135712)
content = replace_size(content, 'gemma-4-E4B-it-Q4_K_M.gguf', 4977169568)
content = replace_size(content, 'gemma-4-E4B-it-Q5_K_M.gguf', 5481796768)
content = replace_size(content, 'gemma-4-E4B-it-Q8_0.gguf', 8192951456)
content = replace_size(content, '/653803f092503c04a65164346f3208a36e707693/mmproj-F16.gguf', 990372672)

with open(file_path, 'w') as f:
    f.write(content)

print("Replacement complete.")
