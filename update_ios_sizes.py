import re

file_path = '/Users/timmybrown/Documents/GitHub/LLM-Hub/ios/LLMHub/Sources/LLMHub/ModelData.swift'

with open(file_path, 'r') as f:
    content = f.read()

def replace_size(content, url_marker, new_size):
    pattern = r'(url:\s*"[^"]*?' + re.escape(url_marker) + r'[^"]*?",\s*category:\s*\.[a-zA-Z]+,\s*sizeBytes:\s*)\d+'
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

print("Sizes replaced")
