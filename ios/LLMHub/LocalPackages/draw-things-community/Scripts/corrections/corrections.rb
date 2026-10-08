# Prompt: Review the following Localizable.strings snippet and correct any issues with style, spelling, grammar, etc. that you see. Do not change keys, only their values. Only correct values that are full sentences or sentence-like. Print out only the key-value pairs for which you changed it. Text:

def hash_from_str(str)
  (str.split("\n") - [""]).map do |line|
    next nil if line.start_with?("//")
    match = line.match(/^"(.*?)"\s*=\s*"(.*)";/)
    raise "Invalid line: #{line}" if !match
    match[1..-1]
  end.compact.to_h
end

def str_from_hash(hash)
  hash.map { |key, value| "\"#{key}\" = \"#{value}\";" }.join("\n\n")
end

path = "Apps/DrawThings/Resources/en.lproj/Localizable.strings"
hash = hash_from_str(IO.read(path))

corrections = hash_from_str(IO.read("corrections.strings"))

for key, value in corrections
  raise "Missing key: #{key}" if !hash[key]
  hash[key] = value
end

IO.write(path, str_from_hash(hash))
