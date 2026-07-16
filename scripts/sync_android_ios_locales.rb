require 'cgi'

root = File.expand_path('..', __dir__)
android = File.join(root, 'android/app/src/main/res')
ios = File.join(root, 'ios/LLMHub/Sources/LLMHub')

mapping = {
  'en' => 'values/strings.xml',
  'ar' => 'values-ar/strings.xml',
  'de' => 'values-de/strings.xml',
  'es' => 'values-es/strings.xml',
  'fa' => 'values-fa/strings.xml',
  'fr' => 'values-fr/strings.xml',
  'he' => 'values-he/strings.xml',
  'id' => 'values-id/strings.xml',
  'it' => 'values-it/strings.xml',
  'ja' => 'values-ja/strings.xml',
  'ko' => 'values-ko/strings.xml',
  'pl' => 'values-pl/strings.xml',
  'pt' => 'values-pt/strings.xml',
  'ru' => 'values-ru/strings.xml',
  'tr' => 'values-tr/strings.xml',
  'uk' => 'values-uk/strings.xml',
  'zh' => 'values-zh/strings.xml',
  'nl' => 'values-nl/strings.xml',
  'da' => 'values-da/strings.xml',
  'th' => 'values-th/strings.xml',
  'vi' => 'values-vi/strings.xml'
}

def parse_android_strings(path)
  content = File.read(path)
  map = {}
  content.scan(/<string\s+name="([^"]+)">(.*?)<\/string>/m) do |key, raw|
    next if raw.include?('<')

    value = CGI.unescapeHTML(raw.strip)
    value = value.gsub(/%([0-9]+)\$s/, '%\1$@')
    value = value.gsub('"', '\\"')
    value = value.gsub(/(?<!\\)"/, '\\"')
    map[key] = value
  end
  map
end

def parse_ios_line(line)
  match = line.match(/^\s*"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"\s*;\s*$/)
  return nil unless match
  [match[1], match[2]]
end

def sync_locale(android, ios, locale, android_rel)
  android_path = File.join(android, android_rel)
  ios_dir = File.join(ios, "#{locale}.lproj")
  ios_path = File.join(ios_dir, 'Localizable.strings')

  unless File.exist?(android_path)
    warn "Skip #{locale}: missing android source #{android_path}"
    return
  end

  android_map = parse_android_strings(android_path)

  Dir.mkdir(ios_dir) unless Dir.exist?(ios_dir)
  ios_lines = File.exist?(ios_path) ? File.readlines(ios_path, chomp: true) : []

  line_index_by_key = {}
  ios_lines.each_with_index do |line, idx|
    parsed = parse_ios_line(line)
    line_index_by_key[parsed[0]] = idx if parsed
  end

  android_map.each do |key, value|
    new_line = "\"#{key}\" = \"#{value}\";"
    if line_index_by_key.key?(key)
      ios_lines[line_index_by_key[key]] = new_line
    else
      ios_lines << new_line
      line_index_by_key[key] = ios_lines.length - 1
    end
  end

  File.write(ios_path, ios_lines.join("\n") + "\n")
  puts "Synced #{locale}: #{android_map.size} keys"
end

mapping.each do |locale, rel|
  sync_locale(android, ios, locale, rel)
end
