#!/usr/bin/env ruby

require 'fileutils'
require 'json'
require 'base64'
require 'open3'
require 'chunky_png'
require 'parallel'
require 'pry-byebug'

def create_tempfile(prefix: "", suffix: "", contents: nil, is_dir: false, keep: false)
  possible_chars = [('a'..'z'), ('0'..'9')].map(&:to_a).flatten
  random_string = Array.new(6) { possible_chars.sample }.join
  path = "/tmp/#{prefix ? "#{prefix}_" : ""}tempfile_#{random_string}#{suffix}"
  # ObjectSpace.define_finalizer(path, proc {  })
  at_exit do
    # TODO: it seems like, when running things in parallel, this maybe doesn't work properly?
    FileUtils.rm_rf(path) if File.exist?(path) && !keep
  end
  if is_dir
    Dir.mkdir(path)
  elsif contents
    IO.write(path, contents)
  end
  path
end

def sh!(args, display_cmd_on_fail: true, can_fail: false, get_output: false)
  if get_output
    stdout, stderr, status = Open3.capture3([args[0], args[0]], *args.drop(1))
    success = status.success?
  else
    success = system(*args)
    stdout = stderr = nil
  end
  if !can_fail && !success
    str = "Command failed"
    str += ": #{args}" if display_cmd_on_fail
    raise str
  end
  [stdout, stderr, success]
end

raise "Usage: ./test_cli_and_server.rb <path to CLI> <path to models dir> <path to repo root>" unless ARGV.size == 3

cli_path = ARGV[0]
models_dir = ARGV[1]
repo_root = ARGV[2]
run_server = false

ENV["BUILD_WORKSPACE_DIRECTORY"] = repo_root

if run_server
  server_pid = Process.spawn(*%W[#{cli_path} --models-dir #{models_dir} --persistent], :out => '/dev/null', :err => '/dev/null')
  puts "Running at #{server_pid}"
  # at_exit { Process.kill('SIGKILL', server_pid) }
  exit
end

params = {
  prompt: "animal",
  model: "sd_v1.5_f16.ckpt",
  seed: 1,
  steps: 29,
  # controls: ["controlnet_canny_1.x_f16.ckpt"],
  # loras: ["anime_lineart_style_v2.0_lora_f16.ckpt"],
  negative_prompt_for_image_prior: true,
  negative_prompt: "cat",
  sampler: "Euler a",
  strength: 0.73,
  width: 512,
  height: 384,
  guidance_scale: 2,
  seed_mode: "Torch CPU Compatible",
  clip_skip: 2,
  image_guidance: 3,
  mask_blur: 4,
  clip_weight: 0.5,
  image_prior_steps: 3,
  hires_fix: true,
  hires_fix_width: 768,
  hires_fix_height: 768,
  hires_fix_strength: 0.65,
  image: "dog.png"
}

# prev_png = nil
# pairs = Parallel.map(1.upto(params.size), in_processes: 4) do |num|
#   next nil if num < 2
#   out_file = create_tempfile(suffix: ".png")
#   sh!(%W[#{cli_path} --models-dir #{models_dir} --output #{out_file}] + params.take(num).flat_map { |k, v| ["--" + k.to_s.gsub("_", "-"), v.to_s] })
#   png = ChunkyPNG::Image.from_file(out_file)
#   [png, params.take(num).last[0]]
# end.compact

# pairs.each_cons(2) do |p1, p2|
#   puts "No change for #{p1[1]}" if p1[0].pixels == p2[0].pixels
# end

use_existing = false

cli_png, server_png = Parallel.map([false, true]) do |is_server|
# cli_png, server_png = [true, false].map do |is_server|
  if is_server
    out_file = "/tmp/server.png"
    next ChunkyPNG::Image.from_file(out_file) if use_existing
    content_type = "Content-Type: application/json"
    out_path = create_tempfile
    params = params.dup
    params[:init_images] = [Base64.strict_encode64(IO.read(params.delete(:image)))] if params[:image]
    params.delete(:prompt) # TODO:
    # params[:model] = "Anime (Waifu Diffusion v1.3)" # TODO: remove
    json_file = create_tempfile
    IO.write(json_file, params.to_json)
    # path = params[:init_images] ? "img2img" : "txt2img"
    path = "img2img"
    sh!(%W[curl --silent -o #{out_path}  -H #{content_type} -X POST http://127.0.0.1:3819/sdapi/v1/#{path} -d @#{json_file}])
    image_base64 = JSON.parse(IO.read(out_path))['images'][0]
    image = Base64.decode64(image_base64)
    IO.write(out_file, image)
    ChunkyPNG::Image.from_file(out_file) # or .from_blob for data
  else
    out_file = "/tmp/cli.png"
    next ChunkyPNG::Image.from_file(out_file) if use_existing
    params = params.dup
    controls = params.delete(:controls) || []
    control_args = controls.flat_map { |control| ["--control", control] }
    loras = params.delete(:loras) || []
    lora_args = loras.flat_map { |lora| ["--lora", lora] }
    param_args = params.flat_map { |k, v| ["--" + k.to_s.gsub("_", "-"), v.to_s] }
    sh!(%W[rm #{out_file}]) if File.file?(out_file)
    sh!(%W[#{cli_path} --models-dir #{models_dir} --output #{out_file}] + lora_args + control_args + param_args, display_cmd_on_fail: false)
    ChunkyPNG::Image.from_file(out_file)
  end
end

diffs = server_png.pixels.zip(cli_png.pixels).map do |a, b|
  p1, p2 = [a, b].map { |p| ChunkyPNG::Color.to_truecolor_alpha_bytes(p) }
  p1.zip(p2).map { |c, d| (c - d).abs }.max
end.sort.reverse

almost_match = diffs.sum / diffs.count < 20

binding.pry
raise "Mismatch" unless almost_match # server_png.pixels == cli_png.pixels

puts "Tests pass"
