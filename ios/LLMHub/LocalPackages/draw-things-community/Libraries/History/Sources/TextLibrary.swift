import Foundation

public struct TextLibrary {
  public static var randomText: String {
    return textLibrary.randomElement()!
  }

  public static let textLibrary: [String] = {
    let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    let customPromptsUrl = urls.first!.appendingPathComponent("custom_prompts.txt")
    if let textContent = try? String(contentsOf: customPromptsUrl, encoding: .utf8) {
      let lines: [String] = textContent.components(separatedBy: .newlines).compactMap {
        let line = $0.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return nil }
        return line
      }
      if !lines.isEmpty {
        return lines
      }
    }
    // Return default list of prompts.
    return [
      "a photograph of an astronaut riding a horse, 4k, volumetric light",
      "a samurai walking towards a mountain, 4k, highly detailed, sharp focus, grayscale",
      "multiple layers of silhouette \"mountains\", with silhouette of \"big rocket in sky\", sharp edges, at sunset, with heavy fog in air, vector style, horizon silhouette landscape wallpaper by alena aenami, firewatch game style, vector style background",
      "closeup face portrait of a black girl wearing crown of flowers, smooth soft skin, big dreamy eyes, beautiful intricate colored hair, symmetrical, anime wide eyes, soft lighting, detailed face, by makoto shinkai, stanley artgerm lau, wlop, rossdraws, concept art, digital painting, looking into camera",
      "3d fluffy llama, closeup cute and adorable, cute big circular reflective eyes, long fuzzy fur, pixar render, unreal engine cinematic smooth, intricate detail, cinematic",
      "professional oil painting of establishing shot of canal surrounded by verdant blue modern curved rustic greek tiled buildings, professional majestic oil painting by ed blinkey, atey ghailan, studio ghibli, by jeremy mann, greg manchess, antonio moro, trending on artstation, trending on cgsociety, volumetric lighting, dramatic lighting, dawn, water, canoes, refraction",
      "cthulhu priestess, biopunk, intricate, headshot, highly detailed, digital painting, artstation, concept art, sharp focus, cinematic lighting, illustration, art by artgerm and greg rutkowski, alphonse mucha, cgsociety",
      "establishing shot of small modern osaka house, pitched roof, trending on pixiv, beautiful gouache painting by hayao miyazaki, by studio ghibli, by atey ghailan, by james jean, trending on artstation, unreal engine 5, cinematic lighting",
      "tiny cute fluffly fantasy creatures on a tropical beach, rainbow fuzz, adorable, unreal engine, octane render, bokeh, vray, houdini render, quixel megascans, arnold render, 8k uhd, raytracing, cgi, lumen reflections, cgsociety, ultra realistic, 100mm, film photography, dslr, cinema4d, studio quality, film grain",
      "war in space, lasers, spacestation in space, people, futuristic, planet outside window, unreal engine, octane render, bokeh, vray, houdini render, quixel megascans, arnold render, 8k uhd, raytracing, cgi, lumen reflections, cgsociety, ultra realistic, 100mm, film photography, dslr, cinema4d, studio quality, film grain, trending on artstation, trending on cgsociety",
      "8k resolution, beautiful, cozy, inviting, bloomcore, decopunk, opulent, hobbit-house, luxurious, enchanted library in giverny flower garden, lily pond, detailed painting, romanticism, warm colors, digital illustration, polished, psychadelic, matte painting trending on artstation",
      "musician disappeared when the music stops, only a few green spires in the distance looming over the long river, misty, storybook illustration, ink wash painting, chinese landscape painting, art of mountains-and-waters painting, large expanse of negative space, art by zhang daqian",
      "beautiful portrait of a female anthropomorphic lion fursona wearing Jedi robes. Leather Gloves. character design by charlie bowater, ross tran, artgerm, and makoto shinkai, detailed, soft lighting, rendered in octane",
      "beautiful magical woman, extracting forbidden spells from a big spellbook, ancient library, unreal engine, magical, portrait, digital painting, artstation, concept art, pixar, rich details, shiny, sharp focus, illustration, art, cgi, by greg rutkowski, gorgeous red gothic witch dress",
      "a painting of a cyborg mecha woman with long straight hair and a smile on her face, greebling, with a landscape in the background, by leonardo da vinci, italian renaissance painting, glowing outlines, cyberpunk girl, gundam girl",
      "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt",
    ]
  }()
}
