import Foundation

public enum LLMVersion: String, Codable, Hashable {
  case qwen_3_5_4b = "qwen_3.5_4b"
  case qwen_3_5_9b = "qwen_3.5_9b"
  case qwen_3_5_27b = "qwen_3.5_27b"
  case deepseek_4_1_flash = "deepseek_4.1_flash"
  case deepseek_4_flash = "deepseek_4_flash"
}
