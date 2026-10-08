import C_Resources
import Foundation

public enum BinaryResources {
  public static let bpe_simple_vocab_16e6_txt = Data(
    bytesNoCopy: C_Resources.bpe_simple_vocab_16e6_txt()!,
    count: C_Resources.bpe_simple_vocab_16e6_txt_size, deallocator: .none)
  public static let merges_txt = Data(
    bytesNoCopy: C_Resources.merges_txt()!, count: C_Resources.merges_txt_size, deallocator: .none)
  public static let vocab_json = Data(
    bytesNoCopy: C_Resources.vocab_json()!, count: C_Resources.vocab_json_size, deallocator: .none)
  public static let vocab_16e6_json = Data(
    bytesNoCopy: C_Resources.vocab_16e6_json()!, count: C_Resources.vocab_16e6_json_size,
    deallocator: .none)
  public static let merges_deepseek4_txt = Data(
    bytesNoCopy: C_Resources.merges_deepseek4_txt()!,
    count: C_Resources.merges_deepseek4_txt_size, deallocator: .none)
  public static let vocab_deepseek4_json = Data(
    bytesNoCopy: C_Resources.vocab_deepseek4_json()!,
    count: C_Resources.vocab_deepseek4_json_size, deallocator: .none)
  public static let chatglm3_spiece_model = Data(
    bytesNoCopy: C_Resources.chatglm3_spiece_model()!,
    count: C_Resources.chatglm3_spiece_model_size,
    deallocator: .none)
  public static let gemma3_spiece_model = Data(
    bytesNoCopy: C_Resources.gemma3_spiece_model()!, count: C_Resources.gemma3_spiece_model_size,
    deallocator: .none)
  public static let pile_t5_spiece_model = Data(
    bytesNoCopy: C_Resources.pile_t5_spiece_model()!, count: C_Resources.pile_t5_spiece_model_size,
    deallocator: .none)
  public static let t5_spiece_model = Data(
    bytesNoCopy: C_Resources.t5_spiece_model()!, count: C_Resources.t5_spiece_model_size,
    deallocator: .none)
  public static let xlmroberta_bpe_model = Data(
    bytesNoCopy: C_Resources.xlmroberta_bpe_model()!, count: C_Resources.xlmroberta_bpe_model_size,
    deallocator: .none)
  public static let server_crt_crt = Data(
    bytesNoCopy: C_Resources.server_crt_crt()!, count: C_Resources.server_crt_crt_size,
    deallocator: .none)
  public static let root_ca_crt = Data(
    bytesNoCopy: C_Resources.root_ca_crt()!, count: C_Resources.root_ca_crt_size,
    deallocator: .none)
  public static let server_key_key = Data(
    bytesNoCopy: C_Resources.server_key_key()!, count: C_Resources.server_key_key_size,
    deallocator: .none)
  public static let isrgrootx1_pem = Data(
    bytesNoCopy: C_Resources.isrgrootx1_pem()!, count: C_Resources.isrgrootx1_pem_size,
    deallocator: .none)
  public static let merges_llama3_txt = Data(
    bytesNoCopy: C_Resources.merges_llama3_txt()!, count: C_Resources.merges_llama3_txt_size,
    deallocator: .none)
  public static let vocab_llama3_json = Data(
    bytesNoCopy: C_Resources.vocab_llama3_json()!, count: C_Resources.vocab_llama3_json_size,
    deallocator: .none)
  public static let umt5_spiece_model = Data(
    bytesNoCopy: C_Resources.umt5_spiece_model()!, count: C_Resources.umt5_spiece_model_size,
    deallocator: .none)
  public static let merges_qwen2_5_txt = Data(
    bytesNoCopy: C_Resources.merges_qwen2_5_txt()!, count: C_Resources.merges_qwen2_5_txt_size,
    deallocator: .none)
  public static let vocab_qwen2_5_json = Data(
    bytesNoCopy: C_Resources.vocab_qwen2_5_json()!, count: C_Resources.vocab_qwen2_5_json_size,
    deallocator: .none)
  public static let merges_qwen3_5_txt = Data(
    bytesNoCopy: C_Resources.merges_qwen3_5_txt()!, count: C_Resources.merges_qwen3_5_txt_size,
    deallocator: .none)
  public static let vocab_qwen3_5_json = Data(
    bytesNoCopy: C_Resources.vocab_qwen3_5_json()!, count: C_Resources.vocab_qwen3_5_json_size,
    deallocator: .none)
  public static let merges_qwen3_txt = Data(
    bytesNoCopy: C_Resources.merges_qwen3_txt()!, count: C_Resources.merges_qwen3_txt_size,
    deallocator: .none)
  public static let vocab_qwen3_json = Data(
    bytesNoCopy: C_Resources.vocab_qwen3_json()!, count: C_Resources.vocab_qwen3_json_size,
    deallocator: .none)
  public static let merges_mistral3_txt = Data(
    bytesNoCopy: C_Resources.merges_mistral3_txt()!, count: C_Resources.merges_mistral3_txt_size,
    deallocator: .none)
  public static let vocab_mistral3_json = Data(
    bytesNoCopy: C_Resources.vocab_mistral3_json()!, count: C_Resources.vocab_mistral3_json_size,
    deallocator: .none)
}
