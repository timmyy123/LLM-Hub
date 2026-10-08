import DataModels
import ImageGenerator
import LocalImageGenerator
import Localization
import ModelZoo
import ScriptDataModels

public protocol CommandLineAbbreviatable {
  var commandLineAbbreviation: String { get }
}

extension SeedMode: CommandLineAbbreviatable {
  public var commandLineAbbreviation: String {
    switch self {
    case .legacy:
      return "Legacy"
    case .scaleAlike:
      return "Scale Alike"
    case .torchCpuCompatible:
      return "Torch CPU Compatible"
    case .nvidiaGpuCompatible:
      return "NVIDIA GPU Compatible"
    }
  }
}

extension SamplerType: CommandLineAbbreviatable {
  public var commandLineAbbreviation: String {
    switch self {
    case .dPMPP2MKarras:
      return "DPM++ 2M Karras"
    case .eulerA:
      return "Euler a"
    case .DDIM:
      return "DDIM"
    case .uniPC:
      return "UniPC"
    case .dPMPPSDEKarras:
      return "DPM++ SDE Karras"
    case .PLMS:
      return "PLMS"
    case .LCM:
      return "LCM"
    case .eulerASubstep:
      return "Euler A Substep"
    case .dPMPPSDESubstep:
      return "DPM++ SDE Substep"
    case .TCD:
      return "TCD"
    case .tCDTrailing:
      return "TCD Trailing"
    case .eulerATrailing:
      return "Euler A Trailing"
    case .dPMPPSDETrailing:
      return "DPM++ SDE Trailing"
    case .DPMPP2MAYS:
      return "DPM++ 2M AYS"
    case .eulerAAYS:
      return "Euler A AYS"
    case .DPMPPSDEAYS:
      return "DPM++ SDE AYS"
    case .dPMPP2MTrailing:
      return "DPM++ 2M Trailing"
    case .dDIMTrailing:
      return "DDIM Trailing"
    case .uniPCTrailing:
      return "UniPC Trailing"
    case .uniPCAYS:
      return "UniPC AYS"
    }
  }
}

extension CompressionMethod: CommandLineAbbreviatable {
  public var commandLineAbbreviation: String {
    switch self {
    case .disabled:
      return "disabled"
    case .H264:
      return "h264"
    case .H265:
      return "h265"
    case .jpeg:
      return "jpeg"
    }
  }
}

extension ColorCalibration: CommandLineAbbreviatable {
  public var commandLineAbbreviation: String {
    switch self {
    case .disabled:
      return "none"
    case .lab:
      return "lab"
    }
  }
}

func fileList(model: ModelZoo.Specification) -> String {
  let lines: [String?] = [
    "model: \(model.file)",
    "text encoder: \(model.textEncoder ?? ImageGeneratorUtils.defaultTextEncoder)",
    "autoencoder: \(model.autoencoder ?? ImageGeneratorUtils.defaultAutoencoder)",
    model.imageEncoder.map { "image encoder: \($0)" },
    model.clipEncoder.map { "CLIP encoder: \($0)" },
    model.diffusionMapping.map { "diffusion mapping: \($0)" },
  ]
  return lines.compactMap { $0 }.joined(separator: "\n")
}

let modelExplanation =
  "Name of the model file in the models directory. Its corresponding text encoder and autoencoder must also reside in the same directory. Models and encoders can be downloaded from https://static.libnnc.org/<name>. Models:\n\(ModelZoo.availableSpecifications.map{ fileList(model: $0) }.joined(separator: "\n\n"))\n\(LocalizedString.forKey("choose_model"))"
let upscalerExplanation =
  "Upscaler - (Optional) Name of upscaler file in the models directory to use. Upscalers can be downloaded from https://static.libnnc.org/<name>. Possible names:\n\(UpscalerZoo.availableSpecifications.map { $0.file }.joined(separator: "\n"))"

public final class Parameters {
  public let defaultConfiguration: GenerationConfiguration

  let modelParameter, upscalerParameter, promptParameter, negativePromptParameter,
    refinerModelParameter, clipLTextParameter, openClipGTextParameter,
    t5TextParameter: StringParameter
  public let widthParameter, heightParameter, seedParameter, stepsParameter, batchCountParameter,
    batchSizeParameter, clipSkipParameter, imagePriorStepsParameter, hiresFixWidthParameter,
    hiresFixHeightParameter, originalWidthParameter, originalHeightParameter, cropTopParameter,
    cropLeftParameter, targetWidthParameter, targetHeightParameter, negativeOriginalWidthParameter,
    negativeOriginalHeightParameter, numFramesParameter, fpsParameter,
    motionScaleParameter, maskBlurOutsetParameter, stage2StepsParameter, decodingTileWidthParameter,
    decodingTileHeightParameter, decodingTileOverlapParameter, diffusionTileWidthParameter,
    diffusionTileHeightParameter, diffusionTileOverlapParameter,
    upscalerScaleFactorParameter, teaCacheStartParameter, teaCacheEndParameter,
    solAttentionStartParameter,
    teaCacheMaxSkipStepsParameter, causalInferenceParameter,
    causalInferencePadParameter, cfgZeroInitStepsParameter: IntParameter
  public let guidanceScaleParameter, strengthParameter, imageGuidanceScaleParameter,
    maskBlurParameter,
    clipWeightParameter, hiresFixStrengthParameter, refinerStartParameter, aestheticScoreParameter,
    negativeAestheticScoreParameter, guidingFrameNoiseParameter,
    startFrameGuidanceParameter, sharpnessParameter, shiftParameter, shiftForAudioParameter,
    stage2CfgParameter,
    stage2ShiftParameter, stochasticSamplingGammaParameter, guidanceEmbedParameter,
    teaCacheThresholdParameter, solAttentionTauParameter,
    compressionArtifactsQualityParameter: DoubleParameter
  let seedModeParameter: EnumParameter<SeedMode>
  let samplerParameter: EnumParameter<SamplerType>
  let compressionArtifactsParameter: EnumParameter<CompressionMethod>
  let colorCalibrationParameter: EnumParameter<ColorCalibration>
  let negativePromptForImagePriorParameter, hiresFixParameter,
    zeroNegativePromptParameter, tiledDecodingParameter,
    preserveOriginalAfterInpaintParameter, tiledDiffusionParameter,
    t5TextEncoderParameter, separateClipLParameter, separateOpenClipGParameter,
    speedUpWithGuidanceEmbedParameter, resolutionDependentShiftParameter,
    teaCacheParameter, usesSolAttentionParameter, separateT5Parameter, cfgZeroStarParameter,
    expandPromptToJsonParameter: BoolParameter
  let lorasParameter: JSONParameter<[JSLoRA]>
  let controlsParameter: JSONParameter<[JSControl]>

  // Note txt2img strength is not covered here, because further on in the process it will get set to 1 for that case
  // if necessary
  public init(
    defaultConfiguration: GenerationConfiguration, defaultSeed: Int, defaultImg2ImgStrength: Double,
    defaultPrompt: String?, defaultNegativePrompt: String
  ) {
    self.defaultConfiguration = defaultConfiguration
    modelParameter = StringParameter(
      title: "model", explanation: modelExplanation, defaultValue: defaultConfiguration.model,
      commandLineFlag: "model")
    upscalerParameter = StringParameter(
      title: "upscaler", explanation: upscalerExplanation,
      defaultValue: defaultConfiguration.upscaler, commandLineFlag: "upscaler")
    upscalerScaleFactorParameter = IntParameter(
      titleKey: "upscaler_scale_factor", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.upscalerScaleFactor), range: 0...4,
      commandLineFlag: "upscaler-scale", additionalJsonKeys: ["upscaler_scale_factor"])
    widthParameter = IntParameter(
      titleKey: "width", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.startWidth * 64),
      range: 128...8192, commandLineFlag: "width")
    heightParameter = IntParameter(
      titleKey: "height", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.startHeight * 64),
      range: 128...8192, commandLineFlag: "height")
    seedParameter = IntParameter(
      titleKey: "seed", explanationKey: "seed_explain_cli", defaultValue: defaultSeed,
      range: -1...Int(UInt32.max),
      commandLineFlag: "seed")
    guidanceScaleParameter = DoubleParameter(
      titleKey: "guidance", explanationKey: "text_guidance_detail_cli",
      defaultValue: Double(defaultConfiguration.guidanceScale),
      range: 0...50,
      commandLineFlag: "guidance-scale", additionalJsonKeys: ["cfg_scale"])
    seedModeParameter = EnumParameter<SeedMode>(
      titleKey: "seed_mode", explanationKey: "seed_mode_detail",
      defaultValue: defaultConfiguration.seedMode, commandLineFlag: "seed-mode")
    stepsParameter = IntParameter(
      titleKey: "steps", explanationKey: "steps_detail",
      defaultValue: Int(defaultConfiguration.steps),
      range: 1...150, commandLineFlag: "steps")
    batchCountParameter = IntParameter(
      titleKey: "batch_count", explanationKey: "batch_count_detail",
      defaultValue: Int(defaultConfiguration.batchCount), range: 1...100,
      commandLineFlag: "batch-count", additionalJsonKeys: ["n_iter"])
    batchSizeParameter = IntParameter(
      titleKey: "batch_size", explanationKey: "batch_size_detail",
      defaultValue: Int(defaultConfiguration.batchSize), range: 1...4,
      commandLineFlag: "batch-size")
    samplerParameter = EnumParameter<SamplerType>(
      titleKey: "sampler", explanationKey: "sampler_detail",
      defaultValue: defaultConfiguration.sampler,
      commandLineFlag: "sampler",
      additionalJsonKeys: ["sampler_name", "sampler_index"])
    strengthParameter = DoubleParameter(
      titleKey: "strength", explanationKey: "strength_detail",
      defaultValue: defaultImg2ImgStrength, range: 0...1.0,
      commandLineFlag: "strength", additionalJsonKeys: ["denoising_strength"])
    clipSkipParameter = IntParameter(
      titleKey: "clip_skip", explanationKey: "clip_skip_detail",
      defaultValue: Int(defaultConfiguration.clipSkip), range: 1...23,
      commandLineFlag: "clip-skip")
    imageGuidanceScaleParameter = DoubleParameter(
      titleKey: "image_guidance", explanationKey: "image_guidance_detail",
      defaultValue: Double(defaultConfiguration.imageGuidanceScale), range: 0...25,
      commandLineFlag: "image-guidance")
    maskBlurParameter = DoubleParameter(
      titleKey: "mask_blur", explanationKey: "mask_blur_detail",
      defaultValue: Double(defaultConfiguration.maskBlur), range: 0...25,
      commandLineFlag: "mask-blur")
    maskBlurOutsetParameter = IntParameter(
      titleKey: "mask_blur_outset", explanationKey: "mask_blur_outset_detail",
      defaultValue: Int(defaultConfiguration.maskBlurOutset), range: -100...1000,
      commandLineFlag: "mask-blur-outset")
    sharpnessParameter = DoubleParameter(
      titleKey: "sharpness", explanationKey: "sharpness_detail",
      defaultValue: Double(defaultConfiguration.sharpness), range: 0...30,
      commandLineFlag: "sharpness")
    shiftParameter = DoubleParameter(
      titleKey: "shift", explanationKey: "shift_detail",
      defaultValue: Double(defaultConfiguration.shift), range: 0.1...12,
      commandLineFlag: "shift")
    shiftForAudioParameter = DoubleParameter(
      titleKey: "shift_for_audio", explanationKey: "shift_for_audio_detail",
      defaultValue: Double(defaultConfiguration.shiftForAudio), range: 0.1...12,
      commandLineFlag: "shift-for-audio")
    stage2StepsParameter = IntParameter(
      titleKey: "stage_2_steps", explanationKey: "stage_2_steps_detail",
      defaultValue: Int(defaultConfiguration.maskBlurOutset), range: 1...150,
      commandLineFlag: "stage-2-steps")
    stage2CfgParameter = DoubleParameter(
      titleKey: "stage_2_cfg", explanationKey: "stage_2_cfg_detail",
      defaultValue: Double(defaultConfiguration.stage2Cfg), range: 0...25,
      commandLineFlag: "stage-2-guidance")
    stage2ShiftParameter = DoubleParameter(
      titleKey: "stage_2_shift", explanationKey: "stage_2_shift_detail",
      defaultValue: Double(defaultConfiguration.stage2Shift), range: 0.1...5,
      commandLineFlag: "stage-2-shift")
    clipWeightParameter = DoubleParameter(
      titleKey: "clip_weight", explanationKey: "clip_weight_detail",
      defaultValue: Double(defaultConfiguration.clipWeight), range: 0...1,
      commandLineFlag: "clip-weight")
    negativePromptForImagePriorParameter = BoolParameter(
      titleKey: "negative_prompt_for_image_prior",
      explanationKey: "negative_prompt_for_image_prior_detail",
      commandLineFlag: "negative-prompt-for-image-prior",
      defaultValue: defaultConfiguration.negativePromptForImagePrior)
    imagePriorStepsParameter = IntParameter(
      titleKey: "image_prior_steps", explanationKey: "image_prior_steps_detail",
      defaultValue: Int(defaultConfiguration.imagePriorSteps), range: 3...60,
      commandLineFlag: "image-prior-steps")
    promptParameter = StringParameter(
      titleKey: "prompt", explanationKey: nil, defaultValue: defaultPrompt,
      commandLineFlag: "prompt")
    negativePromptParameter = StringParameter(
      titleKey: "negative-prompt", explanationKey: nil, defaultValue: defaultNegativePrompt,
      commandLineFlag: "negative-prompt")
    hiresFixParameter = BoolParameter(
      titleKey: "high_resolution_fix", explanationKey: "hires_fix_detail",
      commandLineFlag: "hires-fix",
      additionalJsonKeys: ["enable_hr"], defaultValue: defaultConfiguration.hiresFix)
    hiresFixWidthParameter = IntParameter(
      titleKey: "hires_first_pass_width", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.hiresFixStartWidth * 64),
      range: 128...2048,
      commandLineFlag: "hires-fix-width", additionalJsonKeys: ["firstphase_width"])
    hiresFixHeightParameter = IntParameter(
      titleKey: "hires_first_pass_height", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.hiresFixStartHeight * 64),
      range: 128...2048, commandLineFlag: "hires-fix-height",
      additionalJsonKeys: ["firstphase_height"])
    hiresFixStrengthParameter = DoubleParameter(
      titleKey: "hires_second_pass_strength", explanationKey: nil,
      defaultValue: Double(defaultConfiguration.hiresFixStrength), range: 0.0...1.0,
      commandLineFlag: "hires-fix-strength")
    originalWidthParameter = IntParameter(
      titleKey: "original_width", explanationKey: "original_size_detail",
      defaultValue: Int(defaultConfiguration.originalImageWidth),
      range: 128...2048, commandLineFlag: "original-width")
    originalHeightParameter = IntParameter(
      titleKey: "original_height", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.originalImageHeight),
      range: 128...2048, commandLineFlag: "original-height")
    cropTopParameter = IntParameter(
      titleKey: "crop_top", explanationKey: "crop_detail",
      defaultValue: Int(defaultConfiguration.cropTop),
      range: 0...1024, commandLineFlag: "crop-top")
    cropLeftParameter = IntParameter(
      titleKey: "crop_left", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.cropLeft),
      range: 0...1024, commandLineFlag: "crop-left")
    targetWidthParameter = IntParameter(
      titleKey: "target_width", explanationKey: "target_size_detail",
      defaultValue: Int(defaultConfiguration.targetImageWidth),
      range: 128...2048, commandLineFlag: "target-width")
    targetHeightParameter = IntParameter(
      titleKey: "target_height", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.targetImageHeight),
      range: 128...2048, commandLineFlag: "target-height")
    negativeOriginalWidthParameter = IntParameter(
      titleKey: "negative_original_width", explanationKey: "negative_original_size_detail",
      defaultValue: Int(defaultConfiguration.negativeOriginalImageWidth),
      range: 128...2048, commandLineFlag: "negative-original-width")
    negativeOriginalHeightParameter = IntParameter(
      titleKey: "negative_original_height", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.negativeOriginalImageHeight),
      range: 128...2048, commandLineFlag: "negative-original-height")
    refinerStartParameter = DoubleParameter(
      titleKey: "refiner_start", explanationKey: nil,
      defaultValue: Double(defaultConfiguration.refinerStart), range: 0.0...1.0,
      commandLineFlag: "refiner-start")
    aestheticScoreParameter = DoubleParameter(
      titleKey: "aesthetic_score", explanationKey: "aesthetic_score_detail",
      defaultValue: Double(defaultConfiguration.aestheticScore), range: 0.0...10.0,
      commandLineFlag: "aesthetic-score")
    negativeAestheticScoreParameter = DoubleParameter(
      titleKey: "negative_aesthetic_score", explanationKey: nil,
      defaultValue: Double(defaultConfiguration.negativeAestheticScore), range: 0.0...10.0,
      commandLineFlag: "negative-aesthetic-score")
    zeroNegativePromptParameter = BoolParameter(
      titleKey: "zero_negative_prompt", explanationKey: "zero_negative_prompt_detail",
      commandLineFlag: "zero-negative-prompt",
      defaultValue: defaultConfiguration.zeroNegativePrompt)
    refinerModelParameter = StringParameter(
      title: "refiner_model", explanation: "refiner_model_detail",
      defaultValue: defaultConfiguration.refinerModel, commandLineFlag: "refiner-model")
    numFramesParameter = IntParameter(
      titleKey: "num_frames", explanationKey: "num_frames_detail",
      defaultValue: Int(defaultConfiguration.numFrames), range: 1...201,
      commandLineFlag: "num-frames")
    fpsParameter = IntParameter(
      titleKey: "fps", explanationKey: "fps_detail",
      defaultValue: Int(defaultConfiguration.fpsId), range: 1...30, commandLineFlag: "fps")
    motionScaleParameter = IntParameter(
      titleKey: "motion_scale", explanationKey: "motion_scale_detail",
      defaultValue: Int(defaultConfiguration.motionBucketId), range: 0...255,
      commandLineFlag: "motion-scale")
    guidingFrameNoiseParameter = DoubleParameter(
      titleKey: "cond_aug", explanationKey: "cond_aug_detail",
      defaultValue: Double(defaultConfiguration.condAug), range: 0...1,
      commandLineFlag: "guiding-frame-noise")
    startFrameGuidanceParameter = DoubleParameter(
      titleKey: "start_frame_guidance", explanationKey: "start_frame_guidance_detail",
      defaultValue: Double(defaultConfiguration.startFrameCfg), range: 0...25,
      commandLineFlag: "start-frame-guidance")
    tiledDecodingParameter = BoolParameter(
      titleKey: "tiled_decoding", explanationKey: "tiled_decoding_detail",
      commandLineFlag: "tiled-decoding",
      defaultValue: defaultConfiguration.tiledDecoding)
    decodingTileWidthParameter = IntParameter(
      titleKey: "decoding_tile_width_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.decodingTileWidth * 64),
      range: 128...2048, commandLineFlag: "decoding-tile-width")
    decodingTileHeightParameter = IntParameter(
      titleKey: "decoding_tile_height_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.decodingTileHeight * 64),
      range: 128...2048, commandLineFlag: "decoding-tile-height")
    decodingTileOverlapParameter = IntParameter(
      titleKey: "decoding_tile_overlap_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.decodingTileOverlap * 64),
      range: 64...1024, commandLineFlag: "decoding-tile-overlap")
    lorasParameter = JSONParameter(
      title: "loras", explanationKey: "lora_detail",
      defaultValue: defaultConfiguration.loras.map { JSLoRA(lora: $0) }, commandLineFlag: "loras")
    controlsParameter = JSONParameter(
      title: "controls", explanationKey: "control_detail",
      defaultValue: defaultConfiguration.controls.map { JSControl(control: $0) },
      commandLineFlag: "controls")
    stochasticSamplingGammaParameter = DoubleParameter(
      titleKey: "strategic_stochastic_sampling",
      explanationKey: "strategic_stochastic_sampling_detail", defaultValue: 0.3, range: 0...1,
      commandLineFlag: "stochastic-sampling-gamma")
    preserveOriginalAfterInpaintParameter = BoolParameter(
      titleKey: "preserve_original_after_inpaint",
      explanationKey: "preserve_original_after_inpaint_detail",
      commandLineFlag: "preserve-original-after-inpaint",
      defaultValue: defaultConfiguration.preserveOriginalAfterInpaint)
    tiledDiffusionParameter = BoolParameter(
      titleKey: "tiled_diffusion", explanationKey: "tiled_diffusion_detail",
      commandLineFlag: "tiled-diffusion",
      defaultValue: defaultConfiguration.tiledDiffusion)
    diffusionTileWidthParameter = IntParameter(
      titleKey: "diffusion_tile_width_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.diffusionTileWidth * 64),
      range: 128...2048, commandLineFlag: "diffusion-tile-width")
    diffusionTileHeightParameter = IntParameter(
      titleKey: "diffusion_tile_height_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.diffusionTileHeight * 64),
      range: 128...2048, commandLineFlag: "diffusion-tile-height")
    diffusionTileOverlapParameter = IntParameter(
      titleKey: "diffusion_tile_overlap_explanation", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.diffusionTileOverlap * 64),
      range: 64...1024, commandLineFlag: "diffusion-tile-overlap")
    t5TextEncoderParameter = BoolParameter(
      titleKey: "t5_text_encoder", explanationKey: "t5_text_encoder_detail",
      commandLineFlag: "t5-text-encoder-decoding",
      defaultValue: defaultConfiguration.t5TextEncoder)
    separateClipLParameter = BoolParameter(
      titleKey: "separate_clip_l", explanationKey: "separate_clip_l_detail",
      commandLineFlag: "separate-clip-l",
      defaultValue: defaultConfiguration.separateClipL)
    clipLTextParameter = StringParameter(
      title: "clip_l_text", explanation: "separate_clip_l_detail",
      defaultValue: defaultConfiguration.clipLText, commandLineFlag: "clip-l-text")
    separateOpenClipGParameter = BoolParameter(
      titleKey: "separate_open_clip_g", explanationKey: "separate_open_clip_g_detail",
      commandLineFlag: "separate-open-clip-g",
      defaultValue: defaultConfiguration.separateOpenClipG)
    openClipGTextParameter = StringParameter(
      title: "open_clip_g_text", explanation: "separate_clip_g_detail",
      defaultValue: defaultConfiguration.openClipGText, commandLineFlag: "open-clip-g-text")
    speedUpWithGuidanceEmbedParameter = BoolParameter(
      titleKey: "speed_up_with_guidance_embed",
      explanationKey: "speed_up_with_guidance_embed_detail",
      commandLineFlag: "speed-up-with-guidance-embed",
      defaultValue: defaultConfiguration.speedUpWithGuidanceEmbed)
    guidanceEmbedParameter = DoubleParameter(
      titleKey: "guidance_embed", explanationKey: "speed_up_with_guidance_embed_detail",
      defaultValue: Double(defaultConfiguration.guidanceEmbed),
      range: 0...25, commandLineFlag: "guidance-embed")
    resolutionDependentShiftParameter = BoolParameter(
      titleKey: "resolution_dependent_shift",
      explanationKey: "resolution_dependent_shift_detail",
      commandLineFlag: "resolution-dependent-shift",
      defaultValue: defaultConfiguration.resolutionDependentShift)
    teaCacheStartParameter = IntParameter(
      titleKey: "tea_cache_start", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.teaCacheStart),
      range: 0...1000, commandLineFlag: "tea-cache-start")
    teaCacheEndParameter = IntParameter(
      titleKey: "tea_cache_end", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.teaCacheEnd),
      range: 0...1000, commandLineFlag: "tea-cache-end")
    teaCacheThresholdParameter = DoubleParameter(
      titleKey: "tea_cache_threshold", explanationKey: "tea_cache_threshold_detail",
      defaultValue: Double(defaultConfiguration.teaCacheThreshold),
      range: 0...1, commandLineFlag: "tea-cache-threshold")
    teaCacheMaxSkipStepsParameter = IntParameter(
      titleKey: "tea_cache_max_skip_steps", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.teaCacheMaxSkipSteps),
      range: 1...1000, commandLineFlag: "tea-cache-max-skip-steps")
    usesSolAttentionParameter = BoolParameter(
      titleKey: "sol_attention", explanationKey: "sol_attention_detail",
      commandLineFlag: "uses-sol-attention", defaultValue: defaultConfiguration.usesSolAttention)
    solAttentionStartParameter = IntParameter(
      titleKey: "start_step", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.solAttentionStart),
      range: 0...1000, commandLineFlag: "sol-attention-start")
    solAttentionTauParameter = DoubleParameter(
      titleKey: "tau", explanationKey: "sol_attention_detail",
      defaultValue: Double(defaultConfiguration.solAttentionTau),
      range: 0...4, commandLineFlag: "sol-attention-tau")
    teaCacheParameter = BoolParameter(
      titleKey: "tea_cache", explanationKey: "tea_cache_detail",
      commandLineFlag: "tea-cache",
      defaultValue: defaultConfiguration.teaCache)
    separateT5Parameter = BoolParameter(
      titleKey: "separate_t5", explanationKey: "separate_t5_detail",
      commandLineFlag: "separate-t5",
      defaultValue: defaultConfiguration.separateClipL)
    t5TextParameter = StringParameter(
      title: "t5_text", explanation: "separate_t5_detail",
      defaultValue: defaultConfiguration.clipLText, commandLineFlag: "t5-text")
    causalInferenceParameter = IntParameter(
      titleKey: "causal_inference", explanationKey: nil,
      defaultValue: Int(
        defaultConfiguration.causalInferenceEnabled ? defaultConfiguration.causalInference : 0),
      range: 0...1000, commandLineFlag: "causal-inference")
    causalInferencePadParameter = IntParameter(
      titleKey: "causal_inference_pad", explanationKey: nil,
      defaultValue: Int(
        defaultConfiguration.causalInferenceEnabled ? defaultConfiguration.causalInferencePad : 0),
      range: 0...1000, commandLineFlag: "causal-inference-pad")
    cfgZeroStarParameter = BoolParameter(
      titleKey: "cfg_zero_star", explanationKey: nil, commandLineFlag: "cfg-zero-star",
      defaultValue: defaultConfiguration.cfgZeroStar)
    cfgZeroInitStepsParameter = IntParameter(
      titleKey: "cfg_zero_init_steps", explanationKey: nil,
      defaultValue: Int(defaultConfiguration.cfgZeroInitSteps),
      range: 0...1000, commandLineFlag: "cfg-zero-init-steps")
    compressionArtifactsParameter = EnumParameter<CompressionMethod>(
      titleKey: "compression_artifacts", explanationKey: nil,
      defaultValue: defaultConfiguration.compressionArtifacts,
      commandLineFlag: "compression-artifacts", additionalJsonKeys: ["compression_artifacts"])
    compressionArtifactsQualityParameter = DoubleParameter(
      titleKey: "compression_artifacts_quality", explanationKey: nil,
      defaultValue: Double(defaultConfiguration.compressionArtifactsQuality), range: 0...100,
      commandLineFlag: "compression-artifacts-quality",
      additionalJsonKeys: ["compression_artifacts_quality"])
    colorCalibrationParameter = EnumParameter<ColorCalibration>(
      titleKey: "color_calibration", explanationKey: nil,
      defaultValue: defaultConfiguration.colorCalibration,
      commandLineFlag: "color-calibration", additionalJsonKeys: ["color_calibration"])
    expandPromptToJsonParameter = BoolParameter(
      titleKey: "expand_prompt_to_json", explanationKey: nil,
      commandLineFlag: "expand-prompt-to-json", additionalJsonKeys: ["expand_prompt_to_json"],
      defaultValue: defaultConfiguration.expandPromptToJson)
  }

  public func allParameters() -> [Parameter] {
    return [
      modelParameter,
      widthParameter,
      heightParameter,
      seedParameter,
      guidanceScaleParameter,
      seedModeParameter,
      stepsParameter,
      batchCountParameter,
      batchSizeParameter,
      samplerParameter,
      strengthParameter,
      clipSkipParameter,
      imageGuidanceScaleParameter,
      maskBlurParameter,
      maskBlurOutsetParameter,
      sharpnessParameter,
      clipWeightParameter,
      negativePromptForImagePriorParameter,
      imagePriorStepsParameter,
      promptParameter,
      negativePromptParameter,
      hiresFixParameter,
      hiresFixWidthParameter,
      hiresFixHeightParameter,
      hiresFixStrengthParameter,
      tiledDecodingParameter,
      decodingTileWidthParameter,
      decodingTileHeightParameter,
      decodingTileOverlapParameter,
      originalWidthParameter,
      originalHeightParameter,
      cropTopParameter,
      cropLeftParameter,
      targetWidthParameter,
      targetHeightParameter,
      negativeOriginalWidthParameter,
      negativeOriginalHeightParameter,
      aestheticScoreParameter,
      negativeAestheticScoreParameter,
      zeroNegativePromptParameter,
      refinerModelParameter,
      refinerStartParameter,
      numFramesParameter,
      fpsParameter,
      motionScaleParameter,
      guidingFrameNoiseParameter,
      startFrameGuidanceParameter,
      shiftParameter,
      shiftForAudioParameter,
      stage2CfgParameter,
      stage2ShiftParameter,
      lorasParameter,
      controlsParameter,
      stochasticSamplingGammaParameter,
      preserveOriginalAfterInpaintParameter,
      tiledDiffusionParameter,
      diffusionTileWidthParameter,
      diffusionTileHeightParameter,
      diffusionTileOverlapParameter,
      upscalerParameter,
      upscalerScaleFactorParameter,
      t5TextEncoderParameter,
      separateClipLParameter,
      clipLTextParameter,
      separateOpenClipGParameter,
      openClipGTextParameter,
      speedUpWithGuidanceEmbedParameter,
      guidanceEmbedParameter,
      resolutionDependentShiftParameter,
      teaCacheStartParameter,
      teaCacheEndParameter,
      teaCacheThresholdParameter,
      teaCacheMaxSkipStepsParameter,
      teaCacheParameter,
      usesSolAttentionParameter,
      solAttentionStartParameter,
      solAttentionTauParameter,
      separateT5Parameter,
      t5TextParameter,
      causalInferenceParameter,
      causalInferencePadParameter,
      cfgZeroStarParameter,
      cfgZeroInitStepsParameter,
      compressionArtifactsParameter,
      compressionArtifactsQualityParameter,
      colorCalibrationParameter,
      expandPromptToJsonParameter,
    ]
  }
}
