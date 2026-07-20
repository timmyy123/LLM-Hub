// Ollama Model Manager Service - Complete official variants for Gemma 4, Ministral 3, and LFM2 24B

export const ALLOWED_FAMILIES = {
  GEMMA_4: {
    id: 'gemma4',
    name: 'Gemma 4 Family',
    description: "Google DeepMind's Gemma 4 multimodal series",
    pattern: /^gemma[-_]?4/i,
    popularTags: [
      'gemma4:latest',
      'gemma4:e2b',
      'gemma4:e4b',
      'gemma4:12b',
      'gemma4:26b',
      'gemma4:31b',
    ],
  },
  MINISTRAL_3: {
    id: 'ministral3',
    name: 'Ministral 3 Family',
    description: "Mistral AI's Ministral 3 edge-optimized series",
    pattern: /^ministral[-_]?3/i,
    popularTags: [
      'ministral-3:latest',
      'ministral-3:3b',
      'ministral-3:8b',
      'ministral-3:14b',
    ],
  },
  LFM2_24B: {
    id: 'lfm2',
    name: 'LFM2 24B A4B / A2B',
    description: 'Liquid AI Hybrid SSM-Transformer 24B architecture',
    pattern: /^lfm2.*24b/i,
    popularTags: [
      'lfm2:24b-a4b',
      'lfm2:24b-a2b',
      'lfm2:latest',
    ],
  },
};

export function isAllowedModelTag(modelName) {
  if (!modelName || typeof modelName !== 'string') return false;
  return modelName.trim().length > 0;
}

export function getModelFamily(modelName) {
  if (!modelName) return null;
  const cleanName = modelName.trim().toLowerCase();

  if (ALLOWED_FAMILIES.GEMMA_4.pattern.test(cleanName)) return ALLOWED_FAMILIES.GEMMA_4;
  if (ALLOWED_FAMILIES.MINISTRAL_3.pattern.test(cleanName)) return ALLOWED_FAMILIES.MINISTRAL_3;
  if (ALLOWED_FAMILIES.LFM2_24B.pattern.test(cleanName)) return ALLOWED_FAMILIES.LFM2_24B;
  return null;
}

export function filterAllowedModels(modelsList) {
  if (!Array.isArray(modelsList)) return [];
  return modelsList;
}
