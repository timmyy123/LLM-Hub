// Ollama Model Manager and Strict Family Filter Service

export const ALLOWED_FAMILIES = {
  GEMMA_4: {
    id: 'gemma4',
    name: 'Gemma 4 Family',
    description: "Google's Gemma 4 open weights multimodal architecture",
    pattern: /^gemma[-_]?4/i,
    popularTags: ['gemma4:latest', 'gemma4:e2b', 'gemma4:e4b', 'gemma4:12b', 'gemma4:26b'],
  },
  MINISTRAL_3: {
    id: 'ministral3',
    name: 'Ministral 3 Family',
    description: "Mistral AI's Ministral 3 edge-optimized model series",
    pattern: /^ministral[-_]?3/i,
    popularTags: ['ministral-3:latest', 'ministral-3:3b', 'ministral-3:8b', 'ministral-3:14b'],
  },
  LFM2_24B_A4B: {
    id: 'lfm2',
    name: 'LFM2 24B A4B',
    description: 'Liquid AI Hybrid SSM-Transformer 24B A4B architecture',
    pattern: /^lfm2.*(24b|a4b|a2b)?/i,
    popularTags: ['lfm2:24b-a4b', 'lfm2:24b-a2b', 'lfm2:latest'],
  },
};

/**
 * Checks if a model tag matches one of the 3 strictly allowed model families.
 */
export function isAllowedModelTag(modelName) {
  if (!modelName) return false;
  const cleanName = modelName.trim().toLowerCase();

  return (
    ALLOWED_FAMILIES.GEMMA_4.pattern.test(cleanName) ||
    ALLOWED_FAMILIES.MINISTRAL_3.pattern.test(cleanName) ||
    ALLOWED_FAMILIES.LFM2_24B_A4B.pattern.test(cleanName)
  );
}

/**
 * Categorizes a model tag into its family object, or null if disallowed.
 */
export function getModelFamily(modelName) {
  if (!modelName) return null;
  const cleanName = modelName.trim().toLowerCase();

  if (ALLOWED_FAMILIES.GEMMA_4.pattern.test(cleanName)) return ALLOWED_FAMILIES.GEMMA_4;
  if (ALLOWED_FAMILIES.MINISTRAL_3.pattern.test(cleanName)) return ALLOWED_FAMILIES.MINISTRAL_3;
  if (ALLOWED_FAMILIES.LFM2_24B_A4B.pattern.test(cleanName)) return ALLOWED_FAMILIES.LFM2_24B_A4B;
  return null;
}

/**
 * Filters list of models from Ollama to ONLY include allowed families.
 */
export function filterAllowedModels(modelsList) {
  if (!Array.isArray(modelsList)) return [];
  return modelsList.filter((m) => isAllowedModelTag(m.name || m.model || m));
}
