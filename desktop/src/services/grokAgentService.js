// Grok Build Agent Bridge Service

export async function runGrokAgent({ prompt, model, workspacePath, onStream }) {
  if (window.api && window.api.runGrokPrompt) {
    const unsub = window.api.onGrokStream((data) => {
      onStream(data);
    });
    const res = await window.api.runGrokPrompt({ prompt, model, workspacePath });
    return { ...res, unsub };
  } else {
    // Fallback streaming mock for browser preview mode
    return simulateAgentStream({ prompt, model, workspacePath, onStream });
  }
}

function simulateAgentStream({ prompt, model, workspacePath, onStream }) {
  let isCancelled = false;

  const steps = [
    { type: 'stdout', text: `[Grok Build Agent] Initializing local Ollama session...\n` },
    { type: 'stdout', text: `[Grok Build Agent] Active Model: ${model} (Endpoint: http://localhost:11434/v1)\n` },
    { type: 'stdout', text: `[Grok Build Agent] Workspace Target: ${workspacePath || './'}\n` },
    { type: 'stdout', text: `\n> Thinking: Analyzing user prompt: "${prompt}"...\n` },
    { type: 'stdout', text: `> Tool Call: list_dir(DirectoryPath="${workspacePath || './'}")\n` },
    { type: 'stdout', text: `> Processing local files with ${model}...\n` },
    { type: 'stdout', text: `\n[Grok Build Agent Response]:\nI have examined the codebase using local model ${model}.\nExecuting task steps as requested:\n` },
    { type: 'stdout', text: `✓ Step 1: Resolved local configurations\n✓ Step 2: Generated updated component code\n✓ Step 3: Verified builds with Ollama local backend\n` },
    { type: 'exit', code: 0 },
  ];

  let idx = 0;
  const timer = setInterval(() => {
    if (isCancelled || idx >= steps.length) {
      clearInterval(timer);
      return;
    }
    onStream(steps[idx]);
    idx++;
  }, 600);

  return {
    success: true,
    unsub: () => { isCancelled = true; clearInterval(timer); }
  };
}
