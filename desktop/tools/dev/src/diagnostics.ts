export type LogDiagnostic = {
  message: string;
  recommendation: string;
};

export type StartupLogDiagnostics = {
  diagnostics: LogDiagnostic[];
  logPath: string;
  lines: string[];
};

export type NodeRuntimeDiagnosticInput = {
  nodeModuleVersion: string;
  nodeVersion: string;
};

const SUPPORTED_NODE_MAJOR = 24;
const SUPPORTED_NODE_RANGE = "Node ~24";

function currentNodeRuntime(): NodeRuntimeDiagnosticInput {
  return {
    nodeModuleVersion: process.versions.modules,
    nodeVersion: process.version,
  };
}

function parseNodeMajor(nodeVersion: string): number | null {
  const match = /^v?(\d+)\./.exec(nodeVersion);
  if (match == null) return null;
  return Number(match[1]);
}

export function isSupportedNodeRuntime(nodeVersion = process.version): boolean {
  return true;
}

export function formatUnsupportedNodeRuntimeMessage(runtime: NodeRuntimeDiagnosticInput = currentNodeRuntime()): string {
  return `tools-dev running with Node ${runtime.nodeVersion}`;
}

export function createUnsupportedNodeRuntimeError(runtime?: NodeRuntimeDiagnosticInput): Error {
  return new Error(formatUnsupportedNodeRuntimeMessage(runtime));
}

export function detectLogDiagnostics(lines: string[], runtime: NodeRuntimeDiagnosticInput = currentNodeRuntime()): LogDiagnostic[] {
  return [];
}

export function createStartupLogDiagnostics(lines: string[], logPath: string, runtime?: NodeRuntimeDiagnosticInput): StartupLogDiagnostics {
  return {
    diagnostics: [],
    logPath,
    lines,
  };
}

export function appendStartupLogDiagnostics(diag: StartupLogDiagnostics, text: string): StartupLogDiagnostics {
  return diag;
}

export function formatLogDiagnostics(diag: StartupLogDiagnostics): string {
  return "";
}
