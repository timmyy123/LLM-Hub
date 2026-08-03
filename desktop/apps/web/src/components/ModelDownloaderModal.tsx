import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';
import { useI18n } from '../i18n';

interface ModelDownloaderModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface InstalledModel {
  name: string;
  size: number;
  modified_at: string;
}

const GEMMA4_VARIANTS = [
  { tag: 'gemma4:latest', desc: 'Default · 9B · General purpose' },
  { tag: 'gemma4:e2b',    desc: '2B · Ultra-fast, low memory' },
  { tag: 'gemma4:e4b',    desc: '4B · Balanced for laptops' },
  { tag: 'gemma4:12b',    desc: '12B · Multimodal & agentic' },
  { tag: 'gemma4:26b',    desc: '26B · MoE, high performance' },
  { tag: 'gemma4:31b',    desc: '31B · Dense, complex reasoning' },
];

export function ModelDownloaderModal({ isOpen, onClose }: ModelDownloaderModalProps) {
  const [installedModels, setInstalledModels] = useState<InstalledModel[]>([]);
  const [customTag, setCustomTag] = useState('');
  const [pullingTag, setPullingTag] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { t } = useI18n();

  const fetchInstalled = async () => {
    // First try to fetch directly — if Ollama is already running this is instant.
    let res = await fetch('/api/ollama/models').catch(() => null);
    if (!res || !res.ok) {
      res = await fetch('http://127.0.0.1:11434/api/tags').catch(() => null);
    }
    if (!res || !res.ok) {
      // Ollama isn't running yet — ask the daemon to start the bundled binary
      // then retry. This is the common path on app launch/restart.
      await fetch('/api/ollama/start', { method: 'POST' }).catch(() => null);
      await new Promise((r) => setTimeout(r, 2000));
      res = await fetch('/api/ollama/models').catch(() => null);
      if (!res || !res.ok) {
        res = await fetch('http://127.0.0.1:11434/api/tags').catch(() => null);
      }
    }
    if (res && res.ok) {
      const data = await res.json();
      setInstalledModels(data.models || []);
      setStatusMessage(null);
    } else {
      setInstalledModels([]);
      setStatusMessage(null); // Don't scare the user — Ollama starts on first pull
    }
  };

  const [startingOllama, setStartingOllama] = useState(false);

  // Refresh installed model list every time the modal is opened.
  // Also auto-starts the bundled Ollama service so the list is accurate
  // immediately on first open after a restart (not just after a pull).
  useEffect(() => {
    if (!isOpen) return;
    void fetchInstalled();
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleStartOllama = async () => {
    setStartingOllama(true);
    setStatusMessage('Starting Ollama service…');
    try {
      await fetch('/api/ollama/start', { method: 'POST' }).catch(() => null);
      await new Promise((r) => setTimeout(r, 1500));
      await fetchInstalled();
    } catch {
      setStatusMessage('Could not start Ollama automatically. Please open Ollama app or run "ollama serve" in terminal.');
    } finally {
      setStartingOllama(false);
    }
  };

  if (!isOpen) return null;

  const handlePull = async (tag: string) => {
    tag = tag.trim();
    if (!tag || pullingTag) return;
    setPullingTag(tag);
    setPullProgress('Starting…');
    setStatusMessage(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      let res = await fetch('/api/ollama/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: tag }),
        signal: ctrl.signal,
      }).catch(() => null);

      if (!res || !res.ok) {
        res = await fetch('http://127.0.0.1:11434/api/pull', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: tag, stream: true }),
          signal: ctrl.signal,
        }).catch(() => null);
      }

      if (!res || !res.ok || !res.body) throw new Error(`Pull failed for "${tag}"`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of dec.decode(value).split('\n').filter(Boolean)) {
          try {
            const p = JSON.parse(line);
            if (p.status) {
              setPullProgress(
                p.total && p.completed
                  ? `${p.status} (${Math.round((p.completed / p.total) * 100)}%)`
                  : p.status
              );
            }
          } catch { /* ignore */ }
        }
      }
      setStatusMessage(`${t('modelDownloader.successPrefix')} ${tag}`);
      void fetchInstalled();
    } catch (err) {
      setStatusMessage(
        err instanceof Error && err.name === 'AbortError'
          ? `${t('modelDownloader.cancelledPrefix')} ${tag}`
          : `${t('modelDownloader.errorPrefix')} ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setPullingTag(null);
      setPullProgress('');
      abortRef.current = null;
    }
  };

  const handleDelete = async (tag: string) => {
    try {
      let res = await fetch('/api/ollama/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: tag }),
      }).catch(() => null);

      if (!res || !res.ok) {
        res = await fetch('http://127.0.0.1:11434/api/delete', {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: tag }),
        }).catch(() => null);
      }

      if (!res || !res.ok) throw new Error(`Delete failed for "${tag}"`);
      setStatusMessage(`${t('modelDownloader.deleteSuccess')} ${tag}`);
      void fetchInstalled();
    } catch (err) {
      setStatusMessage(`${t('modelDownloader.errorPrefix')} ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const isInstalled = (tag: string) => installedModels.some((m) => m.name === tag);

  return (
    <div className="model-downloader-backdrop" onClick={onClose}>
      <div className="model-downloader-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="model-downloader__head">
          <div className="model-downloader__icon-wrap">
            <Icon name="download" size={18} />
          </div>
          <div>
            <h2 className="model-downloader__title">{t('modelDownloader.title')}</h2>
            <p className="model-downloader__subtitle">{t('modelDownloader.subtitle')}</p>
          </div>
          <button type="button" className="model-downloader__close" onClick={onClose} aria-label={t('modelDownloader.cancel')}>
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="model-downloader__body">
          {/* Status */}
          {statusMessage && (
            <div style={{
              padding: '10px 14px', borderRadius: '10px', fontSize: '13px',
              backgroundColor: statusMessage.includes('not running') || statusMessage.startsWith('Error') ? 'color-mix(in srgb, var(--danger, #ef4444) 12%, transparent)' : statusMessage.startsWith('Cancelled') ? 'color-mix(in srgb, var(--warning, #eab308) 12%, transparent)' : 'color-mix(in srgb, var(--success, #22c55e) 12%, transparent)',
              border: `1px solid ${statusMessage.includes('not running') || statusMessage.startsWith('Error') ? 'color-mix(in srgb, var(--danger, #ef4444) 30%, transparent)' : statusMessage.startsWith('Cancelled') ? 'color-mix(in srgb, var(--warning, #eab308) 30%, transparent)' : 'color-mix(in srgb, var(--success, #22c55e) 30%, transparent)'}`,
              color: statusMessage.includes('not running') || statusMessage.startsWith('Error') ? 'var(--danger, #dc2626)' : statusMessage.startsWith('Cancelled') ? 'var(--warning, #ca8a04)' : 'var(--success, #16a34a)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap'
            }}>
              <span style={{ flex: 1, minWidth: 0 }}>{statusMessage}</span>
              {statusMessage.includes('not running') && (
                <button
                  type="button"
                  disabled={startingOllama}
                  onClick={() => void handleStartOllama()}
                  className="model-downloader__btn-primary"
                  style={{ padding: '5px 12px', fontSize: '12px' }}
                >
                  {startingOllama ? 'Starting…' : 'Start Ollama Service'}
                </button>
              )}
            </div>
          )}

          {/* Active download */}
          {pullingTag && (
            <div className="model-downloader__card" style={{ backgroundColor: 'color-mix(in srgb, var(--accent, #2563eb) 12%, transparent)', borderColor: 'color-mix(in srgb, var(--accent, #2563eb) 30%, transparent)' }}>
              <Icon name="refresh" size={16} className="animate-spin" />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {t('modelDownloader.downloading')} {pullingTag}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted, #71717a)', marginTop: '1px' }}>{pullProgress}</div>
              </div>
              <button
                type="button"
                className="model-downloader__btn-delete"
                onClick={() => abortRef.current?.abort()}
              >
                {t('modelDownloader.cancel')}
              </button>
            </div>
          )}

          {/* Pull any model */}
          <div>
            <div className="model-downloader__section-label">{t('modelDownloader.pullAnyModel')}</div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                className="model-downloader__input"
                placeholder={t('modelDownloader.pullPlaceholder')}
                value={customTag}
                onChange={(e) => setCustomTag(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void handlePull(customTag); }}
              />
              <button
                type="button"
                className="model-downloader__btn-primary"
                disabled={!customTag.trim() || !!pullingTag}
                onClick={() => void handlePull(customTag)}
              >
                {t('modelDownloader.pull')}
              </button>
            </div>
          </div>

          {/* Gemma 4 variants */}
          <div>
            <div className="model-downloader__section-label">
              {t('modelDownloader.gemma4Section')} <span style={{ textTransform: 'none', fontWeight: 400, letterSpacing: 0, opacity: 0.7 }}>by Google DeepMind</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {GEMMA4_VARIANTS.map((v) => {
                const inst = isInstalled(v.tag);
                return (
                  <div key={v.tag} className="model-downloader__card">
                    {/* Tag */}
                    <code className="model-downloader__code-tag">
                      {v.tag}
                    </code>
                    {/* Description */}
                    <span className="model-downloader__desc">
                      {v.desc}
                    </span>
                    {/* Installed badge */}
                    {inst && (
                      <span className="model-downloader__badge-installed">
                        {t('modelDownloader.installedBadge')}
                      </span>
                    )}
                    {/* Button: Delete if installed, Download if not */}
                    <button
                      type="button"
                      disabled={!!pullingTag}
                      className={inst ? 'model-downloader__btn-delete' : 'model-downloader__btn-download'}
                      onClick={() => inst ? void handleDelete(v.tag) : void handlePull(v.tag)}
                    >
                      {inst ? t('modelDownloader.delete') : t('modelDownloader.download')}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Installed models */}
          <div>
            <div className="model-downloader__section-label">{t('modelDownloader.installedSection')} ({installedModels.length})</div>
            {installedModels.length === 0 ? (
              <div style={{ fontSize: '13px', color: 'var(--text-muted, #71717a)', fontStyle: 'italic' }}>{t('modelDownloader.noModels')}</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '6px' }}>
                {installedModels.map((m) => (
                  <div key={m.name} className="model-downloader__installed-pill">
                    <span className="model-downloader__installed-pill-name">{m.name}</span>
                    <span className="model-downloader__installed-pill-check">✓</span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
