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
    try {
      const res = await fetch('http://localhost:11434/api/tags');
      if (res.ok) {
        const data = await res.json();
        setInstalledModels(data.models || []);
      }
    } catch {
      setStatusMessage(t('modelDownloader.ollamaOffline'));
    }
  };

  useEffect(() => {
    if (isOpen) {
      void fetchInstalled();
      setStatusMessage(null);
    }
  }, [isOpen]);

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
      const res = await fetch('http://localhost:11434/api/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: tag, stream: true }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`Pull failed for "${tag}"`);
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
      const res = await fetch('http://localhost:11434/api/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: tag }),
      });
      if (!res.ok) throw new Error(`Delete failed for "${tag}"`);
      setStatusMessage(`${t('modelDownloader.deleteSuccess')} ${tag}`);
      void fetchInstalled();
    } catch (err) {
      setStatusMessage(`${t('modelDownloader.errorPrefix')} ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const isInstalled = (tag: string) => installedModels.some((m) => m.name === tag);

  // shared style tokens
  const card: React.CSSProperties = {
    padding: '10px 12px',
    borderRadius: '10px',
    backgroundColor: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(255,255,255,0.08)',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    boxSizing: 'border-box',
    width: '100%',
  };

  const sectionLabel: React.CSSProperties = {
    fontSize: '11px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.07em',
    color: '#52525b',
    fontWeight: 600,
    marginBottom: '8px',
  };

  const dlBtn = (installed: boolean, disabled: boolean): React.CSSProperties => ({
    flexShrink: 0,
    padding: '4px 12px',
    borderRadius: '6px',
    fontSize: '12px',
    fontWeight: 600,
    border: installed ? '1px solid rgba(239,68,68,0.3)' : 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    backgroundColor: installed ? 'rgba(239,68,68,0.15)' : '#2563eb',
    color: installed ? '#fca5a5' : '#fff',
    whiteSpace: 'nowrap' as const,
  });


  return (
    <div
      style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(10px)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px' }}
      onClick={onClose}
    >
      <div
        style={{ width: '100%', maxWidth: '640px', maxHeight: '88vh', backgroundColor: '#18181b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', display: 'flex', flexDirection: 'column', boxShadow: '0 32px 64px rgba(0,0,0,0.7)', overflow: 'hidden', color: '#f4f4f5', boxSizing: 'border-box' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <Icon name="download" size={16} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700, fontSize: '16px', lineHeight: 1.2 }}>{t('modelDownloader.title')}</div>
            <div style={{ fontSize: '12px', color: '#71717a', marginTop: '1px' }}>{t('modelDownloader.subtitle')}</div>
          </div>
          <button type="button" onClick={onClose} style={{ background: 'none', border: 'none', color: '#71717a', cursor: 'pointer', padding: '4px', borderRadius: '6px', lineHeight: 0, flexShrink: 0 }}>
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Scrollable body */}
        <div style={{ padding: '18px 20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '18px', boxSizing: 'border-box' }}>

          {/* Status */}
          {statusMessage && (
            <div style={{
              padding: '9px 14px', borderRadius: '8px', fontSize: '13px',
              backgroundColor: statusMessage.startsWith('Error') ? 'rgba(239,68,68,0.12)' : statusMessage.startsWith('Cancelled') ? 'rgba(234,179,8,0.12)' : 'rgba(34,197,94,0.12)',
              border: `1px solid ${statusMessage.startsWith('Error') ? 'rgba(239,68,68,0.3)' : statusMessage.startsWith('Cancelled') ? 'rgba(234,179,8,0.3)' : 'rgba(34,197,94,0.3)'}`,
            }}>
              {statusMessage}
            </div>
          )}

          {/* Active download */}
          {pullingTag && (
            <div style={{ ...card, backgroundColor: 'rgba(37,99,235,0.1)', border: '1px solid rgba(37,99,235,0.3)' }}>
              <Icon name="refresh" size={16} className="animate-spin" />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: '13px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {t('modelDownloader.downloading')} {pullingTag}
                </div>
                <div style={{ fontSize: '12px', color: '#93c5fd', marginTop: '1px' }}>{pullProgress}</div>
              </div>
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                style={{ flexShrink: 0, padding: '4px 12px', borderRadius: '6px', backgroundColor: 'rgba(239,68,68,0.15)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.3)', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}
              >
                {t('modelDownloader.cancel')}
              </button>
            </div>
          )}

          {/* Pull any model */}
          <div>
            <div style={sectionLabel}>{t('modelDownloader.pullAnyModel')}</div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                placeholder={t('modelDownloader.pullPlaceholder')}
                value={customTag}
                onChange={(e) => setCustomTag(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void handlePull(customTag); }}
                style={{ flex: 1, minWidth: 0, padding: '8px 12px', borderRadius: '8px', backgroundColor: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.12)', color: '#fff', fontSize: '13px', outline: 'none', boxSizing: 'border-box' }}
              />
              <button
                type="button"
                disabled={!customTag.trim() || !!pullingTag}
                onClick={() => void handlePull(customTag)}
                style={{ flexShrink: 0, padding: '8px 18px', borderRadius: '8px', backgroundColor: '#2563eb', color: '#fff', border: 'none', fontSize: '13px', fontWeight: 600, cursor: customTag.trim() && !pullingTag ? 'pointer' : 'not-allowed', opacity: customTag.trim() && !pullingTag ? 1 : 0.5 }}
              >
                {t('modelDownloader.pull')}
              </button>
            </div>
          </div>

          {/* Gemma 4 variants */}
          <div>
            <div style={sectionLabel}>
              {t('modelDownloader.gemma4Section')} <span style={{ color: '#3f3f46', textTransform: 'none', fontWeight: 400, letterSpacing: 0 }}>by Google DeepMind</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {GEMMA4_VARIANTS.map((v) => {
                const inst = isInstalled(v.tag);
                return (
                  <div key={v.tag} style={{ ...card }}>
                    {/* Tag */}
                    <code style={{ flexShrink: 0, fontSize: '12px', fontFamily: 'ui-monospace,monospace', backgroundColor: 'rgba(255,255,255,0.07)', padding: '2px 8px', borderRadius: '5px', color: '#e4e4e7', whiteSpace: 'nowrap' }}>
                      {v.tag}
                    </code>
                    {/* Description */}
                    <span style={{ flex: 1, minWidth: 0, fontSize: '13px', color: '#71717a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {v.desc}
                    </span>
                    {/* Installed badge */}
                    {inst && (
                      <span style={{ flexShrink: 0, fontSize: '11px', padding: '2px 7px', borderRadius: '8px', backgroundColor: 'rgba(34,197,94,0.12)', color: '#4ade80', fontWeight: 600, border: '1px solid rgba(34,197,94,0.2)', whiteSpace: 'nowrap' }}>
                        {t('modelDownloader.installedBadge')}
                      </span>
                    )}
                    {/* Button: Delete if installed, Download if not */}
                    <button
                      type="button"
                      disabled={!!pullingTag}
                      onClick={() => inst ? void handleDelete(v.tag) : void handlePull(v.tag)}
                      style={dlBtn(inst, !!pullingTag)}
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
            <div style={sectionLabel}>{t('modelDownloader.installedSection')} ({installedModels.length})</div>
            {installedModels.length === 0 ? (
              <div style={{ fontSize: '13px', color: '#52525b', fontStyle: 'italic' }}>{t('modelDownloader.noModels')}</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '6px' }}>
                {installedModels.map((m) => (
                  <div key={m.name} style={{ padding: '7px 12px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px' }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#d4d4d8' }}>{m.name}</span>
                    <span style={{ color: '#4ade80', flexShrink: 0 }}>✓</span>
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
