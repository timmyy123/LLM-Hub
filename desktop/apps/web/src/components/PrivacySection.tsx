import { Icon } from './Icon';
import { useT } from '../i18n';

export function PrivacySection(): JSX.Element {
  const t = useT();
  return (
    <section className="settings-section" style={{ padding: '24px' }}>
      <div
        style={{
          padding: '24px',
          borderRadius: '16px',
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '12px',
              backgroundColor: 'rgba(34, 197, 94, 0.15)',
              color: '#4ade80',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Icon name="check" size={24} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#fff' }}>
              {t('settings.privacyGuaranteeTitle')}
            </h3>
            <p style={{ margin: '4px 0 0', fontSize: '14px', color: '#a1a1aa' }}>
              {t('settings.privacyGuaranteeSubtitle')}
            </p>
          </div>
        </div>

        <div
          style={{
            fontSize: '15px',
            lineHeight: '1.6',
            color: '#e4e4e7',
            padding: '16px',
            borderRadius: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <strong>{t('settings.privacyGuaranteeStatement')}</strong>
          <br />
          {t('settings.privacyGuaranteeBody')}
        </div>
      </div>
    </section>
  );
}
