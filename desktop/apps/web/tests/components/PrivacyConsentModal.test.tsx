// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { PrivacyConsentModal } from '../../src/components/PrivacyConsentModal';

describe('PrivacyConsentModal', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders nothing as privacy consent popups are disabled', () => {
    const { container } = render(<PrivacyConsentModal />);
    expect(container.firstChild).toBeNull();
  });
});

