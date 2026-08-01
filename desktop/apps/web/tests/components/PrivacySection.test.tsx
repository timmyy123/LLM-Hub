// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { PrivacySection } from '../../src/components/PrivacySection';
import { I18nProvider } from '../../src/i18n';

describe('PrivacySection', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders the zero data collection privacy guarantee', () => {
    render(
      <I18nProvider initial="en">
        <PrivacySection />
      </I18nProvider>,
    );

    expect(screen.getByText('Privacy & Data Guarantee')).toBeTruthy();
    expect(screen.getByText('We never collect or have any interest in user data.')).toBeTruthy();
    expect(
      screen.getByText(
        /All code, prompts, chat messages, and model weights remain strictly on your local machine and offline./,
      ),
    ).toBeTruthy();
  });
});

