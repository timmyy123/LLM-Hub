// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AmrArtifactUpgradeGate } from '../../src/components/AmrArtifactUpgradeGate';

describe('AmrArtifactUpgradeGate', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders null when artifact upgrade gate is called', () => {
    const { container } = render(<AmrArtifactUpgradeGate />);
    expect(container.firstChild).toBeNull();
  });
});
