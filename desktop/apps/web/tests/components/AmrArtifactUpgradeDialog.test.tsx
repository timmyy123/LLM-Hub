// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AmrArtifactUpgradeDialog } from '../../src/components/AmrArtifactUpgradeDialog';

describe('AmrArtifactUpgradeDialog', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders null when artifact upgrade dialog is called', () => {
    const { container } = render(<AmrArtifactUpgradeDialog />);
    expect(container.firstChild).toBeNull();
  });
});
