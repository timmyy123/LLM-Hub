// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AmrLowBalanceDialog } from '../../src/components/AmrLowBalanceDialog';

describe('AmrLowBalanceDialog', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders null when low balance dialog is called', () => {
    const { container } = render(<AmrLowBalanceDialog />);
    expect(container.firstChild).toBeNull();
  });
});
