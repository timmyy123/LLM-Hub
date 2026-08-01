// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AmrBalanceDialog } from '../../src/components/AmrBalanceDialog';

describe('AmrBalanceDialog', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders null when balance dialog is called', () => {
    const { container } = render(<AmrBalanceDialog />);
    expect(container.firstChild).toBeNull();
  });
});
