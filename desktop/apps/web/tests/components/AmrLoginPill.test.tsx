// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AmrLoginPill } from '../../src/components/AmrLoginPill';

describe('AmrLoginPill', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders null when login pill is called', () => {
    const { container } = render(<AmrLoginPill />);
    expect(container.firstChild).toBeNull();
  });
});
