import { describe, expect, it } from 'vitest';

import { createDesignSystemServerServices } from '../../src/design-systems/server-services.js';

describe('design system server services', () => {
  it('hides Chinese built-in design systems from the merged catalog', async () => {
    const paths = {
      PROJECTS_DIR: '/tmp/projects',
      DESIGN_SYSTEMS_DIR: '/tmp/design-systems',
      USER_DESIGN_SYSTEMS_DIR: '/tmp/user-design-systems',
    };

    const services = createDesignSystemServerServices({
      roots: {
        SKILL_ROOTS: [],
        DESIGN_TEMPLATE_ROOTS: [],
        ALL_SKILL_LIKE_ROOTS: [],
      },
      paths,
      skills: {
        listSkills: async () => [],
        findSkillById: () => undefined,
      },
      designSystems: {
        listDesignSystems: async (root: string) => {
          if (root === paths.DESIGN_SYSTEMS_DIR) {
            return [
              { id: 'ant', title: 'Ant', source: 'built-in' },
              { id: 'minimax', title: 'MiniMax', source: 'built-in' },
              { id: 'wechat', title: 'WeChat Design System', source: 'built-in' },
              { id: 'wired', title: 'WIRED', source: 'built-in' },
              { id: 'xiaohongshu', title: 'Xiaohongshu', source: 'built-in' },
            ] as any;
          }
          return [
            { id: 'user:custom', title: 'Custom', source: 'user', updatedAt: '2026-01-01T00:00:00.000Z' },
          ] as any;
        },
        readDesignSystem: async () => null,
        readDesignSystemPackageInfo: async () => null,
        readDesignSystemStaticFile: async () => null,
        listUserDesignSystemFiles: async () => null,
        readUserDesignSystemFile: async () => null,
        linkUserDesignSystemProject: async () => null,
        buildUserDesignSystemArchive: async () => null,
        createUserDesignSystem: async () => ({
          id: 'user:custom',
          title: 'Custom',
          category: 'Custom',
          summary: '',
          swatches: [],
          surface: 'web',
          body: '',
          source: 'user',
          status: 'draft',
          isEditable: true,
        }),
        deleteUserDesignSystem: async () => false,
        ensureUserDesignSystemWorkspaceProject: async () => null,
        listUserDesignSystemRevisions: async () => null,
        prepareDesignTokenContractRebuild: async () => ({ decision: { available: false } as any }),
        renderDesignSystemPreview: () => '',
        renderDesignSystemShowcase: () => '',
        updateUserDesignSystem: async () => null,
        updateUserDesignSystemRevisionStatus: async () => null,
      },
      generationJobs: {
        get: () => null,
        rebuildTokenContract: () => ({}) as any,
        revise: () => ({}) as any,
        start: () => ({}) as any,
      },
      projects: {
        getProject: () => null,
        insertProject: () => null,
        updateProject: () => null,
        readProjectFile: async () => ({ buffer: Buffer.from('') }),
        writeProjectFile: async () => ({}),
        listFiles: async () => [],
        resolveProjectDir: () => '/tmp/project',
        isSafeId: () => true,
      },
    } as any);

    const systems = await services.listAllDesignSystems();

    expect(systems.map((system) => system.id)).toEqual(['user:custom', 'wired']);
    expect(systems.some((system) => ['ant', 'minimax', 'wechat', 'xiaohongshu'].includes(system.id))).toBe(false);
  });
});
