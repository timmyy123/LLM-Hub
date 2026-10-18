#!/usr/bin/env python3
"""Regenerate embedded kernels from this repo's MLX core sources."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
core = root / 'Source/Cmlx/mlx'
generated = root / 'Source/Cmlx/mlx-generated'
metal = core / 'mlx/backend/metal'
clang = subprocess.check_output(['xcrun', '--find', 'clang'], text=True).strip()
kernels = sorted(set(re.findall(r'make_jit_source\(\s*([\w/]+)', (metal / 'CMakeLists.txt').read_text())))

with tempfile.TemporaryDirectory(prefix='llmhub-mlx-kernels-') as temporary:
    output = Path(temporary)
    def generate(kernel):
        subprocess.run(['bash', str(metal / 'make_compiled_preamble.sh'), str(output), clang, str(core), kernel, ''], check=True, stdout=subprocess.DEVNULL)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(generate, kernels))
    subprocess.run(['bash', str(core / 'mlx/backend/cpu/make_compiled_preamble.sh'), str(output / 'compiled_preamble.cpp'), clang, str(core), 'TRUE', 'arm64'], check=True)
    cuda = core / 'mlx/backend/cuda'
    cuda_headers = sorted(str(p.relative_to(cuda)) for p in (cuda / 'device').iterdir() if p.suffix in ('.h', '.cuh'))
    subprocess.run(['cmake', '-DMLX_SOURCE_ROOT=' + str(cuda), '-DMLX_JIT_SOURCES=' + ':'.join(cuda_headers), '-P', str(cuda / 'bin2h.cmake')], cwd=output, check=True)
    for path in generated.glob('*.cpp'):
        path.unlink()
    for path in output.glob('*.cpp'):
        shutil.copy2(path, generated / path.name)
    (generated / 'cuda').mkdir(exist_ok=True)
    shutil.copy2(output / 'gen/cuda_jit_sources.h', generated / 'cuda/cuda_jit_sources.h')
subprocess.run(['bash', str(root / 'tools/fix-metal-includes.sh')], cwd=root, check=True, stdout=subprocess.DEVNULL)
print(f'Regenerated {len(kernels)} Metal kernel sources, the CPU preamble, CUDA header, and Metal include tree.')
