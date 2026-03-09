import os
BASE = '/mnt/c/Users/timmy/Downloads/LLM-Hub/ios/LLMHub/.build/checkouts/mlx-swift'

# 1. lapack.h
lapack = BASE + '/Source/Cmlx/mlx/mlx/backend/cpu/lapack.h'
text = open(lapack, encoding='latin-1').read()
old = '#if defined(MLX_USE_ACCELERATE) || defined(__APPLE__)\n#ifndef ACCELERATE_NEW_LAPACK\n#define ACCELERATE_NEW_LAPACK\n#endif\n#include <Accelerate/Accelerate.h>\n#else\n#include <cblas.h>\n#include <lapack.h>\n#endif'
rep = '#ifdef MLX_USE_ACCELERATE\n#include <Accelerate/Accelerate.h>\n#else\n#include <cblas.h>\n#include <lapack.h>\n#endif'
text2 = text.replace(old, rep)
open(lapack, 'w', encoding='latin-1').write(text2)
print('lapack.h:', 'reverted' if text2 != text else 'already original')

# 2. Metal Swift files
for fname in ['GPU+Metal.swift','MLXArray+Metal.swift','MLXFast.swift','MLXFastKernel.swift']:
    fpath = BASE + '/Source/MLX/' + fname
    text = open(fpath, encoding='utf-8').read()
    guard = '#if canImport(Metal)\n'
    if text.startswith(guard) and text.rstrip('\n').endswith('#endif'):
        text2 = text[len(guard):].rstrip('\n')[:-7].rstrip('\n') + '\n'
        open(fpath, 'w', encoding='utf-8').write(text2)
        print(fname + ': reverted')
    else:
        print(fname + ': already original')

# 3. Package.swift
pkg = BASE + '/Package.swift'
text = open(pkg, encoding='latin-1').read()
EXCL = ['GPU+Metal.swift', 'MLXArray+Metal.swift', 'MLXFast.swift', 'MLXFastKernel.swift']
lines = ',\n'.join('        "' + x + '"' for x in EXCL)
old_pkg = '    let mlxSwiftExcludes: [String] = []\n    let platformExcludes'
new_pkg = '    let mlxSwiftExcludes: [String] = [\n' + lines + ',\n    ]\n    let platformExcludes'
if old_pkg in text:
    text2 = text.replace(old_pkg, new_pkg, 1)
    open(pkg, 'w', encoding='latin-1').write(text2)
    print('Package.swift: reverted')
elif new_pkg in text:
    print('Package.swift: already original')
else:
    idx = text.find('mlxSwiftExcludes')
    print('unrecognized: ' + repr(text[idx:idx+200]))