pkg = '/mnt/c/Users/timmy/Downloads/LLM-Hub/ios/LLMHub/.build/checkouts/mlx-swift/Package.swift'
text = open(pkg, encoding='latin-1').read()
EXCL = ['GPU+Metal.swift', 'MLXArray+Metal.swift', 'MLXFast.swift', 'MLXFastKernel.swift']
indent = '        '
lines = ',\n'.join(indent + '"' + x + '"' for x in EXCL)
old_pkg = '    let mlxSwiftExcludes: [String] = []\n#else'
new_pkg = '    let mlxSwiftExcludes: [String] = [\n' + lines + ',\n    ]\n#else'
if old_pkg in text:
    text2 = text.replace(old_pkg, new_pkg, 1)
    open(pkg, 'w', encoding='latin-1').write(text2)
    print('Package.swift: reverted')
elif new_pkg in text:
    print('Package.swift: already original')
else:
    idx = text.find('mlxSwiftExcludes')
    print('unrecognized: ' + repr(text[idx:idx+100]))