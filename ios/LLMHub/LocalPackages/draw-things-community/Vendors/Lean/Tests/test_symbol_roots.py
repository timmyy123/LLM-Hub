import unittest

from generate_symbol_roots import public_symbols


class SymbolRootsTests(unittest.TestCase):
    def test_only_public_definitions_in_reviewed_abi(self):
        output = """
object.o:
0000000000000000 (__TEXT,__text) external _lean_io_print
0000000000000000 (__DATA,__data) external _l_Lean_constant
0000000000000000 (__TEXT,__text) weak external _lean_weak
0000000000000000 (__TEXT,__text) private external _lean_private
                 (undefined) external _lean_missing
0000000000000000 (__TEXT,__text) external __ZN4lean11check_stackEPKc
0000000000000000 (__TEXT,__text) external _unrelated
0000000000000000 (__TEXT,__text) non-external _lean_local
0000000000000000 (__TEXT,__text) external _lean_native_process_exit
0000000000000000 (__TEXT,__text) external _lean_native_process_force_exit
"""
        self.assertEqual(
            public_symbols(output, ["_lean_*", "_l_*"]),
            {"_lean_io_print", "_l_Lean_constant", "_lean_weak"},
        )

    def test_lto_and_duplicate_definitions(self):
        output = """
---------------- (LTO,CODE) external _runtime_initialize_Init
---------------- (LTO,DATA) external _l_Init_value
---------------- (LTO,CODE) private external _lean_hidden
---------------- (LTO,CODE) external _runtime_initialize_Init
"""
        self.assertEqual(
            public_symbols(output, ["_runtime_initialize_*", "_l_*", "_lean_*"]),
            {"_runtime_initialize_Init", "_l_Init_value"},
        )


if __name__ == "__main__":
    unittest.main()
