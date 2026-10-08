#!/usr/bin/env python3
"""Adapt pinned upstream Z3 sources for in-process execution.

The CLI parser and frontend dispatch remain upstream code. Mechanical stream
and exit substitutions also cover diagnostics in the solver, not just the CLI.
File-specific edits below address process lifetime and repeated invocations.
"""

import argparse
from pathlib import Path
import re


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def adapt(path, text):
    if path == "src/util/memory_manager.h":
        text = replace_once(text, "    static bool is_out_of_memory();", "    static bool is_out_of_memory();\n    static void reset_out_of_memory();")
    if path == "src/util/memory_manager.cpp":
        text += "\nvoid memory::reset_out_of_memory() { g_memory_out_of_memory = false; }\n"
        # Log replay can contain finalize/reset calls, but not tear down the host.
        text = replace_once(text, "void memory::finalize(bool shutdown) {", "void memory::finalize(bool shutdown) {\n    if (z3_embedded::is_active()) return;")
    if path == "src/api/api_context.cpp":
        text = replace_once(text, '        printf("Error: %s\\n", Z3_get_error_msg(ctx, c));',
                            '        std::cerr << "Error: " << Z3_get_error_msg(ctx, c) << "\\n";')
    if path == "src/util/gparams.h":
        text = replace_once(text, "    static void reset();", """    static void reset();
    // The embedding host serializes calls while CLI defaults are in scope.
    static void *begin_scope();
    static void end_scope(void *previous);""")
    if path == "src/util/gparams.cpp":
        text += """
void *gparams::begin_scope() {
    auto *fresh = alloc(imp);
    auto *previous = g_imp;
    g_imp = fresh;
    return previous;
}

void gparams::end_scope(void *previous) {
    auto *finished = g_imp;
    g_imp = static_cast<imp *>(previous);
    dealloc(finished);
}
"""
    if path == "src/shell/main.cpp":
        text = replace_once(text, "int STD_CALL main(int argc, char ** argv)",
                            "int z3_upstream_main(int argc, char ** argv)")
        text = replace_once(text, "        unsigned return_value = 0;", """        g_input_file = nullptr;
        g_drat_input_file = nullptr;
        g_standard_input = false;
        g_input_kind = IN_UNSPECIFIED;
        g_display_statistics = false;
        g_display_model = false;
        g_display_istatistics = false;
        unsigned return_value = 0;""")
        text = text.replace("memory::exit_when_out_of_memory(true,", "memory::exit_when_out_of_memory(false,")
        text = replace_once(text, "set_timeout(timeout * 1000);", "z3_embedded::set_timeout(timeout);")
        text = replace_once(text, "        disable_timeout();", "        z3_embedded::checkpoint();")
        text = replace_once(text, "        memory::finalize();", "        // The application owns the shared Z3 runtime.")
        text = replace_once(text, "        switch (g_input_kind) {", """        if (g_input_file) {
            input_file = z3_embedded::resolve_path(g_input_file, false);
            g_input_file = input_file.c_str();
        }
        switch (g_input_kind) {""")
        # Datalog requires a file/directory, unlike the other frontends.
        text = replace_once(text, "            read_datalog(g_input_file);", """            if (!g_input_file) error("Datalog input requires a file.");
            return_value = read_datalog(g_input_file);""")
    if path.startswith("src/shell/"):
        # App cancellation is polled through reslimit and input streams.
        text = re.sub(r"static void (?:STD_CALL )?on_(?:timeout|ctrl_c)\([^)]*\) \{.*?^\}",
                      "", text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r"^\s*(?:register_on_timeout_proc\(on_timeout\)|signal\(SIGINT, on_ctrl_c\));\n",
                      "\n", text, flags=re.MULTILINE)
    if path == "src/shell/opt_frontend.cpp":
        text = replace_once(text, "    g_first_interrupt = true;", "    g_first_interrupt = true;\n    g_handles.reset();")
    if path == "src/shell/dimacs_frontend.cpp":
        # A tactic must be destroyed before the local AST manager, even on error.
        text = replace_once(text, "static tactic_ref    g_tac;", "static tactic *      g_tac = nullptr;")
        text = replace_once(text, "    g_tac = mk_parallel_qffd_tactic(m, p);", "    tactic_ref tac = mk_parallel_qffd_tactic(m, p);\n    g_tac = tac.get();")
        text = replace_once(text, "    g_start_time = clock();", "    g_start_time = clock();\n    g_st.reset();\n    g_tac = nullptr;")
    if path == "src/shell/datalog_frontend.cpp":
        text = replace_once(text, "    g_overall_time.start();", "    g_overall_time = stopwatch();\n    g_overall_time.start();")
    if path == "src/util/scoped_ctrl_c.cpp":
        text = replace_once(text, "    m_enabled(enabled)", "    m_enabled(false)")
    if path == "src/util/rlimit.h":
        text = replace_once(text,
                            "m_cancel == 0 && m_count <= m_limit && !is_timeout()",
                            "m_cancel == 0 && m_count <= m_limit && !is_timeout() && !z3_embedded::cancelled()")
    if path == "src/parsers/smt2/smt2parser.cpp":
        text = replace_once(text, "                            parse_cmd();", "                            z3_embedded::checkpoint();\n                            parse_cmd();")

    # Keep literals/comments intact, including the upstream help text.
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|R"(?P<delim>[^ (\\\t\r\n]*)\(.*?\)(?P=delim)"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?:std::)?\b(?:cout|cerr|cin|clog|exit|_Exit|ifstream|ofstream|fstream|fopen)\b', re.DOTALL)
    names = {"std::cout": "z3_embedded::out()", "std::cerr": "z3_embedded::err()",
             "std::clog": "z3_embedded::err()", "std::cin": "z3_embedded::in()",
             "exit": "z3_embedded::exit", "_Exit": "z3_embedded::exit",
             "std::exit": "z3_embedded::exit", "std::_Exit": "z3_embedded::exit",
             "std::ifstream": "z3_embedded::ifstream", "std::ofstream": "z3_embedded::ofstream",
             "std::fstream": "z3_embedded::fstream", "fopen": "z3_embedded::fopen"}
    def substitute(match):
        if match[0] in ("exit", "_Exit", "std::exit", "std::_Exit"):
            if not re.match(r"\s*\(", text[match.end():]):
                return match[0]
        return names.get(match[0], match[0])
    text = tokens.sub(substitute, text)
    return '#include "Z3Embedded.h"\n' + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for name in args.files:
        source = Path(args.source_root) / name
        output = Path(args.output_root) / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(adapt(name, source.read_text()), encoding="utf-8")


if __name__ == "__main__":
    main()
