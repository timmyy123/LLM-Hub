"""Use the pinned Lean CLI's option table verbatim, not a parallel flag grammar."""
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
start = source.index("static struct option g_long_options[] = {")
end = source.index("namespace lean {", start)
pathlib.Path(sys.argv[2]).write_text(source[start:end])
