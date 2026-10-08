# Embedded Z3 command

Local Code registers `z3_main` using the generic `ios_registerCommand("z3", z3_main)` API. The same adapter is registered by LocalCodeCLI. The command implementation lives here, not in ios_system's built-in command table.

## Upstream reuse

`@z3//:shell` compiles the pinned Z3 4.15.4 `src/shell` sources, including its original option parser, help, parameter handling, and SMT-LIB, DIMACS, optimization, Datalog, DRAT, and API-log frontends. There is no second argument parser or reduced SMT-LIB interpreter.

`embed.py` generates an adapted source tree during the build. It preserves literals/comments and mechanically redirects Z3's C++ streams and file streams, and replaces process exits with exceptions. Small version-checked edits handle the entry-point name, global state, signal handlers, and runtime ownership. Updating Z3 should include reviewing these adaptations; exact replacements fail the build if their source anchors drift.

## In-process behavior

- `Z3ShellRun` owns invocation streams and fresh CLI parameters. It restores host parameters afterward and never finalizes the shared solver runtime. `Z3BridgeCheckSMT` and shell invocations share a runtime lock; they cannot execute simultaneously because upstream CLI settings are process-global. New direct callers must use the same serialization boundary.
- `z3_main` snapshots the session working directory and `/home`, `/tmp`, `/app_bundle` aliases. Z3 file streams use those paths, including `(include ...)` and output-channel files. `/app_bundle` writes are rejected.
- No process-wide signal handlers or standard-stream buffer swaps are installed. On Darwin, command pipes are temporarily nonblocking and suppress SIGPIPE per descriptor, with original flags restored on return.
- Z3's `-t:` per-query timeout retains its upstream implementation. `-T:` is a cooperative invocation deadline, checked during parsing, resource-limit checks, and command I/O; expiration returns 124. It is not a separate-process hard kill.
- App cancellation uses a generic ios_system cancellation context, returns 130, and unwinds solver objects before pthread cancellation is reenabled. Waiting for the shared runtime lock is cancellable too.
- The host keeps runtime allocations/caches alive between commands; `-memory:` therefore measures shared Z3 allocations, not a fresh process's memory. Flags, optimization handles, and memory-limit failure state are reset for subsequent runs.

Examples inside Local Code:

```sh
z3 -h
z3 -st problem.smt2
printf '(declare-const x Int)(assert (= x 7))(check-sat)(get-model)\n' | z3 -in
z3 -T:10 -t:1000 smt.random_seed=42 problem.smt2
```

## Verification

```sh
bazel test //Vendors/Z3:Z3ShellTests //Vendors/Z3:Z3CommandTests //Vendors/Z3:Z3BridgeTests
bazel test //Apps/LocalCode:CheckSmtToolTests //Vendors/ios_system:OSHIntegrationTests
bazel build //Apps/LocalCode:LocalCode --ios_multi_cpus=arm64
bazel build //Apps/LocalCode:LocalCodeCLI
```

The command tests exercise actual ios_system registration/dispatch, aliases, includes, redirects, errors, and cancellation followed by reuse. Shell tests cover upstream options, incremental SMT-LIB and models, alternate frontends, limits, live host contexts, parameter/stream/signal preservation, and concurrent callers.
