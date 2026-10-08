import Lean

open Lean

-- Read the actual exported artifact's imports, including private/meta edges.
-- The conservative closure is phase-independent and needs no source parsing.
unsafe def main (args : List String) : IO Unit := do
  let [root, listPath] := args | throw (IO.userError "expected artifact root and file list")
  let root ← IO.FS.realPath root
  let stdout ← IO.getStdout
  for line in (← IO.FS.readFile listPath).splitOn "\n" do
    if line.isEmpty then continue
    let path : System.FilePath := root / line
    -- Bazel's individual input files are symlinks outside the sandbox root.
    -- Derive the name from the declared relative path, without resolving links.
    let name := ((System.FilePath.mk line).withExtension "").toString.replace "/" "."
    let (data, region) ← readModuleData path
    let record := Json.mkObj [
      ("module", toJson name),
      ("imports", toJson (data.imports.map (toString ·.module)))]
    stdout.putStrLn record.compress
    region.free
