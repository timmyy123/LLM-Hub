module

prelude
import Lean.Elab.Frontend
import Lean.Elab.BuiltinCommand
import Lean.Elab.BuiltinTerm
import Lean.Elab.Declaration
import Lean.Elab.Extra
import Lean.Elab.MutualDef
import Lean.Elab.Tactic.Basic
import Lean.Elab.Tactic.BuiltinTactic
import Lean.Elab.Tactic.Calc
import Lean.Elab.Tactic.Change
import Lean.Elab.Tactic.Congr
import Lean.Elab.Tactic.Generalize
import Lean.Elab.Tactic.Induction
import Lean.Elab.Tactic.Injection
import Lean.Elab.Tactic.Omega
import Lean.Elab.Tactic.Repeat
import Lean.Elab.Tactic.Rewrite
import Lean.Elab.Tactic.Rfl
import Lean.Elab.Tactic.Simp
import Lean.Elab.Tactic.Simpa
import Lean.Elab.Tactic.SolveByElim
import Lean.Elab.Tactic.Split
import Lean.Elab.Tactic.Symm
import Lean.Elab.Tactic.Unfold
import Lean.Util.CollectAxioms
import Lean.Util.Path
import all Lean.Shell

open Lean Elab

private def standardAxioms : NameSet :=
  NameSet.ofList [`propext, `Classical.choice, `Quot.sound]

-- Keep proof holes and the deprecated native-reduction shortcuts outside this
-- experiment's accepted proof policy, even when other assumptions are reported.
private def unsupportedProofAxioms : NameSet :=
  NameSet.ofList [`sorryAx, `Lean.ofReduceBool, `Lean.ofReduceNat]

private def loadArtifacts (roots : Array System.FilePath) : IO (NameMap ImportArtifacts) := do
  let mut artifacts : NameMap ImportArtifacts := {}
  for root in roots do
    for path in ← root.walkDir do
      if path.extension == some "olean" then
        let name ← moduleNameOfFileName path (some root)
        let irSig := path.withExtension "ir.sig"
        let ir := path.withExtension "ir"
        let irFiles :=
          if (← irSig.pathExists) && (← ir.pathExists) then #[irSig, ir] else #[]
        unless artifacts.contains name do
          artifacts := artifacts.insert name (.ofArrays #[#[path], irFiles])
  pure artifacts

-- Use Lean's parser (including implicit Init and nested comments), not a textual
-- approximation of import syntax. The checker itself reports header diagnostics.
@[export check_lean_source_imports]
def sourceImports (input : String) : IO String := do
  let (header, _, _) ← Parser.parseHeader (Parser.mkInputContext input "<check_lean>")
  pure (toJson ((HeaderSyntax.imports header).map (toString ·.module))).compress

-- Local, unpacked packages may not have an archive manifest. Read only the
-- requested module's import metadata, without executing its initializers.
@[export check_lean_module_imports]
unsafe def moduleImports (path : String) : IO String := do
  let (data, region) ← readModuleData path
  -- An IO reference forces serialization before freeing the mapped region.
  -- A pure `let` may be sunk past `region.free` by the native compiler.
  let result ← IO.mkRef (toJson (data.imports.map (toString ·.module))).compress
  region.free
  result.get

private unsafe def processInput
    (input : String) (roots : Array System.FilePath) (artifactJson : String)
    (fileName : String := "<check_lean>") (options : Options := {})
    (cancelTk? : Option IO.CancelToken := none) :
    IO (Environment × Environment × MessageLog) := do
  let inputCtx := Parser.mkInputContext input fileName
  enableInitializersExecution
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  searchPathRef.set roots.toList
  let artifacts ← if artifactJson.isEmpty then loadArtifacts roots else
    IO.ofExcept (Json.parse artifactJson >>= fromJson? (α := NameMap ImportArtifacts))
  let opts := Elab.async.set options false
  -- Local packages contain only exported artifacts. Treat legacy headers as module
  -- headers as well so `import Mathlib` does not require `.olean.private` files.
  let (initialEnv, messages) ← processHeaderCore
    (HeaderSyntax.startPos header) (HeaderSyntax.imports header) true
    opts messages inputCtx (arts := artifacts) (headerStx? := header)
  if messages.hasErrors then
    return (initialEnv, initialEnv, messages)
  if let some cancelTk := cancelTk? then
    -- The synchronous frontend lets an embedded caller own the command's streams
    -- and cancellation token, without the shell's process-lifetime task tree.
    let mut parserState := parserState
    let mut state := Command.mkState initialEnv messages opts
    let mut allMessages := messages
    repeat
      if ← cancelTk.isSet then throw (IO.userError "interrupted")
      let scope := state.scopes.head!
      let pmctx : Parser.ParserModuleContext := {
        env := state.env
        options := scope.opts
        currNamespace := scope.currNamespace
        openDecls := scope.openDecls }
      let pos := parserState.pos
      let (stx, next, messages) := Parser.parseCommand inputCtx pmctx parserState {}
      parserState := next
      state := { state with messages }
      let ctx : Command.Context := {
        cmdPos := pos
        fileName
        fileMap := inputCtx.fileMap
        snap? := none
        cancelTk? := some cancelTk }
      match ← EIO.toIO' ((Command.elabCommandTopLevel stx #[] ctx).run state) with
      | .error e => throw (IO.userError (← e.toMessageData.toString))
      | .ok (_, next) => state := next
      -- The top-level elaborator resets its log per command, including at EOF.
      allMessages := allMessages ++ messages ++ state.messages
      if Parser.isTerminalCommand stx then break
    -- Auxiliary checking and linters may produce snapshot tasks even when main
    -- elaboration is synchronous. Wait for them and retain unreported diagnostics.
    for task in state.snapshotTasks do
      for snapshot in task.get.getAll do
        for message in snapshot.diagnostics.msgLog.unreported do
          allMessages := allMessages.add message
    pure (initialEnv, state.env, allMessages)
  else
    let state ← IO.processCommands inputCtx parserState (Command.mkState initialEnv messages opts)
    pure (initialEnv, state.commandState.env, state.commandState.messages)

private def newDeclarations (before after : Environment) : Array Name :=
  after.constants.toList.foldl (init := #[]) fun names (name, _) =>
    if before.find? name |>.isSome then names else names.push name

private def collectDeclarationAxioms (env : Environment) (name : Name) : IO (Array Name) :=
  Core.CoreM.toIO' (collectAxioms name)
    { fileName := "<check_lean>", fileMap := default }
    { env }

private def proofResponse (before after : Environment) (messages : MessageLog)
    (errorOnKinds : Array Name := #[]) :
    IO (String × UInt32) := do
  let mut lines : Array String := #[]
  let mut errors := messages.hasErrors
  for message in messages.toList do
    let message := if errorOnKinds.contains message.kind then
      { message with severity := .error } else message
    if message.severity == .error then errors := true
    if message.isSilent then continue
    lines := lines.push (← message.toString)
  if errors then
    lines := lines.push "status: invalid"
    return (String.intercalate "\n" lines.toList, 1)
  let declarations := newDeclarations before after
  let mut assumptions : NameSet := {}
  let mut unsupportedAxioms : NameSet := {}
  for name in declarations do
    let axioms ← collectDeclarationAxioms after name
    for axiomName in axioms do
      if unsupportedProofAxioms.contains axiomName then
        unsupportedAxioms := unsupportedAxioms.insert axiomName
      else unless standardAxioms.contains axiomName do
        assumptions := assumptions.insert axiomName
  if !unsupportedAxioms.isEmpty then
    lines := lines.push "status: invalid"
    for name in unsupportedAxioms.toArray.qsort Name.lt do
      lines := lines.push s!"policy: unsupported proof axiom: {name}"
  else if assumptions.isEmpty then
    lines := lines.push "status: valid"
  else
    lines := lines.push "status: conditional"
  if !assumptions.isEmpty then
    lines := lines.push "assumptions:"
    for name in assumptions.toArray.qsort Name.lt do
      lines := lines.push s!"  {name}"
  return (String.intercalate "\n" lines.toList, if unsupportedAxioms.isEmpty then 0 else 1)

@[export check_lean_bridge]
unsafe def runCheckLean (input baseRoot packageRoot artifactJson : String) : IO String := do
  let roots := (System.SearchPath.parse packageRoot).toArray.push baseRoot
  let (before, after, messages) ← processInput input roots artifactJson
    (options := maxRecDepth.set (maxHeartbeats.set {} 200000) 1000)
  return (← proofResponse before after messages).1

@[export lean_bridge_new_cancel_token]
def newCancelToken : BaseIO IO.CancelToken := IO.CancelToken.new

@[export lean_bridge_cancel_token]
def cancelToken (token : IO.CancelToken) : BaseIO Unit := token.set

@[export lean_bridge_command]
unsafe def runCommand (input fileName baseRoot packageRoot artifactJson : String)
    (shellOpts : ShellOptions) (audit : UInt8) (cancelTk : IO.CancelToken) : IO UInt32 := do
  let opts := shellOpts.leanOpts
  let roots := (System.SearchPath.parse packageRoot).toArray.push baseRoot
  let (before, after, messages) ← processInput input roots artifactJson fileName opts cancelTk
  if audit != 0 then
    let (response, status) ← proofResponse before after messages shellOpts.errorOnKinds
    (← IO.getStdout).putStrLn response
    return status
  let mut errors := messages.hasErrors
  for message in messages.toList do
    let message := if shellOpts.errorOnKinds.contains message.kind then
      { message with severity := .error } else message
    if message.severity == .error then errors := true
    if message.isSilent then continue
    if shellOpts.jsonOutput then
      IO.println (← message.toJson).compress
    else
      (← IO.getStdout).putStr (← message.toString (opts.getBool `printMessageEndPos false))
  if errors then return 1
  return 0

@[export lean_bridge_options_stdin]
def optionsStdin (opts : ShellOptions) : Bool := opts.useStdin

@[export lean_bridge_options_timeout]
def optionsTimeout (opts : ShellOptions) : UInt64 := (timeout.get opts.leanOpts).toUInt64
