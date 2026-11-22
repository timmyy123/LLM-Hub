// The engine's tunables, described once.
//
// Every RunConfig field a person can set is one row of one table: its key, type, bounds, group,
// help text, and how to read and write it. The CLI parses its flags from the table and prints its
// usage from it; a front-end renders its settings form from params_json(). Adding a knob is one row
// here, instead of a flag parser, a usage line, a UI control and a settings-to-argv mapping kept in
// step by hand — the arrangement that let the Android app's settings drift from the engine.
//
// Keys are the long CLI flag without its dashes, which is also the name a planner Decision uses for
// the knob it decided, so a plan and a settings form line up without a mapping table. A flag that
// only turns something off (--no-odirect) gets the positive key (o-direct) and is a switch on it.
//
// The bounds are what a form offers, written with the same named constants validate() checks, so
// they cannot drift apart silently. validate() stays the authority: rules that relate two knobs
// (prefetch needs the cache, route-ahead excludes speculation) live there only, and a front-end
// asks it rather than re-deriving them.
//
// Not in the table, because they are not tunables of a run: the prompt (a request's payload), the
// line-protocol switch (`--progress`, an output mode of the CLI, which reaches the engine as
// RunOutputs::render_text) and the compute-trace granularity (IComputeTraceSink::layers_only(), a
// property of a diagnostics sink). Front-ends own those, and none of them is a RunConfig field.
//
// Pure policy: no llama.cpp, no I/O, no environment.
#pragma once

#include "bmoe/config.h"

#include <functional>
#include <string>
#include <vector>

namespace bmoe {

// Path and Text are both free strings; the difference is only how a form presents them.
enum class ParamType { Bool, Int, Float, Choice, Path, Text };

// Groups in the order a settings form shows them: the enum order IS the form order, sent as
// `group_order` so a form does not depend on where a group's first row sits in the table.
enum class ParamGroup {
    Model,
    Generation,
    Sampling,
    Streaming,
    Prefill,
    Cache,
    Memory,
    Prefetch,
    Speculation,
    Lossy,
    Diagnostics
};

// How far from the defaults a person has to be to want this knob.
enum class ParamLevel {
    Basic,        // what anyone running a model sets
    Advanced,     // measured levers with a known trade-off
    Experimental, // opt-in, pending an A/B on the target device
    Debug,        // tests, diagnostics and A/B baselines; never a speed setting
};

// When a change takes effect in a running session.
enum class ParamScope {
    Request, // per generation (GenerateRequest): no reload
    Session, // fixed when the session opens: changing it reopens the session
};

struct ParamChoice {
    std::string value; // as typed on the command line and stored by a front-end
    std::string label; // as shown in a form
};

// A bare flag that sets its parameter to a fixed value: --no-odirect sets o-direct=false, --mtp sets
// spec=mtp. A deprecated switch still parses but is left out of the usage and of to_args().
struct ParamSwitch {
    std::string flag;
    std::string value;
    bool deprecated = false;
};

// A value of a numeric parameter that means something other than its number: 0 = off, 0 = the
// model's own, 4294967295 = random. A form shows the label in place of the number.
struct ParamSpecial {
    std::string value; // as get() renders it
    std::string label; // "off", "as wide as the context"
};

struct ParamDesc {
    std::string key;   // stable identifier: the long flag without dashes
    std::string label; // short name for a form
    ParamType type = ParamType::Int;
    ParamGroup group = ParamGroup::Generation;
    ParamLevel level = ParamLevel::Advanced;
    ParamScope scope = ParamScope::Session;
    // A settings form offers it: every basic row, every advanced row of a group that shapes the
    // answer (param_group_shapes_answer), and the few deeper ones a person picks rather than the
    // plan (which devices, drafts). The rest stay flags of the CLI: the plan sets them, or they are
    // a value nobody types into a phone. Experimental and debug rows never are.
    bool form = false;
    std::string help; // one paragraph, in the user's terms
    // What a form folds under the help, in a sentence or two of plain words: when to change the
    // setting and what it costs. Every row a form shows has one (#468).
    std::string details;

    std::string flag;       // "--cache-mb": takes a value. Empty when the parameter has only switches
    std::string short_flag; // "-m", or empty
    std::string value_hint; // "N", "F", "PATH", "N|auto", "mmap|warm|..." for the usage line
    std::vector<ParamSwitch> switches;
    std::vector<ParamChoice> choices; // Choice only

    // What a form offers; validate() decides what a run accepts. Unbounded when bounded is false.
    // bounded_below: only `min` is a limit, because the only upper one is the type's, which is no
    // range to show a person.
    bool bounded = false;
    bool bounded_below = false;
    double min = 0.0;
    double max = 0.0;
    std::string unit; // "MiB", "tokens", "" ...

    // What a person usually sets, for a form to show as the guide. Not a limit: a value outside it
    // is accepted, it is just unusual. Absent where the right value depends on the model or the
    // machine, and the plan is the guide instead.
    bool has_typical = false;
    double typical_min = 0.0;
    double typical_max = 0.0;
    std::vector<ParamSpecial> specials;

    bool lossy = false;              // changes the output: a quality trade, not an optimisation
    bool accepts_auto = false;       // an Int that also takes the literal "auto"
    bool exclusive_switches = false; // two different switches on one command line are a contradiction

    // Read the parameter as the string a flag would carry; write it from one. `set` rejects a
    // malformed value with a reason and leaves the config untouched.
    std::function<std::string(const RunConfig &)> get;
    std::function<bool(RunConfig &, const std::string &, std::string & err)> set;
};

// The table, in form order. Built once; the reference stays valid for the process lifetime.
const std::vector<ParamDesc> & params();

// By key, or nullptr.
const ParamDesc * find_param(const std::string & key);

const char * param_type_name(ParamType t);
const char * param_group_name(ParamGroup g);  // stable identifier, as in params_json()
const char * param_group_label(ParamGroup g); // section title for a form or a usage text: short
const char * param_group_help(ParamGroup g);  // the group in one line, under its title
// True for a group a front-end sets with a control of its own (the model file, from a model list):
// a settings form shows its value and does not edit it. The CLI sets it like any other flag.
bool param_group_read_only(ParamGroup g);
// True for a group whose rows decide what the answer is (sampling, quality trades) rather than how
// fast it comes. The plan never sets them, so a person is the only one who will: a form offers their
// advanced rows too, unless a row opts out (the seed, which replays a run rather than choosing one).
bool param_group_shapes_answer(ParamGroup g);
const char * param_level_name(ParamLevel l);
const char * param_scope_name(ParamScope s);

// Match one command-line token against the table. `value` is the token after it, or nullptr when
// there is none. `matched` is false for a token the table does not know (a front-end flag), which
// the caller then handles itself.
struct FlagResult {
    bool matched = false;
    bool consumed_value = false; // the caller must skip `value`
    const ParamDesc * param = nullptr;
    const ParamSwitch * via_switch = nullptr; // set when a bare switch matched
    std::string error;                        // non-empty: matched, but the value was rejected
};
FlagResult apply_flag(RunConfig & cfg, const char * arg, const char * value);

// The table's own integer reading, for a front-end value that is not a parameter (a CLI mode's
// count, a server's port): the whole string must be the number, and it must lie in [lo, hi]. What
// atoi() would make of "4x" or "abc" is an error here, with the reason in `err`.
bool parse_int_strict(const std::string & s, long long lo, long long hi, long long & out, std::string & err);

// The config as command-line arguments: every parameter that differs from a default RunConfig, in
// table order, in the spelling apply_flag() reads back. Parsing the result into a default config
// reproduces `cfg` for every parameter the table describes — the property a "copy as command"
// button and a bench cell both depend on, and which the tests check.
std::vector<std::string> to_args(const RunConfig & cfg);

// How a person types parameter `key` at `value`, as one string in the spelling to_args() emits: the
// switch that says exactly that value ("--overlap"), or the flag and the value ("--ctx-size 4096").
// With no value, the flag that takes one ("--ubatch"). Empty for a key the table does not have.
// This is what prose that tells a person what to type is built from - the planner's rationale - so
// renaming a flag in the table cannot leave advice naming one that no longer parses.
std::string flag_spelling(const std::string & key, const std::string & value = "");

// How a front-end names a parameter in a message the engine wrote. The engine's messages (validate(),
// a session that will not open) name a setting as `{key}`, or `{key=value}` for one value of it, and
// never as a flag or a field: the CLI's user types flags, a settings form's user sees labels, and the
// engine cannot know which one is reading. Each front-end renders once, where the message leaves it.
//   Flags:  {ubatch} -> --ubatch, {spec=mtp} -> --mtp, {ctx-size=4096} -> --ctx-size 4096
//   Labels: {ubatch} -> "Batch width", {spec=mtp} -> "Speculative drafts" set to MTP head
// Braces that do not name a key of the table are left as they are, so a message quoting JSON or a
// file's own text cannot be mangled.
enum class ParamNames { Flags, Labels };
std::string name_params(const std::string & message, ParamNames how);

// The table as a JSON array, one object per parameter, with `default` read from `defaults` — so a
// front-end that resolves its own defaults (the CLI turns the cache on with streaming) can say so.
std::string params_json(const RunConfig & defaults);

// The current value of every parameter as a JSON object keyed by parameter key.
std::string config_json(const RunConfig & cfg);

} // namespace bmoe
