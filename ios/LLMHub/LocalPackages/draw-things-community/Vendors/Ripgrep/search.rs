use std::{
    collections::HashSet,
    ffi::{CStr, c_char, c_void},
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    ptr,
};

use grep::{
    matcher::Matcher,
    regex::{RegexMatcher, RegexMatcherBuilder},
    searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch},
};
use ignore::WalkBuilder;

const MODE_TEXT: i32 = 0;
const MODE_REGEX: i32 = 1;
const MODE_FILES: i32 = 2;

const RESULT_FILE: i32 = 0;
const RESULT_DIRECTORY: i32 = 1;
const RESULT_LINE: i32 = 2;

const STATUS_OK: i32 = 0;
const STATUS_INVALID_REGEX: i32 = 1;
const STATUS_ERROR: i32 = 2;

type ResultCallback = unsafe extern "C" fn(
    context: *mut c_void,
    kind: i32,
    path: *const u8,
    path_length: usize,
    line_number: u64,
    line: *const u8,
    line_length: usize,
);

type ErrorCallback =
    unsafe extern "C" fn(context: *mut c_void, message: *const u8, message_length: usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Text,
    Regex,
    Files,
}

impl Mode {
    fn from_raw(raw: i32) -> Option<Mode> {
        match raw {
            MODE_TEXT => Some(Mode::Text),
            MODE_REGEX => Some(Mode::Regex),
            MODE_FILES => Some(Mode::Files),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum SearchError {
    InvalidRegex(String),
    Error(String),
}

#[derive(Debug)]
struct Options<'a> {
    project_root: &'a Path,
    root: &'a Path,
    query: &'a str,
    glob: Option<&'a str>,
    mode: Mode,
    case_sensitive: bool,
    context_lines: usize,
    max_results: usize,
}

#[derive(Debug)]
struct Entry {
    path: PathBuf,
    relative_path: String,
    is_directory: bool,
}

#[derive(Debug)]
struct Line {
    number: u64,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct FileSink {
    events: Vec<Line>,
    matches: Vec<u64>,
    max_matches: usize,
    context_lines: usize,
    stop_after_line: Option<u64>,
}

impl FileSink {
    fn new(max_matches: usize, context_lines: usize) -> FileSink {
        FileSink {
            events: Vec::new(),
            matches: Vec::new(),
            max_matches,
            context_lines,
            stop_after_line: None,
        }
    }

    fn push_line(&mut self, number: u64, bytes: &[u8]) -> io::Result<()> {
        let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
        std::str::from_utf8(bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.events.push(Line {
            number,
            bytes: bytes.to_vec(),
        });
        Ok(())
    }
}

impl Sink for FileSink {
    type Error = io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        matched: &SinkMatch<'_>,
    ) -> Result<bool, io::Error> {
        let line_number = matched.line_number().unwrap_or(1);
        self.push_line(line_number, matched.bytes())?;
        if self.matches.len() < self.max_matches {
            self.matches.push(line_number);
            if self.matches.len() == self.max_matches {
                if self.context_lines == 0 {
                    return Ok(false);
                }
                self.stop_after_line = Some(line_number.saturating_add(self.context_lines as u64));
            }
            return Ok(true);
        }
        Ok(self.stop_after_line.is_some_and(|line| line_number < line))
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        context: &SinkContext<'_>,
    ) -> Result<bool, io::Error> {
        let line_number = context.line_number().unwrap_or(1);
        self.push_line(line_number, context.bytes())?;
        Ok(self.stop_after_line.is_none_or(|line| line_number < line))
    }
}

fn matcher(query: &str, mode: Mode, case_sensitive: bool) -> Result<RegexMatcher, SearchError> {
    let mut builder = RegexMatcherBuilder::new();
    builder
        .case_insensitive(!case_sensitive)
        .fixed_strings(mode != Mode::Regex);
    if mode != Mode::Files {
        builder.line_terminator(Some(b'\n')).ban_byte(Some(b'\0'));
    }
    builder
        .build(query)
        .map_err(|error| SearchError::InvalidRegex(error.to_string()))
}

fn glob_matcher(glob: Option<&str>) -> Result<Option<RegexMatcher>, SearchError> {
    let Some(glob) = glob.map(str::trim).filter(|glob| !glob.is_empty()) else {
        return Ok(None);
    };
    let mut pattern = String::from("^");
    for character in glob.chars() {
        match character {
            '*' => pattern.push_str(".*"),
            '?' => pattern.push('.'),
            '.' | '\\' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' => {
                pattern.push('\\');
                pattern.push(character);
            }
            _ => pattern.push(character),
        }
    }
    pattern.push('$');
    RegexMatcherBuilder::new()
        .build(&pattern)
        .map(Some)
        .map_err(|error| SearchError::Error(error.to_string()))
}

fn entries(options: &Options<'_>, glob: Option<&RegexMatcher>) -> Result<Vec<Entry>, SearchError> {
    let mut builder = WalkBuilder::new(options.root);
    builder
        .standard_filters(false)
        .hidden(true)
        .follow_links(false);

    let root_is_directory = options.root.is_dir();
    let mut entries = Vec::new();
    for result in builder.build() {
        let entry = result.map_err(|error| SearchError::Error(error.to_string()))?;
        let path = entry.path();
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        let is_directory = file_type.is_dir();
        if is_directory && path == options.root && root_is_directory {
            continue;
        }
        if options.mode == Mode::Files {
            if !file_type.is_file() && !is_directory {
                continue;
            }
        } else if !file_type.is_file() {
            continue;
        }
        let relative_path = path
            .strip_prefix(options.project_root)
            .ok()
            .and_then(Path::to_str)
            .map(|path| if path.is_empty() { "." } else { path })
            .ok_or_else(|| {
                SearchError::Error(format!(
                    "Path is not valid project-relative UTF-8: {}",
                    path.display()
                ))
            })?
            .to_owned();
        if glob.is_some_and(|glob| !glob.is_match(relative_path.as_bytes()).unwrap_or(false)) {
            continue;
        }
        entries.push(Entry {
            path: path.to_path_buf(),
            relative_path,
            is_directory,
        });
    }
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(entries)
}

fn emit(
    callback: ResultCallback,
    context: *mut c_void,
    kind: i32,
    path: &str,
    line_number: u64,
    line: Option<&[u8]>,
) {
    let (line_pointer, line_length) =
        line.map_or((ptr::null(), 0), |line| (line.as_ptr(), line.len()));
    unsafe {
        callback(
            context,
            kind,
            path.as_ptr(),
            path.len(),
            line_number,
            line_pointer,
            line_length,
        );
    }
}

fn search(
    options: &Options<'_>,
    callback: ResultCallback,
    context: *mut c_void,
) -> Result<bool, SearchError> {
    if options.max_results == 0 {
        return Err(SearchError::Error(
            "max_results must be greater than zero".to_owned(),
        ));
    }
    let matcher = matcher(options.query, options.mode, options.case_sensitive)?;
    let glob = glob_matcher(options.glob)?;
    let entries = entries(options, glob.as_ref())?;

    if options.mode == Mode::Files {
        let mut result_count = 0;
        for entry in entries {
            if !matcher
                .is_match(entry.relative_path.as_bytes())
                .map_err(|error| SearchError::Error(error.to_string()))?
            {
                continue;
            }
            emit(
                callback,
                context,
                if entry.is_directory {
                    RESULT_DIRECTORY
                } else {
                    RESULT_FILE
                },
                &entry.relative_path,
                0,
                None,
            );
            result_count += 1;
            if result_count == options.max_results {
                return Ok(true);
            }
        }
        return Ok(false);
    }

    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .before_context(options.context_lines)
        .after_context(options.context_lines)
        .binary_detection(BinaryDetection::quit(b'\0'))
        .bom_sniffing(false)
        .build();
    let mut match_count = 0;
    for entry in entries {
        let remaining = options.max_results - match_count;
        let mut sink = FileSink::new(remaining, options.context_lines);
        match searcher.search_path(&matcher, &entry.path, &mut sink) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::InvalidData => continue,
            Err(error) => return Err(SearchError::Error(error.to_string())),
        }
        if sink.matches.is_empty() {
            continue;
        }
        let ranges = sink
            .matches
            .iter()
            .map(|line| {
                (
                    line.saturating_sub(options.context_lines as u64).max(1),
                    line.saturating_add(options.context_lines as u64),
                )
            })
            .collect::<Vec<_>>();
        let mut emitted_lines = HashSet::new();
        for line in sink.events {
            if !ranges
                .iter()
                .any(|range| range.0 <= line.number && line.number <= range.1)
                || !emitted_lines.insert(line.number)
            {
                continue;
            }
            emit(
                callback,
                context,
                RESULT_LINE,
                &entry.relative_path,
                line.number,
                Some(&line.bytes),
            );
        }
        match_count += sink.matches.len();
        if match_count == options.max_results {
            return Ok(true);
        }
    }
    Ok(false)
}

fn required_string(pointer: *const c_char, name: &str) -> Result<String, SearchError> {
    if pointer.is_null() {
        return Err(SearchError::Error(format!("{name} is null")));
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| SearchError::Error(format!("{name} is not UTF-8: {error}")))
}

fn optional_string(pointer: *const c_char) -> Result<Option<String>, SearchError> {
    if pointer.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map(Some)
        .map_err(|error| SearchError::Error(format!("glob is not UTF-8: {error}")))
}

fn report_error(callback: Option<ErrorCallback>, context: *mut c_void, message: &str) {
    if let Some(callback) = callback {
        unsafe {
            callback(context, message.as_ptr(), message.len());
        }
    }
}

/// Searches synchronously and reports structured results through callbacks.
///
/// # Safety
///
/// `project_root`, `root`, and `query` must point to valid NUL-terminated
/// strings. `glob` must either be null or point to a valid NUL-terminated
/// string. `context` must remain valid for every callback. If non-null,
/// `max_results_reached` must be valid for writes for the duration of this
/// call. Callbacks must not retain the borrowed path, line, or error buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn local_code_ripgrep_search(
    project_root: *const c_char,
    root: *const c_char,
    query: *const c_char,
    glob: *const c_char,
    mode: i32,
    case_sensitive: bool,
    context_lines: usize,
    max_results: usize,
    context: *mut c_void,
    result_callback: Option<ResultCallback>,
    error_callback: Option<ErrorCallback>,
    max_results_reached: *mut bool,
) -> i32 {
    if !max_results_reached.is_null() {
        unsafe {
            *max_results_reached = false;
        }
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let project_root = required_string(project_root, "project_root")?;
        let root = required_string(root, "root")?;
        let query = required_string(query, "query")?;
        let glob = optional_string(glob)?;
        let mode =
            Mode::from_raw(mode).ok_or_else(|| SearchError::Error("Invalid mode".to_owned()))?;
        let callback = result_callback
            .ok_or_else(|| SearchError::Error("result_callback is null".to_owned()))?;
        search(
            &Options {
                project_root: Path::new(&project_root),
                root: Path::new(&root),
                query: &query,
                glob: glob.as_deref(),
                mode,
                case_sensitive,
                context_lines,
                max_results,
            },
            callback,
            context,
        )
    }));
    match result {
        Ok(Ok(reached)) => {
            if !max_results_reached.is_null() {
                unsafe {
                    *max_results_reached = reached;
                }
            }
            STATUS_OK
        }
        Ok(Err(SearchError::InvalidRegex(error))) => {
            report_error(error_callback, context, &error);
            STATUS_INVALID_REGEX
        }
        Ok(Err(SearchError::Error(error))) => {
            report_error(error_callback, context, &error);
            STATUS_ERROR
        }
        Err(_) => {
            report_error(error_callback, context, "Ripgrep search panicked");
            STATUS_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    type TestEvent = (i32, String, u64, String);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> TestDirectory {
            let path = std::env::temp_dir().join(format!(
                "local-code-ripgrep-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            TestDirectory(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn collect(options: &Options<'_>) -> Result<(Vec<TestEvent>, bool), SearchError> {
        unsafe extern "C" fn callback(
            context: *mut c_void,
            kind: i32,
            path: *const u8,
            path_length: usize,
            line_number: u64,
            line: *const u8,
            line_length: usize,
        ) {
            let events = unsafe { &mut *(context as *mut Vec<TestEvent>) };
            let path =
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(path, path_length) });
            let line = if line.is_null() {
                String::new()
            } else {
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(line, line_length) })
                    .into_owned()
            };
            events.push((kind, path.into_owned(), line_number, line));
        }

        let mut events = Vec::new();
        let reached = search(
            options,
            callback,
            (&mut events as *mut Vec<TestEvent>).cast(),
        )?;
        Ok((events, reached))
    }

    #[test]
    fn content_search_preserves_context_and_global_limit() {
        let directory = TestDirectory::new();
        fs::write(directory.0.join("a.txt"), "before\nneedle\nafter\n").unwrap();
        fs::write(directory.0.join("b.txt"), "needle\n").unwrap();
        fs::write(directory.0.join(".hidden.txt"), "needle\n").unwrap();
        fs::write(directory.0.join(".gitignore"), "b.txt\n").unwrap();

        let (events, reached) = collect(&Options {
            project_root: &directory.0,
            root: &directory.0,
            query: "needle",
            glob: Some("*.txt"),
            mode: Mode::Text,
            case_sensitive: true,
            context_lines: 1,
            max_results: 2,
        })
        .unwrap();

        assert!(reached);
        assert_eq!(
            events,
            vec![
                (RESULT_LINE, "a.txt".to_owned(), 1, "before".to_owned()),
                (RESULT_LINE, "a.txt".to_owned(), 2, "needle".to_owned()),
                (RESULT_LINE, "a.txt".to_owned(), 3, "after".to_owned()),
                (RESULT_LINE, "b.txt".to_owned(), 1, "needle".to_owned()),
            ]
        );
    }

    #[test]
    fn files_search_includes_directories() {
        let directory = TestDirectory::new();
        fs::create_dir_all(directory.0.join("src")).unwrap();
        fs::write(directory.0.join("src/file.swift"), "").unwrap();

        let (events, reached) = collect(&Options {
            project_root: &directory.0,
            root: &directory.0,
            query: "src",
            glob: None,
            mode: Mode::Files,
            case_sensitive: true,
            context_lines: 0,
            max_results: 10,
        })
        .unwrap();

        assert!(!reached);
        assert_eq!(
            events,
            vec![
                (RESULT_DIRECTORY, "src".to_owned(), 0, String::new()),
                (RESULT_FILE, "src/file.swift".to_owned(), 0, String::new(),),
            ]
        );
    }

    #[test]
    fn final_match_context_does_not_count_or_extend_later_matches() {
        let directory = TestDirectory::new();
        fs::write(directory.0.join("a.txt"), "needle\nneedle\nneedle\n").unwrap();

        let (events, reached) = collect(&Options {
            project_root: &directory.0,
            root: &directory.0,
            query: "needle",
            glob: None,
            mode: Mode::Text,
            case_sensitive: true,
            context_lines: 1,
            max_results: 1,
        })
        .unwrap();

        assert!(reached);
        assert_eq!(
            events,
            vec![
                (RESULT_LINE, "a.txt".to_owned(), 1, "needle".to_owned()),
                (RESULT_LINE, "a.txt".to_owned(), 2, "needle".to_owned()),
            ]
        );
    }

    #[test]
    fn invalid_regex_is_reported() {
        let directory = TestDirectory::new();
        let result = collect(&Options {
            project_root: &directory.0,
            root: &directory.0,
            query: "[",
            glob: None,
            mode: Mode::Regex,
            case_sensitive: true,
            context_lines: 0,
            max_results: 10,
        });
        assert!(matches!(result, Err(SearchError::InvalidRegex(_))));
    }
}
