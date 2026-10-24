//! Segmented download: one file of known size fetched as fixed-size byte
//! ranges over several connections at once.
//!
//! - The `.part` is preallocated to its full size, so its length no longer
//!   says how much has arrived. `<dest>.part.segments` does: bytes written
//!   from the start of every segment. It is rewritten atomically about once
//!   a second and only ever claims bytes already written, so a killed
//!   process resumes each segment where it stopped.
//! - Segments are handed out in ascending order (NTFS zero-fills the gap
//!   before a write past the valid data length, so keep that gap small).
//!   Each connection writes at its own offsets through its own handle.
//! - The first range request is the probe: a 206 whose Content-Range
//!   matches the pinned size turns on the parallel path; a 200 from a
//!   server that ignores Range, or an error status, is handed back for the
//!   single-stream path to deal with, after the `.part` has been cut back
//!   to its contiguous prefix. (The probe body is dropped and its segment
//!   fetched again: a WinHTTP response cannot move to another thread.)
//! - Only for files with a pinned sha256: the hash is what makes it safe
//!   that a full-length `.part` whose state file was lost reads as done.
//! - A legacy single-stream `.part` (no state file) is upgraded: its length
//!   is a done prefix.
//! - Progress, cancel, the artifact-lock heartbeat and the state file stay
//!   on the calling thread; connections only write and count.

use super::{ArtifactLock, DownloadProgress, Downloader};
use crate::backend::CancelToken;
use crate::error::AssetAiError;
use crate::http_client::{http_fetch, BearerAuth, HttpClientRequest, HttpClientResponse};
use crate::registry::FileSpec;
use makepad_micro_serde::*;
use std::collections::VecDeque;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const STATE_VERSION: u32 = 1;
/// Consecutive failed attempts on one segment (no byte written in between)
/// before the download gives up; the partial stays for a later resume.
const MAX_ATTEMPTS: u32 = 6;
const POLL: Duration = Duration::from_millis(100);
const SAVE_EVERY: Duration = Duration::from_secs(1);

pub(super) enum Outcome {
    /// Every byte is in the `.part`; it still needs its hash check.
    Complete,
    /// The server did not answer the first range with a matching 206. This
    /// is its response, for the single-stream path.
    Single(HttpClientResponse),
}

#[derive(Clone, Debug, SerJson, DeJson)]
struct SegmentState {
    version: u32,
    url: String,
    size: u64,
    sha256: Option<String>,
    segment_bytes: u64,
    /// Bytes written from the start of each segment.
    done: Vec<u64>,
}

pub(super) fn state_path(part: &Path) -> PathBuf {
    let mut os = part.as_os_str().to_os_string();
    os.push(".segments");
    PathBuf::from(os)
}

fn read_state(part: &Path) -> Option<SegmentState> {
    let text = fs::read_to_string(state_path(part)).ok()?;
    SegmentState::deserialize_json(&text).ok()
}

/// Write then rename over the old state: `fs::rename` replaces atomically on
/// every platform, so there is no moment without a state file (a lost one
/// would make the preallocated `.part` read as complete).
fn save_state(part: &Path, state: &SegmentState) -> Result<(), AssetAiError> {
    let path = state_path(part);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp-{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, state.serialize_json().as_bytes())
        .map_err(|e| AssetAiError::Download(format!("write {}: {e}", tmp.display())))?;
    fs::rename(&tmp, &path).map_err(|e| {
        AssetAiError::Download(format!("rename {} -> {}: {e}", tmp.display(), path.display()))
    })
}

impl SegmentState {
    fn segments(&self) -> usize {
        self.size.div_ceil(self.segment_bytes) as usize
    }

    fn span(&self, index: usize) -> (u64, u64) {
        let start = index as u64 * self.segment_bytes;
        (start, (start + self.segment_bytes).min(self.size))
    }

    /// Bytes from offset 0 with no gap.
    fn contiguous_prefix(&self) -> u64 {
        let mut prefix = 0;
        for (index, &done) in self.done.iter().enumerate() {
            let (start, end) = self.span(index);
            prefix = start + done;
            if start + done < end {
                break;
            }
        }
        prefix
    }
}

/// Turns a segmented `.part` back into a plain prefix the single-stream
/// and peer paths can append to; no-op without a state file.
pub(super) fn collapse_to_prefix(part: &Path) -> Result<(), AssetAiError> {
    let state_file = state_path(part);
    if !state_file.exists() {
        return Ok(());
    }
    let prefix = read_state(part).map_or(0, |state| state.contiguous_prefix());
    if part.exists() {
        let file = fs::OpenOptions::new()
            .write(true)
            .open(part)
            .map_err(|e| AssetAiError::Download(format!("open {}: {e}", part.display())))?;
        file.set_len(prefix)
            .map_err(|e| AssetAiError::Download(format!("truncate {}: {e}", part.display())))?;
    }
    let _ = fs::remove_file(&state_file);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fetch(
    downloader: &Downloader,
    url: &str,
    file: &FileSpec,
    size: u64,
    part: &Path,
    progress: &mut dyn FnMut(DownloadProgress),
    cancel: &CancelToken,
    artifact_lock: &ArtifactLock,
) -> Result<Outcome, AssetAiError> {
    let segment_bytes = downloader.segment_bytes.max(1);
    let fresh = |done: Vec<u64>| SegmentState {
        version: STATE_VERSION,
        url: url.to_string(),
        size,
        sha256: file.sha256.clone(),
        segment_bytes,
        done,
    };
    let part_len = fs::metadata(part).map(|meta| meta.len()).ok();
    let state = match read_state(part) {
        Some(state)
            if state.version == STATE_VERSION
                && state.url == url
                && state.size == size
                && state.sha256 == file.sha256
                && state.segment_bytes == segment_bytes
                && state.done.len() == state.segments()
                && part_len == Some(size) =>
        {
            state
        }
        Some(_) => {
            // Another file, or another segment layout: nothing in the
            // preallocated `.part` can be trusted.
            let _ = fs::remove_file(part);
            let _ = fs::remove_file(state_path(part));
            fresh(vec![0; size.div_ceil(segment_bytes) as usize])
        }
        None => {
            // A single-stream `.part` is a done prefix (the caller has
            // already refused one longer than the pinned size).
            let mut state = fresh(vec![0; size.div_ceil(segment_bytes) as usize]);
            let prefix = part_len.unwrap_or(0).min(size);
            for index in 0..state.done.len() {
                let (start, end) = state.span(index);
                state.done[index] = prefix.clamp(start, end) - start;
            }
            state
        }
    };
    let total_done: u64 = state.done.iter().sum();
    let pending: VecDeque<usize> = (0..state.segments())
        .filter(|&index| {
            let (start, end) = state.span(index);
            start + state.done[index] < end
        })
        .collect();
    let Some(&first) = pending.front() else {
        let _ = fs::remove_file(state_path(part));
        return Ok(Outcome::Complete);
    };

    // The probe: the first missing range. Its 206 is the first segment's
    // body; anything else goes back to the single stream.
    let (start, end) = state.span(first);
    let from = start + state.done[first];
    let response = range_request(downloader, url, from, end)?;
    if response.status == 206 && !range_matches(&response, from, end, size) {
        return Err(AssetAiError::Download(format!(
            "{url}: asked for bytes {from}-{}, got content-range {:?}",
            end - 1,
            response.header("content-range")
        )));
    }
    if response.status != 206 {
        collapse_to_prefix(part)?;
        return Ok(Outcome::Single(response));
    }
    drop(response);

    // From here the `.part` length says nothing: the state file must exist
    // before the file grows.
    save_state(part, &state)?;
    let out = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(part)
        .map_err(|e| AssetAiError::Download(format!("open {}: {e}", part.display())))?;
    out.set_len(size)
        .map_err(|e| AssetAiError::Download(format!("preallocate {}: {e}", part.display())))?;
    drop(out);

    let shared = Shared {
        downloader,
        url,
        part,
        state: &state,
        done: state.done.iter().map(|&done| AtomicU64::new(done)).collect(),
        total: AtomicU64::new(total_done),
        queue: Mutex::new(pending),
        stop: AtomicBool::new(false),
        error: Mutex::new(None),
        running: AtomicUsize::new(0),
    };
    let connections = downloader
        .connections
        .clamp(1, shared.queue.lock().unwrap().len());
    let mut snapshot = state.clone();
    let result = std::thread::scope(|scope| -> Result<(), AssetAiError> {
        shared.running.store(connections, Ordering::SeqCst);
        for _ in 0..connections {
            let shared = &shared;
            scope.spawn(move || {
                shared.connection(cancel);
                shared.running.fetch_sub(1, Ordering::SeqCst);
            });
        }
        let mut last_save = Instant::now();
        let mut outcome = Ok(());
        loop {
            let running = shared.running.load(Ordering::SeqCst);
            if cancel.is_cancelled() {
                shared.stop.store(true, Ordering::SeqCst);
            }
            progress(DownloadProgress {
                file: file.path.clone(),
                done: shared.total.load(Ordering::SeqCst),
                total: Some(size),
            });
            if running == 0 {
                break;
            }
            if last_save.elapsed() >= SAVE_EVERY {
                last_save = Instant::now();
                shared.snapshot(&mut snapshot);
                // A missed save only costs resume progress; the lock
                // heartbeat is what keeps other processes out.
                let _ = save_state(part, &snapshot);
                if let Err(error) = artifact_lock.heartbeat() {
                    shared.stop.store(true, Ordering::SeqCst);
                    outcome = Err(error);
                }
            }
            std::thread::sleep(POLL);
        }
        outcome
    });
    shared.snapshot(&mut snapshot);
    if snapshot.done.iter().enumerate().all(|(index, &done)| {
        let (start, end) = snapshot.span(index);
        start + done == end
    }) {
        // Every byte is written: from here the full-length `.part` is
        // complete on its own, like a finished single stream.
        let _ = fs::remove_file(state_path(part));
        result?;
        return Ok(Outcome::Complete);
    }
    save_state(part, &snapshot)?;
    result?;
    cancel.check()?;
    if let Some(error) = shared.error.lock().unwrap().take() {
        return Err(error);
    }
    Err(AssetAiError::Download(format!(
        "{url}: segmented download stopped early (partial file kept for resume)"
    )))
}

struct Shared<'a> {
    downloader: &'a Downloader,
    url: &'a str,
    part: &'a Path,
    state: &'a SegmentState,
    done: Vec<AtomicU64>,
    total: AtomicU64,
    queue: Mutex<VecDeque<usize>>,
    stop: AtomicBool,
    error: Mutex<Option<AssetAiError>>,
    running: AtomicUsize,
}

enum Attempt {
    Finished,
    /// Worth another try: the connection broke, timed out, or the server
    /// asked to slow down.
    Retry(AssetAiError),
    Fatal(AssetAiError),
}

impl Shared<'_> {
    fn snapshot(&self, into: &mut SegmentState) {
        for (slot, done) in into.done.iter_mut().zip(&self.done) {
            *slot = done.load(Ordering::SeqCst);
        }
    }

    fn fail(&self, error: AssetAiError) {
        self.stop.store(true, Ordering::SeqCst);
        self.error.lock().unwrap().get_or_insert(error);
    }

    fn stopped(&self, cancel: &CancelToken) -> bool {
        self.stop.load(Ordering::SeqCst) || cancel.is_cancelled()
    }

    /// One connection: segments off the queue until it is empty, the
    /// download is stopped, or a segment fails for good.
    fn connection(&self, cancel: &CancelToken) {
        let out = match fs::OpenOptions::new().write(true).open(self.part) {
            Ok(out) => out,
            Err(e) => return self.fail(AssetAiError::Download(format!("open {}: {e}", self.part.display()))),
        };
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            let Some(index) = self.queue.lock().unwrap().pop_front() else {
                return;
            };
            let mut failures = 0;
            loop {
                if self.stopped(cancel) {
                    return;
                }
                let before = self.done[index].load(Ordering::SeqCst);
                match self.attempt(index, &out, &mut buf, cancel) {
                    Attempt::Finished => break,
                    Attempt::Fatal(error) => return self.fail(error),
                    Attempt::Retry(error) => {
                        if self.done[index].load(Ordering::SeqCst) > before {
                            failures = 0;
                        }
                        failures += 1;
                        if failures >= MAX_ATTEMPTS {
                            return self.fail(AssetAiError::Download(format!(
                                "{}: {error} (gave up after {MAX_ATTEMPTS} tries; partial file kept for resume)",
                                self.url
                            )));
                        }
                        // 1, 2, 4, 8, 16 s, cut short by a stop.
                        let wait = Instant::now();
                        let backoff = Duration::from_secs(1 << (failures - 1));
                        while wait.elapsed() < backoff && !self.stopped(cancel) {
                            std::thread::sleep(POLL);
                        }
                    }
                }
            }
        }
    }

    fn attempt(
        &self,
        index: usize,
        out: &fs::File,
        buf: &mut [u8],
        cancel: &CancelToken,
    ) -> Attempt {
        let (start, end) = self.state.span(index);
        let mut offset = start + self.done[index].load(Ordering::SeqCst);
        if offset >= end {
            return Attempt::Finished;
        }
        let response = match range_request(self.downloader, self.url, offset, end) {
            Ok(response) => response,
            Err(error) => return Attempt::Retry(error),
        };
        if !range_matches(&response, offset, end, self.state.size) {
            let error = AssetAiError::Download(format!(
                "{}: http {} for bytes {offset}-{} (content-range {:?})",
                self.url,
                response.status,
                end - 1,
                response.header("content-range")
            ));
            return match response.status {
                206 | 408 | 429 | 500..=599 => Attempt::Retry(error),
                _ => Attempt::Fatal(error),
            };
        }
        let mut body = response.body;
        while offset < end {
            if self.stopped(cancel) {
                return Attempt::Finished;
            }
            let want = buf.len().min((end - offset) as usize);
            let n = match body.read(&mut buf[..want]) {
                Ok(0) => {
                    return Attempt::Retry(AssetAiError::Download(format!(
                        "{}: connection ended at {offset} of segment end {end}",
                        self.url
                    )))
                }
                Ok(n) => n,
                Err(e) => {
                    return Attempt::Retry(AssetAiError::Download(format!("{}: body read: {e}", self.url)))
                }
            };
            if let Err(e) = write_all_at(out, &buf[..n], offset) {
                return Attempt::Fatal(AssetAiError::Download(format!(
                    "write {}: {e}",
                    self.part.display()
                )));
            }
            offset += n as u64;
            // Counted only once written, so the state file never claims
            // a byte that is not in the `.part`.
            self.done[index].fetch_add(n as u64, Ordering::SeqCst);
            self.total.fetch_add(n as u64, Ordering::SeqCst);
        }
        Attempt::Finished
    }
}

fn range_request(
    downloader: &Downloader,
    url: &str,
    from: u64,
    end: u64,
) -> Result<HttpClientResponse, AssetAiError> {
    http_fetch(&HttpClientRequest {
        method: "GET",
        url,
        range_from: Some(from),
        range_to: Some(end - 1),
        bearer: downloader.token.as_deref().map(|token| BearerAuth {
            token,
            host_suffix: &downloader.token_host_suffix,
        }),
        body: None,
        extra_headers: &[],
    })
}

/// A 206 for exactly `from..end` of a `size`-byte file.
fn range_matches(response: &HttpClientResponse, from: u64, end: u64, size: u64) -> bool {
    if response.status != 206 {
        return false;
    }
    let Some(value) = response.header("content-range") else {
        return false;
    };
    let Some(rest) = value.trim().strip_prefix("bytes ") else {
        return false;
    };
    let Some((range, total)) = rest.split_once('/') else {
        return false;
    };
    let Some((first, last)) = range.split_once('-') else {
        return false;
    };
    first.trim().parse::<u64>().ok() == Some(from)
        && last.trim().parse::<u64>().ok() == Some(end - 1)
        && total.trim().parse::<u64>().ok() == Some(size)
}

#[cfg(unix)]
fn write_all_at(file: &fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn write_all_at(file: &fs::File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn write_all_at(file: &fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(buf)
}
