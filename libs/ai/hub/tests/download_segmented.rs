//! Segmented downloads against a local fixture HTTP server that answers
//! connections in parallel: ranges over several connections, fallback when
//! the server ignores Range, resume from the segment state file and from a
//! single-stream `.part`, retry after a dropped connection, sha256 check.

use makepad_ai_hub::backend::CancelToken;
use makepad_ai_hub::download::{part_path, Downloader};
use makepad_ai_hub::error::AssetAiError;
use makepad_ai_hub::registry::FileSpec;
use makepad_ai_hub::sha256::sha256_hex;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SEGMENT: u64 = 1000;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Honors `Range: bytes=a-b` with 206 + Content-Range.
    Honor,
    /// Ignores Range and always answers 200 with the full body.
    Ignore,
    /// Like Honor, but the first ranged answer that starts past 0 breaks
    /// off halfway through its body.
    DropOnce,
}

struct Fixture {
    addr: SocketAddr,
    /// Request heads, lowercased.
    requests: Arc<Mutex<Vec<String>>>,
    /// Most connections open at the same time.
    peak: Arc<AtomicUsize>,
}

impl Fixture {
    fn ranges(&self) -> Vec<(u64, u64)> {
        self.requests.lock().unwrap().iter().filter_map(|head| parse_range(head)).collect()
    }
}

fn spawn_fixture(data: Vec<u8>, mode: Mode) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let peak = Arc::new(AtomicUsize::new(0));
    let open = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let data = Arc::new(data);
    let (requests_thread, peak_thread) = (requests.clone(), peak.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let (requests, peak, open, dropped, data) = (
                requests_thread.clone(),
                peak_thread.clone(),
                open.clone(),
                dropped.clone(),
                data.clone(),
            );
            std::thread::spawn(move || {
                let now = open.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                let head = read_head(&mut stream).to_ascii_lowercase();
                requests.lock().unwrap().push(head.clone());
                // Hold the connection a moment so parallel ones overlap.
                std::thread::sleep(std::time::Duration::from_millis(20));
                match (mode, parse_range(&head)) {
                    (Mode::Honor | Mode::DropOnce, Some((from, to))) => {
                        let to = to.min(data.len() as u64 - 1);
                        let body = &data[from as usize..=to as usize];
                        let response = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{to}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            data.len(),
                            body.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                        if mode == Mode::DropOnce && from > 0 && dropped.fetch_add(1, Ordering::SeqCst) == 0 {
                            let _ = stream.write_all(&body[..body.len() / 2]);
                        } else {
                            let _ = stream.write_all(body);
                        }
                    }
                    _ => {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            data.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.write_all(&data);
                    }
                }
                open.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Fixture { addr, requests, peak }
}

fn read_head(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => buf.push(byte[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

/// `range: bytes=a-b` (or `a-`, open-ended to the end).
fn parse_range(head: &str) -> Option<(u64, u64)> {
    let rest = head.lines().find_map(|line| line.strip_prefix("range: bytes="))?;
    let (from, to) = rest.trim().split_once('-')?;
    Some((from.parse().ok()?, to.parse().unwrap_or(u64::MAX)))
}

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "makepad-asset-ai-segmented-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn test_data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 % 251) as u8).collect()
}

fn spec_for(data: &[u8]) -> FileSpec {
    FileSpec {
        role: None,
        repo: "test-org/test-repo".to_string(),
        path: "file.bin".to_string(),
        revision: None,
        cache_as: "llm/file.bin".to_string(),
        size: Some(data.len() as u64),
        sha256: Some(sha256_hex(data)),
        local: false,
        optional: false,
        converts_to: None,
        conversion: None,
    }
}

fn downloader_for(fixture: &Fixture, connections: usize) -> Downloader {
    let mut downloader = Downloader::new(&format!("http://{}", fixture.addr), None).unwrap();
    downloader.connections = connections;
    downloader.segment_bytes = SEGMENT;
    downloader
}

fn segments_path(dest: &Path) -> PathBuf {
    let mut os = part_path(dest).into_os_string();
    os.push(".segments");
    PathBuf::from(os)
}

#[test]
fn large_file_downloads_in_parallel_ranges() {
    let data = test_data(10_500);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("parallel");
    let spec = spec_for(&data);
    let mut reports = Vec::new();
    let dest = downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |p| reports.push((p.done, p.total)), &CancelToken::new())
        .unwrap();

    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(!part_path(&dest).exists());
    assert!(!segments_path(&dest).exists());
    let mut ranges = fixture.ranges();
    ranges.sort();
    // The probe, then each of the 11 segments once.
    assert_eq!(ranges[0], (0, 999));
    assert_eq!(ranges.len(), 12);
    assert!(ranges.contains(&(10_000, 10_499)));
    assert!(fixture.peak.load(Ordering::SeqCst) > 1, "ranges must overlap");
    // Progress never goes backwards and ends at the total.
    assert!(reports.windows(2).all(|w| w[0].0 <= w[1].0));
    assert_eq!(*reports.last().unwrap(), (10_500, Some(10_500)));
}

#[test]
fn server_without_ranges_gets_one_stream() {
    let data = test_data(5_000);
    let fixture = spawn_fixture(data.clone(), Mode::Ignore);
    let cache = test_dir("no-ranges");
    let spec = spec_for(&data);
    let dest = downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    // The probe's 200 is the download: one request, no state left.
    assert_eq!(fixture.requests.lock().unwrap().len(), 1);
    assert!(!segments_path(&dest).exists());
}

#[test]
fn resume_continues_each_segment_from_the_state_file() {
    let data = test_data(4_000);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("resume-state");
    let spec = spec_for(&data);
    let dest = spec.dest_path(&cache);
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    // A killed run: segment 0 done, segment 1 untouched, segment 2 half,
    // segment 3 done. Unwritten bytes are zeros in the preallocated file.
    let mut part = vec![0u8; data.len()];
    part[..1000].copy_from_slice(&data[..1000]);
    part[2000..2500].copy_from_slice(&data[2000..2500]);
    part[3000..].copy_from_slice(&data[3000..]);
    std::fs::write(part_path(&dest), &part).unwrap();
    let url = downloader_for(&fixture, 4).file_url(&spec);
    std::fs::write(
        segments_path(&dest),
        format!(
            r#"{{"version":1,"url":"{url}","size":4000,"sha256":"{}","segment_bytes":1000,"done":[1000,0,500,1000]}}"#,
            spec.sha256.as_ref().unwrap()
        ),
    )
    .unwrap();

    let mut first = None;
    let dest = downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |p| {
            first.get_or_insert(p.done);
        }, &CancelToken::new())
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    let mut ranges = fixture.ranges();
    ranges.sort();
    ranges.dedup();
    // Only the missing bytes: segment 1 whole, segment 2 from its middle.
    assert_eq!(ranges, vec![(1000, 1999), (2500, 2999)]);
    assert_eq!(first, Some(2500));
}

#[test]
fn single_stream_partial_is_a_done_prefix() {
    let data = test_data(4_000);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("legacy-prefix");
    let spec = spec_for(&data);
    let dest = spec.dest_path(&cache);
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(part_path(&dest), &data[..1500]).unwrap();
    let dest = downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    let ranges = fixture.ranges();
    assert!(ranges.iter().all(|&(from, _)| from >= 1500), "{ranges:?}");
}

#[test]
fn dropped_connection_is_retried() {
    let data = test_data(4_000);
    let fixture = spawn_fixture(data.clone(), Mode::DropOnce);
    let cache = test_dir("retry");
    let spec = spec_for(&data);
    let dest = downloader_for(&fixture, 2)
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    // The broken segment continued from where its bytes stopped.
    assert!(fixture.ranges().iter().any(|&(from, to)| from % SEGMENT == 500 && to % SEGMENT == 999));
}

#[test]
fn hash_mismatch_discards_the_segmented_partial() {
    let data = test_data(4_000);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("mismatch");
    let mut spec = spec_for(&data);
    spec.sha256 = Some("00".repeat(32));
    let error = downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap_err();
    match error {
        AssetAiError::Download(message) => assert!(message.contains("sha256 mismatch"), "{message}"),
        other => panic!("unexpected {other:?}"),
    }
    let dest = spec.dest_path(&cache);
    assert!(!dest.exists());
    assert!(!part_path(&dest).exists());
    assert!(!segments_path(&dest).exists());
}

#[test]
fn cancel_keeps_the_state_for_resume() {
    // 40 segments at 20 ms each over two connections: far longer than the
    // first progress poll.
    let data = test_data(40_000);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("cancel");
    let spec = spec_for(&data);
    let cancel = CancelToken::new();
    let downloader = downloader_for(&fixture, 2);
    let error = downloader
        .ensure_file(&spec, &cache, &mut |p| {
            if p.done >= 3_000 {
                cancel.cancel();
            }
        }, &cancel)
        .unwrap_err();
    assert_eq!(error, AssetAiError::Cancelled);
    let dest = spec.dest_path(&cache);
    assert_eq!(std::fs::metadata(part_path(&dest)).unwrap().len(), data.len() as u64);
    assert!(segments_path(&dest).exists());

    // The next run finishes without fetching the first segments again.
    let before = fixture.requests.lock().unwrap().len();
    let dest = downloader
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    let again = fixture.ranges()[before..].to_vec();
    assert!(again.iter().all(|&(from, _)| from >= 2_000), "{again:?}");
}

#[test]
fn small_or_unhashed_files_stay_on_one_stream() {
    let data = test_data(1_500);
    let fixture = spawn_fixture(data.clone(), Mode::Honor);
    let cache = test_dir("small");
    let spec = spec_for(&data);
    downloader_for(&fixture, 4)
        .ensure_file(&spec, &cache, &mut |_| {}, &CancelToken::new())
        .unwrap();
    // Under two segments: the plain request, no Range header.
    assert!(fixture.ranges().is_empty());
}
