//! Times `Downloader::ensure_file` on one pinned file:
//! `fetch_bench <url> <size> <sha256> <cache_dir> [seconds]`. With a time
//! limit it cancels after that many seconds (the `.part` stays for resume).

use makepad_ai_hub::backend::CancelToken;
use makepad_ai_hub::download::Downloader;
use makepad_ai_hub::registry::FileSpec;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (url, size, sha256, cache) = (&args[1], args[2].parse::<u64>().unwrap(), &args[3], &args[4]);
    let limit = args.get(5).map(|s| Duration::from_secs(s.parse().unwrap()));
    let spec = FileSpec {
        role: None,
        repo: String::new(),
        path: url.clone(),
        revision: None,
        cache_as: url.rsplit('/').next().unwrap().to_string(),
        size: Some(size),
        sha256: Some(sha256.clone()),
        local: false,
        optional: false,
        converts_to: None,
        conversion: None,
    };
    let downloader = Downloader::from_env().unwrap();
    let cancel = CancelToken::new();
    if let Some(limit) = limit {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(limit);
            cancel.cancel();
        });
    }
    let started = Instant::now();
    let mut first: Option<u64> = None;
    let mut last = Instant::now();
    let mut done = 0;
    let result = downloader.ensure_file(&spec, std::path::Path::new(cache), &mut |p| {
        let start = *first.get_or_insert(p.done);
        done = p.done;
        if last.elapsed() >= Duration::from_secs(5) {
            last = Instant::now();
            let secs = started.elapsed().as_secs_f64();
            println!("{:>6.0}s {:>6} MB of {} MB, {:.2} MB/s", secs, p.done >> 20, size >> 20, (p.done - start) as f64 / secs / 1e6);
        }
    }, &cancel);
    let secs = started.elapsed().as_secs_f64();
    let start = first.unwrap_or(0);
    println!("{:?} after {secs:.1}s: {} MB new, {:.2} MB/s", result.map(|_| "ok"), (done - start) >> 20, (done - start) as f64 / secs / 1e6);
}
