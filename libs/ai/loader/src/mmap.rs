//! Read-only memory-mapped file region for weight data.
//!
//! Weights served from a file-backed `PROT_READ` mapping stay CLEAN pages
//! (evictable, re-faultable from disk) instead of jetsam-dirty malloc
//! memory — the llama.cpp iOS pattern. This is the ONLY platform-gated
//! module: everything else treats the region as an opaque `&[u8]`. Unix
//! uses `mmap`, Windows `CreateFileMappingW`/`MapViewOfFile`; on any other
//! target `map_file` simply fails so callers fall back to the owned-arena
//! path.

use std::path::Path;

#[cfg(unix)]
mod imp {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    const O_RDONLY: c_int = 0;
    const SEEK_END: c_int = 2;
    const PROT_READ: c_int = 1;
    const MAP_SHARED: c_int = 1;
    // MADV_DONTNEED is 4 on Linux, Android and macOS.
    const MADV_DONTNEED: c_int = 4;
    /// macOS: issue an async advisory read with no copy to user space.
    #[cfg(target_os = "macos")]
    const F_RDADVISE: c_int = 44;

    /// macOS `struct radvisory` — the argument to `fcntl(F_RDADVISE)`.
    #[cfg(target_os = "macos")]
    #[repr(C)]
    struct Radvisory {
        ra_offset: i64,
        ra_count: c_int,
    }

    extern "C" {
        fn open(path: *const c_char, flags: c_int, ...) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn lseek(fd: c_int, offset: i64, whence: c_int) -> i64;
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn getpagesize() -> c_int;
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        fn mincore(addr: *const c_void, len: usize, vec: *mut c_char) -> c_int;
        #[cfg(target_os = "macos")]
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    }

    /// Read-only mapping of a whole file. The mapped length is the file
    /// size rounded UP to a page multiple (the tail stays within the
    /// file's zero-filled last partial page), so the base pointer and
    /// length both satisfy page granularity — required by the Metal
    /// no-copy buffer path.
    pub struct MappedRegion {
        ptr: *mut c_void,
        len: usize,
        /// Kept open for the lifetime of the mapping so `prefault` can ask
        /// the kernel to read ahead into the page cache. Costs one fd.
        fd: c_int,
    }

    // Safety: the mapping is PROT_READ and never written or remapped;
    // sharing immutable bytes across threads is safe.
    unsafe impl Send for MappedRegion {}
    unsafe impl Sync for MappedRegion {}

    impl MappedRegion {
        pub fn map_file(path: &Path) -> Result<Self, String> {
            let page = unsafe { getpagesize() };
            if page <= 0 {
                return Err("mmap: getpagesize failed".to_string());
            }
            let page = page as usize;

            let c_path = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| format!("mmap: path {:?} contains a NUL byte", path))?;
            let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
            if fd < 0 {
                return Err(format!("mmap: failed to open {:?}", path));
            }

            let file_size = unsafe { lseek(fd, 0, SEEK_END) };
            if file_size <= 0 {
                unsafe { close(fd) };
                return Err(format!("mmap: failed to size {:?}", path));
            }
            let Some(len) = usize::try_from(file_size)
                .ok()
                .and_then(|size| size.checked_next_multiple_of(page))
            else {
                unsafe { close(fd) };
                return Err(format!("mmap: file size {} overflows usize", file_size));
            };

            let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, fd, 0) };
            if ptr.is_null() || ptr as isize == -1 {
                unsafe { close(fd) };
                return Err(format!("mmap: mapping {:?} ({} bytes) failed", path, len));
            }
            // The mapping itself no longer needs the fd, but `prefault`
            // does, so it is held until Drop instead of closed here.
            Ok(Self { ptr, len, fd })
        }

        /// Pages of this mapping currently resident in the unified page
        /// cache, as `(resident, total)`. Straight from `mincore(2)` — the
        /// only honest way to tell a warm mapping from a cold one, since a
        /// cold mapping is indistinguishable from a warm one until it is
        /// touched.
        pub fn residency(&self) -> (usize, usize) {
            let page = unsafe { getpagesize() }.max(1) as usize;
            let pages = self.len / page;
            let mut vec = vec![0i8; pages];
            let rc =
                unsafe { mincore(self.ptr, self.len, vec.as_mut_ptr() as *mut c_char) };
            if rc != 0 {
                return (0, pages);
            }
            (vec.iter().filter(|b| **b & 1 != 0).count(), pages)
        }

        /// Force the whole mapping resident before anything reads it.
        ///
        /// Demand-paging a cold mapping is the slowest way to get weights
        /// off an Apple SSD: every 16 KB page is a synchronous fault, which
        /// measures ~0.8 GB/s against ~6.3 GB/s for the same file read
        /// sequentially. `F_RDADVISE` fixes that — it queues an async
        /// advisory read that populates the page cache with no copy to user
        /// space — so this walks the file hinting a window ahead of the
        /// cursor it is touching, and the faults land on pages that are
        /// already there. Measured 6.2 GB/s, i.e. device speed, while
        /// leaving a fully populated file-backed mapping (which is what the
        /// Metal no-copy buffer needs).
        ///
        /// Cheap to call on an already-warm mapping (~26 GB/s of soft
        /// faults), and a no-op beyond that.
        #[cfg(target_os = "macos")]
        pub fn prefault(&self) {
            const WINDOW: usize = 16 << 20;
            const AHEAD: usize = 4 * WINDOW;
            let page = unsafe { getpagesize() }.max(1) as usize;
            let mut hinted = 0usize;
            let mut touched = 0usize;
            while touched < self.len {
                // Keep the device busy ahead of where we are reading.
                while hinted < self.len && hinted < touched + AHEAD {
                    let count = WINDOW.min(self.len - hinted);
                    let ra = Radvisory {
                        ra_offset: hinted as i64,
                        ra_count: count as c_int,
                    };
                    unsafe { fcntl(self.fd, F_RDADVISE, &ra as *const Radvisory) };
                    hinted += count;
                }
                let end = (touched + WINDOW).min(self.len);
                while touched < end {
                    // Safety: `touched` stays inside the live mapping.
                    unsafe {
                        std::ptr::read_volatile((self.ptr as *const u8).add(touched));
                    }
                    touched += page;
                }
            }
        }

        /// Non-macOS unix: no `F_RDADVISE`, so just fault the pages in.
        #[cfg(not(target_os = "macos"))]
        pub fn prefault(&self) {
            let page = unsafe { getpagesize() }.max(1) as usize;
            let mut off = 0usize;
            while off < self.len {
                // Safety: `off` stays inside the live mapping.
                unsafe {
                    std::ptr::read_volatile((self.ptr as *const u8).add(off));
                }
                off += page;
            }
        }

        pub fn len(&self) -> usize {
            self.len
        }

        /// Drops this mapping's pages in `[offset, offset + len)` (whole
        /// pages inside it) from the process: a backend that copied the
        /// bytes to device memory no longer pays for them twice. A later
        /// read faults them back in from the file.
        pub fn release(&self, offset: usize, len: usize) {
            let page = unsafe { getpagesize() }.max(1) as usize;
            let start = offset.next_multiple_of(page);
            let end = (offset + len).min(self.len) / page * page;
            if end > start {
                // Safety: the range lies inside the live mapping.
                unsafe { madvise((self.ptr as *mut u8).add(start) as *mut c_void, end - start, MADV_DONTNEED) };
            }
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn as_slice(&self) -> &[u8] {
            // Safety: ptr/len describe a live PROT_READ mapping owned by self.
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for MappedRegion {
        fn drop(&mut self) {
            unsafe {
                munmap(self.ptr, self.len);
                close(self.fd);
            }
        }
    }

    impl std::fmt::Debug for MappedRegion {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MappedRegion")
                .field("ptr", &self.ptr)
                .field("len", &self.len)
                .finish()
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    type Handle = *mut c_void;

    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 0x1;
    const FILE_SHARE_DELETE: u32 = 0x4;
    const OPEN_EXISTING: u32 = 3;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;
    const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;

    #[repr(C)]
    struct SystemInfo {
        processor_architecture: u16,
        reserved: u16,
        page_size: u32,
        minimum_application_address: *mut c_void,
        maximum_application_address: *mut c_void,
        active_processor_mask: usize,
        number_of_processors: u32,
        processor_type: u32,
        allocation_granularity: u32,
        processor_level: u16,
        processor_revision: u16,
    }

    #[repr(C)]
    struct MemoryRangeEntry {
        virtual_address: *mut c_void,
        number_of_bytes: usize,
    }

    #[repr(C)]
    struct WorkingSetExInformation {
        virtual_address: *mut c_void,
        /// Bit 0 is `Valid`: the page is in this process's working set.
        virtual_attributes: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn GetFileSizeEx(file: Handle, size: *mut i64) -> i32;
        fn CreateFileMappingW(
            file: Handle,
            security: *mut c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
        fn GetLastError() -> u32;
        fn GetSystemInfo(info: *mut SystemInfo);
        fn GetCurrentProcess() -> Handle;
        fn VirtualUnlock(address: *mut c_void, size: usize) -> i32;
        fn PrefetchVirtualMemory(
            process: Handle,
            entries: usize,
            ranges: *const MemoryRangeEntry,
            flags: u32,
        ) -> i32;
        fn K32QueryWorkingSetEx(process: Handle, info: *mut c_void, size: u32) -> i32;
    }

    fn page_size() -> usize {
        let mut info = std::mem::MaybeUninit::<SystemInfo>::zeroed();
        // Safety: GetSystemInfo fills the struct and cannot fail.
        let info = unsafe {
            GetSystemInfo(info.as_mut_ptr());
            info.assume_init()
        };
        (info.page_size as usize).max(1)
    }

    /// Read-only view of a whole file. As on unix, the length is the file
    /// size rounded UP to a page multiple (the tail of the last page reads
    /// as zeros), so the base pointer and length both satisfy page
    /// granularity.
    ///
    /// The file and mapping handles are closed once the view exists: the
    /// view keeps the section alive. While it is mapped the file cannot be
    /// overwritten or truncated (Windows refuses writes to a mapped
    /// section), but it is opened with `FILE_SHARE_DELETE`, so it can be
    /// renamed or marked for deletion.
    pub struct MappedRegion {
        ptr: *mut c_void,
        len: usize,
    }

    // Safety: the view is FILE_MAP_READ and never written or remapped;
    // sharing immutable bytes across threads is safe.
    unsafe impl Send for MappedRegion {}
    unsafe impl Sync for MappedRegion {}

    impl MappedRegion {
        pub fn map_file(path: &Path) -> Result<Self, String> {
            let page = page_size();
            let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            let file = unsafe {
                CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_DELETE,
                    std::ptr::null_mut(),
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL,
                    std::ptr::null_mut(),
                )
            };
            if file == INVALID_HANDLE_VALUE {
                return Err(format!("mmap: failed to open {:?} (error {})", path, unsafe {
                    GetLastError()
                }));
            }

            let mut file_size = 0i64;
            if unsafe { GetFileSizeEx(file, &mut file_size) } == 0 || file_size <= 0 {
                unsafe { CloseHandle(file) };
                return Err(format!("mmap: failed to size {:?}", path));
            }
            let Some(len) = usize::try_from(file_size)
                .ok()
                .and_then(|size| size.checked_next_multiple_of(page))
            else {
                unsafe { CloseHandle(file) };
                return Err(format!("mmap: file size {} overflows usize", file_size));
            };

            // Size 0 maps the whole file; a read-only section cannot be
            // larger than the file anyway.
            let mapping = unsafe {
                CreateFileMappingW(
                    file,
                    std::ptr::null_mut(),
                    PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                )
            };
            unsafe { CloseHandle(file) };
            if mapping.is_null() {
                return Err(format!("mmap: CreateFileMappingW {:?} failed (error {})", path, unsafe {
                    GetLastError()
                }));
            }
            let ptr = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
            let error = unsafe { GetLastError() };
            unsafe { CloseHandle(mapping) };
            if ptr.is_null() {
                return Err(format!(
                    "mmap: mapping {:?} ({} bytes) failed (error {})",
                    path, len, error
                ));
            }
            Ok(Self { ptr, len })
        }

        /// Pages of this view currently in the process's working set, as
        /// `(resident, total)`, from `QueryWorkingSetEx`. Unlike unix
        /// `mincore` this does not count pages that are in the system file
        /// cache but not yet faulted into this process.
        pub fn residency(&self) -> (usize, usize) {
            let page = page_size();
            let pages = self.len / page;
            let mut info: Vec<WorkingSetExInformation> = (0..pages)
                .map(|index| WorkingSetExInformation {
                    virtual_address: unsafe { (self.ptr as *mut u8).add(index * page) }
                        as *mut c_void,
                    virtual_attributes: 0,
                })
                .collect();
            let Ok(bytes) =
                u32::try_from(info.len() * std::mem::size_of::<WorkingSetExInformation>())
            else {
                return (0, pages);
            };
            let ok = unsafe {
                K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr() as *mut c_void, bytes)
            };
            if ok == 0 {
                return (0, pages);
            }
            (info.iter().filter(|entry| entry.virtual_attributes & 1 != 0).count(), pages)
        }

        /// Force the whole view resident before anything reads it.
        /// `PrefetchVirtualMemory` queues large reads of the file into the
        /// page cache so the touches that follow are soft faults rather
        /// than one synchronous disk read per page.
        pub fn prefault(&self) {
            let range = MemoryRangeEntry {
                virtual_address: self.ptr,
                number_of_bytes: self.len,
            };
            unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) };
            let page = page_size();
            let mut off = 0usize;
            while off < self.len {
                // Safety: `off` stays inside the live view.
                unsafe {
                    std::ptr::read_volatile((self.ptr as *const u8).add(off));
                }
                off += page;
            }
        }

        pub fn len(&self) -> usize {
            self.len
        }

        /// Drops this view's pages in `[offset, offset + len)` (whole pages
        /// inside it) from the process's working set: a backend that copied
        /// the bytes to device memory no longer pays for them twice. A later
        /// read faults them back in from the file. `VirtualUnlock` on pages
        /// that were never locked does exactly this (and reports
        /// ERROR_NOT_LOCKED, which is expected).
        pub fn release(&self, offset: usize, len: usize) {
            let page = page_size();
            let start = offset.next_multiple_of(page);
            let end = (offset + len).min(self.len) / page * page;
            if end > start {
                // Safety: the range lies inside the live view.
                unsafe { VirtualUnlock((self.ptr as *mut u8).add(start) as *mut c_void, end - start) };
            }
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn as_slice(&self) -> &[u8] {
            // Safety: ptr/len describe a live read-only view owned by self.
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for MappedRegion {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(self.ptr);
            }
        }
    }

    impl std::fmt::Debug for MappedRegion {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MappedRegion")
                .field("ptr", &self.ptr)
                .field("len", &self.len)
                .finish()
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::path::Path;

    /// Stub: mapping is unavailable, so `map_file` always fails and
    /// callers take the owned-arena path.
    #[derive(Debug)]
    pub struct MappedRegion {
        _private: (),
    }

    impl MappedRegion {
        pub fn map_file(path: &Path) -> Result<Self, String> {
            Err(format!(
                "mmap is unavailable on this platform (cannot map {:?})",
                path
            ))
        }

        pub fn len(&self) -> usize {
            0
        }

        pub fn release(&self, _offset: usize, _len: usize) {}

        pub fn is_empty(&self) -> bool {
            true
        }

        pub fn as_slice(&self) -> &[u8] {
            &[]
        }

        pub fn residency(&self) -> (usize, usize) {
            (0, 0)
        }

        pub fn prefault(&self) {}
    }
}

pub use imp::MappedRegion;

impl MappedRegion {
    /// Convenience wrapper taking anything path-like.
    pub fn map(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::map_file(path.as_ref())
    }
}

#[cfg(all(test, any(unix, windows)))]
mod tests {
    use super::MappedRegion;

    #[test]
    fn a_mapped_file_reads_back_its_bytes_and_survives_a_release() {
        let path = std::env::temp_dir().join(format!("makepad-mmap-test-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i.wrapping_mul(7) >> 3) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        {
            let region = MappedRegion::map(&path).unwrap();
            // Rounded up to whole pages, the tail reading as zeros.
            assert!(region.len() >= bytes.len() && region.len() % 4096 == 0, "{}", region.len());
            assert_eq!(&region.as_slice()[..bytes.len()], &bytes[..]);
            assert!(region.as_slice()[bytes.len()..].iter().all(|b| *b == 0));

            region.prefault();
            let (resident, pages) = region.residency();
            assert!(pages > 0 && resident > 0, "{resident}/{pages}");

            // Released pages fault back in from the file, unchanged.
            region.release(0, region.len());
            assert_eq!(&region.as_slice()[..bytes.len()], &bytes[..]);
        }
        // Unmapped: the file can go.
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_empty_or_missing_file_does_not_map() {
        let path = std::env::temp_dir().join(format!("makepad-mmap-empty-{}.bin", std::process::id()));
        std::fs::write(&path, b"").unwrap();
        assert!(MappedRegion::map(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(MappedRegion::map(&path).is_err());
    }
}
