//! Link our pdfium build into whoever depends on this crate.
//!
//! This lives here, and not in paddock-runner's build.rs or in
//! `.cargo/config.toml`, because this is the crate that declares the FFI. A
//! crate that says `extern "C" { fn FPDF_... }` should also say where those
//! symbols come from; splitting the two is what produced a pile of unresolved
//! `FPDF_*` at the end of a long link when the declarations moved here and the
//! link directive did not.
//!
//! It could not live here before. `pdfium-render` declared crate-types `["lib",
//! "staticlib", "cdylib"]`, so cargo also built a `pdfium_render.dll` whose
//! link line no consumer's build script could reach - which is why the path
//! travelled as the `PDFIUM_STATIC_LIB_PATH` env var that crate's own build
//! script reads. This crate is a plain `lib`, so that whole workaround is gone
//! and the linking is ordinary again.

fn main() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/<crate> is two levels below the repo root");

    // Windows and Linux only, deliberately. An unknown
    // target should say so here rather than fail as a pile of unresolved
    // FPDF_* symbols at the end of a long link.
    // Linux splits by CPU: an aarch64 box (DGX Spark's Grace, Jetson) links
    // the arm64 library, cross-built by the same recipe with target_cpu set.
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let (dir, file) = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => ("win-x64", "pdfium.lib"),
        Ok("linux") if arch == "aarch64" => ("linux-arm64", "libpdfium.a"),
        Ok("linux") => ("linux-x64", "libpdfium.a"),
        Ok(other) => {
            panic!("paddock does not target {other}: pdfium is built for windows and linux only")
        }
        Err(e) => panic!("CARGO_CFG_TARGET_OS unreadable: {e}"),
    };

    let lib_dir = repo.join("packs").join("pdfium").join(dir);
    let lib = lib_dir.join(file);
    if !lib.is_file() {
        // Loud and actionable: a missing prebuilt is a two-command fix, and a
        // raw linker error would send the reader looking for a bug that is not
        // there.
        let build = if dir == "win-x64" {
            "packs/pdfium/build/build-windows.ps1"
        } else {
            "packs/pdfium/build/build-linux.sh"
        };
        let fetch = if dir == "win-x64" {
            "powershell -File packs/pdfium/fetch.ps1"
        } else {
            "bash packs/pdfium/fetch.sh"
        };
        panic!(
            "pdfium is not staged at {}\n\
             \n\
             download ours:           {fetch}   (reads packs/pdfium/prebuilt.json, \
             checks the sha256)\n\
             or build it:             {build}   (needs depot_tools, ~15 min)\n\
             \n\
             It is linked INTO this binary, so there is nothing to ship \
             separately - see packs/pdfium/build/ for how it is made.",
            lib.display(),
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=pdfium");

    // What pdfium itself calls and Rust does not bring on its own.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => {
            // GDI32 is the real one: pdfium's Windows font and device path
            // (CreateFontIndirect, GetOutlineTextMetrics, CreateCompatibleDC,
            // SelectObject) is GDI. Same set the prebuilt pdfium.dll imported,
            // minus what Rust already links.
            println!("cargo:rustc-link-lib=dylib=gdi32");
            println!("cargo:rustc-link-lib=dylib=user32");
            // winmm for timeGetTime, which partition_alloc's Windows clock uses.
            println!("cargo:rustc-link-lib=dylib=winmm");
            // The MSVC C++ runtime, /MT flavour to match pdfium's own build and
            // our +crt-static. Rust links the C runtime and stops there, so
            // pdfium's C++ objects arrive wanting the STL's out-of-line helpers
            // - `__std_rotate`, `__std_min_element_f_` and friends, which are
            // real functions in libcpmt.lib rather than header templates.
            println!("cargo:rustc-link-lib=dylib=libcpmt");
        }
        Ok("linux") => {
            // args-linux.gn builds pdfium with use_custom_libcxx = false, so its
            // objects reference the SYSTEM libstdc++ (GCC-ABI std::_Rb_tree_*,
            // iostreams, __cxxabiv1 vtables), which Rust's link driver does not
            // pull in by itself.
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        _ => {}
    }

    println!("cargo:rerun-if-changed={}", lib.display());
}
