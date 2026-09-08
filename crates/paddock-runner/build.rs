// Windows VERSIONINFO + the taskbar icon.
//
// Without this the runner is a nameless exe: blank ProductName, blank
// FileVersion, blank publisher, and the generic Windows icon in Task Manager,
// the Details tab and every "unknown program" prompt. That is not cosmetic on
// a program users are asked to trust with a firewall exception - a signed
// binary that will not say who it is reads exactly like something that would
// rather not.
//
// Kept identical to the manager's build.rs deliberately (the two are one
// product); the only differences are the per-binary description and filename.

fn main() {
    pdfium_static();
    #[cfg(windows)]
    windows_resource();
}

/// Check that our own pdfium build is staged, and say what to do if it is not.
///
/// The LINKING is not done here - `.cargo/config.toml` points pdfium-render's
/// own build script at the library, because that script's directives also cover
/// the extra crate-types Cargo builds for it (see the config for the measured
/// reason). What is left for us is the part that script does badly: a missing
/// file becomes a page of unresolved `FPDF_*` symbols, which reads like a bug
/// in the binding rather than "you have not built pdfium yet".
///
/// The libraries come from `packs/pdfium/build/build-{windows.ps1,linux.sh}`
/// and are not in git (~25 MB build artifacts).
fn pdfium_static() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/<crate> is two levels below the repo root");

    // Windows and Linux only, deliberately. An unknown
    // target should say so here rather than fail as a pile of unresolved
    // FPDF_* symbols at the end of a long link.
    // Same platform split as crates/paddock-pdfium/build.rs: Linux picks the
    // library by CPU, so an aarch64 box (DGX Spark, Jetson) wants linux-arm64.
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
        // Loud and actionable, in the spirit of the rc.exe panic below: a
        // missing prebuilt is a two-command fix, and a raw linker error would
        // send the reader looking for a bug that is not there.
        let build = if dir == "win-x64" {
            "packs/pdfium/build/build-windows.ps1"
        } else {
            "packs/pdfium/build/build-linux.sh"
        };
        panic!(
            "pdfium is not staged at {}\n\
             \n\
             build it:                {build}   (needs depot_tools, ~15 min)\n\
             or download ours:        the URLs and sha256s are in \
             packs/pdfium/prebuilt.json\n\
             \n\
             It is linked INTO this binary, so there is nothing to ship \
             separately - see packs/pdfium/build/ for how it is made.",
            lib.display(),
        );
    }

    println!("cargo:rerun-if-changed={}", lib.display());
}

#[cfg(windows)]
fn windows_resource() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/<crate> is two levels below the repo root");
    let icon = repo.join("assets").join("paddock.ico");
    println!("cargo:rerun-if-changed={}", icon.display());

    let mut res = winresource::WindowsResource::new();
    res.set_icon(icon.to_str().expect("icon path is UTF-8"));
    // FileVersion/ProductVersion come from CARGO_PKG_VERSION automatically, in
    // the numeric a.b.c.0 form Windows requires - the commit stamp is not
    // representable there (four integers, nothing else) and lives in
    // `--version` instead.
    res.set("ProductName", "Paddock");
    res.set("FileDescription", "Paddock runner - model serving");
    res.set("CompanyName", "Truespar");
    res.set(
        "LegalCopyright",
        "Copyright (c) 2026 Truespar. MIT OR Apache-2.0.",
    );
    res.set("OriginalFilename", "paddock-runner.exe");
    res.set("InternalName", "paddock-runner");

    // Loud rather than lenient. winresource's own default on a missing rc.exe
    // is to warn and carry on, which would ship an unbranded binary that looks
    // fine everywhere except the one place anyone checks.
    if let Err(e) = res.compile() {
        panic!(
            "could not compile the Windows resource ({e}).\n\
             rc.exe comes with the Windows SDK - build from a VS2022 vcvars64 \
             environment."
        );
    }
}
