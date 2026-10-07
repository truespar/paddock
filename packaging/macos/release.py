#!/usr/bin/env python3
"""Public-source Apple Silicon release builder. Python 3.9+, no Python packages.

Build never installs, publishes, reads signing secrets, or changes the user's app.
Finalize requires explicit Developer ID identities and a Keychain notary profile.
See README.md for the credential boundary and the remaining manual release gates.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as XML

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ORIGINS = {
    "git@github.com:truespar/paddock.git",
    "https://github.com/truespar/paddock.git",
    "https://github.com/truespar/paddock",
    "ssh://git@github.com/truespar/paddock.git",
}
MIN_OS = "26.0"  # The runner compiles Metal Shading Language 4.0 at runtime.
UPDATE_ACCOUNT = "com.truespar.paddock.sparkle"
UPDATE_FEED = "https://github.com/truespar/paddock/releases/latest/download/appcast.xml"
TARGET = "aarch64-apple-darwin"
SWIFT_TARGET = "arm64-apple-macosx" + MIN_OS
LICENSES = ("LICENSE", "LICENSE-MIT", "LICENSE-APACHE", "THIRD-PARTY-NOTICES")
FEATURES = ",".join(("paddock-manager/hardened", "paddock-desktop/hardened",
                     "paddock-runner/hardened", "paddock-runner/metal"))


class ReleaseError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def run(*args, cwd=None, env=None, capture=False, timeout=None):
    args = [str(arg) for arg in args]
    print("+ " + " ".join(args), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, check=True,
                            stdout=subprocess.PIPE if capture else None,
                            text=True, timeout=timeout)
    return result.stdout.strip() if capture else None


def git(*args, root=ROOT):
    return run("git", "-C", root, *args, capture=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def public_source(root):
    origin = git("remote", "get-url", "origin", root=root)
    require(origin in PUBLIC_ORIGINS, "Release source must be truespar/paddock, not an internal checkout.")
    private = root / "crates/paddock-manager/models.private.toml"
    require(not private.exists() and not private.is_symlink(),
            "Private catalog present: refusing to build a public release.")
    return origin


def relative_path(name):
    path = Path(name)
    require(name and not path.is_absolute() and ".." not in path.parts and path != Path("."),
            "Unsafe relative path: " + name)
    return path


def inventory(root):
    require(root.is_dir() and not root.is_symlink(), "Missing or symlinked package root: " + str(root))
    result = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            target = os.readlink(path)
            require(not Path(target).is_absolute() and path.resolve().is_relative_to(root.resolve()),
                    "External package symlink: " + name)
            require(path.exists(), "Broken package symlink: " + name)
            result[name] = {"symlink": target, "mode": mode}
        elif path.is_file():
            result[name] = {"sha256": digest(path), "size": path.stat().st_size, "mode": mode}
        elif path.is_dir():
            result[name] = {"directory": True, "mode": mode}
        else:
            raise ReleaseError("Unsupported package entry: " + name)
    return result


def version(root):
    # Only the workspace version; no third-party TOML parser required on macOS.
    cargo = (root / "Cargo.toml").read_text()
    section = cargo.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$', section, re.M)
    require(match is not None, "A numeric x.y.z release version is required.")
    return match.group(1)


def build_environment(work):
    # No inherited PADDOCK/VITE overrides, loader injection, compiler flags or
    # cloud credentials. HOME keeps standard tool caches; it is never rewritten.
    allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "DEVELOPER_DIR", "SDKROOT")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update({"MACOSX_DEPLOYMENT_TARGET": MIN_OS, "CARGO_TARGET_DIR": str(work / "cargo"),
                "CI": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "npm_config_userconfig": os.devnull})
    mappings = ["--remap-path-prefix=" + str(work) + "=/build"]
    if "HOME" in env:
        mappings.append("--remap-path-prefix=" + env["HOME"] + "=/build-user")
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(mappings)
    return env


def doctor():
    require(sys.platform == "darwin" and os.uname().machine == "arm64",
            "This release target requires an Apple Silicon Mac.")
    public_source(ROOT)
    commands = {"macOS": ("sw_vers",), "Xcode": ("xcodebuild", "-version"),
                "Swift": ("xcrun", "swift", "--version"), "Rust": ("rustc", "-Vv"),
                "Cargo": ("cargo", "--version"), "Node": ("node", "--version"),
                "npm": ("npm", "--version"), "SDK": ("xcrun", "--show-sdk-version"),
                "notarytool": ("xcrun", "notarytool", "--version"), "CMake": ("cmake", "--version")}
    versions = {name: run(*args, capture=True) for name, args in commands.items()}
    for program in ("codesign", "security", "ditto", "hdiutil", "pkgbuild", "productbuild",
                    "pkgutil", "spctl", "lipo", "otool", "install_name_tool"):
        require(shutil.which(program), "Missing tool: " + program)
    run("bash", ROOT / "apps/macos/scripts/check-toolchain.sh")
    print(json.dumps(versions, indent=2))
    print(run("security", "find-identity", "-v", "-p", "codesigning", capture=True))
    print("Free disk: %.1f GiB" % (shutil.disk_usage(ROOT).free / 2**30))
    return versions


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def snapshot(root, source, allow_dirty):
    origin = public_source(root)
    dirty = bool(git("status", "--porcelain", "--untracked-files=all", root=root))
    require(not dirty or allow_dirty, "Working tree is dirty. --allow-dirty is for unsigned candidates only.")
    commit = git("rev-parse", "HEAD", root=root)
    public_commit = bool(git("for-each-ref", "--format=%(refname)", "--contains", commit,
                             "refs/remotes/origin/", root=root))
    # Copy Git objects, not references into another worktree. Preserve the real
    # commit stamp without importing existing target/, static/, models or caches.
    run("git", "clone", "--no-local", "--no-checkout", root, source)
    run("git", "-C", source, "checkout", "--detach", commit)
    run("git", "-C", source, "remote", "set-url", "origin", origin)
    if dirty:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z",
                                       "--cached", "--others", "--exclude-standard"])
        for name in sorted(set(raw.decode().rstrip("\0").split("\0"))):
            relative = relative_path(name)
            require(relative.parts[0] != ".git", "Cannot overlay Git metadata.")
            require(relative.suffix.lower() not in (".p12", ".p8", ".key") and
                    relative.name not in (".env", "models.private.toml"),
                    "Potential private input in source snapshot: " + name)
            original, copied = root / relative, source / relative
            require(not original.is_symlink(), "Source symlinks need a release review: " + name)
            require(original.resolve().is_relative_to(root.resolve()), "Source path escaped the checkout: " + name)
            if original.is_file():
                copy_file(original, copied)
            elif copied.is_file():
                copied.unlink()  # A tracked deletion, only inside our new snapshot.
    public_source(source)
    # Include untracked candidate tooling in provenance as well as Git's diff.
    # No generated build outputs or Git objects enter this source inventory.
    inputs = {}
    for path in sorted(source.rglob("*")):
        if ".git" in path.relative_to(source).parts:
            continue
        require(not path.is_symlink(), "Source symlinks require a release review.")
        if path.is_file():
            inputs[path.relative_to(source).as_posix()] = digest(path)
    return {"origin": origin, "commit": commit, "dirty": dirty,
            "public_commit": public_commit, "files": inputs,
            "diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(source), "diff", "--binary", "HEAD"])).hexdigest()}


def swift_args(source, sdk_version):
    require(bool(re.fullmatch(r"\d+\.\d+(?:\.\d+)?", sdk_version))
            and int(sdk_version.split(".")[0]) >= 26, "A macOS 26+ SDK is required.")
    return ["xcrun", "swift", "build", "--package-path", str(source / "apps/macos"),
            "--configuration", "release", "--product", "PaddockMac", "--triple", SWIFT_TARGET,
            "--force-resolved-versions", "-Xswiftc", "-warnings-as-errors",
            # Swift Build must not substitute the deployment floor for the SDK;
            # AppKit uses the latter to enable native scroll-edge behavior.
            "-Xlinker", "-platform_version", "-Xlinker", "macos", "-Xlinker", MIN_OS,
            "-Xlinker", sdk_version,
            "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks",
            # Map source inputs, not .build/out's SDK module cache: dsymutil
            # must still resolve those binary modules while linking symbols.
            "-Xswiftc", "-file-prefix-map", "-Xswiftc",
            str(source / "apps/macos/Sources") + "=/src/paddock/apps/macos/Sources",
            "-Xswiftc", "-file-prefix-map", "-Xswiftc",
            str(source / "apps/macos/.build/checkouts") + "=/src/dependencies"]


def assemble(source, work, stage, release_version, build_number, env, sdk_version):
    app = stage / "Paddock.app"
    contents = app / "Contents"
    resources = contents / "Resources"
    rust = work / "cargo" / TARGET / "release"
    swift = Path(run(*swift_args(source, sdk_version), "--show-bin-path", env=env, capture=True))
    run("bash", source / "apps/macos/scripts/check-linked-sdk.sh", swift / "PaddockMac", sdk_version)
    copy_file(swift / "PaddockMac", contents / "MacOS/Paddock")
    copy_file(rust / "paddock-runner", contents / "Helpers/paddock-runner")
    library = contents / "Frameworks/libpaddock_desktop.dylib"
    copy_file(rust / library.name, library)
    sparkle = source / "apps/macos/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
    require(sparkle.is_dir(), "Pinned Sparkle framework is missing.")
    shutil.copytree(sparkle, contents / "Frameworks/Sparkle.framework", symlinks=True)
    run("install_name_tool", "-id", "@rpath/" + library.name, library)
    resources.mkdir(parents=True)
    bundles = sorted(swift.glob("*.bundle"))
    require(bool(bundles), "SwiftPM resource bundles are missing.")
    for bundle in bundles:
        shutil.copytree(bundle, resources / bundle.name, symlinks=True)
    viewers = source / "apps/macos/.build/studio-workspace"
    audit = json.loads((viewers / "manifest.json").read_text())
    require(audit["surface"] == "native-embedded-viewers" and audit["renderer"] == "native"
            and audit["webUI"] == ["Lector", "Scriptor", "Traverse"]
            and not audit["forbiddenWebUI"] and audit["auditedModules"] > 0,
            "The native viewer boundary audit failed.")
    shutil.copytree(viewers, resources / "StudioWorkspace")
    copy_file(source / "packaging/macos/Paddock.icns", resources / "Paddock.icns")
    checkouts = source / "apps/macos/.build/checkouts"
    notices = resources / "NativeMarkdownNotices"
    deps = {name: name + "/LICENSE" for name in
            ("MarkdownView", "beautiful-mermaid-swift", "elk-swift", "RichText", "Highlightr", "SwiftMath")}
    deps.update({"swift-markdown": "swift-markdown/LICENSE.txt", "swift-cmark": "swift-cmark/COPYING",
                 "Sparkle": "Sparkle/LICENSE",
                 "highlight-js": "Highlightr/src/assets/highlighter/LICENSE",
                 "math-fonts": "SwiftMath/Sources/SwiftMath/mathFonts.bundle/LICENSE"})
    for name, path in deps.items():
        copy_file(checkouts / path, notices / (name + ".txt"))
    for name in LICENSES:
        copy_file(source / name, resources / "Licenses" / name)
        copy_file(source / name, stage / "cli/share/paddock" / name)
    for name in ("paddock", "paddock-runner"):
        copy_file(rust / name, stage / "cli/bin" / name)
    info = plistlib.loads((source / "packaging/macos/Info.plist").read_bytes())
    info.update(CFBundleShortVersionString=release_version, CFBundleVersion=build_number,
                LSMinimumSystemVersion=MIN_OS)
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    copy_file(source / "packaging/macos/App.entitlements", stage / "App.entitlements")
    return app


def binaries(stage):
    return [stage / name for name in (
        "Paddock.app/Contents/Frameworks/libpaddock_desktop.dylib",
        "Paddock.app/Contents/Helpers/paddock-runner",
        "Paddock.app/Contents/MacOS/Paddock", "cli/bin/paddock", "cli/bin/paddock-runner")]


def sign(stage, identity):
    # Nested code first. Never --deep for signing, no JIT or library-validation
    # exceptions. Ad-hoc is allowed ONLY for the explicitly unsigned candidate.
    options = ["--force", "--sign", identity, "--options", "runtime"]
    options += ["--timestamp"] if identity != "-" else ["--timestamp=none"]
    sparkle = stage / "Paddock.app/Contents/Frameworks/Sparkle.framework"
    for nested in ("Autoupdate", "XPCServices/Downloader.xpc", "XPCServices/Installer.xpc", "Updater.app"):
        run("codesign", *options, "--preserve-metadata=entitlements", sparkle / "Versions/B" / nested)
    run("codesign", *options, sparkle)
    for binary in binaries(stage):
        if binary.name != "Paddock":
            run("codesign", *options, binary)
    run("codesign", *options, "--entitlements", stage / "App.entitlements", stage / "Paddock.app")
    run("codesign", "--verify", "--deep", "--strict", stage / "Paddock.app")
    for binary in binaries(stage):
        run("codesign", "--verify", "--strict", binary)


def smoke(stage, env):
    for binary in binaries(stage):
        require(run("lipo", "-archs", binary, capture=True) == "arm64", "Unexpected architecture.")
        linked = run("otool", "-L", binary, capture=True).splitlines()[1:]
        for line in linked:
            dependency = line.strip().split(" (", 1)[0]
            require(dependency.startswith(("/System/Library/", "/usr/lib/", "@rpath/",
                                           "@loader_path/", "@executable_path/")),
                    "Nonportable Mach-O dependency: " + dependency)
        verify_load_commands(run("otool", "-l", binary, capture=True))
    with tempfile.TemporaryDirectory(prefix="paddock-release-smoke-") as temp:
        isolated = dict(env, PADDOCK_DATA=temp + "/data", XDG_RUNTIME_DIR=temp + "/runtime")
        for name in ("paddock", "paddock-runner"):
            run(stage / "cli/bin" / name, "--version", env=isolated, timeout=30)
        capabilities = json.loads(run(stage / "cli/bin/paddock-runner", "--capabilities",
                                      env=isolated, capture=True, timeout=30))
        require("version" in capabilities, "Invalid runner capability response.")
        run(sys.executable, ROOT / "packaging/macos/smoke-desktop.py",
            stage / "Paddock.app/Contents/Frameworks/libpaddock_desktop.dylib",
            env=isolated, timeout=90)


def verify_load_commands(commands):
    minimums = re.findall(r"^\s*minos\s+(\d+(?:\.\d+)*)\s*$", commands, re.M)
    require(len(minimums) == 1, "Expected one arm64 Mach-O deployment target.")

    def components(value):
        parts = tuple(int(part) for part in value.split("."))
        return parts + (0,) * (3 - len(parts))

    require(components(minimums[0]) <= components(MIN_OS),
            "Binary requires a newer macOS than the package declares: " + minimums[0])
    for block in commands.split("Load command"):
        if "cmd LC_RPATH\n" not in block:
            continue
        match = re.search(r"^\s*path (.+) \(offset \d+\)\s*$", block, re.M)
        require(match is not None, "Cannot parse runtime search path.")
        path = match.group(1)
        require(path.startswith(("@executable_path", "@loader_path", "/usr/lib/", "/System/Library/")),
                "Build-machine runtime search path in package: " + path)


def packages(stage, destination, name, release_version, installer_identity=None):
    destination.mkdir(parents=True, exist_ok=False)
    image_root = destination / "image-root"
    image_root.mkdir()
    run("ditto", stage / "Paddock.app", image_root / "Paddock.app")
    (image_root / "Applications").symlink_to("/Applications")
    dmg = destination / (name + ".dmg")
    run("hdiutil", "create", "-quiet", "-fs", "HFS+", "-format", "UDZO", "-volname", "Paddock",
        "-srcfolder", image_root, dmg)
    component = destination / "cli-component.pkg"
    run("pkgbuild", "--root", stage / "cli", "--install-location", "/usr/local",
        "--identifier", "io.truespar.paddock.cli", "--version", release_version,
        "--ownership", "recommended", component)
    # A distribution wrapper enforces architecture/OS. The CLI installer does
    # not install a launch daemon, download weights, or start a service.
    distribution = destination / "Distribution.xml"
    distribution.write_text('''<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>Paddock command-line tools</title>
  <options customize="never" require-scripts="false" hostArchitectures="arm64"/>
  <volume-check><allowed-os-versions><os-version min="26.0"/></allowed-os-versions></volume-check>
  <choices-outline><line choice="cli"/></choices-outline>
  <choice id="cli" visible="false"><pkg-ref id="io.truespar.paddock.cli"/></choice>
  <pkg-ref id="io.truespar.paddock.cli">cli-component.pkg</pkg-ref>
</installer-gui-script>
''')
    pkg = destination / (name + "-cli.pkg")
    signing = ["--sign", installer_identity, "--timestamp"] if installer_identity else []
    run("productbuild", "--distribution", distribution, "--package-path", destination, *signing, pkg)
    archive = destination / (name + "-cli.tar.gz")
    run("tar", "-czf", archive, "-C", stage / "cli", "bin", "share")
    return [dmg, pkg, archive]


def checksums(paths, destination):
    (destination / "SHA256SUMS").write_text("".join(digest(p) + "  " + p.name + "\n" for p in paths))


def build(args):
    tools = doctor()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        require(output.is_relative_to(ROOT / "dist"), "In-repository output must be under ignored dist/.")
    require(not output.exists(), "Choose a new output directory; existing builds are never overwritten.")
    require(shutil.disk_usage(ROOT).free > 40 * 2**30, "At least 40 GiB free is required for a cold build.")
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()
    source = work / "source"
    provenance = snapshot(ROOT, source, args.allow_dirty)
    release_version = version(source)
    env = build_environment(work)
    locks = ("Cargo.lock", "studio/package-lock.json", "apps/macos/Package.resolved")
    hashes = {name: digest(source / name) for name in locks}
    # Build the manager's embedded web Studio before Cargo, and the separate
    # viewer-only surface before assembling the native app.
    run("npm", "ci", "--no-audit", "--no-fund", cwd=source / "studio", env=env)
    run("npm", "run", "build", "--", "--logLevel", "warn", cwd=source / "studio", env=env)
    require((source / "crates/paddock-manager/static/index.html").is_file(), "Web Studio assets missing.")
    run("node_modules/.bin/vue-tsc", "--noEmit", "-p", "native-workspace/tsconfig.json",
        cwd=source / "studio", env=env)
    run("node_modules/.bin/vite", "build", "--config", "native-workspace/vite.config.ts", "--logLevel", "warn",
        cwd=source / "studio", env=env)
    run("cargo", "build", "--locked", "--release", "--target", TARGET,
        "-p", "paddock-manager", "-p", "paddock-desktop", "-p", "paddock-runner",
        "--no-default-features", "--features", FEATURES, cwd=source, env=env)
    run(*swift_args(source, tools["SDK"]), env=env)
    require(hashes == {name: digest(source / name) for name in locks}, "A dependency lockfile changed.")
    stage = output / "stage"
    assemble(source, work, stage, release_version, args.build_number, env, tools["SDK"])
    if args.update_public_key:
        import base64
        try:
            require(len(base64.b64decode(args.update_public_key, validate=True)) == 32,
                    "The Sparkle public key must decode to 32 bytes.")
        except ValueError as error:
            raise ReleaseError("Invalid Sparkle public key.") from error
        info_path = stage / "Paddock.app/Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info["SUPublicEDKey"] = args.update_public_key
        info_path.write_bytes(plistlib.dumps(info))
    sign(stage, "-")
    smoke(stage, env)
    name = "Paddock-" + release_version + "-macos-arm64-UNSIGNED"
    artifacts = packages(stage, output / "unsigned", name, release_version)
    checksums(artifacts, output / "unsigned")
    manifest = {"schema": 1, "status": "unsigned-candidate", "source": provenance,
                "version": release_version, "build_number": args.build_number,
                "minimum_macos": MIN_OS, "target": TARGET, "tools": tools,
                "locks": hashes, "stage": inventory(stage),
                "checks": ["web-studio-build", "native-viewer-boundary", "locked-dependencies",
                           "optimized-metal-and-desktop", "swift-release-warnings-as-errors",
                           "arm64-linkage", "adhoc-runtime-signatures", "cli-and-desktop-smoke"]}
    write_json(output / "manifest.json", manifest)
    print("Unsigned candidate: " + str(output) + "\nNot ready for distribution. Nothing installed or published.")


def identity_team(identity, kind):
    match = re.fullmatch(r"Developer ID " + kind + r": .+ \(([A-Z0-9]{10})\)", identity)
    require(match is not None, "Expected the full Developer ID " + kind + " identity, not Apple Development/ad-hoc.")
    return match.group(1)


def valid_build_number(value):
    # CFBundleVersion: up to four digits, then optional two-digit components.
    require(bool(re.fullmatch(r"[1-9][0-9]{0,3}(?:\.(?:0|[1-9][0-9]?)){0,2}", value)),
            "Build number must be N[.N[.N]], with at most 4/2/2 digits and no leading zeros.")
    return value


def validate_candidate(candidate):
    manifest = json.loads((candidate / "manifest.json").read_text())
    require(manifest["schema"] == 1 and manifest["status"] == "unsigned-candidate", "Unknown candidate format.")
    require(manifest["source"]["origin"] in PUBLIC_ORIGINS and not manifest["source"]["dirty"]
            and manifest["source"]["public_commit"], "Signing requires a clean, public-source commit.")
    require(manifest["stage"] == inventory(candidate / "stage"), "Candidate files changed after validation.")
    return manifest


def notarize(path, profile, logs):
    result = json.loads(run("xcrun", "notarytool", "submit", path, "--keychain-profile", profile,
                            "--output-format", "json", capture=True))
    # Persist the submission before waiting, so interruption cannot lose its ID.
    write_json(logs / (path.name + ".submission.json"), result)
    submission = result.get("id")
    require(bool(submission), "Notarization returned no submission ID.")
    result = json.loads(run("xcrun", "notarytool", "wait", submission, "--keychain-profile", profile,
                            "--timeout", "30m", "--output-format", "json", capture=True))
    write_json(logs / (path.name + ".result.json"), result)
    run("xcrun", "notarytool", "log", submission, "--keychain-profile", profile,
        logs / (path.name + ".notary-log.json"))
    require(result.get("status") == "Accepted", "Notarization was not accepted; see " + str(logs))


def finalize(args):
    doctor()
    team = identity_team(args.application_identity, "Application")
    require(team == identity_team(args.installer_identity, "Installer"), "Signing teams must match.")
    identities = run("security", "find-identity", "-v", capture=True)
    for identity in (args.application_identity, args.installer_identity):
        require('"' + identity + '"' in identities, "Missing valid identity/private key: " + identity)
    candidate = args.input.expanduser().resolve()
    manifest = validate_candidate(candidate)
    update_tools = candidate / "work/source/apps/macos/.build/artifacts/sparkle/Sparkle/bin"
    validate_update_identity(candidate / "stage/Paddock.app", update_tools, args.update_account)
    release = candidate / "release"
    require(not release.exists(), "Release output already exists; use a fresh candidate to retry.")
    release.mkdir()
    stage = release / "stage"
    shutil.copytree(candidate / "stage", stage, symlinks=True)
    logs = release / "notarization"
    logs.mkdir()
    sign(stage, args.application_identity)
    app = stage / "Paddock.app"
    zip_path = release / "Paddock-notarization.zip"
    run("ditto", "-c", "-k", "--keepParent", app, zip_path)
    notarize(zip_path, args.notary_profile, logs)
    run("xcrun", "stapler", "staple", app)
    run("xcrun", "stapler", "validate", app)
    name = "Paddock-" + manifest["version"] + "-macos-arm64"
    artifacts = packages(stage, release / "artifacts", name, manifest["version"], args.installer_identity)
    dmg, pkg, _ = artifacts
    run("codesign", "--sign", args.application_identity, "--timestamp", dmg)
    for artifact in (dmg, pkg):
        notarize(artifact, args.notary_profile, logs)
        run("xcrun", "stapler", "staple", artifact)
        run("xcrun", "stapler", "validate", artifact)
    run("codesign", "--verify", "--deep", "--strict", app)
    run("spctl", "--assess", "--type", "execute", "--verbose=2", app)
    run("spctl", "--assess", "--type", "install", "--verbose=2", pkg)
    run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", dmg)
    run("pkgutil", "--check-signature", pkg)
    artifacts.append(signed_appcast(app, dmg, update_tools, args.update_account, manifest))
    checksums(artifacts, release / "artifacts")  # After stapling, never before.
    manifest.update(status="notarized-awaiting-manual-qualification", team=team, stage=inventory(stage))
    write_json(release / "manifest.json", manifest)
    print("Notarized artifacts: " + str(release / "artifacts") + "\nNothing published. Complete the README's manual gates.")


def validate_update_identity(app, tools, account):
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    key = info.get("SUPublicEDKey", "")
    try:
        valid = len(base64.b64decode(key, validate=True)) == 32
    except (ValueError, TypeError):
        valid = False
    require(valid and info.get("SUFeedURL") == UPDATE_FEED
            and info.get("SURequireSignedFeed") is True
            and info.get("SUVerifyUpdateBeforeExtraction") is True,
            "Release must embed the public update key and require signed feeds/archives.")
    for name in ("generate_keys", "generate_appcast", "sign_update"):
        require((tools / name).is_file(), "Missing pinned Sparkle tool: " + name)
    public = run(tools / "generate_keys", "--account", account, "-p", capture=True).strip()
    require(public == key, "The Keychain update signing identity does not match this app. Do not rotate it implicitly.")
    return info


def validate_appcast(path, dmg, manifest):
    namespace = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"
    root = XML.parse(path).getroot()
    items = root.findall("./channel/item")
    require(len(items) == 1, "Expected exactly one release in the new appcast.")
    item = items[0]
    enclosure = item.find("enclosure")
    require(enclosure is not None, "Missing signed update enclosure.")
    expected_url = "https://github.com/truespar/paddock/releases/download/v" + manifest["version"] + "/" + dmg.name
    require(enclosure.get("url") == expected_url, "Update URL must identify the exact public release asset.")
    require(item.findtext(namespace + "version") == manifest["build_number"]
            and item.findtext(namespace + "shortVersionString") == manifest["version"],
            "Appcast version differs from the built release.")
    require(enclosure.get("length") == str(dmg.stat().st_size), "Appcast size differs from the stapled DMG.")
    signature = enclosure.get(namespace + "edSignature", "")
    try:
        valid = len(base64.b64decode(signature, validate=True)) == 64
    except ValueError:
        valid = False
    require(valid, "Missing or invalid Ed25519 archive signature.")
    require(item.findtext(namespace + "minimumSystemVersion") == MIN_OS, "Appcast OS floor differs from the release.")
    return signature


def signed_appcast(app, dmg, tools, account, manifest):
    validate_update_identity(app, tools, account)
    before = digest(dmg)
    destination = dmg.parent / "appcast.xml"
    require(not destination.exists(), "Never overwrite a published/signed appcast implicitly.")
    # Isolate the archive scanner: the CLI pkg is NOT an app update, and no old
    # release files may be pruned/moved by generate_appcast.
    with tempfile.TemporaryDirectory(prefix="paddock-appcast-") as folder:
        archives = Path(folder)
        shutil.copy2(dmg, archives / dmg.name)
        run(tools / "generate_appcast", "--account", account, "--maximum-deltas", "0",
            "--maximum-versions", "1", "--versions", manifest["build_number"],
            "--download-url-prefix", "https://github.com/truespar/paddock/releases/download/v" + manifest["version"] + "/",
            "-o", destination, archives)
    require(digest(dmg) == before, "Update generation changed the notarized archive.")
    signature = validate_appcast(destination, dmg, manifest)
    run(tools / "sign_update", "--account", account, "--verify", dmg, signature)
    run(tools / "sign_update", "--account", account, "--verify", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--build-number", required=True, help="Monotonically increasing app build number, N[.N[.N]]")
    build_parser.add_argument("--allow-dirty", action="store_true", help="Unsigned local candidate only; cannot finalize")
    build_parser.add_argument("--update-public-key", help="Public Ed25519 key for the signed Sparkle appcast (never a private key)")
    final_parser = commands.add_parser("finalize")
    final_parser.add_argument("--input", type=Path, required=True)
    final_parser.add_argument("--application-identity", required=True)
    final_parser.add_argument("--installer-identity", required=True)
    final_parser.add_argument("--notary-profile", required=True)
    final_parser.add_argument("--update-account", default=UPDATE_ACCOUNT,
                              help="Keychain account holding the Sparkle signing key; never a private key value")
    args = parser.parse_args()
    if args.command == "build":
        valid_build_number(args.build_number)
        build(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        doctor()


if __name__ == "__main__":
    try:
        main()
    except (ReleaseError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        print("Release stopped: " + str(error), file=sys.stderr)
        sys.exit(1)
