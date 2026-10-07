"""Release safety/packaging tests. No certificates, network or GUI required."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("macos_release", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paddock-packaging-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_exact_public_origins(self):
        for origin in release.PUBLIC_ORIGINS:
            with patch.object(release, "git", return_value=origin):
                self.assertEqual(release.public_source(self.root), origin)
        for origin in ("git@github.com:truespar/paddock-mirror.git",
                       "https://example.org/truespar/paddock.git",
                       "https://github.com/truespar/paddock.git.evil"):
            with patch.object(release, "git", return_value=origin):
                with self.assertRaises(release.ReleaseError):
                    release.public_source(self.root)

    def update_fixture(self):
        app = self.root / "Paddock.app"
        (app / "Contents/MacOS").mkdir(parents=True)
        info = plistlib.loads((release.ROOT / "packaging/macos/Info.plist").read_bytes())
        info.update(CFBundleShortVersionString="9.9.9", CFBundleVersion="9999",
                    LSMinimumSystemVersion=release.MIN_OS)
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        return app, info

    def test_update_requires_matching_key_and_signed_feed(self):
        app, info = self.update_fixture()
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("generate_keys", "generate_appcast", "sign_update"):
            (tools / name).touch()
        with patch.object(release, "run", return_value=info["SUPublicEDKey"]):
            self.assertEqual(release.validate_update_identity(app, tools, "test"), info)
        with patch.object(release, "run", return_value="another-key"):
            with self.assertRaises(release.ReleaseError):
                release.validate_update_identity(app, tools, "test")
        info.pop("SUPublicEDKey")
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        with self.assertRaises(release.ReleaseError):
            release.validate_update_identity(app, tools, "test")

    @unittest.skipUnless(os.environ.get("PADDOCK_TEST_SPARKLE_SIGNING") == "1",
                         "Opt-in local Keychain signing diagnostic; never uploads or launches an app")
    def test_real_signed_feed_and_archive_reject_tampering(self):
        import shutil
        app, _ = self.update_fixture()
        shutil.copyfile("/usr/bin/true", app / "Contents/MacOS/Paddock")
        (app / "Contents/MacOS/Paddock").chmod(0o755)
        release.run("codesign", "--force", "--sign", "-", app)
        archive = self.root / "Paddock-9.9.9-macos-arm64.zip"
        release.run("ditto", "-c", "-k", "--keepParent", app, archive)
        tools = release.ROOT / "apps/macos/.build/artifacts/sparkle/Sparkle/bin"
        manifest = {"version": "9.9.9", "build_number": "9999"}
        feed = release.signed_appcast(app, archive, tools, release.UPDATE_ACCOUNT, manifest)
        signature = release.validate_appcast(feed, archive, manifest)
        archive.write_bytes(archive.read_bytes() + b"tampered")
        with self.assertRaises(release.subprocess.CalledProcessError):
            release.run(tools / "sign_update", "--account", release.UPDATE_ACCOUNT,
                        "--verify", archive, signature)
        feed.write_bytes(feed.read_bytes().replace(b"9.9.9", b"9.9.8"))
        with self.assertRaises(release.subprocess.CalledProcessError):
            release.run(tools / "sign_update", "--account", release.UPDATE_ACCOUNT,
                        "--verify", feed)

    def test_private_catalog_blocks_even_when_untracked(self):
        private = self.root / "crates/paddock-manager/models.private.toml"
        private.parent.mkdir(parents=True)
        private.write_text("private")
        with patch.object(release, "git", return_value=next(iter(release.PUBLIC_ORIGINS))):
            with self.assertRaises(release.ReleaseError):
                release.public_source(self.root)

    def test_broken_private_catalog_symlink_blocks(self):
        private = self.root / "crates/paddock-manager/models.private.toml"
        private.parent.mkdir(parents=True)
        private.symlink_to("missing")
        with patch.object(release, "git", return_value=next(iter(release.PUBLIC_ORIGINS))):
            with self.assertRaises(release.ReleaseError):
                release.public_source(self.root)

    def test_unsafe_relative_paths(self):
        for name in ("", ".", "/tmp/file", "../file", "a/../../file"):
            with self.assertRaises(release.ReleaseError):
                release.relative_path(name)
        self.assertEqual(release.relative_path("app/Contents/file"), Path("app/Contents/file"))

    def test_inventory_detects_bytes_modes_and_extra_files(self):
        entry = self.root / "binary"
        entry.write_bytes(b"first")
        initial = release.inventory(self.root)
        entry.write_bytes(b"other")
        self.assertNotEqual(initial, release.inventory(self.root))
        entry.write_bytes(b"first")
        entry.chmod(0o700)
        self.assertNotEqual(initial, release.inventory(self.root))
        (self.root / "injected").write_text("extra")
        self.assertIn("injected", release.inventory(self.root))

    def test_inventory_preserves_internal_symlinks(self):
        (self.root / "real").write_text("resource")
        (self.root / "link").symlink_to("real")
        self.assertEqual(release.inventory(self.root)["link"]["symlink"], "real")

    def test_inventory_rejects_external_symlinks(self):
        (self.root / "outside").symlink_to("/etc/hosts")
        with self.assertRaises(release.ReleaseError):
            release.inventory(self.root)

    def test_inventory_rejects_missing_root(self):
        with self.assertRaises(release.ReleaseError):
            release.inventory(self.root / "absent")

    def test_workspace_version_not_dependency_version(self):
        (self.root / "Cargo.toml").write_text(
            '[workspace.package]\nversion = "1.2.3"\n[dependencies]\nversion = "9.9.9"\n')
        self.assertEqual(release.version(self.root), "1.2.3")

    def test_environment_drops_credentials_and_build_overrides(self):
        with patch.dict(os.environ, {"VITE_SECRET": "secret", "AWS_SECRET_ACCESS_KEY": "secret",
                                     "PADDOCK_TEST": "1", "DYLD_INSERT_LIBRARIES": "bad",
                                     "RUSTFLAGS": "bad", "HOME": "/test-user"}, clear=True):
            env = release.build_environment(self.root)
        self.assertEqual(env["HOME"], "/test-user")
        for key in ("VITE_SECRET", "AWS_SECRET_ACCESS_KEY", "PADDOCK_TEST", "DYLD_INSERT_LIBRARIES", "RUSTFLAGS"):
            self.assertNotIn(key, env)
        self.assertEqual(env["MACOSX_DEPLOYMENT_TARGET"], "26.0")

    def test_identity_kind_is_not_interchangeable(self):
        application = "Developer ID Application: Example Ltd (ABCDEFGHIJ)"
        self.assertEqual(release.identity_team(application, "Application"), "ABCDEFGHIJ")
        for identity in ("-", "Apple Development: Example (ABCDEFGHIJ)", application):
            with self.assertRaises(release.ReleaseError):
                release.identity_team(identity, "Installer")

    def test_build_number_contract(self):
        for value in ("1", "12.3", "9999.99.99"):
            self.assertEqual(release.valid_build_number(value), value)
        for value in ("0", "01", "10000", "1.100", "1.01", "1.2.3.4", "beta"):
            with self.assertRaises(release.ReleaseError):
                release.valid_build_number(value)

    def test_sign_inside_out_no_deep_signing(self):
        with patch.object(release, "run") as command:
            release.sign(self.root, "Developer ID Application: Example (ABCDEFGHIJ)")
        signing = [call.args for call in command.call_args_list if "--sign" in call.args]
        self.assertEqual(len(signing), 10)
        self.assertEqual(signing[-1][-1], self.root / "Paddock.app")
        for args in signing:
            self.assertNotIn("--deep", args)
            self.assertIn("--timestamp", args)
            self.assertIn("runtime", args)

    def test_minimal_capture_entitlements(self):
        value = plistlib.loads(Path(__file__).with_name("App.entitlements").read_bytes())
        self.assertEqual(value, {
            "com.apple.security.device.audio-input": True,
            "com.apple.security.device.camera": True,
        })

    def test_app_icon_is_the_packaged_icns(self):
        info = plistlib.loads(Path(__file__).with_name("Info.plist").read_bytes())
        icon = Path(__file__).with_name(info["CFBundleIconFile"] + ".icns")
        self.assertTrue(icon.is_file())
        self.assertEqual(icon.read_bytes()[:4], b"icns")

    def candidate(self, dirty=False):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "binary").write_text("payload")
        manifest = {"schema": 1, "status": "unsigned-candidate",
                    "source": {"origin": next(iter(release.PUBLIC_ORIGINS)),
                               "dirty": dirty, "public_commit": True},
                    "stage": release.inventory(stage)}
        release.write_json(self.root / "manifest.json", manifest)
        return manifest

    def test_clean_candidate_verifies(self):
        expected = self.candidate()
        self.assertEqual(release.validate_candidate(self.root), expected)

    def test_dirty_candidate_cannot_be_signed(self):
        self.candidate(dirty=True)
        with self.assertRaises(release.ReleaseError):
            release.validate_candidate(self.root)

    def test_modified_candidate_cannot_be_signed(self):
        self.candidate()
        (self.root / "stage/binary").write_text("replacement")
        with self.assertRaises(release.ReleaseError):
            release.validate_candidate(self.root)

    def test_invalid_notarization_never_passes(self):
        with patch.object(release, "run", return_value=json.dumps({"id": "submission", "status": "Invalid"})):
            with self.assertRaises(release.ReleaseError):
                release.notarize(self.root / "app.zip", "profile-name", self.root)
        self.assertEqual(json.loads((self.root / "app.zip.submission.json").read_text())["status"], "Invalid")

    def test_accepted_notarization_fetches_log(self):
        with patch.object(release, "run", return_value=json.dumps({"id": "submission", "status": "Accepted"})) as command:
            release.notarize(self.root / "app.zip", "profile-name", self.root)
        self.assertTrue(any("log" in call.args for call in command.call_args_list))

    def test_interrupted_notarization_preserves_submission(self):
        with patch.object(release, "run", side_effect=[json.dumps({"id": "submission"}), OSError("interrupted")]):
            with self.assertRaises(OSError):
                release.notarize(self.root / "app.zip", "profile-name", self.root)
        self.assertEqual(json.loads((self.root / "app.zip.submission.json").read_text())["id"], "submission")
        self.assertFalse((self.root / "app.zip.result.json").exists())

    def test_installer_does_not_start_service(self):
        with patch.object(release, "run") as command:
            release.packages(self.root / "stage", self.root / "packages", "test", "1.2.3")
        import xml.etree.ElementTree as xml
        distribution = xml.parse(self.root / "packages/Distribution.xml").getroot()
        self.assertEqual(distribution.find("volume-check/allowed-os-versions/os-version").get("min"), release.MIN_OS)
        for call in command.call_args_list:
            self.assertNotIn("--scripts", call.args)
            self.assertNotIn("launchctl", call.args)

    def test_swift_only_builds_release_product(self):
        args = release.swift_args(self.root, "27.0")
        self.assertEqual(args[args.index("--product") + 1], "PaddockMac")
        self.assertIn("release", args)
        self.assertIn("--force-resolved-versions", args)
        self.assertNotIn("--skip-update", args)
        self.assertNotIn(str(self.root) + "=/src/paddock", args)

    def test_swift_records_sdk_separately_from_deployment_floor(self):
        args = release.swift_args(self.root, "27.0")
        start = args.index("-platform_version")
        self.assertEqual(args[start:start + 7],
                         ["-platform_version", "-Xlinker", "macos", "-Xlinker",
                          release.MIN_OS, "-Xlinker", "27.0"])
        for invalid in ("15.0", "", "27.0 extra"):
            with self.assertRaises(release.ReleaseError):
                release.swift_args(self.root, invalid)

    def test_binary_deployment_floor_matches_package(self):
        release.verify_load_commands("Load command 1\n cmd LC_BUILD_VERSION\n minos 26.0\n")
        with self.assertRaises(release.ReleaseError):
            release.verify_load_commands("Load command 1\n cmd LC_BUILD_VERSION\n minos 27.0\n")

    def test_build_machine_rpath_rejected(self):
        with self.assertRaises(release.ReleaseError):
            release.verify_load_commands("Load command 1\n minos 26.0\nLoad command 2\n"
                                         " cmd LC_RPATH\n path /opt/homebrew/lib (offset 12)\n")
        release.verify_load_commands("Load command 1\n minos 26.0\nLoad command 2\n"
                                     " cmd LC_RPATH\n path @executable_path/../Frameworks (offset 12)\n")


if __name__ == "__main__":
    unittest.main()
