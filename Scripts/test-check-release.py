#!/usr/bin/env python3
"""Regression tests for Scripts/check-release.py.

    python3 Scripts/test-check-release.py

The release checker guards a one-way door, so its own failures are expensive: a
hole in it means a tag that says the wrong thing, permanently. Every hole it has
had so far was found by someone reading it rather than by running it, and the
hand-written controls that "verified" the fixes had holes of their own - one used
a fork named AudioToolSwiftFork, so it never exercised the same-name fork that a
real mistake produces, and the check went on accepting any owner.

So the controls live here, run in CI, and each one is a mutation of a
release-ready tree that must be rejected for a stated reason.
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = "9.9.9"

VALID_SNIPPET = f'.package(url: "https://github.com/starkdmi/AudioToolSwift.git", from: "{VERSION}")'

# The fixture README is written here rather than derived from the repository's,
# and that is the whole point: the first version of this harness edited the real
# README by replacing its `branch: "main"` snippet. On the release commit that
# snippet is already gone - it has become `from: "0.1.0"` - so the edit became a
# no-op, the baseline case ran the checker for 9.9.9 against a README saying
# 0.1.0, and this test went red. CI red means the Release workflow refuses to tag,
# which made the correctly prepared release commit the one commit that could never
# be released. A fixture that depends on the state it is meant to be tested in is
# not a fixture.
FIXTURE_README = f"""# AudioToolSwift

On-device speech and audio ML for Apple platforms.

## Speed

Numbers go here.

## Install

```swift
{VALID_SNIPPET}
```

Then depend on the libraries you need:

```swift
.product(name: "AudioTool", package: "AudioToolSwift"),
```

## Requirements

- iOS 18+ / macOS 15+
"""

FIXTURE_CHANGELOG = f"""# Changelog

## {VERSION} — 2026-08-15

Everything.
"""


def release_ready(directory):
    """A minimal tree in the state a release commit should be in.

    Package.resolved and Docs/licenses.md are copied from the repository, since
    neither changes at tag time and the licence cases need the real table's
    shape - if those two ever disagree, this harness says so, which is the same
    thing the release itself would say.
    """
    (directory / "Scripts").mkdir(parents=True, exist_ok=True)
    (directory / "Docs").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "Scripts" / "check-release.py", directory / "Scripts")
    shutil.copy(ROOT / "Package.resolved", directory)
    shutil.copy(ROOT / "Docs" / "licenses.md", directory / "Docs")

    (directory / "README.md").write_text(FIXTURE_README)
    (directory / "CHANGELOG.md").write_text(FIXTURE_CHANGELOG)
    return directory


def edit(directory, name, replacement, original=None):
    path = directory / name
    text = path.read_text()
    if original is None:
        path.write_text(replacement)
    else:
        assert original in text, f"{name} does not contain {original!r}"
        path.write_text(text.replace(original, replacement, 1))


def run(directory, version=VERSION):
    result = subprocess.run(
        [sys.executable, str(directory / "Scripts" / "check-release.py"), version],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


# Each case mutates the release-ready tree and names what must be reported.
# `None` means the tree must pass.
CASES = [
    ("release-ready tree", lambda d: None, None),
    (
        "install snippet points at a branch",
        lambda d: edit(
            d, "README.md",
            '.package(url: "https://github.com/starkdmi/AudioToolSwift.git", branch: "main")',
            VALID_SNIPPET,
        ),
        "installs from branch",
    ),
    (
        "declaration is prose, not the install block",
        lambda d: edit(
            d, "README.md",
            f"Do not use `{VALID_SNIPPET}` for anything.",
            f"```swift\n{VALID_SNIPPET}\n```",
        ),
        "no .package declaration whose url is this repository",
    ),
    (
        "declaration lives under a different heading",
        lambda d: edit(
            d, "README.md",
            f"## Elsewhere\n\n```swift\n{VALID_SNIPPET}\n```\n\n## Install\n\n```swift\n// nothing to see\n```",
            f"## Install\n\n```swift\n{VALID_SNIPPET}\n```",
        ),
        "no .package declaration whose url is this repository",
    ),
    (
        "install section has no code block",
        lambda d: edit(
            d, "README.md",
            "## Install\n\nSee the docs.\n\n## Requirements",
            f"## Install\n\n```swift\n{VALID_SNIPPET}\n```\n\nThen depend on the libraries you need:\n\n```swift\n.product(name: \"AudioTool\", package: \"AudioToolSwift\"),\n```\n\n## Requirements",
        ),
        "no code block",
    ),
    (
        "no install section at all",
        lambda d: edit(
            d, "README.md",
            f"## Something else\n\n```swift\n{VALID_SNIPPET}\n```",
            f"## Install\n\n```swift\n{VALID_SNIPPET}\n```",
        ),
        "no '## Install' section",
    ),
    (
        "README still says there is no tagged release",
        lambda d: edit(
            d, "README.md",
            "## Install\n\nThere is no tagged release yet; track `main`.\n",
            "## Install\n",
        ),
        "no tagged release yet",
    ),
    (
        "the only fence is quoted inside a blockquote",
        lambda d: edit(
            d, "README.md",
            "## Install\n\n> Do not copy this:\n>\n> ```swift\n> " + VALID_SNIPPET
            + "\n> ```\n\n## Requirements",
            f"## Install\n\n```swift\n{VALID_SNIPPET}\n```\n\nThen depend on the libraries you need:\n\n```swift\n.product(name: \"AudioTool\", package: \"AudioToolSwift\"),\n```\n\n## Requirements",
        ),
        "no code block",
    ),
    (
        "install snippet is commented out of the rendered page",
        lambda d: edit(d, "README.md", f"<!-- {VALID_SNIPPET} -->", VALID_SNIPPET),
        "no .package declaration whose url is this repository",
    ),
    (
        "declaration carries no url",
        lambda d: edit(
            d, "README.md",
            f'.package(name: "AudioToolSwift", from: "{VERSION}")', VALID_SNIPPET,
        ),
        "no .package declaration whose url is this repository",
    ),
    (
        "same-name fork under another owner",
        lambda d: edit(
            d, "README.md",
            f'.package(url: "https://github.com/someone-else/AudioToolSwift.git", from: "{VERSION}")',
            VALID_SNIPPET,
        ),
        "no .package declaration whose url is this repository",
    ),
    (
        "canonical path as the tail of another url",
        lambda d: edit(
            d, "README.md",
            f'.package(url: "https://evil.example.com/https://github.com/starkdmi/AudioToolSwift", from: "{VERSION}")',
            VALID_SNIPPET,
        ),
        "no .package declaration whose url is this repository",
    ),
    (
        "install snippet names a different version",
        lambda d: edit(
            d, "README.md",
            '.package(url: "https://github.com/starkdmi/AudioToolSwift.git", from: "0.0.1")',
            VALID_SNIPPET,
        ),
        f'no .package declaration using from: "{VERSION}"',
    ),
    (
        "changelog heading is undated",
        lambda d: edit(d, "CHANGELOG.md", f"## {VERSION}", f"## {VERSION} — 2026-08-15"),
        "carries no ISO date",
    ),
    (
        "changelog heading says unreleased",
        lambda d: edit(
            d, "CHANGELOG.md", f"## {VERSION} — unreleased", f"## {VERSION} — 2026-08-15"
        ),
        "carries no ISO date",
    ),
    (
        "changelog date is impossible",
        lambda d: edit(
            d, "CHANGELOG.md", f"## {VERSION} — 2026-99-99", f"## {VERSION} — 2026-08-15"
        ),
        "not a real date",
    ),
    (
        "changelog has no entry at all",
        lambda d: edit(d, "CHANGELOG.md", "# Changelog\n"),
        "has no entry",
    ),
    (
        "licence row names a stale version",
        lambda d: edit(d, "Docs/licenses.md", "| `yyjson` | 0.0.1 |", "| `yyjson` | 0.12.0 |"),
        "but Package.resolved has",
    ),
    (
        "licence row names an imprecise version",
        lambda d: edit(d, "Docs/licenses.md", "| `yyjson` | 0.12 |", "| `yyjson` | 0.12.0 |"),
        "but Package.resolved has",
    ),
    (
        "licence row names no licence",
        lambda d: edit(d, "Docs/licenses.md", "| `yyjson` | 0.12.0 |  |", "| `yyjson` | 0.12.0 | MIT |"),
        "names no licence",
    ),
    (
        "package documented twice",
        lambda d: edit(
            d, "Docs/licenses.md",
            "| `yyjson` | 0.12.0 | MIT |\n| `yyjson` | 0.12.0 | MIT |",
            "| `yyjson` | 0.12.0 | MIT |",
        ),
        "twice",
    ),
    ("dependency with no licence row", lambda d: add_dependency(d), "documents no licence for"),
    ("licence row for a dropped dependency", lambda d: drop_dependency(d), "no longer a resolved dependency"),
]

# Versions the checker must refuse outright, before looking at any file.
BAD_VERSIONS = ["01.0.0", "1.0", "1.0.0-beta", "٠.١.٠", "1.0.0 ; rm -rf /"]


def add_dependency(directory):
    path = directory / "Package.resolved"
    resolved = json.loads(path.read_text())
    resolved["pins"].append({
        "identity": "brand-new-dependency",
        "kind": "remoteSourceControl",
        "location": "https://example.com/new",
        "state": {"version": "1.0.0"},
    })
    path.write_text(json.dumps(resolved, indent=2))


def drop_dependency(directory):
    path = directory / "Package.resolved"
    resolved = json.loads(path.read_text())
    resolved["pins"] = [pin for pin in resolved["pins"] if pin["identity"] != "yyjson"]
    path.write_text(json.dumps(resolved, indent=2))


def main():
    failures = []

    for name, mutate, expected in CASES:
        with tempfile.TemporaryDirectory() as temporary:
            directory = release_ready(pathlib.Path(temporary))
            mutate(directory)
            code, output = run(directory)

            if expected is None:
                if code != 0:
                    failures.append(f"{name}: expected a pass, got:\n{output}")
            elif code == 0:
                failures.append(f"{name}: expected a failure mentioning {expected!r}, but it passed")
            elif expected not in output:
                failures.append(f"{name}: expected {expected!r} in:\n{output}")

    for version in BAD_VERSIONS:
        with tempfile.TemporaryDirectory() as temporary:
            directory = release_ready(pathlib.Path(temporary))
            code, output = run(directory, version)
            if code == 0:
                failures.append(f"version {version!r}: accepted, should have been refused")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"\n{len(failures)} of {len(CASES) + len(BAD_VERSIONS)} checks failed")
        return 1

    print(f"All {len(CASES) + len(BAD_VERSIONS)} release-checker cases behave as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
