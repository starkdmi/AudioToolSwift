#!/usr/bin/env python3
"""Check that the tree describes the release it is about to become.

    python3 Scripts/check-release.py 0.1.0

Run this *before* creating the tag. A tag is immutable in every consumer's
resolver cache: SwiftPM records it in Package.resolved and CI on a pushed tag
starts only once GitHub has accepted it, so a red run there reports a mistake
rather than preventing one. The only enforcement that works is the kind that
happens first, which is why this is a script and not just a workflow step.

Three things go stale the instant a tag exists:

  * the README's install snippet, which points at a branch and says there is no
    release yet
  * the changelog heading, which says unreleased and carries no date
  * the resolved dependency table in Docs/licenses.md

The version is matched literally, not as a regex - "0.1.0" should not match
"0X1Y0".
"""

import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


# A declaration is this package's install snippet only if its url is *this*
# repository - canonical owner included, both ends anchored.
#
# Every looser version of this let something through. Matching the name anywhere
# inside the parentheses accepted `.package(name: "AudioToolSwift", ...)`, which
# has no URL to install from. Allowing any owner accepted
# https://github.com/anyone/AudioToolSwift.git - a same-name fork, which is the
# shape a mistake actually takes, and the shape the hand-written control missed
# by using a differently named fork. Leaving the start unanchored accepted that
# path as the tail of some other URL.
CANONICAL_REPOSITORY = "https://github.com/starkdmi/AudioToolSwift"
PACKAGE_URL = re.compile(r'url:\s*"(?P<url>[^"]+)"')
OUR_REPOSITORY = re.compile(
    r"\A" + re.escape(CANONICAL_REPOSITORY) + r"(\.git)?/?\Z"
)


INSTALL_HEADING = re.compile(r"^##\s+Install\s*$", re.MULTILINE)
NEXT_HEADING = re.compile(r"^##\s+", re.MULTILINE)
# Anchored to line starts, so a fence inside a blockquote or a nested example
# is not one of this page's install snippets: "> ```swift" is quoted text
# telling you about a declaration, not a block offering you one.
CODE_FENCE = re.compile(r"^```[a-z]*[ \t]*$\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def install_snippets(readme, problems):
    """The code fences under `## Install`, which is the only place that counts.

    Every looser reading of "the README says how to install this" has let
    something through, in this order: the string `from: "0.1.0"` anywhere, which
    the sentence "then pin with from: \"0.1.0\"" satisfied on its own; any
    `.package(...)` mentioning the name, which a declaration inside an HTML
    comment satisfied; and any *visible* declaration, which a sentence like "do
    not use `.package(url: ..., from: ...)`" would satisfy even with the install
    block deleted. Instructions are what is in the install block, so that is what
    is read.
    """
    # Commented-out instructions are not instructions.
    visible = re.sub(r"<!--.*?-->", "", readme, flags=re.DOTALL)

    heading = INSTALL_HEADING.search(visible)
    if heading is None:
        problems.append("README.md has no '## Install' section")
        return []

    rest = visible[heading.end():]
    following = NEXT_HEADING.search(rest)
    section = rest[:following.start()] if following else rest

    fences = CODE_FENCE.findall(section)
    if not fences:
        problems.append("README.md's Install section contains no code block")
        return []
    return fences


def check_readme(version, problems):
    """The install snippet must actually install this version."""
    readme = (ROOT / "README.md").read_text()

    if re.search(r"no tagged release yet", readme, re.IGNORECASE):
        problems.append("README.md still says there is no tagged release yet")

    declarations = []
    for fence in install_snippets(readme, problems):
        for body in re.findall(r"\.package\(([^)]*)\)", fence, re.DOTALL):
            found = PACKAGE_URL.search(body)
            if found and OUR_REPOSITORY.search(found.group("url")):
                declarations.append(" ".join(body.split()))

    if not declarations:
        problems.append(
            "README.md's Install section has no .package declaration whose url is "
            "this repository, so there are no install instructions to follow"
        )
        return

    for body in declarations:
        unstable = re.search(r"\b(branch|revision)\s*:", body)
        if unstable:
            problems.append(
                f"README.md installs from {unstable.group(1)} rather than a "
                f"version: .package({body})"
            )

    wanted = re.compile(r"from:\s*\"" + re.escape(version) + r"\"")
    if not any(wanted.search(body) for body in declarations):
        problems.append(
            f'README.md has no .package declaration using from: "{version}"'
        )


def check_changelog(version, problems):
    changelog = (ROOT / "CHANGELOG.md").read_text()

    # `## 0.1.0 — 2026-08-15`, in either dash. An undated heading is the failure
    # this exists to catch, so the date is required rather than merely preferred,
    # and "unreleased" is only the most obvious way to lack one.
    heading = re.compile(
        r"^##\s+" + re.escape(version) + r"\s*[-–—]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$",
        re.MULTILINE,
    )
    dated = heading.search(changelog)
    if dated:
        try:
            datetime.date.fromisoformat(dated.group(1))
            return
        except ValueError:
            problems.append(
                f"CHANGELOG.md heading for {version} carries {dated.group(1)!r}, "
                "which is not a real date"
            )
            return

    any_heading = re.compile(r"^##\s+" + re.escape(version) + r"\b.*$", re.MULTILINE)
    found = any_heading.search(changelog)
    if found:
        problems.append(
            f"CHANGELOG.md heading for {version} carries no ISO date: {found.group(0)!r}"
        )
    else:
        problems.append(f"CHANGELOG.md has no entry for {version}")


def check_licenses(problems):
    """The dependency table and Package.resolved must describe the same set.

    Both directions matter. A dependency added without a row goes undocumented,
    and a row left behind after a dependency is dropped claims a licence
    obligation that no longer exists. Checking only "every row that names a
    version names the right one" catches neither.
    """
    resolved = json.loads((ROOT / "Package.resolved").read_text())
    licenses = (ROOT / "Docs" / "licenses.md").read_text()

    # Only the SwiftPM dependency table. The document also has tables of model and
    # fixture licences, and one of those rows begins "FluidAudio Parakeet ..." -
    # matching on the name alone picks it up and compares a package version against
    # a model's licence.
    section = licenses.split("## SwiftPM dependencies", 1)
    if len(section) != 2:
        problems.append("Docs/licenses.md has no '## SwiftPM dependencies' section")
        return
    table = section[1].split("\n## ", 1)[0]

    documented = {}
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue  # separator row
        # A row may cover several packages: the Apple row names four. Take every
        # backticked identifier in the first cell.
        if len(cells) < 3 or not cells[2]:
            problems.append(
                f"Docs/licenses.md row names no licence: {line.strip()!r}"
            )
            continue
        names = re.findall(r"`([^`]+)`", cells[0])
        for name in names:
            identity = name.lower()
            if identity in documented:
                problems.append(
                    f"Docs/licenses.md documents {identity} twice; the second row "
                    f"hides the first: {line.strip()!r}"
                )
            documented[identity] = (line.strip(), cells[1])

    pinned = {}
    for pin in resolved.get("pins", []):
        pinned[pin["identity"].lower()] = pin.get("state", {}).get("version")

    for identity, version in sorted(pinned.items()):
        entry = documented.get(identity)
        if entry is None:
            problems.append(
                f"Docs/licenses.md documents no licence for {identity}, "
                "which Package.resolved pins"
            )
            continue
        row, version_cell = entry
        if version is None:
            continue  # a revision pin; check-publishable.sh rejects those anyway
        # Rows that defer ("see `Package.resolved`") have nothing to go stale.
        if "package.resolved" in version_cell.lower():
            continue
        # The cell may carry a parenthetical, as the fork rows do:
        # "1.0.5-static.1 (`1ecaf9a...`)". Compare the version itself, exactly,
        # so "1.5" does not satisfy "1.5.0".
        stated = re.sub(r"\(.*?\)", "", version_cell).strip().strip("`").strip()
        if stated != version:
            problems.append(
                f"Docs/licenses.md lists {identity} at {stated!r}, "
                f"but Package.resolved has {version!r}: {row!r}"
            )

    for identity in sorted(set(documented) - set(pinned)):
        problems.append(
            f"Docs/licenses.md documents {identity}, which is no longer a "
            "resolved dependency"
        )


# Semantic version, the subset SwiftPM accepts as a release tag: ASCII digits, no
# leading zeros, no prerelease or build suffix. `\d` is deliberately avoided -
# without re.ASCII it matches Unicode digits, so "٠.١.٠" would pass a naive check
# and then fail to resolve for everyone.
SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def main(version):
    if not SEMVER.match(version):
        sys.exit(
            f"usage: check-release.py <version>, e.g. 0.1.0 - got {version!r}, "
            "which is not a release semantic version"
        )

    problems = []
    check_readme(version, problems)
    check_changelog(version, problems)
    check_licenses(problems)

    if problems:
        print(f"Not ready to tag {version}:")
        for problem in problems:
            print(f"  error: {problem}")
        return 1

    print(f"Ready to tag {version}: README, changelog and licence table agree.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check-release.py <version>")
    sys.exit(main(sys.argv[1]))
