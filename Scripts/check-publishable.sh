#!/bin/bash
#
# Fail if this package could not be consumed as a versioned SwiftPM dependency.
#
# Two manifest properties make `.package(url: ..., from: "0.1.0")` impossible for
# anyone depending on us, and neither shows up in a normal build - the package
# itself compiles and tests fine either way. They only surface in someone else's
# resolution, which is the one place we never look:
#
#   1. Unsafe build flags in a target reachable from a product:
#        error: 'app': the target 'AudioTool' in product 'AudioTool'
#               contains unsafe build flags
#
#   2. A branch or revision dependency requirement:
#        error: package 'audiotoolswift' is required using a stable-version but
#               'audiotoolswift' depends on an unstable-version package 'fluidaudio'
#
# Both were true of this package before 0.1.0. Run this in CI so they cannot come
# back the next time someone reaches for -warnings-as-errors or pins a SHA.
#
# Usage: Scripts/check-publishable.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# The strict-warnings flag is opt-in precisely because it is unsafe, so a caller
# who has it exported would otherwise see this script fail on their own setting.
# Judge the manifest as a consumer sees it: without it.
manifest="$(env -u AUDIOTOOL_STRICT_WARNINGS swift package dump-package)"

status=0

unsafe="$(printf '%s' "$manifest" | python3 -c '
import json, sys
package = json.load(sys.stdin)
for target in package["targets"]:
    for setting in target.get("settings", []):
        flags = setting.get("kind", {}).get("unsafeFlags")
        if flags is not None:
            print("  " + target["name"] + ": " + " ".join(flags.get("values", [])))
')"

if [ -n "$unsafe" ]; then
    echo "error: targets carry unsafe build flags, which bars every product from"
    echo "       versioned consumption:"
    echo "$unsafe"
    echo "       Gate them behind an environment variable, as commonSwiftSettings does."
    status=1
fi

unstable="$(printf '%s' "$manifest" | python3 -c '
import json, sys
package = json.load(sys.stdin)
for dependency in package["dependencies"]:
    source = dependency.get("sourceControl") or []
    for entry in source:
        location = entry.get("location", {})
        urls = location.get("remote", [])
        url = urls[0].get("urlString", "?") if urls else "?"
        requirement = entry.get("requirement", {})
        for kind in ("branch", "revision"):
            if kind in requirement:
                value = requirement[kind]
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                print("  " + url + ": " + kind + " " + str(value))
')"

if [ -n "$unstable" ]; then
    echo "error: dependencies pinned by branch or revision, which bars this package"
    echo "       from being depended on by version:"
    echo "$unstable"
    echo "       Tag the dependency (a fork tag is enough) and use exact: or a range."
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "Publishable: no unsafe flags, no branch or revision dependencies."
fi

exit "$status"
