#!/bin/bash
#
# Compile the package with the Swift version the manifest declares, locally.
#
# The CI job of the same purpose picks the oldest Xcode on the runner image whose
# Swift is 6.2.x. On a machine that has only a newer Xcode - which any machine on
# macOS Tahoe does, since Xcode 26.4.1 dropped Sequoia and with it the 6.2
# toolchains - the equivalent is a swift.org toolchain installed alongside it:
#
#   https://download.swift.org/swift-6.2-release/xcode/swift-6.2-RELEASE/swift-6.2-RELEASE-osx.pkg
#   installer -pkg swift-6.2-RELEASE-osx.pkg -target CurrentUserHomeDirectory
#
# That needs no sudo; it lands in ~/Library/Developer/Toolchains.
#
# Worth having because the floor is not theoretical: it has caught two breaks
# that no other job could see, both in code that compiles happily on 6.3 - an
# expression 6.2's type checker gives up on, and `Test.cancel`, which does not
# exist in the Swift Testing 6.2 ships. Each cost a CI round trip to find.
#
# Usage: Scripts/check-swift-floor.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

FLOOR="6.2"

toolchain=""
for candidate in "$HOME/Library/Developer/Toolchains"/*.xctoolchain \
                 "/Library/Developer/Toolchains"/*.xctoolchain; do
    [ -d "$candidate" ] || continue
    identifier="$(plutil -extract CFBundleIdentifier raw "$candidate/Info.plist" 2>/dev/null || true)"
    [ -n "$identifier" ] || continue
    version="$(TOOLCHAINS="$identifier" swift --version 2>/dev/null \
        | sed -n 's/.*Swift version \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
    case "$version" in
        "$FLOOR"|"$FLOOR".*)
            toolchain="$identifier"
            echo "Using Swift $version from $(basename "$candidate")"
            break
            ;;
    esac
done

if [ -z "$toolchain" ]; then
    echo "error: no Swift $FLOOR.x toolchain installed."
    echo "       See the header of this script for the one-line install."
    exit 1
fi

# -disable-experimental-parser-round-trip is a property of open-source toolchains,
# not of the floor. They enable a check that re-parses each file with the new
# parser and reports where it disagrees with the old one; Apple's shipped 6.2 does
# not, which is why CI compiles this package unaided. Without the flag the build
# stops inside MisakiSwift, whose `MToken` has a member named `_` - `$0._.stress`
# parses under Xcode's 6.2 and under 6.3, and trips the round-trip check here. CI
# does not pass this flag and should not.
#
# Scratch path under .build so the tree stays clean and the 6.3 build is not
# invalidated on every switch.
exec env TOOLCHAINS="$toolchain" AUDIOTOOL_STRICT_WARNINGS=1 \
    swift build --build-tests \
    --scratch-path .build/swift-floor \
    -Xswiftc -Xfrontend -Xswiftc -disable-experimental-parser-round-trip
