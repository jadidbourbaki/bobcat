#!/bin/sh
# install.sh installs bobcat on a Mac with Apple silicon.
#
#   curl -fsSL https://raw.githubusercontent.com/jadidbourbaki/bobcat/main/install.sh | sh
#
# The script downloads the bobcat binary from the latest GitHub release,
# checks its SHA-256 checksum, and installs it into ~/.local/bin.
#
# BOBCAT_VERSION selects a release tag such as v0.1.0 in place of the
# latest release. BOBCAT_INSTALL_DIR selects the install directory.

# The whole script runs inside main, so a download cut off partway runs
# nothing.
main() {
    set -eu

    repo="jadidbourbaki/bobcat"
    asset="bobcat-aarch64-apple-darwin.tar.gz"
    install_dir="${BOBCAT_INSTALL_DIR:-$HOME/.local/bin}"

    status() { echo ">>> $*" >&2; }
    fail() {
        echo "bobcat install: $*" >&2
        exit 1
    }

    [ "$(uname -s)" = "Darwin" ] || fail "bobcat runs on macOS only"
    [ "$(uname -m)" = "arm64" ] || fail "bobcat needs a Mac with Apple silicon"
    macos_major="$(sw_vers -productVersion | cut -d. -f1)"
    [ "$macos_major" -ge 26 ] || fail "bobcat needs macOS 26 or newer"
    for tool in curl shasum tar; do
        command -v "$tool" >/dev/null || fail "bobcat needs $tool"
    done

    if [ -n "${BOBCAT_VERSION:-}" ]; then
        base="https://github.com/$repo/releases/download/$BOBCAT_VERSION"
    else
        base="https://github.com/$repo/releases/latest/download"
    fi

    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' EXIT

    status "Downloading bobcat ${BOBCAT_VERSION:-(latest release)}"
    curl --fail --show-error --location --progress-bar \
        -o "$temp_dir/$asset" "$base/$asset"
    curl --fail --silent --show-error --location \
        -o "$temp_dir/$asset.sha256" "$base/$asset.sha256"

    status "Checking the download"
    (cd "$temp_dir" && shasum -a 256 -c "$asset.sha256" >/dev/null) ||
        fail "the download does not match its checksum"

    status "Installing bobcat into $install_dir"
    tar -xzf "$temp_dir/$asset" -C "$temp_dir"
    mkdir -p "$install_dir"
    install -m 755 "$temp_dir/bobcat" "$install_dir/bobcat"

    case ":$PATH:" in
    *":$install_dir:"*) ;;
    *)
        status "Add $install_dir to your PATH, for example with"
        echo "    echo 'export PATH=\"$install_dir:\$PATH\"' >> ~/.zshrc" >&2
        ;;
    esac
    status "Installed. Try: bobcat chat -m lfm2.5:1.2b"
}

main "$@"
