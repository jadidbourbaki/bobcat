#!/bin/sh
# install.sh installs gip on a Mac with Apple silicon.
#
#   curl -fsSL https://raw.githubusercontent.com/jadidbourbaki/gip/main/install.sh | sh
#
# The script downloads the gip binary from the latest GitHub release,
# checks its SHA-256 checksum, and installs it into ~/.local/bin.
#
# GIP_VERSION selects a release tag such as v0.1.0 in place of the
# latest release. GIP_INSTALL_DIR selects the install directory.

# The whole script runs inside main, so a download cut off partway runs
# nothing.
main() {
    set -eu

    repo="jadidbourbaki/gip"
    asset="gip-aarch64-apple-darwin.tar.gz"
    install_dir="${GIP_INSTALL_DIR:-$HOME/.local/bin}"

    status() { echo ">>> $*" >&2; }
    fail() {
        echo "gip install: $*" >&2
        exit 1
    }

    [ "$(uname -s)" = "Darwin" ] || fail "gip runs on macOS only"
    [ "$(uname -m)" = "arm64" ] || fail "gip needs a Mac with Apple silicon"
    macos_major="$(sw_vers -productVersion | cut -d. -f1)"
    [ "$macos_major" -ge 15 ] || fail "gip needs macOS 15 or newer"
    for tool in curl shasum tar; do
        command -v "$tool" >/dev/null || fail "gip needs $tool"
    done

    if [ -n "${GIP_VERSION:-}" ]; then
        base="https://github.com/$repo/releases/download/$GIP_VERSION"
    else
        base="https://github.com/$repo/releases/latest/download"
    fi

    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' EXIT

    status "Downloading gip ${GIP_VERSION:-(latest release)}"
    curl --fail --show-error --location --progress-bar \
        -o "$temp_dir/$asset" "$base/$asset"
    curl --fail --silent --show-error --location \
        -o "$temp_dir/$asset.sha256" "$base/$asset.sha256"

    status "Checking the download"
    (cd "$temp_dir" && shasum -a 256 -c "$asset.sha256" >/dev/null) ||
        fail "the download does not match its checksum"

    status "Installing gip into $install_dir"
    tar -xzf "$temp_dir/$asset" -C "$temp_dir"
    mkdir -p "$install_dir"
    install -m 755 "$temp_dir/gip" "$install_dir/gip"

    case ":$PATH:" in
    *":$install_dir:"*) ;;
    *)
        status "Add $install_dir to your PATH, for example with"
        echo "    echo 'export PATH=\"$install_dir:\$PATH\"' >> ~/.zshrc" >&2
        ;;
    esac
    status "Installed. Try: gip chat -m lfm2.5:1.2b"
}

main "$@"
