#!/bin/bash
# This script should run `rustfmt` on all changed rust files in the commit

set -eu -o pipefail

find_changed_files() {
    local root_dir="$(git rev-parse --show-toplevel)"
    local pattern='\.rs$'
    # filter=ACMR shows only added, changed, modified, or renamed files.
    # Get only pattern matching files and prepend the root directory to make the paths absolute.
    echo "$( git diff --cached --name-only --diff-filter=ACMR | grep -E "${pattern}" | sed "s:^:${root_dir}/:" )"
}

find_and_format_changed_files() {
    local changed_files="$(find_changed_files)"
    # If we have changed files, format them!
    if [[ -n "$changed_files" ]]; then
        rustfmt $changed_files
        git add $changed_files
    fi
}

find_and_format_changed_files