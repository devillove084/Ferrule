# Repository instructions

## Commit messages

Write short, clear Git commit messages that summarize the change.

- Use the imperative mood in the subject.
- Capitalize the subject.
- Do not end the subject with punctuation.
- Keep the subject near 50 characters when practical.
- Use only the subject when it fully explains the change.
- Add a body only when it provides useful context that is not already in the subject.
- Separate the body from the subject with a blank line.
- Wrap body lines at 72 characters.
- Do not include raw diff output or meta-commentary.

## Repository inspection

Delegate all viewing, reading, searching, and investigation of repository
contents to subagents. The main agent should coordinate the work, synthesize
subagent findings, and implement changes instead of inspecting repository
contents directly.

- The main agent should not use `read_file`, `grep`, `find_path`,
  `list_directory`, or `fetch` to inspect content.
- Give subagents focused scopes and ask them to return the evidence needed for
  implementation.

## Local inspection

Subagents inspecting local content should use the terminal with `rg` and
`rg --files` first, adding local tools such as `sed` when useful. Do not default
to editor search or file-viewing tools when these terminal commands are
sufficient.

## Remote research

When practical, download remote resources to temporary local files and inspect
them with `rg` and other local tools. Keep temporary research files out of
version control. Use `fetch` directly only when downloading is unsuitable, and
state why direct fetching is necessary.

## Validation

Do not call `diagnostics` mechanically after routine changes or invoke it
repeatedly. Prefer the narrowest relevant tests, builds, formatting checks, and
linters, such as `cargo test`, `cargo check`, `cargo fmt`, or `cargo clippy`.
Use `diagnostics` only when the user explicitly requests it or when command-line
tests and builds cannot locate a suspected editor or LSP issue. This guidance
does not prohibit necessary validation; choose checks that are proportionate
to the change.
