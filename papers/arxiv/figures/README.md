# Ferrule paper protocol figures

Three restrained-color vector PDFs, built locally with the installed
pdfLaTeX/TikZ, xcolor and Latin Modern fonts. No new dependency, network access,
GPU query or model experiment is required. PNGs in `build/` are review
artifacts, not manuscript inputs. The existing `/papers/` ignore rule covers
these files; this figure pass did not force-add or commit them.

## Deliverables and build conventions

| Label | Editable TikZ source | Manuscript leaf / vector PDF |
| --- | --- | --- |
| `fig:architecture` | `architecture-diagram.tex` | `architecture.tex` / `architecture.pdf` |
| `fig:transaction-lifecycle` | `transaction-lifecycle-diagram.tex` | `transaction-lifecycle.tex` / `transaction-lifecycle.pdf` |
| `fig:operation-state` | `operation-state-diagram.tex` | `operation-state.tex` / `operation-state.pdf` |

Each existing `*-source.tex` wrapper loads `figure-driver.tex`, which crops a
single-page PDF. The palette and role styles are local to the three editable
diagrams; `figure-style.tex` continues to supply the common font and line
styles. No shared driver, style, Makefile, overview, manuscript, production or
experiment source was changed. Manuscript leaves use the generated PDFs, so
TikZ is needed only when regenerating them.

## Palette and print accessibility

All three diagrams repeat the same semantic mapping:

| Role | Color | Non-color encoding |
| --- | --- | --- |
| Control / planning / validation | Blue `#315E85` | Named plan, admission, journal and validation nodes |
| Storage / materialization | Ochre `#9A6B25` | Named read, staging, transfer and installation nodes |
| Execution / provider resources | Teal `#2E786D` | Named physical resources and compute nodes |
| Publication / custody / retirement | Violet `#6C588A` | Named live, retirement, quiescence and custody nodes |
| Unknown / failure | Brick `#A34F4A` | Explicit outcome labels; dashed Unknown border, solid known-failure border |

Dark neutral text sits on pale 8--10% tinted fills. Meaning is never encoded by
color alone: every node has a role/state label and every figure has labeled
swatches. Solid arrows denote dependencies or a known outcome; dashed arrows
denote evidence/owner-control gates. Unknown has no outgoing recovery edge.
The state abstraction does not introduce a separate storage-credit state;
its storage swatch documents the same cross-figure palette.

Connectors are drawn on TikZ's existing background layer, behind opaque nodes
and label backplates. Named anchors, orthogonal lanes and reserved whitespace
keep the physical path, publication gates and failure/Unknown branches apart.
All diagrams retain vector text and lines in the PDFs.

## Semantic scope

These are conceptual, profile-scoped boundaries, not one implementation enum
or a universal runner diagram:

- Source catalogs are immutable metadata, not residency registries or waiter
  owners. Covered-domain admission does not hard-cap OS page caches or remote
  storage-server caches.
- Physical providers retain their resources; the appropriate scoped owner
  controls publication and custody. CUDA is a concrete provider instance, not
  the hardware-neutral design boundary. No default NCCL/device-collective
  route is drawn.
- The selected lifecycle journals provisional ownership **before** backend
  preparation can submit physical work, even if no handle is returned.
  Candidate readiness is a later validation gate, not journal creation.
- The lifecycle explicitly includes candidate preflight, prepared-KV
  installation and exact generation-matched ACK before publication. This ACK
  gate applies to the selected prepared-KV protocol, not every publication
  path; the diagram is not a hardware-overlap timeline.
- Validated success is distinct from known non-success. Exact operation, key,
  stage, bytes, generations and the applicable retained fence must match.
  Stale/mismatched events cannot mutate newer state or return its credits.
- Failure/cancel follows drain, not new publication; cancellation preserves any
  irreversible ending intent. Producer readiness is not consumer retirement.
  Only relevant final fences and scoped quiescence justify custody return;
  explicit release or ownership transfer is still required for claims.
- Physical-owner Unknown retains payloads, pins, slots and claims and remains
  sticky. A later exact completion may close a registry-stage obligation
  without healing a separately poisoned owner. Idle leases do not make live
  resident bytes free, and quiescence alone is not a credit refund.
- In-memory owner journals imply neither durable database ACID/crash recovery,
  global commit nor general cross-rank atomicity. No universal registry owns
  all materialization, execution, KV and pipeline protocols.

Grounding consulted for the figure pass: `docs/ARCHITECTURE.md`,
`crates/ferrule-common/src/io_protocol.rs`,
`crates/ferrule-runtime/src/io/registry.rs`,
`crates/ferrule-runtime/src/parallel/pipeline/owner.rs`, and
`crates/ferrule-backend/src/cuda/operators/kv/transaction/{prepared,chunked_protocol}.rs`.

## Reproduce and review (CPU only)

From the repository root:

```sh
make -C papers/arxiv/figures all preview render
```

The existing Makefile disables shell escape, rebuilds the three cropped PDFs,
and compiles `build/figure-preview.pdf` with the actual manuscript leaves,
captions and labels at paper text width. `render` uses installed Ghostscript
to create the three color review PNGs. The unchanged preview harness still calls the earlier style "monochrome" in
its introductory prose; the actual generated figures use the color design
specified here. Its captions are the actual manuscript leaves.

This command was run with a finite 120-second timeout. The full manuscript
was not compiled or edited by this figure pass; manuscript pagination,
algorithm layouts and production/GPU correctness were not validated.

Review artifacts:

- `build/architecture-preview.png`
- `build/transaction-lifecycle-preview.png`
- `build/operation-state-preview.png`
- `build/figure-preview.pdf`: figure-only paper-width integration preview
- `build/integration-preview-2.png` through `build/integration-preview-4.png`:
  paper-width leaves and captions
- `build/architecture-grayscale.png`, `build/transaction-lifecycle-grayscale.png`
  and `build/operation-state-grayscale.png`: print-oriented review renders

## Final review record

- Changed figure sources: `architecture-diagram.tex`,
  `transaction-lifecycle-diagram.tex`, and `operation-state-diagram.tex`.
  Changed leaves/captions: `architecture.tex`, `transaction-lifecycle.tex`,
  and `operation-state.tex`. Changed figure documentation: this `README.md`.
  The existing `*-source.tex` wrappers and all shared build/preview sources
  remain unchanged. No `main.tex`, production code, manuscript section,
  experiment or Makefile was edited.
- Generated PDFs: `architecture.pdf`, `transaction-lifecycle.pdf`, and
  `operation-state.pdf`. Rendered standalone previews are
  `build/architecture-preview.png`, `build/transaction-lifecycle-preview.png`,
  and `build/operation-state-preview.png`.
- The finite `make -C papers/arxiv/figures all preview render` command completed
  successfully. Final standalone and integration-preview logs contain no
  LaTeX/package warnings, overfull/underfull boxes, undefined references or
  errors. Ghostscript generated the three color previews, three grayscale
  checks, and the paper-width integration pages. Visual review found no
  clipping, connector/text occlusion, or crowded caption label.
- Review-only grayscale checks are
  `build/architecture-grayscale.png`,
  `build/transaction-lifecycle-grayscale.png`, and
  `build/operation-state-grayscale.png`. These verify that labels, dashed
  borders and line styles preserve semantic distinctions without relying on
  color alone.
