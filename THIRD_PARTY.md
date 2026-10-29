# Third-party provenance

本文件记录 Ferrule 的第三方来源边界，不构成法律意见。

## 当前状态

本次 DeepInfer 参考阶段未将 `target/DeepInfer` 源码、vendor 内容或其二进制加入 Ferrule 的 tracked 源码或发布物。`target/DeepInfer` 下既有或被 Git 忽略的 third-party artifacts 另按各自 provenance 管理；本记录不声称整个 checkout 不含 third-party。

Ferrule 的相关实现是独立的 Rust clean-room adaptation；DeepInfer 仅作为 behavior reference。由于本阶段没有向 Ferrule tracked 源码或发布物加入第三方内容，本阶段不修改 `LICENSE` 或 `NOTICE`，也不新增具体第三方版权归属，以免误称已经分发第三方代码。

## 参考来源与本地审查

- canonical URL（本地 clone 来源记录）：`https://github.com/deepseek-ai/DeepInfer`
- 参考 commit：`553741557a57df44a46fb3ab72b6aafeaf945337`；这是**本地浅克隆记录的参考commit**，对应本地 `target/DeepInfer`。
- 许可证：Apache-2.0；依据本地浅克隆审查的 `target/DeepInfer/LICENSE` 与 `target/DeepInfer/pyproject.toml`（`project.license` 和 `license-files`）。

本记录不声明当前远程可解析性，不提供 commit 的远程链接。标注为 vLLM-derived 的文件，以及尚未初始化的 vendor 子模块，均需分别检查其上游 commit、`NOTICE`、许可证和归属要求；不得仅依据 DeepInfer 的顶层许可推定这些内容可直接复制或分发。

详细的架构映射、numeric 支持矩阵和能力边界见 [`docs/provenance/deepinfer.md`](docs/provenance/deepinfer.md)。

## 后续待办

若未来直接复制标注为 Derived from vLLM 的文件或 vendor 内容，必须在复制前完成上游 commit、`NOTICE`、许可证和归属核对，并保留所需声明。本阶段不作性能或支持承诺。
