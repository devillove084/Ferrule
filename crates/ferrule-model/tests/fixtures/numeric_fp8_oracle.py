"""Bounded local-checkpoint CPU oracle; no model load and no vLLM dependency.

Called by checkpoint_numeric_fp8::local_35b_expert_matches_torch_fp8_dequantization.
Only safetensors headers and one expert's intersecting weight/scale rectangles
are read. weight_scale_inv is a numeric multiplier, never a reciprocal.
"""
import json
import pathlib
import struct
import sys

import torch


def tensor_metadata(directory, index, name):
    path = directory / index[name]
    with path.open("rb") as file:
        header_bytes = struct.unpack("<Q", file.read(8))[0]
        assert header_bytes <= 16 * 1024 * 1024
        header = json.loads(file.read(header_bytes))
    tensor = header[name]
    begin, end = tensor["data_offsets"]
    return dict(
        name=name,
        path=str(path),
        offset=8 + header_bytes + begin,
        bytes=end - begin,
        dtype=tensor["dtype"],
        shape=tensor["shape"],
    )


def rectangle(tensor, rows, columns, element_bytes, dtype):
    width = tensor["shape"][1]
    data = bytearray()
    with open(tensor["path"], "rb") as file:
        for row in range(*rows):
            file.seek(tensor["offset"] + (row * width + columns[0]) * element_bytes)
            count = (columns[1] - columns[0]) * element_bytes
            payload = file.read(count)
            assert len(payload) == count
            data.extend(payload)
    return torch.frombuffer(data, dtype=dtype).reshape(
        rows[1] - rows[0], columns[1] - columns[0]
    )


def bits(tensor):
    return [int(value) & 0xFFFFFFFF for value in tensor.flatten().view(torch.int32)]


def main():
    torch.set_num_threads(1)
    directory = pathlib.Path(sys.argv[1]).resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
    weight = tensor_metadata(directory, index, name)
    scale = tensor_metadata(directory, index, name.removesuffix(".weight") + ".weight_scale_inv")
    assert weight["dtype"] == "F8_E4M3"
    assert scale["dtype"] in ("BF16", "F32")
    assert scale["shape"] == [(dim + 127) // 128 for dim in weight["shape"]]
    rows, columns = [123, 260], [125, 259]
    assert rows[1] <= weight["shape"][0] and columns[1] <= weight["shape"][1]
    scale_rows = [rows[0] // 128, (rows[1] + 127) // 128]
    scale_columns = [columns[0] // 128, (columns[1] + 127) // 128]
    scale_bytes, scale_dtype = (2, torch.bfloat16) if scale["dtype"] == "BF16" else (4, torch.float32)
    w = rectangle(weight, rows, columns, 1, torch.float8_e4m3fn).float()
    s = rectangle(scale, scale_rows, scale_columns, scale_bytes, scale_dtype).float()
    assert torch.isfinite(w).all() and torch.isfinite(s).all() and (s > 0).all()
    expanded = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    row_origin, col_origin = rows[0] % 128, columns[0] % 128
    result = w * expanded[
        row_origin : row_origin + w.shape[0], col_origin : col_origin + w.shape[1]
    ]
    assert torch.isfinite(result).all()
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    print(json.dumps(dict(
        weight=weight, scale=scale, rows=rows, columns=columns,
        read_bytes=w.numel() + s.numel() * scale_bytes,
        expected_bits=bits(result), e4m3_bits=bits(table), torch_version=torch.__version__,
    )))


if __name__ == "__main__":
    main()
