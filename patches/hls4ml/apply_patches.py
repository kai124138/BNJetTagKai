#!/usr/bin/env python3
"""Apply the three required hls4ml LayerNormalization patches to a local clone.

The patches are documented as before/after snippets in the .md files in this
directory. Line numbers drift across upstream hls4ml versions, so this applier
uses anchored string replacement instead of a line-based diff. Each change is
idempotent: running twice is a no-op and reports "already applied".

Usage:
    python patches/hls4ml/apply_patches.py --hls4ml-root software/hls4ml
    # then reinstall:
    cd software/hls4ml && pip install -e .

What it does (see the .md files for the full rationale):
  1. nnet_layernorm.h     : table_range_power2 unsigned -> int; UB bit-shift ->
                            float pow() for the inverse-sqrt LUT index.
  2. core_templates.py    : emit `int` instead of `unsigned` for table_range_power2.
  3. model/layers.py      : plumb table_t through the Precision config dict by
                            adding TypeAttribute('table_t') + _set_type_t('table').
"""
import argparse
import sys
from pathlib import Path


class PatchError(RuntimeError):
    pass


def _read(path: Path) -> str:
    if not path.is_file():
        raise PatchError(f"file not found: {path}")
    return path.read_text()


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count == 0:
        raise PatchError(f"anchor not found in {where}:\n    {old!r}")
    if count > 1:
        raise PatchError(f"anchor is ambiguous ({count}x) in {where}:\n    {old!r}")
    return text.replace(old, new)


def patch_nnet_layernorm(root: Path) -> str:
    path = root / "hls4ml" / "templates" / "vivado" / "nnet_utils" / "nnet_layernorm.h"
    if not path.is_file():  # some layouts omit the leading hls4ml/ package dir
        path = root / "templates" / "vivado" / "nnet_utils" / "nnet_layernorm.h"
    text = _read(path)

    if "static const int table_range_power2" in text and \
       "pow(2.0f, (float)(int)CONFIG_T::table_range_power2)" in text:
        return f"  [skip] {path.name}: already patched"

    # Change 1: unsigned -> int on the config field
    if "static const unsigned table_range_power2" in text:
        text = text.replace(
            "static const unsigned table_range_power2",
            "static const int table_range_power2",
        )

    # Change 2: UB bit-shift inverse-range -> float pow
    if "unsigned inv_range_inv = 1 << (-CONFIG_T::table_range_power2);" in text:
        text = text.replace(
            "unsigned inv_range_inv = 1 << (-CONFIG_T::table_range_power2);",
            "float inv_range_inv = pow(2.0f, (float)(int)CONFIG_T::table_range_power2);",
        )

    # Change 3: integer-shift index -> float multiply using inv_range_inv
    if "(var * CONFIG_T::table_size) >> (-CONFIG_T::table_range_power2)" in text:
        text = text.replace(
            "int index = (var * CONFIG_T::table_size) >> (-CONFIG_T::table_range_power2);",
            "int index = (float)(var) * (float)(CONFIG_T::table_size) * inv_range_inv;",
        )

    path.write_text(text)
    return f"  [ok]   {path.name}: table_range_power2 -> int, LUT index -> float pow"


def patch_core_templates(root: Path) -> str:
    path = root / "hls4ml" / "backends" / "vivado" / "passes" / "core_templates.py"
    if not path.is_file():
        path = root / "backends" / "vivado" / "passes" / "core_templates.py"
    text = _read(path)

    if "static const int table_range_power2 = {table_range_power2};" in text:
        return f"  [skip] {path.name}: already patched"

    text = _replace_once(
        text,
        "static const unsigned table_range_power2 = {table_range_power2};",
        "static const int table_range_power2 = {table_range_power2};  // negative = larger variance range",
        path.name,
    )
    path.write_text(text)
    return f"  [ok]   {path.name}: emit int for table_range_power2"


def patch_layers(root: Path) -> str:
    path = root / "hls4ml" / "model" / "layers.py"
    if not path.is_file():
        path = root / "model" / "layers.py"
    text = _read(path)

    already_attr = "TypeAttribute('table_t')" in text or 'TypeAttribute("table_t")' in text
    already_set = "_set_type_t('table')" in text or '_set_type_t("table")' in text
    if already_attr and already_set:
        return f"  [skip] {path.name}: already patched"

    # Locate the LayerNormalization class block to keep edits scoped to it.
    anchor = "class LayerNormalization(Layer):"
    if anchor not in text:
        raise PatchError(f"{path.name}: could not find `{anchor}`")

    # Add TypeAttribute('table_t') right after the bias TypeAttribute inside the
    # LayerNormalization _expected_attributes list. We find the first
    # "TypeAttribute('bias')" / 'TypeAttribute("bias")' that occurs after the class.
    cls_idx = text.index(anchor)
    head, tail = text[:cls_idx], text[cls_idx:]

    if not already_attr:
        for bias_attr in ("TypeAttribute('bias'),", 'TypeAttribute("bias"),'):
            if bias_attr in tail:
                # indentation of that line
                line_start = tail.rindex("\n", 0, tail.index(bias_attr)) + 1
                indent = tail[line_start:tail.index(bias_attr)]
                tail = tail.replace(
                    bias_attr,
                    bias_attr + "\n" + indent + "TypeAttribute('table_t'),  # patched: configurable LN inv-sqrt LUT precision",
                    1,
                )
                break
        else:
            raise PatchError(
                f"{path.name}: could not find TypeAttribute('bias') in LayerNormalization "
                "to anchor the table_t attribute insertion"
            )

    if not already_set:
        # Add self._set_type_t('table') after the bias weights variable is added
        # inside initialize(). Anchor on the bias add_weights_variable call.
        for bias_w in (
            "self.add_weights_variable(name='bias'",
            'self.add_weights_variable(name="bias"',
        ):
            if bias_w in tail:
                idx = tail.index(bias_w)
                line_end = tail.index("\n", idx)
                line_start = tail.rindex("\n", 0, idx) + 1
                indent = tail[line_start:idx]
                insertion = "\n" + indent + "self._set_type_t('table')  # patched: honor configured LN table precision"
                tail = tail[:line_end] + insertion + tail[line_end:]
                break
        else:
            raise PatchError(
                f"{path.name}: could not find bias add_weights_variable in "
                "LayerNormalization.initialize() to anchor _set_type_t('table')"
            )

    text = head + tail
    path.write_text(text)
    return f"  [ok]   {path.name}: table_t TypeAttribute + _set_type_t('table')"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hls4ml-root", default="software/hls4ml",
                    help="path to the local hls4ml clone (default: software/hls4ml)")
    args = ap.parse_args()

    root = Path(args.hls4ml_root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: --hls4ml-root does not exist: {root}", file=sys.stderr)
        return 2

    print(f"Applying LayerNormalization patches to: {root}")
    try:
        for fn in (patch_nnet_layernorm, patch_core_templates, patch_layers):
            print(fn(root))
    except PatchError as e:
        print(f"\nPATCH FAILED: {e}", file=sys.stderr)
        print("Upstream code may have moved. See the .md files in this directory "
              "for the manual edits.", file=sys.stderr)
        return 1

    print("\nAll patches applied. Now reinstall hls4ml:")
    print(f"    cd {root} && pip install -e .")
    print("\nVerify after conversion with:")
    print('    grep "_table_t" models/hls4ml_deepsets_v2/firmware/defines.h')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
