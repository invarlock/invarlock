#!/usr/bin/env python3
"""Exercise installed hardened Accelerate with temporary CPU checkpoints.

These small synthetic fixtures test loading behavior, not model quality or GPU
qualification. Run with the candidate wheel installed in an isolated environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXPECTED_VERSION = "1.14.0+invarlock.1"


class InstalledAccelerateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import accelerate
        import accelerate.utils.modeling as modeling
        import torch
        from safetensors.torch import save_file

        if importlib.metadata.version("accelerate") != EXPECTED_VERSION:
            raise RuntimeError(f"Install accelerate=={EXPECTED_VERSION} first")
        if accelerate.__version__ != EXPECTED_VERSION:
            raise RuntimeError("Imported Accelerate version differs from metadata")
        if (
            not Path(accelerate.__file__)
            .resolve()
            .is_relative_to(Path(sys.prefix).resolve())
        ):
            raise RuntimeError("Accelerate must be installed inside this environment")
        cls.accelerate = accelerate
        cls.modeling = modeling
        cls.torch = torch
        cls.save_file = staticmethod(save_file)
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="accelerate-installed-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = {
            "weight": self.torch.arange(6, dtype=self.torch.float32).reshape(2, 3),
            "bias": self.torch.tensor([2.0, 3.0]),
        }

    def save(self, path, state=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state if state is None else state
        if path.suffix == ".safetensors":
            self.save_file(state, str(path), metadata={"format": "pt"})
        else:
            self.torch.save(state, path)
        return path

    def index(self, root, mapping):
        root.mkdir(parents=True, exist_ok=True)
        index = root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
        return index

    def load(self, loader, path):
        if loader == "state":
            return self.modeling.load_state_dict(str(path))
        model = self.torch.nn.Linear(3, 2)
        if loader == "model":
            self.accelerate.load_checkpoint_in_model(
                model, str(path), device_map={"": "cpu"}, strict=True
            )
        else:
            model = self.accelerate.load_checkpoint_and_dispatch(
                model, str(path), device_map={"": "cpu"}, strict=True
            )
        return model.state_dict()

    def equal_state(self, state):
        self.assertEqual(set(state), set(self.state))
        for name, expected in self.state.items():
            self.torch.testing.assert_close(state[name], expected, rtol=0, atol=0)

    def reject(self, loader, path):
        # Unsafe input must fail before any tensor deserializer sees its bytes.
        with patch.object(
            self.modeling,
            "_load_state_dict_from_file",
            side_effect=AssertionError("unsafe input reached deserializer"),
        ):
            with self.assertRaises((ValueError, OSError)):
                self.load(loader, path)

    def test_regular_single_files(self):
        for suffix in ("bin", "safetensors"):
            path = self.save(self.root / f"state.{suffix}")
            for loader in ("state", "model", "dispatch"):
                with self.subTest(format=suffix, loader=loader):
                    self.equal_state(self.load(loader, path))

    def test_regular_single_directories(self):
        for filename in ("pytorch_model.bin", "model.safetensors"):
            root = self.root / filename.replace(".", "-")
            self.save(root / filename)
            for loader in ("model", "dispatch"):
                with self.subTest(filename=filename, loader=loader):
                    self.equal_state(self.load(loader, root))

    def test_regular_shards_and_nested_shards(self):
        for suffix in ("bin", "safetensors"):
            for nested in (False, True):
                root = self.root / f"{suffix}-{nested}"
                prefix = "nested/" if nested else ""
                mapping = {key: f"{prefix}{key}.{suffix}" for key in self.state}
                for key, value in self.state.items():
                    self.save(root / mapping[key], {key: value})
                index = self.index(root, mapping)
                for loader in ("model", "dispatch"):
                    for source in (root, index):
                        with self.subTest(
                            format=suffix,
                            nested=nested,
                            loader=loader,
                            source=source.name,
                        ):
                            self.equal_state(self.load(loader, source))

    def test_unsafe_index_members(self):
        outside = self.save(self.root / "outside.bin")
        cases = (
            "../outside.bin",
            str(outside),
            "nested/../../outside.bin",
            "nested/../weight.bin",
            "C:\\outside.bin",
            "nested\\outside.bin",
            "",
            ".",
            "nested//weight.bin",
        )
        for number, member in enumerate(cases):
            root = self.root / f"case-{number}"
            index = self.index(root, dict.fromkeys(self.state, member))
            for loader in ("model", "dispatch"):
                for source in (root, index):
                    with self.subTest(member=member, loader=loader, source=source.name):
                        self.reject(loader, source)

    def test_symlink_files_directories_and_indexes(self):
        real = self.root / "real"
        self.save(real / "model.safetensors")
        link = self.root / "linked.safetensors"
        link.symlink_to(real / "model.safetensors")
        parent = self.root / "linked-parent"
        parent.symlink_to(real, target_is_directory=True)
        for loader in ("state", "model", "dispatch"):
            with self.subTest(loader=loader, source=str(link)):
                self.reject(loader, link)
            # The caller selects the parent; ancestor links such as macOS /tmp
            # are allowed. The final entry and index descendants are protected.
            self.equal_state(self.load(loader, parent / "model.safetensors"))
        for loader in ("model", "dispatch"):
            self.reject(loader, parent)
        indexed = self.root / "indexed"
        index = self.index(indexed, dict.fromkeys(self.state, "weights.safetensors"))
        self.save(indexed / "weights.safetensors")
        saved = index.with_suffix(".saved")
        index.rename(saved)
        index.symlink_to(saved)
        for loader in ("model", "dispatch"):
            for source in (index, indexed):
                self.reject(loader, source)

    def test_symlink_shards_and_nested_escape(self):
        outside = self.root / "outside"
        self.save(outside / "weights.bin")
        for nested in (False, True):
            root = self.root / f"shards-{nested}"
            name = "linked/weights.bin" if nested else "linked.bin"
            index = self.index(root, dict.fromkeys(self.state, name))
            (root / ("linked" if nested else "linked.bin")).symlink_to(
                outside if nested else outside / "weights.bin",
                target_is_directory=nested,
            )
            for loader in ("model", "dispatch"):
                for source in (root, index):
                    self.reject(loader, source)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX named pipes")
    def test_fifo_and_directory_members(self):
        for kind in ("fifo", "directory"):
            direct = self.root / f"{kind}.bin"
            if kind == "fifo":
                os.mkfifo(direct)
            else:
                direct.mkdir()
            for loader in ("state", "model", "dispatch"):
                self.reject(loader, direct)
            root = self.root / kind
            index = self.index(root, dict.fromkeys(self.state, "shard.bin"))
            if kind == "fifo":
                os.mkfifo(root / "shard.bin")
            else:
                (root / "shard.bin").mkdir()
            for loader in ("model", "dispatch"):
                for source in (root, index):
                    self.reject(loader, source)
        root = self.root / "fifo-index"
        root.mkdir()
        index = root / "model.safetensors.index.json"
        os.mkfifo(index)
        for loader in ("model", "dispatch"):
            for source in (root, index):
                self.reject(loader, source)

    def test_open_file_survives_late_replacement(self):
        original = self.modeling._load_state_dict_from_file
        for suffix in ("bin", "safetensors"):
            for loader in ("state", "model", "dispatch"):
                for replacement_kind in ("regular", "symlink", "fifo"):
                    if replacement_kind == "fifo" and not hasattr(os, "mkfifo"):
                        continue
                    root = self.root / f"{suffix}-{loader}-{replacement_kind}"
                    path = self.save(root / f"weights.{suffix}")
                    calls = []

                    def replaced(
                        *args,
                        path=path,
                        root=root,
                        calls=calls,
                        replacement_kind=replacement_kind,
                        **kwargs,
                    ):
                        calls.append(args[0])
                        path.rename(root / "original.saved")
                        if replacement_kind == "regular":
                            path.write_bytes(b"replacement must not be read")
                        elif replacement_kind == "symlink":
                            path.symlink_to(root / "missing")
                        else:
                            os.mkfifo(path)
                        return original(*args, **kwargs)

                    with self.subTest(
                        format=suffix, loader=loader, replacement=replacement_kind
                    ):
                        with patch.object(
                            self.modeling,
                            "_load_state_dict_from_file",
                            side_effect=replaced,
                        ):
                            self.equal_state(self.load(loader, path))
                        self.assertEqual(len(calls), 1)
                        self.assertNotEqual(str(calls[0]), str(path))

    def test_open_directory_survives_late_replacement(self):
        original = self.modeling.load_state_dict
        for loader in ("model", "dispatch"):
            for suffix in ("bin", "safetensors"):
                root = self.root / f"{loader}-{suffix}"
                mapping = {key: f"{key}.{suffix}" for key in self.state}
                for key, value in self.state.items():
                    self.save(root / mapping[key], {key: value})
                self.index(root, mapping)
                calls = []

                def replaced(*args, root=root, calls=calls, mapping=mapping, **kwargs):
                    if not calls:
                        root.rename(root.with_name(root.name + "-saved"))
                        root.mkdir()
                        for filename in mapping.values():
                            (root / filename).write_bytes(
                                b"replacement must not be read"
                            )
                    calls.append(args[0])
                    return original(*args, **kwargs)

                with self.subTest(loader=loader, format=suffix):
                    with patch.object(
                        self.modeling, "load_state_dict", side_effect=replaced
                    ):
                        self.equal_state(self.load(loader, root))
                    self.assertEqual(len(calls), 2)

    def test_deserializer_failure_closes_descriptors(self):
        fd_root = Path("/proc/self/fd")
        if not fd_root.is_dir():
            fd_root = Path("/dev/fd")
        if not fd_root.is_dir():
            self.skipTest("descriptor inventory unavailable")
        path = self.save(self.root / "weights.bin")
        for loader in ("state", "model", "dispatch"):
            before = set(fd_root.iterdir())
            for _ in range(20):
                with patch.object(
                    self.modeling,
                    "_load_state_dict_from_file",
                    side_effect=RuntimeError("deserializer failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "deserializer failed"):
                        self.load(loader, path)
            self.assertEqual(set(fd_root.iterdir()), before)

    def test_transformers_plain_local_save_reload(self):
        from transformers import BertConfig, BertModel
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer,
        )
        from transformers.utils import is_accelerate_available

        self.assertTrue(is_accelerate_available())
        self.assertTrue(FineGrainedFP8HfQuantizer.requires_calibration is False)
        config = BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        model = BertModel(config).eval()
        path = self.root / "transformers"
        model.save_pretrained(path)
        loaded = BertModel.from_pretrained(path, local_files_only=True).eval()
        inputs = self.torch.tensor([[1, 2, 3]])
        with self.torch.no_grad():
            self.torch.testing.assert_close(
                model(inputs).last_hidden_state,
                loaded(inputs).last_hidden_state,
                rtol=0,
                atol=0,
            )
        weights = path / "model.safetensors"
        materialized = self.root / "blob.safetensors"
        weights.rename(materialized)
        weights.symlink_to(materialized)
        linked = BertModel.from_pretrained(path, local_files_only=True).eval()
        with self.torch.no_grad():
            self.torch.testing.assert_close(
                model(inputs).last_hidden_state,
                linked(inputs).last_hidden_state,
                rtol=0,
                atol=0,
            )

    def test_peft_local_adapter_save_reload(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import BertConfig, BertModel

        config = BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        base = BertModel(config)
        base_path = self.root / "base"
        base.save_pretrained(base_path)
        model = get_peft_model(
            base, LoraConfig(r=2, lora_alpha=2, target_modules=["query", "value"])
        )
        with self.torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.125)
        model.eval()
        adapter = self.root / "adapter"
        model.save_pretrained(adapter)
        loaded = PeftModel.from_pretrained(
            BertModel.from_pretrained(base_path, local_files_only=True),
            adapter,
            local_files_only=True,
        ).eval()
        inputs = self.torch.tensor([[1, 2, 3]])
        with self.torch.no_grad():
            self.torch.testing.assert_close(
                model(inputs).last_hidden_state,
                loaded(inputs).last_hidden_state,
                rtol=0,
                atol=0,
            )


def check_harness():
    """Compare a real local Harness likelihood against direct Torch inference."""
    import torch
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    InstalledAccelerateChecks.setUpClass()
    if importlib.metadata.version("lm_eval") != "0.4.12+invarlock.exactmatch.1":
        raise RuntimeError("Install the maintained cache-free lm_eval wheel first")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="accelerate-harness-") as temporary:
        root = Path(temporary)
        implementation = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
            )
        )
        implementation.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=implementation,
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[EOS]",
        )
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=4,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=1,
            )
        ).eval()
        model.save_pretrained(root)
        tokenizer.save_pretrained(root)
        harness = HFLM(
            pretrained=str(root),
            tokenizer=tokenizer,
            device="cpu",
            batch_size=1,
            local_files_only=True,
        )
        result = harness.loglikelihood(
            [
                Instance(
                    request_type="loglikelihood",
                    doc={},
                    arguments=("hello", " world"),
                    idx=0,
                )
            ]
        )
        with torch.no_grad():
            expected = model(torch.tensor([[2]])).logits[0, -1].log_softmax(-1)[3]
        if len(result) != 1:
            raise AssertionError("Harness returned the wrong number of scores")
        torch.testing.assert_close(
            torch.tensor(result[0][0]), expected, rtol=0, atol=1e-6
        )


def main(argv=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        choices=unittest.defaultTestLoader.getTestCaseNames(InstalledAccelerateChecks),
    )
    parser.add_argument(
        "--harness",
        action="store_true",
        help="Run the maintained Harness local CPU likelihood check",
    )
    args = parser.parse_args(argv)
    if args.harness and args.check:
        parser.error("--harness and --check are mutually exclusive")
    suite = (
        unittest.TestSuite([InstalledAccelerateChecks(args.check)])
        if args.check
        else unittest.defaultTestLoader.loadTestsFromTestCase(InstalledAccelerateChecks)
    )
    if args.harness:
        suite = unittest.TestSuite([unittest.FunctionTestCase(check_harness)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "accelerate": importlib.metadata.version("accelerate"),
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "scope": "temporary synthetic CPU checkpoints; no GPU qualification",
            },
            sort_keys=True,
        )
    )
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
