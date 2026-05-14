# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Sequence

try:
    from .config import PrismalWaveConfig
except ImportError:  # pragma: no cover - supports direct script-style launching.
    from config import PrismalWaveConfig


ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"
DATASETS_DIR = ROOT_DIR / "Datasets"
DEFAULT_DATASET_DIR = Path.home()
PROCESS_GRACE_SECONDS = 5.0
DEFAULT_CFG = PrismalWaveConfig()


def _decimal_str(value: float) -> str:
    text = format(float(value), ".10f").rstrip("0").rstrip(".")
    return text if text else "0"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "run"


def _join_cmd(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _split_extra_args(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = base.parent / f"{base.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _source_name(path_text: str) -> str:
    if not path_text.strip():
        return "dataset"
    path = Path(path_text.strip())
    if path.name.lower() == "model.pt":
        return path.parent.name or path.stem
    if path.is_dir() and path.name.lower() in {"data", "dataset", "datasets"} and path.parent.name:
        return path.parent.name
    return path.stem or path.name or "dataset"


def _default_post_output(save_dir: Path) -> str:
    return str(save_dir / "prompt.txt")


@dataclass
class _RunSpec:
    command: list[str]
    capture_output: bool = False


class PrismalWaveGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EPIC-SHARC MOHTE Trainer")
        self.geometry("1200x860")
        self.minsize(980, 720)

        self._process: subprocess.Popen[str] | None = None
        self._process_thread: threading.Thread | None = None
        self._stop_thread: threading.Thread | None = None
        self._queue: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self._capture_output = False
        self._capture_lines: list[str] = []

        self.dataset_var = tk.StringVar(value="")
        self.resume_var = tk.StringVar(value="")
        self.save_dir_var = tk.StringVar(value="")
        self.epochs_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="200")
        self.batch_var = tk.StringVar(value="2")
        self.max_samples_var = tk.StringVar(value=str(DEFAULT_CFG.max_samples))
        self.dataset_streaming_var = tk.BooleanVar(value=True)
        self.tokenizer_workers_var = tk.StringVar(value="1")
        self.lr_var = tk.StringVar(value=_decimal_str(DEFAULT_CFG.lr))
        self.optimizer_var = tk.StringVar(value=DEFAULT_CFG.optimizer)
        self.muon_lr_var = tk.StringVar(value=str(DEFAULT_CFG.muon_lr))
        self.muon_weight_decay_var = tk.StringVar(value=str(DEFAULT_CFG.muon_weight_decay))
        self.muon_momentum_beta_var = tk.StringVar(value=str(DEFAULT_CFG.muon_momentum_beta))
        self.muon_ns_steps_var = tk.StringVar(value=str(DEFAULT_CFG.muon_ns_steps))
        self.muon_extra_scale_factor_var = tk.StringVar(value="0.8")
        self.muon_scalar_optimizer_var = tk.StringVar(value=DEFAULT_CFG.muon_scalar_optimizer)
        self.muon_nesterov_var = tk.BooleanVar(value=DEFAULT_CFG.muon_nesterov)
        self.val_fraction_var = tk.StringVar(value="0.1")
        self.min_token_frequency_var = tk.StringVar(value="2")
        self.post_prompt_var = tk.StringVar(value="What is a cat?")
        self.post_max_new_tokens_var = tk.StringVar(value="80")
        self.post_min_new_tokens_var = tk.StringVar(value="1")
        self.post_temperature_var = tk.StringVar(value="0.15")
        self.post_top_k_var = tk.StringVar(value="8")
        self.post_top_p_var = tk.StringVar(value="0.92")
        self.post_repetition_penalty_var = tk.StringVar(value="1.1")
        self.post_no_repeat_ngram_size_var = tk.StringVar(value="4")
        self.post_beam_size_var = tk.StringVar(value="2")
        self.extra_train_args_var = tk.StringVar(value="")
        self.extra_infer_args_var = tk.StringVar(value="")
        self.torch_compile_var = tk.BooleanVar(value=False)
        self.amp_var = tk.BooleanVar(value=True)
        self.use_bitsandbytes_leaf_precision_var = tk.BooleanVar(value=DEFAULT_CFG.use_bitsandbytes_leaf_precision)
        self.torus_local_field_radius_var = tk.StringVar(value=str(DEFAULT_CFG.torus_local_field_radius))
        self.torus_global_bus_slots_var = tk.StringVar(value=str(DEFAULT_CFG.torus_global_bus_slots))
        self.torus_global_bus_decay_var = tk.StringVar(value=str(DEFAULT_CFG.torus_global_bus_decay))
        self.torus_global_bus_write_scale_var = tk.StringVar(value=str(DEFAULT_CFG.torus_global_bus_write_scale))
        self.torus_sharc_router_var = tk.BooleanVar(value=DEFAULT_CFG.Torus_SHARC_Router)
        self.use_signature_lattice_attention_var = tk.BooleanVar(value=DEFAULT_CFG.use_signature_lattice_attention)
        self.signature_lattice_dim_var = tk.StringVar(value=str(DEFAULT_CFG.signature_lattice_dim))
        self.signature_lattice_buckets_var = tk.StringVar(value=str(DEFAULT_CFG.signature_lattice_buckets))
        self.signature_lattice_candidates_var = tk.StringVar(value=str(DEFAULT_CFG.signature_lattice_candidates))
        self.signature_lattice_weight_var = tk.StringVar(value=str(DEFAULT_CFG.signature_lattice_weight))
        self.signature_lattice_decay_var = tk.StringVar(value=str(DEFAULT_CFG.signature_lattice_decay))
        self.use_signature_lattice_generation_cache_var = tk.BooleanVar(value=DEFAULT_CFG.use_signature_lattice_generation_cache)
        self.use_topk_mot_var = tk.BooleanVar(value=DEFAULT_CFG.use_topk_mot)
        self.mot_top_k_var = tk.StringVar(value=str(DEFAULT_CFG.mot_top_k))
        self.infer_checkpoint_var = tk.StringVar(value="")
        self.infer_prompt_var = tk.StringVar(value="What is a cat?")
        self.infer_max_new_tokens_var = tk.StringVar(value="64")
        self.infer_min_new_tokens_var = tk.StringVar(value="1")
        self.infer_temperature_var = tk.StringVar(value="0.9")
        self.infer_top_k_var = tk.StringVar(value="8")
        self.infer_top_p_var = tk.StringVar(value="0.92")
        self.infer_repetition_penalty_var = tk.StringVar(value="1.1")
        self.infer_no_repeat_ngram_size_var = tk.StringVar(value="4")
        self.infer_beam_size_var = tk.StringVar(value="3")

        self._build_ui()
        self._bind_traces()
        self._refresh_run_dir()
        self.after(100, self._poll_queue)

    def _bind_traces(self) -> None:
        for var in (
            self.dataset_var,
            self.resume_var,
            self.epochs_var,
            self.steps_var,
            self.batch_var,
            self.max_samples_var,
            self.dataset_streaming_var,
        ):
            var.trace_add("write", lambda *_: self._refresh_run_dir())

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        content = ttk.Frame(main)
        log_panel = ttk.Frame(main)
        main.add(content, weight=4)
        main.add(log_panel, weight=1)

        notebook = ttk.Notebook(content)
        notebook.pack(fill="both", expand=True)

        self.train_tab = ttk.Frame(notebook, padding=10)
        self.infer_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.train_tab, text="Training")
        notebook.add(self.infer_tab, text="Inference")

        self._build_training_tab(self.train_tab)
        self._build_inference_tab(self.infer_tab)

        log_header = ttk.Frame(log_panel)
        log_header.pack(fill="x", pady=(0, 6))
        ttk.Label(log_header, text="Log", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(log_header, text="Clear Log", command=self._clear_log).pack(side="right")

        log_frame = ttk.Frame(log_panel)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", font=("Consolas", 10))
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.status_var = tk.StringVar(value="Idle")
        status = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", pady=(10, 0))

    def _build_training_tab(self, tab: ttk.Frame) -> None:
        top = ttk.Frame(tab)
        top.pack(fill="x")

        dataset_row = ttk.Frame(top)
        dataset_row.pack(fill="x", pady=4)
        ttk.Label(dataset_row, text="Dataset", width=18).pack(side="left")
        ttk.Entry(dataset_row, textvariable=self.dataset_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(dataset_row, text="Browse File", command=self._browse_dataset_file).pack(side="left")
        ttk.Button(dataset_row, text="Browse Folder", command=self._browse_dataset_folder).pack(side="left", padx=(6, 0))

        resume = self._path_row(top, "Checkpoint", self.resume_var, self._browse_checkpoint)
        resume.pack(fill="x", pady=4)

        save_row = ttk.Frame(top)
        save_row.pack(fill="x", pady=4)
        ttk.Label(save_row, text="Output dir", width=18).pack(side="left")
        save_entry = ttk.Entry(save_row, textvariable=self.save_dir_var, state="readonly")
        save_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(save_row, text="Refresh", command=self._refresh_run_dir).pack(side="left")

        params = ttk.LabelFrame(tab, text="Training settings", padding=8)
        params.pack(fill="x", pady=(10, 0))
        self._grid_params(
            params,
            [
                ("Epochs", self.epochs_var),
                ("Steps per epoch", self.steps_var),
                ("Batch size", self.batch_var),
                ("Max samples", self.max_samples_var),
                ("Tokenizer workers", self.tokenizer_workers_var),
                ("LR", self.lr_var),
                ("Val fraction", self.val_fraction_var),
                ("Min token freq", self.min_token_frequency_var),
            ],
        )
        ttk.Checkbutton(
            params,
            text="Dataset Streaming",
            variable=self.dataset_streaming_var,
        ).grid(row=3, column=0, columnspan=6, sticky="w", pady=(6, 0))

        opt_box = ttk.LabelFrame(tab, text="Optimizer", padding=8)
        opt_box.pack(fill="x", pady=(10, 0))
        opt_top = ttk.Frame(opt_box)
        opt_top.pack(fill="x")
        ttk.Label(opt_top, text="Optimizer", width=18).pack(side="left")
        opt_choice = ttk.Combobox(opt_top, textvariable=self.optimizer_var, values=("muon", "adamw"), state="readonly", width=14)
        opt_choice.pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opt_top, text="Muon Nesterov", variable=self.muon_nesterov_var).pack(side="left")
        opt_grid = ttk.Frame(opt_box)
        opt_grid.pack(fill="x", pady=(8, 0))
        self._grid_params(
            opt_grid,
            [
                ("Muon LR", self.muon_lr_var),
                ("Muon WD", self.muon_weight_decay_var),
                ("Muon beta", self.muon_momentum_beta_var),
                ("Muon NS", self.muon_ns_steps_var),
                ("Muon scale", self.muon_extra_scale_factor_var),
                ("Scalar opt", self.muon_scalar_optimizer_var),
            ],
        )

        arch_box = ttk.LabelFrame(tab, text="SHARC / routing", padding=8)
        arch_box.pack(fill="x", pady=(10, 0))
        arch_flags = ttk.Frame(arch_box)
        arch_flags.pack(fill="x")
        ttk.Checkbutton(
            arch_flags,
            text="SHARC signature lattice attention",
            variable=self.use_signature_lattice_attention_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            arch_flags,
            text="Carry SHARC cache in generation",
            variable=self.use_signature_lattice_generation_cache_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            arch_flags,
            text="Top-k MoT experts",
            variable=self.use_topk_mot_var,
        ).pack(side="left")
        arch_grid = ttk.Frame(arch_box)
        arch_grid.pack(fill="x", pady=(8, 0))
        self._grid_params(
            arch_grid,
            [
                ("Lattice dim", self.signature_lattice_dim_var),
                ("Lattice buckets", self.signature_lattice_buckets_var),
                ("Candidates", self.signature_lattice_candidates_var),
                ("Lattice weight", self.signature_lattice_weight_var),
                ("Lattice decay", self.signature_lattice_decay_var),
                ("MoT top-k", self.mot_top_k_var),
            ],
        )

        torus_box = ttk.LabelFrame(tab, text="Torus field", padding=8)
        torus_box.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            torus_box,
            text="Torus SHARC router augmentation",
            variable=self.torus_sharc_router_var,
        ).pack(anchor="w")
        self._grid_params(
            torus_box,
            [
                ("Local field r", self.torus_local_field_radius_var),
                ("Bus slots", self.torus_global_bus_slots_var),
                ("Bus decay", self.torus_global_bus_decay_var),
                ("Bus write", self.torus_global_bus_write_scale_var),
            ],
        )

        prompt_box = ttk.LabelFrame(tab, text="Post-run prompt", padding=8)
        prompt_box.pack(fill="x", pady=(10, 0))
        prompt_row = ttk.Frame(prompt_box)
        prompt_row.pack(fill="x")
        ttk.Label(prompt_row, text="Prompt", width=18).pack(side="left")
        ttk.Entry(prompt_row, textvariable=self.post_prompt_var).pack(side="left", fill="x", expand=True)

        post_grid = ttk.Frame(prompt_box)
        post_grid.pack(fill="x", pady=(8, 0))
        self._grid_params(
            post_grid,
            [
                ("Max new tokens", self.post_max_new_tokens_var),
                ("Min new tokens", self.post_min_new_tokens_var),
                ("Temp", self.post_temperature_var),
                ("Top-k", self.post_top_k_var),
                ("Top-p", self.post_top_p_var),
                ("Beam size", self.post_beam_size_var),
                ("Repetition", self.post_repetition_penalty_var),
                ("No-repeat ngram", self.post_no_repeat_ngram_size_var),
            ],
        )

        flags = ttk.Frame(tab)
        flags.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(flags, text="Torch compile", variable=self.torch_compile_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(flags, text="AMP", variable=self.amp_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(flags, text="Leaf 4-bit", variable=self.use_bitsandbytes_leaf_precision_var).pack(side="left", padx=(0, 10))

        extra = ttk.LabelFrame(tab, text="Extra training args", padding=8)
        extra.pack(fill="x", pady=(10, 0))
        ttk.Entry(extra, textvariable=self.extra_train_args_var).pack(fill="x")
        ttk.Label(
            extra,
            text="Example: --d-model 512 --torus-depth 12 --factorized-embedding-dim 16",
            foreground="#555555",
        ).pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Start Fresh Training", command=self._start_fresh_train).pack(side="left")
        ttk.Button(buttons, text="Continue Training", command=self._start_resume_train).pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(buttons, text="Stop Training", command=self._stop_running_process, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))

    def _build_inference_tab(self, tab: ttk.Frame) -> None:
        top = ttk.Frame(tab)
        top.pack(fill="x")
        checkpoint = self._path_row(top, "Checkpoint", self.infer_checkpoint_var, self._browse_checkpoint)
        checkpoint.pack(fill="x", pady=4)

        prompt = ttk.LabelFrame(tab, text="Prompt", padding=8)
        prompt.pack(fill="x", pady=(10, 0))
        ttk.Entry(prompt, textvariable=self.infer_prompt_var).pack(fill="x")

        grid = ttk.LabelFrame(tab, text="Generation settings", padding=8)
        grid.pack(fill="x", pady=(10, 0))
        self._grid_params(
            grid,
            [
                ("Max new tokens", self.infer_max_new_tokens_var),
                ("Min new tokens", self.infer_min_new_tokens_var),
                ("Temp", self.infer_temperature_var),
                ("Top-k", self.infer_top_k_var),
                ("Top-p", self.infer_top_p_var),
                ("Beam size", self.infer_beam_size_var),
                ("Repetition", self.infer_repetition_penalty_var),
                ("No-repeat ngram", self.infer_no_repeat_ngram_size_var),
            ],
        )
        extra = ttk.LabelFrame(tab, text="Extra inference args", padding=8)
        extra.pack(fill="x", pady=(10, 0))
        ttk.Entry(extra, textvariable=self.extra_infer_args_var).pack(fill="x")
        ttk.Label(extra, text="Example: --no-torch-compile", foreground="#555555").pack(anchor="w", pady=(4, 0))

        result_box = ttk.LabelFrame(tab, text="Last answer", padding=8)
        result_box.pack(fill="both", expand=True, pady=(10, 0))
        self.result_text = tk.Text(result_box, height=10, wrap="word", font=("Consolas", 10))
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scroll = ttk.Scrollbar(result_box, orient="vertical", command=self.result_text.yview)
        result_scroll.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=result_scroll.set)

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Run Inference", command=self._start_inference).pack(side="left")
        ttk.Button(buttons, text="Clear Result", command=self._clear_result).pack(side="left", padx=(8, 0))

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, browse_cb) -> ttk.Frame:
        row = ttk.Frame(parent)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse", command=browse_cb).pack(side="left")
        return row

    def _grid_params(self, parent: ttk.Frame, pairs: Sequence[tuple[str, tk.StringVar]]) -> None:
        for idx, (label, var) in enumerate(pairs):
            r = idx // 3
            c = (idx % 3) * 2
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=4)
            ttk.Entry(parent, textvariable=var, width=14).grid(row=r, column=c + 1, sticky="we", padx=(0, 12), pady=4)
        for col in range(6):
            parent.columnconfigure(col, weight=1 if col % 2 == 1 else 0)

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(
            title="Select dataset folder",
            initialdir=str(DATASETS_DIR if DATASETS_DIR.exists() else DEFAULT_DATASET_DIR),
        )
        if path:
            self.dataset_var.set(path)

    def _browse_dataset_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select dataset file",
            initialdir=str(DATASETS_DIR if DATASETS_DIR.exists() else DEFAULT_DATASET_DIR),
            filetypes=[
                ("Data files", "*.parquet *.jsonl *.txt *.md *.markdown *.rst"),
                ("Parquet files", "*.parquet"),
                ("JSONL files", "*.jsonl"),
                ("Text files", "*.txt *.md *.markdown *.rst"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.dataset_var.set(path)

    def _browse_dataset_folder(self) -> None:
        self._browse_dataset()

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            initialdir=str(CHECKPOINTS_DIR if CHECKPOINTS_DIR.exists() else ROOT_DIR),
            filetypes=[("Model checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.resume_var.set(path)
            self.infer_checkpoint_var.set(path)
            self._refresh_run_dir()

    def _refresh_run_dir(self) -> None:
        source = _source_name(self.dataset_var.get())
        mode = "continue" if self.resume_var.get().strip() else "fresh"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = _slugify(
            f"{source}_{mode}_bs{self.batch_var.get().strip() or '0'}_ep{self.epochs_var.get().strip() or '0'}"
            f"_st{self.steps_var.get().strip() or '0'}_ms{self.max_samples_var.get().strip() or '0'}_{stamp}"
        )
        self.save_dir_var.set(str(_unique_dir(CHECKPOINTS_DIR / run_name)))

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _clear_result(self) -> None:
        self.result_text.delete("1.0", "end")

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        if not text.endswith("\n"):
            self.log_text.insert("end", "\n")
        self.log_text.see("end")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _run_spec_for_training(self, *, resume: bool) -> _RunSpec:
        dataset = self.dataset_var.get().strip()
        if not dataset:
            raise ValueError("Please select a dataset file or folder before training.")
        save_dir = Path(self.save_dir_var.get().strip() or "")
        if not save_dir:
            raise ValueError("Save directory could not be generated.")
        if resume and not self.resume_var.get().strip():
            raise ValueError("Please select a checkpoint to resume from.")

        cmd = [
            sys.executable,
            "-u",
            str(ROOT_DIR / "cli.py"),
            "train",
            "--data",
            dataset,
            "--save-dir",
            str(save_dir),
            "--epochs",
            self.epochs_var.get().strip() or "0",
            "--steps",
            self.steps_var.get().strip() or "200",
            "--batch-size",
            self.batch_var.get().strip() or "1",
            "--max-samples",
            self.max_samples_var.get().strip() or "0",
            "--tokenizer-workers",
            self.tokenizer_workers_var.get().strip() or "1",
            "--lr",
            self.lr_var.get().strip() or "0.0001",
            "--optimizer",
            self.optimizer_var.get().strip() or "muon",
            "--muon-lr",
            self.muon_lr_var.get().strip() or "0.02",
            "--muon-weight-decay",
            self.muon_weight_decay_var.get().strip() or "0.01",
            "--muon-momentum-beta",
            self.muon_momentum_beta_var.get().strip() or "0.95",
            "--muon-ns-steps",
            self.muon_ns_steps_var.get().strip() or "5",
            "--muon-extra-scale-factor",
            self.muon_extra_scale_factor_var.get().strip() or "1.0",
            "--muon-scalar-optimizer",
            self.muon_scalar_optimizer_var.get().strip() or "adamw",
            "--val-fraction",
            self.val_fraction_var.get().strip() or "0.1",
            "--min-token-frequency",
            self.min_token_frequency_var.get().strip() or "2",
            "--dataset-streaming" if self.dataset_streaming_var.get() else "--no-dataset-streaming",
            "--post-prompt",
            self.post_prompt_var.get().strip(),
            "--post-output",
            _default_post_output(save_dir),
            "--post-max-new-tokens",
            self.post_max_new_tokens_var.get().strip() or "80",
            "--post-min-new-tokens",
            self.post_min_new_tokens_var.get().strip() or "1",
            "--post-temperature",
            self.post_temperature_var.get().strip() or "0.9",
            "--post-top-k",
            self.post_top_k_var.get().strip() or "8",
            "--post-top-p",
            self.post_top_p_var.get().strip() or "0.92",
            "--post-repetition-penalty",
            self.post_repetition_penalty_var.get().strip() or "1.1",
            "--post-no-repeat-ngram-size",
            self.post_no_repeat_ngram_size_var.get().strip() or "4",
            "--post-beam-size",
            self.post_beam_size_var.get().strip() or "3",
            "--signature-lattice-dim",
            self.signature_lattice_dim_var.get().strip() or str(DEFAULT_CFG.signature_lattice_dim),
            "--signature-lattice-buckets",
            self.signature_lattice_buckets_var.get().strip() or str(DEFAULT_CFG.signature_lattice_buckets),
            "--signature-lattice-candidates",
            self.signature_lattice_candidates_var.get().strip() or str(DEFAULT_CFG.signature_lattice_candidates),
            "--signature-lattice-weight",
            self.signature_lattice_weight_var.get().strip() or str(DEFAULT_CFG.signature_lattice_weight),
            "--signature-lattice-decay",
            self.signature_lattice_decay_var.get().strip() or str(DEFAULT_CFG.signature_lattice_decay),
            "--mot-top-k",
            self.mot_top_k_var.get().strip() or str(DEFAULT_CFG.mot_top_k),
            "--torus-local-field-radius",
            self.torus_local_field_radius_var.get().strip() or str(DEFAULT_CFG.torus_local_field_radius),
            "--torus-global-bus-slots",
            self.torus_global_bus_slots_var.get().strip() or str(DEFAULT_CFG.torus_global_bus_slots),
            "--torus-global-bus-decay",
            self.torus_global_bus_decay_var.get().strip() or str(DEFAULT_CFG.torus_global_bus_decay),
            "--torus-global-bus-write-scale",
            self.torus_global_bus_write_scale_var.get().strip() or str(DEFAULT_CFG.torus_global_bus_write_scale),
        ]
        if self.muon_nesterov_var.get():
            cmd.append("--muon-nesterov")
        else:
            cmd.append("--no-muon-nesterov")
        if self.torch_compile_var.get():
            cmd.append("--torch-compile")
        else:
            cmd.append("--no-torch-compile")
        if self.amp_var.get():
            cmd.append("--amp")
        else:
            cmd.append("--no-amp")
        if self.use_bitsandbytes_leaf_precision_var.get():
            cmd.append("--use-bitsandbytes-leaf-precision")
        else:
            cmd.append("--no-bitsandbytes-leaf-precision")
        if self.use_signature_lattice_attention_var.get():
            cmd.append("--use-signature-lattice-attention")
        else:
            cmd.append("--no-signature-lattice-attention")
        if self.use_signature_lattice_generation_cache_var.get():
            cmd.append("--use-signature-lattice-generation-cache")
        else:
            cmd.append("--no-signature-lattice-generation-cache")
        if self.use_topk_mot_var.get():
            cmd.append("--use-topk-mot")
        else:
            cmd.append("--no-topk-mot")
        if self.torus_sharc_router_var.get():
            cmd.append("--torus-sharc-router")
        else:
            cmd.append("--no-torus-sharc-router")
        if resume:
            cmd.extend(["--continue-checkpoint", self.resume_var.get().strip()])
        cmd.extend(_split_extra_args(self.extra_train_args_var.get()))
        return _RunSpec(command=cmd, capture_output=False)

    def _run_spec_for_inference(self) -> _RunSpec:
        checkpoint = self.infer_checkpoint_var.get().strip()
        if not checkpoint:
            raise ValueError("Please select a checkpoint first.")
        prompt = self.infer_prompt_var.get().strip()
        if not prompt:
            raise ValueError("Please enter a prompt to ask.")
        cmd = [
            sys.executable,
            "-u",
            str(ROOT_DIR / "cli.py"),
            "infer",
            "--checkpoint",
            checkpoint,
            "--prompt",
            prompt,
            "--max-new-tokens",
            self.infer_max_new_tokens_var.get().strip() or "64",
            "--min-new-tokens",
            self.infer_min_new_tokens_var.get().strip() or "1",
            "--temperature",
            self.infer_temperature_var.get().strip() or "0.9",
            "--top-k",
            self.infer_top_k_var.get().strip() or "8",
            "--top-p",
            self.infer_top_p_var.get().strip() or "0.92",
            "--repetition-penalty",
            self.infer_repetition_penalty_var.get().strip() or "1.1",
            "--no-repeat-ngram-size",
            self.infer_no_repeat_ngram_size_var.get().strip() or "4",
            "--beam-size",
            self.infer_beam_size_var.get().strip() or "3",
        ]
        cmd.extend(_split_extra_args(self.extra_infer_args_var.get()))
        return _RunSpec(command=cmd, capture_output=True)

    def _start_fresh_train(self) -> None:
        try:
            self.resume_var.set("")
            self._refresh_run_dir()
            self._start_process(self._run_spec_for_training(resume=False), mode_label="fresh training")
        except Exception as exc:
            messagebox.showerror("Training setup failed", str(exc))

    def _start_resume_train(self) -> None:
        try:
            self._refresh_run_dir()
            self._start_process(self._run_spec_for_training(resume=True), mode_label="resumed training")
        except Exception as exc:
            messagebox.showerror("Training setup failed", str(exc))

    def _start_inference(self) -> None:
        try:
            self._start_process(self._run_spec_for_inference(), mode_label="inference")
        except Exception as exc:
            messagebox.showerror("Inference setup failed", str(exc))

    def _start_process(self, spec: _RunSpec, *, mode_label: str) -> None:
        if self._process is not None and self._process.poll() is None:
            messagebox.showwarning("Busy", "A training or inference process is already running.")
            return

        self._capture_output = spec.capture_output
        self._capture_lines = []
        self._append_log(f"$ {_join_cmd(spec.command)}")
        self._set_status(f"Running {mode_label} ...")

        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        path_bits = [str(ROOT_DIR)]
        if pythonpath:
            path_bits.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(path_bits)
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            spec.command,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._process_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._process_thread.start()
        self._toggle_controls(False)

    def _stop_running_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        if self._stop_thread is not None and self._stop_thread.is_alive():
            return
        self._set_status("Stopping run ...")
        self._append_log("[GUI] stop requested; terminating process ...")
        self._stop_thread = threading.Thread(target=self._terminate_process, daemon=True)
        self._stop_thread.start()

    def _terminate_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=PROCESS_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=PROCESS_GRACE_SECONDS)
            except Exception:
                pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._queue.put(("log", line))
            if self._capture_output:
                self._queue.put(("capture", line))
        returncode = self._process.wait()
        self._queue.put(("done", returncode))

    def _toggle_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.winfo_children():
            self._set_tree_state(widget, state)
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state="disabled" if enabled else "normal")
        if enabled:
            self._set_status("Idle")

    def _set_tree_state(self, widget: tk.Widget, state: str) -> None:
        try:
            if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton)):
                widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_tree_state(child, state)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "capture":
                    self._capture_lines.append(str(payload))
                elif kind == "done":
                    self._finish_process(int(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish_process(self, returncode: int) -> None:
        self._process = None
        self._process_thread = None
        self._toggle_controls(True)
        if returncode == 0:
            self._set_status("Done")
        else:
            self._set_status(f"Exited with code {returncode}")
        if self._capture_output and self._capture_lines:
            output = "".join(self._capture_lines).strip()
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", output)
            self.result_text.see("1.0")
            self._capture_lines = []


def launch_gui() -> None:
    app = PrismalWaveGUI()
    app.mainloop()


def main() -> None:
    launch_gui()


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    main()

