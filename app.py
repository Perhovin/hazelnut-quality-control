"""Hazelnut Quality Control app.

Step 2 (Pandas): the "1. Data" tab runs the ready-made step2_pandas.py
script and shows the resulting dataset_<category>.csv table.
Step 3 (Training): the "2. Training" tab runs train.py and draws the
learning process live.
Step 4 (Check): the "3. Check" tab runs the trained model on photos
and explains each verdict.
"""

import csv
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, font as tkfont
from tkinter import messagebox, scrolledtext, ttk

# pythonw.exe (used by app.bat) has no console, so sys.stdout/stderr are
# None there. Point them at devnull so prints or warnings written by
# libraries (e.g. torch) can never crash the GUI.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from PIL import Image, ImageTk

import paths

BASE = paths.BASE
ERROR_LOG = BASE / "app_error.log"
FONT = ("Segoe UI", 10)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Child scripts must run under python.exe: when the app itself is started
# by pythonw.exe (app.bat), sys.executable is pythonw.exe and its stdout
# can be None, which breaks the child script's prints.
_py_sibling = Path(sys.executable).with_name("python.exe")
PYTHON = str(_py_sibling) if _py_sibling.exists() else sys.executable

def list_categories():
    """Subfolders of the MVTec root that contain train+test folders."""
    root = paths.get_mvtec_root()
    cats = []
    if root.exists():
        cats = sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()
        )
    if "hazelnut" not in cats:
        cats.insert(0, "hazelnut")
    return cats


class DataTab(ttk.Frame):
    """The '1. Data' tab: build and browse dataset_<category>.csv."""

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.df = None
        self.counts = {}
        self._photo_img = None
        self._build_running = False
        self._build_returncode = None
        self._build_proc = None
        self._build_all_summary = ""
        self._overview_win = None
        self._overview_tree = None
        self._overview_total = None
        self._overview_data = {}

        self.category_var = tk.StringVar()
        self.mvtec_var = tk.StringVar(value="")
        self.part_var = tk.StringVar(value="all")
        self.mark_var = tk.StringVar(value="all")
        self.status_var = tk.StringVar()
        self.shown_var = tk.StringVar(value="Shown 0 of 0")
        self.tables_built_var = tk.StringVar(value="Tables built: 0/0")
        self.caption_var = tk.StringVar(
            value="Select a row in the table on the left - its photo and details "
                  "will appear here."
        )

        style = ttk.Style(self)
        style.configure("StatusInfo.TLabel", foreground="#333333")
        style.configure("StatusOk.TLabel", foreground="#1a6b1a")
        style.configure("StatusWarn.TLabel", foreground="#a05a00")
        style.configure("StatusErr.TLabel", foreground="#b00020")

        self._build_flow_row()
        self._build_control_row()
        self._build_bottom()
        self._build_middle()

        self.category_var.set("hazelnut")
        self.update_tables_built()
        self.load_csv()
        # last, so a missing MVTec folder is not overwritten by the
        # "Loaded ..." status above
        self.refresh_mvtec()

    def refresh_mvtec(self):
        """Re-read the MVTec folder, rebuild the category list, and
        disable the build buttons when the folder is missing."""
        root = paths.get_mvtec_root()
        self.mvtec_var.set(self._elide(str(root), 70))
        cats = list_categories()
        self.category_box.configure(values=cats)
        ok = root.is_dir() and bool(cats)
        state = "normal" if ok else "disabled"
        self.build_btn.configure(state=state)
        self.build_all_btn.configure(state=state)
        if not ok:
            self.set_status(
                "MVTec folder not found. Click 'MVTec folder...' and "
                "select the folder that contains the category folders "
                "(hazelnut, wood, ...).", "StatusErr.TLabel")
        elif self.status_var.get().startswith("MVTec folder"):
            self.set_status("Idle")

    @staticmethod
    def _elide(text, width):
        return text if len(text) <= width else "..." + text[-(width - 3):]

    def choose_mvtec(self):
        d = filedialog.askdirectory(
            parent=self,
            title="Select the folder with the MVTec category folders")
        if d:
            paths.set_mvtec_root(d)
            self.refresh_mvtec()
            self.set_status(f"MVTec folder set to {d}", "StatusOk.TLabel")
            self.load_csv()

    # ---------- UI construction ----------

    def _build_flow_row(self):
        ttk.Label(
            self,
            text="Photo folders  ->  step2_pandas.py  ->  dataset_<category>.csv"
                 "  ->  table below",
            font=("Segoe UI", 10, "bold"),
        ).pack(fill="x", pady=(0, 4))

    def _build_control_row(self):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Category:").pack(side="left")
        self.category_box = ttk.Combobox(
            row, textvariable=self.category_var, state="readonly", width=14
        )
        self.category_box.pack(side="left", padx=(4, 6))
        self.category_box.bind("<<ComboboxSelected>>", lambda e: self.load_csv())
        ttk.Label(row, textvariable=self.tables_built_var).pack(
            side="left", padx=(0, 10)
        )
        self.build_btn = ttk.Button(
            row,
            text="Build table (run step2_pandas.py)",
            command=self.run_build,
        )
        self.build_btn.pack(side="left")
        self.build_all_btn = ttk.Button(
            row, text="Build ALL categories", command=self.run_build_all
        )
        self.build_all_btn.pack(side="left", padx=(6, 0))
        ttk.Button(
            row, text="Categories overview", command=self.open_overview
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            row, text="Open files folder", command=lambda: os.startfile(BASE)
        ).pack(side="left", padx=(6, 0))

        mv_row = ttk.Frame(self)
        mv_row.pack(fill="x", pady=(0, 4))
        ttk.Button(
            mv_row, text="MVTec folder...", command=self.choose_mvtec
        ).pack(side="left")
        ttk.Label(
            mv_row, textvariable=self.mvtec_var, foreground="#555555",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=8)

    def _build_middle(self):
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)
        self._build_left(mid)
        self._build_right(mid)

    def _build_left(self, parent):
        left = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(0, 4))
        ttk.Label(filters, text="Split:").pack(side="left")
        part_box = ttk.Combobox(
            filters,
            textvariable=self.part_var,
            state="readonly",
            width=7,
            values=("all", "train", "val", "test"),
        )
        part_box.pack(side="left", padx=(4, 10))
        part_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())
        ttk.Label(filters, text="Label:").pack(side="left")
        mark_box = ttk.Combobox(
            filters,
            textvariable=self.mark_var,
            state="readonly",
            width=7,
            values=("all", "ok", "ng"),
        )
        mark_box.pack(side="left", padx=(4, 10))
        mark_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_table())
        ttk.Label(filters, textvariable=self.shown_var).pack(side="right")

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        cols = ("file", "mark", "defect", "split")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        self.tree.heading("file", text="File")
        self.tree.heading("mark", text="Label")
        self.tree.heading("defect", text="Defect")
        self.tree.heading("split", text="Split")
        self.tree.column("file", width=230, anchor="w")
        self.tree.column("mark", width=75, anchor="center")
        self.tree.column("defect", width=100, anchor="w")
        self.tree.column("split", width=55, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("ng", foreground="#b00020")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_right(self, parent):
        right = ttk.Frame(parent, width=400)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        photo_frame = ttk.LabelFrame(right, text="Selected row photo")
        photo_frame.pack(fill="x")
        inner = ttk.Frame(photo_frame, width=372, height=366)
        inner.pack_propagate(False)
        inner.pack(padx=4, pady=4)
        self.preview_label = ttk.Label(inner, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        ttk.Label(
            right, textvariable=self.caption_var, wraplength=380, justify="left"
        ).pack(fill="x", pady=(4, 6))

        summ_frame = ttk.LabelFrame(right, text="Summary: ok / NG per split")
        summ_frame.pack(fill="x")
        scols = ("part", "ok", "ng", "all")
        self.summary_tree = ttk.Treeview(
            summ_frame, columns=scols, show="headings", height=4
        )
        for col, text, w in (
            ("part", "Split", 90),
            ("ok", "ok", 90),
            ("ng", "NG", 90),
            ("all", "Total", 80),
        ):
            self.summary_tree.heading(col, text=text)
            self.summary_tree.column(col, width=w, anchor="center")
        self.summary_tree.column("part", anchor="w")
        self.summary_tree.pack(fill="x", padx=4, pady=4)
        self.prop_bar = ttk.Progressbar(summ_frame, mode="determinate", maximum=100)
        self.prop_bar.pack(fill="x", padx=4, pady=(0, 2))
        self.prop_var = tk.StringVar(value="")
        ttk.Label(summ_frame, textvariable=self.prop_var).pack(
            anchor="w", padx=4, pady=(0, 4)
        )

    def _build_bottom(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="bottom")

        what = ttk.LabelFrame(bottom, text="What is happening")
        what.pack(side="left", fill="both", padx=(0, 6))
        ttk.Label(
            what,
            text=(
                "The step2_pandas.py script walks the train and test folders, "
                "assigns label 0 (ok) to the 'good' folder and 1 (NG) to all "
                "other folders, splits the photos into train/val/test and saves "
                "the table dataset_<category>.csv.\n"
                "The split is done on the original photos with a fixed seed=42, "
                "so the result is always the same."
            ),
            wraplength=370,
            justify="left",
        ).pack(anchor="w", padx=4, pady=(2, 0))
        self.status_label = ttk.Label(
            what,
            textvariable=self.status_var,
            style="StatusInfo.TLabel",
            wraplength=370,
            justify="left",
        )
        self.status_label.pack(anchor="w", padx=4, pady=(2, 4))

        log_frame = ttk.LabelFrame(bottom, text="Script log")
        log_frame.pack(side="left", fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=8, state="disabled", font=("Consolas", 9)
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    # ---------- data and rendering ----------

    def set_status(self, text, style="StatusInfo.TLabel"):
        self.status_var.set(text)
        self.status_label.configure(style=style)

    def update_tables_built(self):
        cats = list_categories()
        n = sum(1 for c in cats if (BASE / f"dataset_{c}.csv").exists())
        self.tables_built_var.set(f"Tables built: {n}/{len(cats)}")

    def load_csv(self):
        cat = self.category_var.get()
        path = BASE / f"dataset_{cat}.csv"
        if path.exists():
            try:
                self.df = pd.read_csv(path)
            except Exception:
                self.df = None
                err = traceback.format_exc()
                try:
                    with open(ERROR_LOG, "a", encoding="utf-8") as f:
                        f.write(err + "\n")
                except OSError:
                    pass
                self.set_status(
                    f"Could not read {path.name} - the error was written "
                    "to app_error.log.",
                    "StatusErr.TLabel",
                )
                self.refresh_all()
                return
            self.set_status(
                f"Loaded dataset_{cat}.csv - {len(self.df)} rows.",
                "StatusOk.TLabel",
            )
        else:
            self.df = None
            self.set_status(
                f"No table for category '{cat}' yet - press 'Build table'.",
                "StatusWarn.TLabel",
            )
        self.refresh_all()
        self.update_tables_built()

    def filtered_df(self):
        if self.df is None:
            return None
        df = self.df
        part = self.part_var.get()
        mark = self.mark_var.get()
        if part != "all":
            df = df[df["split"] == part]
        if mark == "ok":
            df = df[df["label"] == 0]
        elif mark == "ng":
            df = df[df["label"] == 1]
        return df

    def refresh_all(self):
        for iid in self.summary_tree.get_children():
            self.summary_tree.delete(iid)
        self.counts = {}
        df = self.df
        if df is not None and len(df):
            ct = pd.crosstab(
                df["split"], df["label"], margins=True, margins_name="Total"
            )
            ct = ct.reindex(
                index=["train", "val", "test", "Total"],
                columns=[0, 1, "Total"],
                fill_value=0,
            )
            for part in ("train", "val", "test", "Total"):
                ok = int(ct.loc[part, 0])
                ng = int(ct.loc[part, 1])
                all_ = int(ct.loc[part, "Total"])
                self.summary_tree.insert("", "end", values=(part, ok, ng, all_))
                self.counts[part] = {"ok": ok, "ng": ng, "all": all_}
            total_ok = self.counts["Total"]["ok"]
            total = self.counts["Total"]["all"]
            pct = 100.0 * total_ok / total if total else 0.0
            self.prop_bar["value"] = pct
            self.prop_var.set(
                f"ok {pct:.0f}% / ng {100 - pct:.0f}%  ({total} photos)"
            )
        else:
            self.prop_bar["value"] = 0
            self.prop_var.set("no data")
        self.refresh_table()

    def refresh_table(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        df = self.filtered_df()
        total = 0 if self.df is None else len(self.df)
        shown = 0
        if df is not None:
            cat_root = paths.get_mvtec_root() / self.category_var.get()
            for idx, row in df.iterrows():
                p = Path(str(row["path"]))
                try:
                    rel = str(p.relative_to(cat_root))
                except ValueError:
                    rel = p.name
                label = int(row["label"])
                mark = "ok (0)" if label == 0 else "NG (1)"
                tags = ("ng",) if label else ()
                self.tree.insert(
                    "", "end", iid=str(idx),
                    values=(rel, mark, row["defect"], row["split"]), tags=tags,
                )
                shown += 1
        self.shown_var.set(f"Shown {shown} of {total}")

    def _on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if sel:
            self.show_preview(sel[0])

    def show_preview(self, iid):
        if self.df is None:
            return
        row = self.df.loc[int(iid)]
        path = str(row["path"])
        label = int(row["label"])
        mark = "ok (0)" if label == 0 else "NG (1)"
        try:
            img = Image.open(path)
            img.thumbnail((360, 360))
            self._photo_img = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._photo_img, text="")
        except Exception as e:
            self._photo_img = None
            self.preview_label.configure(
                image="", text=f"could not open photo:\n{e}"
            )
        self.caption_var.set(
            f"{path}\nLabel: {mark}    Defect: {row['defect']}    "
            f"Split: {row['split']}"
        )

    # ---------- categories overview window ----------

    def open_overview(self):
        if self._overview_win is not None and self._overview_win.winfo_exists():
            self._overview_win.lift()
            return
        win = tk.Toplevel(self)
        win.title("Categories overview")
        win.geometry("900x480")

        tree_frame = ttk.Frame(win)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        cols = ("category", "photos", "ok", "ng", "ngpct", "train", "val",
                "test", "size", "mode", "built")
        heads = ("Category", "Photos", "ok", "NG", "NG %", "train", "val",
                 "test", "Image size", "Mode", "Table built")
        widths = (110, 60, 55, 55, 55, 55, 55, 55, 90, 60, 80)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        for c, h, w in zip(cols, heads, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="w" if c == "category" else "center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        bottom = ttk.Frame(win)
        bottom.pack(fill="x", side="bottom", padx=6, pady=6)
        self._overview_total = tk.StringVar()
        ttk.Label(bottom, textvariable=self._overview_total).pack(side="left")
        ttk.Button(bottom, text="Refresh", command=self.refresh_overview).pack(
            side="right"
        )

        self._overview_win = win
        self._overview_tree = tree
        win.bind("<Destroy>", self._on_overview_close)
        self.refresh_overview()

    def _on_overview_close(self, event):
        if event.widget is self._overview_win:
            self._overview_win = None
            self._overview_tree = None

    def refresh_overview(self):
        tree = self._overview_tree
        if tree is None:
            return
        for iid in tree.get_children():
            tree.delete(iid)
        data = {}
        total_photos = 0
        built = 0
        for cat in list_categories():
            path = BASE / f"dataset_{cat}.csv"
            df = None
            if path.exists():
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = None
            if df is None or not len(df):
                tree.insert(
                    "", "end",
                    values=(cat, "-", "-", "-", "-", "-", "-", "-", "-", "-", "no"),
                )
                data[cat] = {"built": False}
                continue
            n = len(df)
            ok = int((df["label"] == 0).sum())
            ng = n - ok
            n_train = int((df["split"] == "train").sum())
            n_val = int((df["split"] == "val").sum())
            n_test = int((df["split"] == "test").sum())
            size_txt, size_px, mode = "-", None, "-"
            try:
                with Image.open(str(df["path"].iloc[0])) as im:
                    size_txt = f"{im.size[0]}x{im.size[1]}"
                    size_px = im.size[0]
                    mode = im.mode
            except Exception:
                pass
            tree.insert(
                "", "end",
                values=(cat, n, ok, ng, f"{100.0 * ng / n:.0f}%",
                        n_train, n_val, n_test, size_txt, mode, "yes"),
            )
            data[cat] = {
                "built": True, "photos": n, "ok": ok, "ng": ng,
                "size": size_px, "mode": mode,
            }
            total_photos += n
            built += 1
        self._overview_data = data
        self._overview_total.set(
            f"Total: {total_photos} photos in {built} categories"
        )
        self.update_tables_built()

    # ---------- running step2_pandas.py ----------

    def log_append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_build_ui(self, running):
        mv_ok = paths.get_mvtec_root().is_dir() and bool(list_categories())
        state = "disabled" if running or not mv_ok else "normal"
        self.build_btn.configure(state=state)
        self.build_all_btn.configure(state=state)
        self.category_box.configure(state="disabled" if running else "readonly")

    def kill_proc_now(self):
        """Used when the whole app is closing: kill a running script."""
        p = self._build_proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def _spawn_step2(self, cat):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._build_proc = subprocess.Popen(
            [PYTHON, str(BASE / "step2_pandas.py"), cat],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        return self._build_proc

    @staticmethod
    def _stream_proc(proc, q):
        for line in proc.stdout:
            q.put(("line", line.rstrip("\n")))
        return proc.wait()

    def run_build(self):
        if self._build_running:
            return
        cat = self.category_var.get()
        self._build_running = True
        self._build_returncode = None
        self._set_build_ui(True)
        self.log_clear()
        self.set_status(
            f"Running: python step2_pandas.py {cat} ...", "StatusInfo.TLabel"
        )
        self.log_append(f"> step2_pandas.py {cat}")

        q = queue.Queue()

        def worker():
            try:
                proc = self._spawn_step2(cat)
                q.put(("done", self._stream_proc(proc, q)))
            except Exception:
                q.put(("fail", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, lambda: self._poll_build(q))

    def run_build_all(self):
        if self._build_running:
            return
        cats = list(self.category_box.cget("values"))
        self._build_running = True
        self._build_returncode = None
        self._build_all_summary = ""
        self._set_build_ui(True)
        self.log_clear()
        self.set_status(
            f"Building tables for all {len(cats)} categories...",
            "StatusInfo.TLabel",
        )
        self.log_append(f"> step2_pandas.py <all {len(cats)} categories>")

        q = queue.Queue()

        def worker():
            ok_n = 0
            failed = []
            for i, cat in enumerate(cats, 1):
                q.put(("line", f"=== {cat} ({i}/{len(cats)}) ==="))
                q.put(("status", f"Building {i}/{len(cats)}: {cat}"))
                try:
                    proc = self._spawn_step2(cat)
                    rc = self._stream_proc(proc, q)
                except Exception:
                    rc = -1
                    q.put(("line", traceback.format_exc()))
                if rc == 0:
                    ok_n += 1
                else:
                    failed.append(cat)
                q.put(("line", ""))
            q.put(("done_all", (ok_n, failed)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, lambda: self._poll_build(q))

    def _poll_build(self, q):
        done_kind = None
        done_payload = None
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "line":
                    self.log_append(payload)
                elif kind == "status":
                    self.set_status(payload, "StatusInfo.TLabel")
                elif kind in ("done", "done_all"):
                    done_kind, done_payload = kind, payload
                elif kind == "fail":
                    done_kind = "fail"
                    self.log_append(payload)
        except queue.Empty:
            pass
        if done_kind is None:
            self.after(100, lambda: self._poll_build(q))
            return
        self._build_running = False
        self._set_build_ui(False)
        self.update_tables_built()
        if done_kind == "done":
            self._build_returncode = done_payload
            if done_payload == 0:
                self.load_csv()
            else:
                self.set_status(
                    f"Script failed (code {done_payload}) - see the log below.",
                    "StatusErr.TLabel",
                )
        elif done_kind == "done_all":
            ok_n, failed = done_payload
            self._build_returncode = 0 if not failed else 1
            if failed:
                summary = (f"Done: {ok_n} ok, {len(failed)} failed "
                           f"({', '.join(failed)})")
                self.set_status(summary, "StatusErr.TLabel")
            else:
                summary = f"Done: {ok_n} ok, 0 failed"
                self.set_status(summary, "StatusOk.TLabel")
            self._build_all_summary = summary
            self.load_csv()
        else:  # fail
            self._build_returncode = -1
            self.set_status(
                "Could not start the script - see the log below.",
                "StatusErr.TLabel",
            )


class PlaceholderTab(ttk.Frame):
    def __init__(self, master, title, text):
        super().__init__(master, padding=20)
        box = ttk.Frame(self)
        box.pack(expand=True)
        ttk.Label(box, text=title, font=("Segoe UI", 18, "bold")).pack()
        ttk.Label(
            box, text=text, font=("Segoe UI", 11), wraplength=700,
            justify="center",
        ).pack(pady=(10, 0))


def fmt_seconds(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class ScrollableTab(tk.Frame):
    """Vertical scroll wrapper for one tab's content.

    The tab frame goes into ``self.inner``. The inner frame is always as
    wide as the canvas and at least as tall: when the window is large the
    content stretches exactly as before (charts/tables keep expanding),
    when it is small - or when Windows DPI scaling makes every font and
    widget bigger - a scrollbar appears and nothing becomes unreachable.
    """

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._fit)
        self.canvas.bind("<Configure>", self._fit)
        # bind_all is per-application: the handler only acts when the
        # wheel event happened inside THIS tab's content, and never over
        # widgets that scroll on their own (Treeview, Text, ...).
        self.bind_all("<MouseWheel>", self._on_wheel)

    def _fit(self, _event=None):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(),
                 self.inner.winfo_reqheight())
        self.canvas.itemconfigure(self._win, width=cw, height=ch)
        self.canvas.configure(scrollregion=(0, 0, cw, ch))

    def _on_wheel(self, event):
        blocked = (ttk.Treeview, tk.Text, tk.Listbox, ttk.Scrollbar,
                   tk.Scrollbar, ttk.Combobox, tk.Spinbox, ttk.Spinbox,
                   tk.Scale, ttk.Scale)
        w = event.widget
        while w is not None:
            if isinstance(w, blocked):
                return
            if w is self.canvas or w is self.inner or w is self:
                break
            w = getattr(w, "master", None)
        if w is None:
            return
        # one canvas "unit" is ~2 px; 30 units ~= 60 px per wheel notch
        self.canvas.yview_scroll(int(-event.delta / 120) * 30, "units")


class LineChart:
    """Tiny canvas line chart: axes, grid, labels, legend, best-epoch marker."""

    def __init__(self, parent, title, percent=False):
        self.title = title
        self.percent = percent
        self.series = []           # {"name": str, "color": str, "values": []}
        self.best_epoch = None
        self.empty_text = "No history yet"
        self.canvas = tk.Canvas(
            parent, width=430, height=200, bg="white",
            highlightthickness=1, highlightbackground="#c8c8c8",
        )
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def add_series(self, name, color):
        self.series.append({"name": name, "color": color, "values": []})

    def clear(self, empty_text="No history yet"):
        for s in self.series:
            s["values"] = []
        self.best_epoch = None
        self.empty_text = empty_text
        self.redraw()

    def push(self, values):
        """Append one point per named series (dict name -> value)."""
        for s in self.series:
            s["values"].append(values.get(s["name"]))
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 150)
        L, R, T, B = 46, 10, 30, 24
        c.create_text(w / 2, 13, text=self.title, font=("Segoe UI", 9, "bold"))
        self._legend(c, w, T)
        vals = [v for s in self.series for v in s["values"] if v is not None]
        if not vals:
            c.create_text(w / 2, (h + T) / 2, text=self.empty_text,
                          fill="#999999", font=("Segoe UI", 10))
            return
        n = max(len(s["values"]) for s in self.series)
        if self.percent:
            ymin, ymax = 0.0, 1.0
        else:
            ymin, ymax = min(vals), max(vals)
            pad = (ymax - ymin) * 0.1 or max(abs(ymax) * 0.1, 0.01)
            ymin, ymax = ymin - pad, ymax + pad
        xmax = max(n, 2)

        def X(i):
            return L + (i - 1) / (xmax - 1) * (w - L - R) if xmax > 1 else L

        def Y(v):
            return T + (ymax - v) / (ymax - ymin) * (h - T - B)

        for j in range(6):
            v = ymin + j * (ymax - ymin) / 5
            y = Y(v)
            c.create_line(L, y, w - R, y, fill="#e8e8e8")
            txt = f"{v:.0%}" if self.percent else f"{v:.2f}"
            c.create_text(L - 5, y, text=txt, anchor="e", font=("Segoe UI", 7))
        step = max(1, -(-n // 10))
        for i in range(1, n + 1):
            if i == 1 or i == n or i % step == 0:
                c.create_text(X(i), h - B + 11, text=str(i),
                              font=("Segoe UI", 7))
        c.create_text(14, h - B + 11, text="ep", font=("Segoe UI", 7))
        c.create_line(L, T, L, h - B, fill="#888888")
        c.create_line(L, h - B, w - R, h - B, fill="#888888")
        if self.best_epoch:
            x = X(self.best_epoch)
            c.create_line(x, T, x, h - B, fill="#2ca02c", dash=(4, 3))
            c.create_text(x + 3, T + 9, text="best", anchor="w",
                          fill="#2ca02c", font=("Segoe UI", 7))
        for s in self.series:
            pts = [(X(i + 1), Y(v)) for i, v in enumerate(s["values"])
                   if v is not None]
            if len(pts) >= 2:
                flat = [c_ for p in pts for c_ in p]
                c.create_line(*flat, fill=s["color"], width=2)
            for x, y in pts:
                c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5,
                              fill=s["color"], outline="")

    def _legend(self, c, w, T):
        lx = w - 10
        for s in reversed(self.series):
            c.create_text(lx, T - 9, text=s["name"], anchor="e",
                          font=("Segoe UI", 7), fill=s["color"])
            lx -= len(s["name"]) * 5.5 + 24


TRAIN_FLOW = [
    ("Photos", "training photos from the table, shuffled each epoch"),
    ("Augmentation", "random flips, rotation, brightness - a new variant "
     "every epoch (training only)"),
    ("ResNet18 (GPU)", "pretrained network, last layer replaced by 1 output = "
     "probability of defect; runs on the GPU"),
    ("Loss", "how wrong the answer is; defect mistakes count more "
     "(pos_weight)"),
    ("Optimizer", "AdamW nudges the network numbers to reduce the loss"),
    ("Repeat", "one pass over all training photos = 1 epoch; after each "
     "epoch we check on the val photos"),
]

GLOSSARY = (
    "epoch - one full pass over all training photos.\n"
    "batch - photos go to the network in small groups (batch size).\n"
    "loss - a number for 'how wrong' the network is; smaller is better.\n"
    "accuracy - the share of correct answers.\n"
    "recall NG - how many real defects were found (missed ones are "
    "dangerous!).\n"
    "precision NG - how many flagged photos really were defects.\n"
    "balanced accuracy - average of ok-recall and NG-recall; fair when one "
    "class is rare.\n"
    "train / val / test - train = the model learns; val = picks the best "
    "epoch; test = the final exam.\n"
    "pos_weight - defects are rare, so each defect mistake counts more in "
    "the loss.\n"
    "threshold 0.5 - defect probability >= 0.5 means the photo is NG."
)


class TrainTab(ttk.Frame):
    """The '2. Training' tab: run train.py and watch the model learn."""

    def __init__(self, master):
        super().__init__(master, padding=8)
        self._train_proc = None
        self._train_running = False
        self._stop_requested = False
        self._run_dir = None
        self._run_cat = None
        self._last_run = None
        self._epoch_seconds = []
        self._run_start = None
        self._final_test = None
        self._gpu_q = queue.Queue()

        self.category_var = tk.StringVar()
        self.epochs_var = tk.StringVar(value="25")
        self.size_var = tk.StringVar(value="256")
        self.batch_var = tk.StringVar(value="32")
        self.lr_var = tk.StringVar(value="0.0001")
        self.summary_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.progress_var = tk.StringVar(value="")
        self.hint_var = tk.StringVar(value="Hover a block to see what it does.")
        self.active_var = tk.StringVar(value="")
        self.final_var = tk.StringVar(
            value="waiting for the final exam to finish..."
        )
        self.gpu_name_var = tk.StringVar(value="")
        self.gpu_util_var = tk.StringVar(value="")
        self.gpu_mem_var = tk.StringVar(value="")
        self.gpu_temp_var = tk.StringVar(value="")

        style = ttk.Style(self)
        style.configure("StatusInfo.TLabel", foreground="#333333")
        style.configure("StatusOk.TLabel", foreground="#1a6b1a")
        style.configure("StatusWarn.TLabel", foreground="#a05a00")
        style.configure("StatusErr.TLabel", foreground="#b00020")

        self._build_flow()
        self._build_bottom()
        self._build_middle()

        self.refresh_categories()
        self.category_var.set("hazelnut")
        self.refresh_category()

        threading.Thread(target=self._gpu_worker, daemon=True).start()
        self._gpu_after = self.after(500, self._gpu_tick)
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event):
        if event.widget is self:
            try:
                self.after_cancel(self._gpu_after)
            except tk.TclError:
                pass

    # ---------- UI construction ----------

    def _build_flow(self):
        strip = ttk.Frame(self)
        strip.pack(fill="x", pady=(0, 2))
        for i, (name, hint) in enumerate(TRAIN_FLOW):
            if i:
                ttk.Label(strip, text="->").pack(side="left", padx=2)
            b = tk.Label(
                strip, text=name, relief="groove", padx=8, pady=3,
                bg="#f4f6f8", font=FONT,
            )
            b.pack(side="left", padx=2)
            b.bind(
                "<Enter>",
                lambda e, n=name, h=hint: self.hint_var.set(f"{n}: {h}"),
            )
        ttk.Label(self, textvariable=self.hint_var,
                  foreground="#555555").pack(anchor="w", pady=(0, 4))

    def _build_middle(self):
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)
        left = ttk.Frame(mid, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        self._build_settings(left)
        self._build_gpu(left)
        self._build_active(left)
        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True)
        self._build_right(right)

    def _build_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Settings")
        box.pack(fill="x", pady=(0, 6))
        g = ttk.Frame(box)
        g.pack(fill="x", padx=6, pady=4)

        def row(r, label, widget):
            ttk.Label(g, text=label).grid(row=r, column=0, sticky="w", pady=1)
            widget.grid(row=r, column=1, sticky="w", padx=(6, 0), pady=1)

        self.category_box = ttk.Combobox(
            g, textvariable=self.category_var, state="readonly", width=16
        )
        row(0, "Category", self.category_box)
        self.category_box.bind(
            "<<ComboboxSelected>>", lambda e: self.refresh_category()
        )
        self.epochs_spin = ttk.Spinbox(
            g, from_=1, to=200, width=6, textvariable=self.epochs_var
        )
        row(1, "Epochs", self.epochs_spin)
        self.size_box = ttk.Combobox(
            g, textvariable=self.size_var, state="readonly", width=8,
            values=("128", "192", "224", "256", "320", "384", "512"),
        )
        row(2, "Image size", self.size_box)
        self.batch_box = ttk.Combobox(
            g, textvariable=self.batch_var, state="readonly", width=8,
            values=("8", "16", "32", "64"),
        )
        row(3, "Batch size", self.batch_box)
        self.lr_box = ttk.Combobox(
            g, textvariable=self.lr_var, state="readonly", width=8,
            values=("0.001", "0.0003", "0.0001", "0.00003"),
        )
        row(4, "Learning rate", self.lr_box)
        ttk.Label(
            box, textvariable=self.summary_var, wraplength=310, justify="left"
        ).pack(anchor="w", padx=6, pady=(2, 4))
        btns = ttk.Frame(box)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        self.start_btn = ttk.Button(
            btns, text="Start training", command=self.start_training
        )
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            btns, text="Stop", command=self.stop_training, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(8, 0))

    def _build_gpu(self, parent):
        box = ttk.LabelFrame(parent, text="GPU")
        box.pack(fill="x", pady=(0, 6))
        inner = ttk.Frame(box)
        inner.pack(fill="x", padx=6, pady=4)
        ttk.Label(inner, textvariable=self.gpu_name_var,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.gpu_util_bar = ttk.Progressbar(
            inner, mode="determinate", maximum=100
        )
        self.gpu_util_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(inner, textvariable=self.gpu_util_var).pack(anchor="w")
        self.gpu_mem_bar = ttk.Progressbar(
            inner, mode="determinate", maximum=100
        )
        self.gpu_mem_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(inner, textvariable=self.gpu_mem_var).pack(anchor="w")
        ttk.Label(inner, textvariable=self.gpu_temp_var).pack(
            anchor="w", pady=(2, 2)
        )

    def _build_active(self, parent):
        box = ttk.LabelFrame(parent, text="Active model")
        box.pack(fill="x")
        ttk.Label(
            box, textvariable=self.active_var, wraplength=310, justify="left"
        ).pack(anchor="w", padx=6, pady=4)
        self.active_btn = ttk.Button(
            box, text="Set last run as active model",
            command=self.set_active_model, state="disabled",
        )
        self.active_btn.pack(anchor="w", padx=6, pady=(0, 6))

    def _build_right(self, parent):
        prog = ttk.Frame(parent)
        prog.pack(fill="x", pady=(0, 4))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(prog, textvariable=self.progress_var, width=46).pack(
            side="left"
        )

        charts = ttk.Frame(parent)
        charts.pack(fill="x")
        self.loss_chart = LineChart(charts, "Loss")
        self.loss_chart.add_series("train loss", "#1f77b4")
        self.loss_chart.add_series("val loss", "#d62728")
        self.loss_chart.canvas.pack(
            side="left", padx=(0, 6), fill="x", expand=True
        )
        self.quality_chart = LineChart(
            charts, "Quality on val photos", percent=True
        )
        self.quality_chart.add_series("bal acc", "#2ca02c")
        self.quality_chart.add_series("recall NG", "#ff7f0e")
        self.quality_chart.add_series("precision NG", "#9467bd")
        self.quality_chart.canvas.pack(side="left", fill="x", expand=True)

        ec = ("epoch", "tr_loss", "tr_acc", "va_loss", "va_acc", "va_rec",
              "va_prec", "va_bal", "best", "sec")
        eh = ("Epoch", "train loss", "train acc", "val loss", "val acc",
              "val recall NG", "val precision NG", "val bal acc", "best",
              "seconds")
        ew = (50, 80, 75, 80, 70, 95, 105, 85, 45, 65)
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="x", pady=4)
        self.epoch_tree = ttk.Treeview(
            tree_frame, columns=ec, show="headings", height=6
        )
        for c, h_, w_ in zip(ec, eh, ew):
            self.epoch_tree.heading(c, text=h_)
            self.epoch_tree.column(c, width=w_, anchor="center")
        vsb = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.epoch_tree.yview
        )
        self.epoch_tree.configure(yscrollcommand=vsb.set)
        self.epoch_tree.pack(side="left", fill="x", expand=True)
        vsb.pack(side="right", fill="y")

        fin = ttk.LabelFrame(
            parent,
            text="Final exam on test photos (the model has never seen these)",
        )
        fin.pack(fill="x")
        inner = ttk.Frame(fin)
        inner.pack(fill="x", padx=6, pady=4)
        ttk.Label(inner, textvariable=self.final_var,
                  justify="left").pack(side="left", anchor="n")
        mtx = ttk.Frame(inner)
        mtx.pack(side="right")
        ttk.Label(mtx, text="").grid(row=0, column=0)
        ttk.Label(mtx, text="predicted OK",
                  font=("Segoe UI", 8, "bold")).grid(row=0, column=1, padx=2)
        ttk.Label(mtx, text="predicted NG",
                  font=("Segoe UI", 8, "bold")).grid(row=0, column=2, padx=2)
        ttk.Label(mtx, text="truth OK",
                  font=("Segoe UI", 8, "bold")).grid(row=1, column=0,
                                                    sticky="e", padx=2)
        ttk.Label(mtx, text="truth NG",
                  font=("Segoe UI", 8, "bold")).grid(row=2, column=0,
                                                    sticky="e", padx=2)
        self.mtx_labels = [w for w in mtx.winfo_children()]
        self.cell_tn = self._cell(mtx, 1, 1)
        self.cell_fp = self._cell(mtx, 1, 2)
        self.cell_fn = self._cell(mtx, 2, 1)
        self.cell_tp = self._cell(mtx, 2, 2)
        self._reset_final()

    def _cell(self, parent, r, c):
        lab = tk.Label(
            parent, text="-", width=14, height=3, bg="#eeeeee",
            relief="groove", font=("Segoe UI", 9), justify="center",
        )
        lab.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
        return lab

    def _build_bottom(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="bottom")

        gl = ttk.LabelFrame(bottom, text="What the words mean")
        gl.pack(side="left", fill="both", padx=(0, 6))
        ttk.Label(gl, text=GLOSSARY, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=(2, 0))
        self.status_label = ttk.Label(
            gl, textvariable=self.status_var, style="StatusInfo.TLabel"
        )
        self.status_label.pack(anchor="w", padx=6, pady=(2, 4))

        log_frame = ttk.LabelFrame(bottom, text="Script log")
        log_frame.pack(side="left", fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=7, state="disabled", font=("Consolas", 9)
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    # ---------- helpers ----------

    def set_status(self, text, style="StatusInfo.TLabel"):
        self.status_var.set(text)
        self.status_label.configure(style=style)

    def log_append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def refresh_categories(self):
        cats = [
            c for c in list_categories()
            if (BASE / f"dataset_{c}.csv").exists()
        ]
        if not cats:
            cats = ["hazelnut"]
        self.category_box.configure(values=cats)

    def refresh_category(self, *_):
        cat = self.category_var.get()
        path = BASE / f"dataset_{cat}.csv"
        df = None
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None
        if df is None:
            self.summary_var.set(
                "No table for this category - build it on the Data tab"
            )
            self.start_btn.configure(state="disabled")
        else:
            n_tr = int((df["split"] == "train").sum())
            n_va = int((df["split"] == "val").sum())
            n_te = int((df["split"] == "test").sum())
            tr = df[df["split"] == "train"]
            n_ng = int((tr["label"] == 1).sum())
            n_ok = len(tr) - n_ng
            pw = n_ok / max(n_ng, 1)
            self.summary_var.set(
                f"Table: {len(df)} photos | train {n_tr} / val {n_va} / "
                f"test {n_te} | ok {n_ok} / NG {n_ng} (train) -> "
                f"pos_weight {pw:.2f}"
            )
            if not self._train_running:
                self.start_btn.configure(state="normal")
        self.refresh_active_model()
        self._load_history(cat)
        self._update_active_btn()

    def refresh_active_model(self):
        cat = self.category_var.get()
        pth = BASE / f"model_{cat}.pth"
        js = BASE / f"model_{cat}.json"
        if not pth.exists():
            self.active_var.set("No model for this category yet")
            return
        if not js.exists():
            self.active_var.set(f"model_{cat}.pth (no info file)")
            return
        try:
            j = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            self.active_var.set(f"model_{cat}.json could not be read")
            return
        t = j.get("test") or {}
        created = str(j.get("created", "?"))[:10]
        if t:
            det = (f"found {t.get('tp', '?')}/"
                   f"{t.get('tp', 0) + t.get('fn', 0)} defects, "
                   f"{t.get('fp', '?')} false alarms")
        else:
            det = "no test data"
        self.active_var.set(
            f"model_{cat}.pth | best epoch {j.get('best_epoch', '?')} | "
            f"test: {det} | trained {created}"
        )

    def _load_history(self, cat):
        for iid in self.epoch_tree.get_children():
            self.epoch_tree.delete(iid)
        self._final_test = None
        hist = None
        js = BASE / f"model_{cat}.json"
        if js.exists():
            try:
                j = json.loads(js.read_text(encoding="utf-8"))
                if j.get("epochs"):
                    hist = j
            except Exception:
                pass
        if hist is None:
            self.loss_chart.clear("No history yet")
            self.quality_chart.clear("No history yet")
            self._reset_final()
            return
        self.loss_chart.clear("")
        self.quality_chart.clear("")
        for row in hist["epochs"]:
            self._push_epoch(row)
        self.loss_chart.best_epoch = hist.get("best_epoch")
        self.quality_chart.best_epoch = hist.get("best_epoch")
        if hist.get("test"):
            self._show_final(hist["test"])
        self.loss_chart.redraw()
        self.quality_chart.redraw()

    def _push_epoch(self, row):
        tr, va = row["train"], row["val"]
        self.epoch_tree.insert(
            "", "end",
            values=(
                row["epoch"], f"{tr['loss']:.4f}", f"{tr['acc']:.1%}",
                f"{va['loss']:.4f}", f"{va['acc']:.1%}",
                f"{va['recall_ng']:.1%}", f"{va['precision_ng']:.1%}",
                f"{va['bal_acc']:.1%}", "*" if row["best"] else "",
                f"{row['seconds']:.1f}s",
            ),
        )
        self.loss_chart.push(
            {"train loss": tr["loss"], "val loss": va["loss"]}
        )
        self.quality_chart.push(
            {"bal acc": va["bal_acc"], "recall NG": va["recall_ng"],
             "precision NG": va["precision_ng"]}
        )
        if row["best"]:
            self.loss_chart.best_epoch = row["epoch"]
            self.quality_chart.best_epoch = row["epoch"]

    def _reset_final(self):
        self.final_var.set("waiting for the final exam to finish...")
        for lab in (self.cell_tn, self.cell_fp, self.cell_fn, self.cell_tp):
            lab.configure(text="-", bg="#eeeeee")

    def _fill_cell(self, lab, count, caption, good):
        lab.configure(
            text=f"{count}\n{caption}",
            bg="#d9f2d9" if good else "#f2b8b8",
        )

    def _show_final(self, t):
        self._final_test = t
        self.final_var.set(
            f"accuracy {t['acc']:.1%}   recall NG {t['recall_ng']:.1%}\n"
            f"precision NG {t['precision_ng']:.1%}   "
            f"balanced acc {t['bal_acc']:.1%}"
        )
        self._fill_cell(self.cell_tn, t["tn"], "good\npassed", good=True)
        self._fill_cell(self.cell_fp, t["fp"], "good flagged\nas defect",
                        good=(t["fp"] == 0))
        self._fill_cell(self.cell_fn, t["fn"], "defects\nmissed",
                        good=(t["fn"] == 0))
        self._fill_cell(self.cell_tp, t["tp"], "defects\nfound", good=True)

    def _update_active_btn(self):
        ok = (self._last_run is not None
              and self._last_run["cat"] == self.category_var.get()
              and not self._train_running)
        self.active_btn.configure(state="normal" if ok else "disabled")

    # ---------- GPU polling ----------

    def _gpu_worker(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while True:
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3,
                    creationflags=CREATE_NO_WINDOW,
                )
                self._gpu_q.put(r.stdout.strip() if r.returncode == 0 else None)
            except Exception:
                self._gpu_q.put(None)
            time.sleep(1)

    def _gpu_tick(self):
        try:
            while True:
                line = self._gpu_q.get_nowait()
                if line is None:
                    self.gpu_name_var.set("nvidia-smi not available")
                    self.gpu_util_var.set("")
                    self.gpu_mem_var.set("")
                    self.gpu_temp_var.set("")
                    self.gpu_util_bar["value"] = 0
                    self.gpu_mem_bar["value"] = 0
                else:
                    try:
                        name, util, mu, mt, temp = [
                            p.strip() for p in line.split(",")
                        ]
                        self.gpu_name_var.set(name)
                        self.gpu_util_bar["value"] = float(util)
                        self.gpu_util_var.set(f"Utilization {util}%")
                        self.gpu_mem_bar["value"] = (
                            100.0 * float(mu) / float(mt)
                        )
                        self.gpu_mem_var.set(f"Memory {mu} / {mt} MB")
                        self.gpu_temp_var.set(f"Temperature {temp} C")
                    except Exception:
                        pass
        except queue.Empty:
            pass
        try:
            if self.winfo_exists():
                self._gpu_after = self.after(1000, self._gpu_tick)
        except tk.TclError:
            pass

    # ---------- running train.py ----------

    def _set_train_ui(self, running):
        csv_ok = (BASE / f"dataset_{self.category_var.get()}.csv").exists()
        self.start_btn.configure(
            state="normal" if (csv_ok and not running) else "disabled"
        )
        self.category_box.configure(
            state="disabled" if running else "readonly"
        )
        self.epochs_spin.configure(state="disabled" if running else "normal")
        for b in (self.size_box, self.batch_box, self.lr_box):
            b.configure(state="disabled" if running else "readonly")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self._update_active_btn()

    def start_training(self):
        if self._train_running:
            return
        cat = self.category_var.get()
        if not (BASE / f"dataset_{cat}.csv").exists():
            self.set_status(
                "No table for this category - build it on the Data tab",
                "StatusWarn.TLabel",
            )
            return
        try:
            epochs = int(self.epochs_var.get())
            size = int(self.size_var.get())
            batch = int(self.batch_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            self.set_status(
                "Check the settings - a number is wrong", "StatusErr.TLabel"
            )
            return
        run_dir = BASE / "runs" / f"{cat}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_out = run_dir / "model.pth"
        self._run_dir, self._run_cat = run_dir, cat
        self._last_run = None
        self._final_test = None
        self._epoch_seconds = []
        self._run_start = time.time()
        self._stop_requested = False
        self._train_running = True
        self._set_train_ui(True)
        for iid in self.epoch_tree.get_children():
            self.epoch_tree.delete(iid)
        self.loss_chart.clear("Waiting for the first epoch...")
        self.quality_chart.clear("Waiting for the first epoch...")
        self._reset_final()
        self.progress["maximum"] = epochs
        self.progress["value"] = 0
        self.progress_var.set(
            f"Epoch 0/{epochs} | loading photos into memory..."
        )
        self.log_clear()
        cmd = [PYTHON, str(BASE / "train.py"), cat,
               "--epochs", str(epochs), "--img-size", str(size),
               "--batch-size", str(batch), "--lr", str(lr),
               "--out", str(model_out)]
        self.log_append("> " + " ".join(cmd))
        self.set_status("Loading", "StatusInfo.TLabel")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        q = queue.Queue()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(BASE), env=env, text=True, encoding="utf-8",
                errors="replace", creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            self._train_running = False
            self._train_proc = None
            self._set_train_ui(False)
            self.set_status(
                "Could not start train.py - see the log below",
                "StatusErr.TLabel",
            )
            self.log_append(traceback.format_exc())
            return
        self._train_proc = proc

        def reader():
            try:
                for line in proc.stdout:
                    q.put(("line", line.rstrip("\n")))
                q.put(("done", proc.wait()))
            except Exception:
                q.put(("fail", traceback.format_exc()))

        threading.Thread(target=reader, daemon=True).start()
        self.after(100, lambda: self._poll_train(q))
        self._eta_tick()

    def stop_training(self):
        if not self._train_running or self._train_proc is None:
            return
        self._stop_requested = True
        self.set_status("Stopping...", "StatusWarn.TLabel")
        try:
            self._train_proc.terminate()
        except Exception:
            pass
        self.after(3000, self._kill_train_proc)

    def _kill_train_proc(self):
        p = self._train_proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def kill_proc_now(self):
        """Used when the whole app is closing: kill a running script."""
        p = self._train_proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def _eta_tick(self):
        if not self._train_running:
            return
        elapsed = time.time() - self._run_start
        done = len(self._epoch_seconds)
        epochs = self.progress["maximum"]
        if done:
            avg = sum(self._epoch_seconds) / done
            eta = avg * (epochs - done)
            self.progress_var.set(
                f"Epoch {done}/{epochs} | elapsed {fmt_seconds(elapsed)} | "
                f"ETA {fmt_seconds(eta)}"
            )
        else:
            self.progress_var.set(
                f"Epoch 0/{epochs} | elapsed {fmt_seconds(elapsed)} | "
                "loading photos into memory..."
            )
        self.after(1000, self._eta_tick)

    def _poll_train(self, q):
        done, rc = False, None
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "line":
                    self._handle_train_line(payload)
                elif kind == "done":
                    done, rc = True, payload
                elif kind == "fail":
                    done, rc = True, -1
                    self.log_append(payload)
        except queue.Empty:
            pass
        if not done:
            self.after(100, lambda: self._poll_train(q))
            return
        self._train_running = False
        self._train_proc = None
        self._set_train_ui(False)
        if self._stop_requested:
            self.set_status("Stopped", "StatusWarn.TLabel")
        elif rc == 0:
            self.set_status("Finished", "StatusOk.TLabel")
        else:
            self.set_status(
                f"Error (code {rc}) - see the log below", "StatusErr.TLabel"
            )
        self._update_active_btn()

    def _handle_train_line(self, s):
        if not s.startswith("@@"):
            self.log_append(s)
            return
        tag, _, payload = s.partition(" ")
        try:
            data = json.loads(payload)
        except Exception:
            self.log_append(s)
            return
        if tag == "@@INFO":
            self.progress["maximum"] = data.get(
                "epochs", self.progress["maximum"]
            )
        elif tag == "@@EPOCH":
            self._epoch_seconds.append(data.get("seconds", 0))
            self._push_epoch(data)
            self.progress["value"] = data["epoch"]
            self.set_status(
                f"Training epoch {data['epoch']}", "StatusInfo.TLabel"
            )
        elif tag == "@@FINAL":
            self._show_final(data["test"])
            self.progress["value"] = self.progress["maximum"]
            self._last_run = {
                "cat": self._run_cat,
                "model": self._run_dir / "model.pth",
                "json": self._run_dir / "model.json",
                "test": data["test"],
            }
            self._update_active_btn()
        else:
            self.log_append(s)

    # ---------- set last run as the active model ----------

    def set_active_model(self):
        run = self._last_run
        cat = self.category_var.get()
        if run is None or run["cat"] != cat:
            return
        dst_pth = BASE / f"model_{cat}.pth"
        dst_json = BASE / f"model_{cat}.json"
        old_test = None
        if dst_json.exists():
            try:
                old_test = json.loads(
                    dst_json.read_text(encoding="utf-8")
                ).get("test")
            except Exception:
                old_test = None

        def fmt(t):
            if not t:
                return "no test info"
            return (f"balanced acc {t['bal_acc']:.0%}, "
                    f"defects found {t['tp']}/{t['tp'] + t['fn']}")

        if dst_pth.exists():
            msg = (f"Replace the active model for '{cat}'?\n\n"
                   f"Current: {fmt(old_test)}\n"
                   f"New run: {fmt(run['test'])}\n\n"
                   f"The current model will be kept as "
                   f"model_{cat}.backup.pth.")
        else:
            msg = (f"Set the last run as the active model for '{cat}'?\n\n"
                   f"New run: {fmt(run['test'])}")
        if not messagebox.askyesno("Set active model", msg):
            return
        try:
            if dst_pth.exists():
                shutil.copy2(dst_pth, BASE / f"model_{cat}.backup.pth")
                if dst_json.exists():
                    shutil.copy2(dst_json, BASE / f"model_{cat}.backup.json")
            shutil.copy2(run["model"], dst_pth)
            if run["json"].exists():
                shutil.copy2(run["json"], dst_json)
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy files:\n{e}")
            return
        self.refresh_active_model()
        self.set_status(
            f"model_{cat}.pth is now the active model", "StatusOk.TLabel"
        )


CHECK_FLOW = [
    ("Photo", "any photo of the same kind of object the model was "
     "trained on"),
    ("Resize + normalize", "photo is resized to NxN and its numbers are "
     "rescaled exactly like during training"),
    ("ResNet18", "the trained network gives one raw score (logit)"),
    ("Probability", "sigmoid turns the score into 0-100% probability "
     "that the photo is defective"),
    ("Threshold", "if probability >= threshold the photo is called NG, "
     "otherwise OK"),
    ("OK / NG", "the final verdict"),
]

CHECK_NOTES = (
    "The model only knows the object it was trained on (here: "
    "hazelnuts). Photos of other objects give meaningless answers.\n"
    "Photos are squeezed to a square NxN image - use photos taken like "
    "the training ones (same camera, light, distance).\n"
    "A very high confidence does not guarantee a correct answer; check "
    "the mistakes with your own eyes.\n"
    "Photos from the dataset's train part were seen by the model during "
    "training - use test photos for an honest check."
)


class CheckTab(ttk.Frame):
    """The '3. Check' tab: run the active model on photos.

    app.py never imports torch. The check.py module is imported lazily
    inside ONE worker thread, which also owns the loaded model; the UI
    only sends jobs to the worker and receives results back via queues.
    The model is cached per category and reloaded when model_<cat>.pth
    changes on disk (its mtime differs from the cached one).
    """

    PHOTO_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, master):
        super().__init__(master, padding=8)
        self._check_mod = None      # check.py module, imported in worker
        self._model = None          # (model, tf, img_size, threshold)
        self._model_cat = None
        self._model_mtime = None
        self._model_img_size = None
        self._model_thr = 0.5       # threshold stored in the model/json
        self._worker = None
        self._jobs = queue.Queue()      # UI -> worker
        self._results = queue.Queue()   # worker -> UI
        self._gen = 0               # job generation; stale jobs dropped
        self._pending = 0           # jobs sent but not yet answered
        self._predict_calls = 0     # real model invocations
        self._rows = []             # folder results, one dict per photo
        self._folder_total = 0
        self._errors = 0
        self._df = None             # dataset_<cat>.csv for truth lookup
        self._truth = {}            # normalized path -> label 0/1
        self._sort_col = "file"
        self._sort_rev = False
        self._photo_img = None
        self._single_path = None
        self._single_p = None
        self._single_truth = None
        self._single_waiting = False

        self.category_var = tk.StringVar()
        self.model_info_var = tk.StringVar(value="")
        self.hint_var = tk.StringVar(value="Hover a block to see what it does.")
        self.thr_var = tk.DoubleVar(value=0.5)
        self.thr_label_var = tk.StringVar(value="Threshold: 50%")
        self.verdict_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="")
        self.truth_var = tk.StringVar(value="")
        self.explain_var = tk.StringVar(
            value="Pick a photo to see how the verdict is made."
        )
        self.progress_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.mtx_caption_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")

        style = ttk.Style(self)
        style.configure("StatusInfo.TLabel", foreground="#333333")
        style.configure("StatusOk.TLabel", foreground="#1a6b1a")
        style.configure("StatusWarn.TLabel", foreground="#a05a00")
        style.configure("StatusErr.TLabel", foreground="#b00020")

        self._build_flow()
        self._build_controls()
        self._build_threshold()
        self._build_middle()
        self._build_bottom()

        self.refresh_categories()
        self.category_var.set("hazelnut")
        self.refresh_category()
        self._poll_after = self.after(100, self._poll_results)
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event):
        if event.widget is self:
            try:
                self.after_cancel(self._poll_after)
            except tk.TclError:
                pass

    # ---------- UI construction ----------

    def _build_flow(self):
        strip = ttk.Frame(self)
        strip.pack(fill="x", pady=(0, 2))
        for i, (name, hint) in enumerate(CHECK_FLOW):
            if i:
                ttk.Label(strip, text="->").pack(side="left", padx=2)
            b = tk.Label(
                strip, text=name, relief="groove", padx=8, pady=3,
                bg="#f4f6f8", font=FONT,
            )
            b.pack(side="left", padx=2)
            b.bind(
                "<Enter>",
                lambda e, n=name, h=hint: self.hint_var.set(f"{n}: {h}"),
            )
        ttk.Label(self, textvariable=self.hint_var,
                  foreground="#555555").pack(anchor="w", pady=(0, 4))

    def _build_controls(self):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Category:").pack(side="left")
        self.category_box = ttk.Combobox(
            row, textvariable=self.category_var, width=12, state="readonly"
        )
        self.category_box.pack(side="left", padx=(2, 10))
        self.category_box.bind(
            "<<ComboboxSelected>>", lambda e: self.refresh_category())
        self.open_btn = ttk.Button(
            row, text="Open photo...", command=self.open_photo)
        self.open_btn.pack(side="left", padx=(0, 4))
        self.folder_btn = ttk.Button(
            row, text="Open folder...", command=self.open_folder)
        self.folder_btn.pack(side="left", padx=(0, 4))
        self.random_btn = ttk.Button(
            row, text="Random test photo", command=self.random_test_photo)
        self.random_btn.pack(side="left", padx=(0, 10))
        ttk.Label(row, textvariable=self.model_info_var,
                  foreground="#333333",
                  font=("Segoe UI", 9)).pack(side="left", fill="x",
                                             expand=True)

    def _build_threshold(self):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Threshold").pack(side="left")
        self.thr_scale = ttk.Scale(
            row, from_=0.05, to=0.95, variable=self.thr_var,
            command=self._on_threshold, length=260,
        )
        self.thr_scale.pack(side="left", padx=6)
        ttk.Label(row, textvariable=self.thr_label_var,
                  width=14).pack(side="left")
        ttk.Button(row, text="Reset",
                   command=self._reset_threshold).pack(side="left",
                                                       padx=(0, 10))
        ttk.Label(
            row,
            text="Lower threshold -> fewer missed defects but more false "
                 "alarms. Higher threshold -> the opposite.",
            foreground="#555555", font=("Segoe UI", 9),
        ).pack(side="left")

    def _build_middle(self):
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)
        left = ttk.Frame(mid, width=520)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True)

        pv = tk.Frame(left, width=500, height=340, bg="#e8e8e8",
                      relief="sunken", bd=1)
        pv.pack(pady=(0, 4))
        pv.pack_propagate(False)
        self.preview = tk.Label(pv, bg="#e8e8e8", text="no photo")
        self.preview.pack(expand=True)

        self.verdict_label = tk.Label(
            left, textvariable=self.verdict_var,
            font=("Segoe UI", 28, "bold"))
        self.verdict_label.pack()
        ttk.Label(left, textvariable=self.conf_var,
                  font=("Segoe UI", 12)).pack()
        ttk.Label(left, text="Probability of defect (bar) "
                  "vs threshold (line)",
                  font=("Segoe UI", 8),
                  foreground="#555555").pack(anchor="w", pady=(4, 0))
        self.thermo = tk.Canvas(left, height=56, bg="white",
                                highlightthickness=0)
        self.thermo.pack(fill="x")
        self.thermo.bind("<Configure>", lambda e: self._draw_thermo())

        ex = ttk.LabelFrame(left, text="How this verdict was made")
        ex.pack(fill="x", pady=(4, 0))
        ttk.Label(ex, textvariable=self.explain_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w", padx=6, pady=4)
        self.truth_label = ttk.Label(ex, textvariable=self.truth_var,
                                     font=("Segoe UI", 10, "bold"))
        self.truth_label.pack(anchor="w", padx=6, pady=(0, 4))

        # right: folder results
        pr = ttk.Frame(right)
        pr.pack(fill="x")
        self.progress = ttk.Progressbar(pr, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        ttk.Label(pr, textvariable=self.progress_var,
                  width=16).pack(side="left", padx=6)

        tf = ttk.Frame(right)
        tf.pack(fill="both", expand=True, pady=(2, 0))
        cols = ("file", "prob", "verdict", "truth")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings",
                                 height=15)
        heads = {"file": ("File", 220, "w"),
                 "prob": ("Defect probability", 120, "center"),
                 "verdict": ("Verdict", 70, "center"),
                 "truth": ("Truth", 60, "center")}
        for c, (txt, w, anchor) in heads.items():
            self.tree.heading(c, text=txt,
                              command=lambda c=c: self._sort_by(c))
            self.tree.column(c, width=w, anchor=anchor)
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self.tree.tag_configure("ng", foreground="#b00020")
        self.tree.tag_configure(
            "wrong", background="#fff0a8", font=("Segoe UI", 10, "bold"))
        self.tree.tag_configure("err", foreground="#888888")

        srow = ttk.Frame(right)
        srow.pack(fill="x", pady=(4, 0))
        ttk.Label(srow, textvariable=self.summary_var,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        self.save_btn = ttk.Button(srow, text="Save results to CSV...",
                                   command=self.save_results)
        self.save_btn.pack(side="right")

        ttk.Label(right, textvariable=self.mtx_caption_var,
                  font=("Segoe UI", 9)).pack(anchor="w")
        mtx = ttk.Frame(right)
        mtx.pack(anchor="w")
        ttk.Label(mtx, text="").grid(row=0, column=0)
        ttk.Label(mtx, text="predicted OK",
                  font=("Segoe UI", 8, "bold")).grid(row=0, column=1)
        ttk.Label(mtx, text="predicted NG",
                  font=("Segoe UI", 8, "bold")).grid(row=0, column=2)
        ttk.Label(mtx, text="truth OK",
                  font=("Segoe UI", 8, "bold")).grid(row=1, column=0,
                                                    sticky="e", padx=2)
        ttk.Label(mtx, text="truth NG",
                  font=("Segoe UI", 8, "bold")).grid(row=2, column=0,
                                                    sticky="e", padx=2)
        self.mtx_labels = [w for w in mtx.winfo_children()]
        self.cell_tn = self._cell(mtx, 1, 1)
        self.cell_fp = self._cell(mtx, 1, 2)
        self.cell_fn = self._cell(mtx, 2, 1)
        self.cell_tp = self._cell(mtx, 2, 2)
        self._reset_matrix()

    def _cell(self, parent, r, c):
        lab = tk.Label(
            parent, text="-", width=14, height=3, bg="#eeeeee",
            relief="groove", font=("Segoe UI", 9), justify="center",
        )
        lab.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
        return lab

    def _build_bottom(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", side="bottom", pady=(4, 0))
        gl = ttk.LabelFrame(bottom, text="Good to know")
        gl.pack(side="left", fill="x", expand=True)
        self.notes_frame = gl
        ttk.Label(gl, text=CHECK_NOTES, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=2)
        self.status_label = ttk.Label(
            gl, textvariable=self.status_var, style="StatusInfo.TLabel")
        self.status_label.pack(anchor="w", padx=6, pady=(0, 4))

    # ---------- category and model info ----------

    def set_status(self, text, style="StatusInfo.TLabel"):
        self.status_var.set(text)
        self.status_label.configure(style=style)

    def refresh_categories(self):
        cats = [c for c in list_categories()
                if (BASE / f"model_{c}.pth").exists()]
        self.category_box.configure(values=cats)
        if not cats:
            self.model_info_var.set(
                "No trained models found - train one on the Training tab")

    def refresh_category(self, *_):
        """Category changed: reload truth map, drop pending jobs, reset."""
        cat = self.category_var.get()
        self._gen += 1
        self._load_truth(cat)
        self._clear_results()
        self._refresh_model_block()
        thr = self._model_threshold_of(cat)
        self._model_thr = thr
        self.thr_var.set(thr)
        self._on_threshold()

    def _refresh_model_block(self):
        """Update model info text and button states (no result clearing)."""
        cat = self.category_var.get()
        pth = BASE / f"model_{cat}.pth"
        buttons = (self.open_btn, self.folder_btn, self.random_btn,
                   self.save_btn)
        if not pth.exists():
            self.model_info_var.set(
                "No trained model for this category - train one on the "
                "Training tab")
            for b in buttons:
                b.configure(state="disabled")
            return
        for b in buttons:
            b.configure(state="normal")
        js = BASE / f"model_{cat}.json"
        size_mb = pth.stat().st_size / 1e6
        j = None
        if js.exists():
            try:
                j = json.loads(js.read_text(encoding="utf-8"))
            except Exception:
                j = None
        if j is None:
            self.model_info_var.set(
                f"Model: model_{cat}.pth | {size_mb:.1f} MB")
            return
        img_size = (j.get("settings") or {}).get("img_size", "?")
        created = str(j.get("created", "?"))[:10]
        t = j.get("test") or {}
        if t:
            det = (f"found {t.get('tp', '?')}/"
                   f"{t.get('tp', 0) + t.get('fn', 0)} defects, "
                   f"{t.get('fp', '?')} false alarms")
        else:
            det = "no test data"
        self.model_info_var.set(
            f"Model: model_{cat}.pth | image size {img_size} | "
            f"trained {created} | exam on test: {det}")

    def _model_threshold_of(self, cat):
        js = BASE / f"model_{cat}.json"
        if js.exists():
            try:
                j = json.loads(js.read_text(encoding="utf-8"))
                return float(
                    (j.get("settings") or {}).get("threshold", 0.5))
            except Exception:
                pass
        return 0.5

    def _load_truth(self, cat):
        self._df = None
        self._truth = {}
        path = BASE / f"dataset_{cat}.csv"
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        self._df = df
        for p, lab in zip(df["path"], df["label"]):
            key = os.path.normcase(os.path.normpath(str(p)))
            self._truth[key] = int(lab)

    def _truth_of(self, path):
        key = os.path.normcase(os.path.normpath(str(path)))
        return self._truth.get(key)

    def _clear_results(self):
        self._rows = []
        self._folder_total = 0
        self._errors = 0
        self._refill_tree()
        self.summary_var.set("")
        self.progress_var.set("")
        self.progress.configure(value=0)
        self.mtx_caption_var.set("")
        self._reset_matrix()
        self._single_path = None
        self._single_p = None
        self._single_truth = None
        self._single_waiting = False
        self._photo_img = None
        self.preview.configure(image="", text="no photo")
        self._render_single()

    def _reset_matrix(self):
        for lab in (self.cell_tn, self.cell_fp, self.cell_fn, self.cell_tp):
            lab.configure(text="-", bg="#eeeeee")

    def _fill_cell(self, lab, count, caption, good):
        lab.configure(
            text=f"{count}\n{caption}",
            bg="#d9f2d9" if good else "#f2b8b8",
        )

    # ---------- worker thread (the only place torch lives) ----------

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True)
            self._worker.start()

    def _enqueue(self, kind, payload):
        self._ensure_worker()
        self._jobs.put((self._gen, kind, payload))
        self._pending += 1

    def _worker_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            gen, kind, payload = job
            if gen != self._gen:
                self._results.put((gen, "drop", payload, None))
                continue
            try:
                if kind == "load":
                    img_size, thr = self._ensure_model(payload)
                    self._results.put(
                        (gen, "load_ok", payload, (img_size, thr)))
                elif kind == "predict":
                    cat, path = payload
                    self._ensure_model(cat)
                    _is_ng, p = self._check_mod.predict(
                        self._model[0], self._model[1], self._model[2],
                        self._model[3], str(path))
                    self._predict_calls += 1
                    self._results.put((gen, "pred_ok", str(path), p))
            except Exception as e:
                self._results.put((gen, kind + "_err", payload, str(e)))

    def _ensure_model(self, cat):
        """Load check.py and model_<cat>.pth; worker thread only."""
        if self._check_mod is None:
            if str(BASE) not in sys.path:
                sys.path.insert(0, str(BASE))
            import check as check_mod
            self._check_mod = check_mod
        pth = BASE / f"model_{cat}.pth"
        mtime = pth.stat().st_mtime
        if (self._model is None or self._model_cat != cat
                or mtime != self._model_mtime):
            self._model = self._check_mod.load_model(cat)
            self._model_cat = cat
            self._model_mtime = mtime
            self._model_img_size = self._model[2]
        return self._model[2], self._model[3]

    def _model_stale(self, cat):
        try:
            mtime = (BASE / f"model_{cat}.pth").stat().st_mtime
        except OSError:
            mtime = None
        return (self._model is None or self._model_cat != cat
                or mtime != self._model_mtime)

    def _poll_results(self):
        try:
            while True:
                gen, kind, payload, data = self._results.get_nowait()
                self._pending = max(0, self._pending - 1)
                if gen != self._gen:
                    continue
                if kind == "load_ok":
                    self._model_img_size = data[0]
                    if self._pending == 0 and not self._folder_busy():
                        self.set_status("Done", "StatusOk.TLabel")
                elif kind == "load_err":
                    self.set_status(f"Error loading model: {data}",
                                    "StatusErr.TLabel")
                elif kind == "pred_ok":
                    self._on_predicted(payload, data)
                elif kind == "predict_err":
                    self._on_predict_error(payload, data)
        except queue.Empty:
            pass
        try:
            self._poll_after = self.after(100, self._poll_results)
        except tk.TclError:
            pass

    # ---------- actions ----------

    def preload_model(self):
        """Load the model in the worker without checking a photo."""
        cat = self.category_var.get()
        if not (BASE / f"model_{cat}.pth").exists():
            return
        self._enqueue("load", cat)
        self.set_status("Loading model...")

    def check_photo(self, path, truth=None):
        """Check one photo; truth=0/1 marks it as a known OK/NG photo."""
        cat = self.category_var.get()
        path = str(path)
        self._single_path = path
        self._single_p = None
        self._single_truth = (truth if truth is not None
                              else self._truth_of(path))
        self._single_waiting = True
        self._show_preview(path)
        self._render_single()
        self._enqueue("predict", (cat, path))
        self.set_status(
            "Loading model..." if self._model_stale(cat) else "Checking...")

    def check_folder(self, paths):
        """Check a list of photo paths; fills the results table."""
        cat = self.category_var.get()
        self._gen += 1
        self._single_waiting = False
        self._rows = []
        self._errors = 0
        self._folder_total = len(paths)
        for p in paths:
            path = str(p)
            self._rows.append({
                "path": path, "name": Path(path).name, "p": None,
                "state": "pending", "error": "",
                "truth": self._truth_of(path),
            })
        self._sort_col = "file"
        self._sort_rev = False
        self._refill_tree()
        self.progress.configure(maximum=max(len(paths), 1), value=0)
        self.progress_var.set(f"Checking 0/{len(paths)}")
        self.set_status(
            "Loading model..." if self._model_stale(cat) else "Checking...")
        for p in paths:
            self._enqueue("predict", (cat, str(p)))

    def _folder_busy(self):
        return any(r["state"] == "pending" for r in self._rows)

    def open_photo(self):
        path = filedialog.askopenfilename(
            parent=self, title="Open photo",
            filetypes=[("Photos", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if path:
            self.check_photo(path)

    def open_folder(self):
        d = filedialog.askdirectory(
            parent=self, title="Open folder with photos")
        if not d:
            return
        paths = sorted(
            p for p in Path(d).iterdir()
            if p.is_file() and p.suffix.lower() in self.PHOTO_EXTS)
        if not paths:
            self.set_status(
                "No photos (.png .jpg .jpeg .bmp) in that folder",
                "StatusWarn.TLabel")
            return
        if len(paths) > 5000 and not messagebox.askyesno(
                "Many files",
                f"{len(paths)} photos found. Check them all?"):
            return
        self.check_folder(paths)

    def random_test_photo(self):
        if self._df is None or "split" not in self._df.columns:
            self.set_status("No dataset table for this category - "
                            "build it on the Data tab", "StatusWarn.TLabel")
            return
        test = self._df[self._df["split"] == "test"]
        if test.empty:
            self.set_status("No test photos in the dataset table",
                            "StatusWarn.TLabel")
            return
        row = test.sample(1).iloc[0]
        self.check_photo(str(row["path"]), truth=int(row["label"]))

    # ---------- results from the worker ----------

    def _on_predicted(self, path, p):
        row = next((r for r in self._rows if r["path"] == path), None)
        if row is not None:
            row["p"] = p
            row["state"] = "done"
            self._update_row(row)
            done = sum(1 for r in self._rows
                       if r["state"] in ("done", "error"))
            self.progress_var.set(
                f"Checking {done}/{self._folder_total}")
            self.progress["value"] = done
            if done == self._folder_total:
                self._folder_finished()
        if self._single_path == path:
            self._single_p = p
            self._single_waiting = False
            self._render_single()
            if not self._folder_busy():
                self.set_status("Done", "StatusOk.TLabel")

    def _on_predict_error(self, payload, msg):
        _cat, path = payload if isinstance(payload, tuple) else ("?", payload)
        path = str(path)
        row = next((r for r in self._rows if r["path"] == path), None)
        if row is not None:
            row["state"] = "error"
            row["error"] = msg
            self._errors += 1
            self._update_row(row)
            done = sum(1 for r in self._rows
                       if r["state"] in ("done", "error"))
            self.progress_var.set(
                f"Checking {done}/{self._folder_total}")
            self.progress["value"] = done
            self.set_status(
                f"Error reading {row['name']}: {msg}", "StatusErr.TLabel")
            if done == self._folder_total:
                self._folder_finished()
        if self._single_path == path:
            self._single_p = None
            self._single_waiting = False
            self._render_single()
            self.set_status(
                f"Error reading {Path(path).name}: {msg}",
                "StatusErr.TLabel")

    def _folder_finished(self):
        self._update_summary()
        if self._errors:
            self.set_status(
                f"Done - {self._errors} file(s) could not be read",
                "StatusWarn.TLabel")
        else:
            self.set_status("Done", "StatusOk.TLabel")

    # ---------- rendering (no model calls here) ----------

    def _verdict_of(self, row):
        if row["state"] == "error":
            return "ERROR"
        if row["p"] is None:
            return "..."
        return "NG" if row["p"] >= float(self.thr_var.get()) else "OK"

    def _row_tag(self, row):
        verdict = self._verdict_of(row)
        if (row["truth"] is not None and row["p"] is not None
                and row["state"] == "done"):
            pred_ng = row["p"] >= float(self.thr_var.get())
            if pred_ng != (row["truth"] == 1):
                return "wrong"
        if verdict == "ERROR":
            return "err"
        if verdict == "NG":
            return "ng"
        return ""

    def _row_values(self, row):
        p = "-" if row["p"] is None else f"{row['p']:.1%}"
        truth = ("-" if row["truth"] is None
                 else ("NG" if row["truth"] == 1 else "OK"))
        verdict = self._verdict_of(row)
        if row["state"] == "error":
            verdict = "ERROR"
        return (row["name"], p, verdict, truth)

    def _refill_tree(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for row in self._rows:
            tag = self._row_tag(row)
            self.tree.insert(
                "", "end", iid=row["path"],
                values=self._row_values(row),
                tags=(tag,) if tag else ())

    def _update_row(self, row):
        if self.tree.exists(row["path"]):
            tag = self._row_tag(row)
            self.tree.item(row["path"], values=self._row_values(row),
                           tags=(tag,) if tag else ())
        self._update_summary()

    def _sort_by(self, col):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = (col == "prob")
        if col == "prob":
            have = [r for r in self._rows if r["p"] is not None]
            none = [r for r in self._rows if r["p"] is None]
            have.sort(key=lambda r: r["p"], reverse=self._sort_rev)
            self._rows = have + none
        elif col == "file":
            self._rows.sort(key=lambda r: r["name"].lower(),
                            reverse=self._sort_rev)
        elif col == "verdict":
            self._rows.sort(key=self._verdict_of, reverse=self._sort_rev)
        elif col == "truth":
            self._rows.sort(
                key=lambda r: (r["truth"] is None, r["truth"]),
                reverse=self._sort_rev)
        self._refill_tree()

    def _on_row_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        row = next((r for r in self._rows if r["path"] == sel[0]), None)
        if row is None:
            return
        self._single_path = row["path"]
        self._single_p = row["p"]
        self._single_truth = row["truth"]
        self._single_waiting = row["state"] == "pending"
        self._show_preview(row["path"])
        self._render_single()

    def _on_threshold(self, *_):
        thr = round(float(self.thr_var.get()), 2)
        thr = min(0.95, max(0.05, thr))
        self.thr_var.set(thr)
        self.thr_label_var.set(f"Threshold: {thr:.0%}")
        # verdicts are recomputed from stored probabilities - no re-run
        self._render_single()
        for row in self._rows:
            self._update_row(row)
        self._update_summary()

    def _reset_threshold(self):
        self.thr_var.set(self._model_thr)
        self._on_threshold()

    def _render_single(self):
        self._draw_thermo()
        p = self._single_p
        thr = float(self.thr_var.get())
        if p is None:
            if self._single_path:
                self.verdict_var.set("...")
                self.verdict_label.configure(fg="#333333")
                self.conf_var.set("")
                self.explain_var.set("checking...")
            else:
                self.verdict_var.set("-")
                self.verdict_label.configure(fg="#333333")
                self.conf_var.set("")
                self.explain_var.set(
                    "Pick a photo to see how the verdict is made.")
            self.truth_var.set("")
            return
        is_ng = p >= thr
        self.verdict_var.set("NG" if is_ng else "OK")
        self.verdict_label.configure(
            fg="#b00020" if is_ng else "#1a6b1a")
        conf = p if is_ng else 1 - p
        self.conf_var.set(f"Confidence {conf:.1%}")
        pc = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(pc / (1 - pc))
        size = self._model_img_size or "?"
        cmp_txt = "is at/above" if is_ng else "is below"
        self.explain_var.set(
            f"1. Photo resized to {size}x{size} and normalized.\n"
            f"2. The network gave a raw score (logit) = {logit:.2f}.\n"
            f"3. Sigmoid turns the score into probability of defect = "
            f"{p:.1%}.\n"
            f"4. {p:.1%} {cmp_txt} the threshold {thr:.0%} -> "
            f"{'NG' if is_ng else 'OK'}"
        )
        if self._single_truth is not None:
            truth_ng = self._single_truth == 1
            ok = truth_ng == is_ng
            self.truth_var.set(
                f"Truth: {'NG' if truth_ng else 'OK'} | model: "
                f"{'NG' if is_ng else 'OK'} -> "
                f"{'correct' if ok else 'WRONG'}")
            self.truth_label.configure(
                foreground="#1a6b1a" if ok else "#b00020")
        else:
            self.truth_var.set("")

    def _draw_thermo(self):
        c = self.thermo
        c.delete("all")
        w = max(c.winfo_width(), 120)
        p = self._single_p if self._single_p is not None else 0.0
        thr = float(self.thr_var.get())
        x0, x1 = 12, w - 12
        y0, y1 = 14, 32
        c.create_rectangle(x0, y0, x1, y1, outline="#888888")
        if p > 0:
            fx = x0 + (x1 - x0) * min(p, 1.0)
            # green for low probability, red for high
            r = int(0x30 + p * (0xC0 - 0x30))
            g = int(0x9E - p * (0x9E - 0x20))
            color = f"#{r:02x}{g:02x}30"
            c.create_rectangle(x0, y0, fx, y1, fill=color, outline="")
        if self._single_p is not None:
            c.create_text(x1 - 4, (y0 + y1) // 2, text=f"{p:.1%}",
                          anchor="e", font=("Segoe UI", 9, "bold"))
        tx = x0 + (x1 - x0) * thr
        c.create_line(tx, y0 - 3, tx, y1 + 3, fill="#000000", width=2)
        c.create_text(tx, y0 - 4, text="threshold", anchor="s",
                      font=("Segoe UI", 7))
        for f, lab in ((0.0, "0%"), (0.25, "25%"), (0.5, "50%"),
                       (0.75, "75%"), (1.0, "100%")):
            lx = x0 + (x1 - x0) * f
            c.create_line(lx, y1, lx, y1 + 3, fill="#888888")
            c.create_text(lx, y1 + 4, text=lab, anchor="n",
                          font=("Segoe UI", 7))

    def _counts(self):
        """(ok, ng, errors) at the current threshold - no model calls."""
        thr = float(self.thr_var.get())
        ok = ng = err = 0
        for r in self._rows:
            if r["state"] == "error":
                err += 1
            elif r["p"] is not None:
                if r["p"] >= thr:
                    ng += 1
                else:
                    ok += 1
        return ok, ng, err

    def _update_summary(self):
        ok, ng, err = self._counts()
        thr = float(self.thr_var.get())
        if not self._rows:
            self.summary_var.set("")
            self.mtx_caption_var.set("")
            self._reset_matrix()
            return
        txt = f"Photos: {len(self._rows)} | OK: {ok} | NG: {ng}"
        if err:
            txt += f" | errors: {err}"
        self.summary_var.set(txt)
        tp = tn = fp = fn = known = 0
        for r in self._rows:
            if r["truth"] is None or r["p"] is None:
                continue
            known += 1
            pred_ng = r["p"] >= thr
            if r["truth"] == 1 and pred_ng:
                tp += 1
            elif r["truth"] == 1:
                fn += 1
            elif pred_ng:
                fp += 1
            else:
                tn += 1
        if known:
            self.mtx_caption_var.set(
                f"for the {known} photo(s) with a known truth:")
            self._fill_cell(self.cell_tn, tn, "good\npassed", good=True)
            self._fill_cell(self.cell_fp, fp, "good flagged\nas defect",
                            good=(fp == 0))
            self._fill_cell(self.cell_fn, fn, "defects\nmissed",
                            good=(fn == 0))
            self._fill_cell(self.cell_tp, tp, "defects\nfound", good=True)
        else:
            self.mtx_caption_var.set("")
            self._reset_matrix()

    def _show_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((496, 336))
            self._photo_img = ImageTk.PhotoImage(img)
            self.preview.configure(image=self._photo_img, text="")
        except Exception:
            self._photo_img = None
            self.preview.configure(image="",
                                   text="cannot preview this file")

    def _write_results_csv(self, path):
        thr = float(self.thr_var.get())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "p_defect", "verdict", "threshold",
                        "truth"])
            for r in self._rows:
                p = "" if r["p"] is None else f"{r['p']:.6f}"
                verdict = self._verdict_of(r)
                if verdict == "...":
                    verdict = ""
                truth = ("" if r["truth"] is None
                         else ("NG" if r["truth"] == 1 else "ok"))
                w.writerow([r["path"], p, verdict, f"{thr:.2f}", truth])

    def save_results(self):
        if not self._rows:
            self.set_status("Nothing to save - check a folder first",
                            "StatusWarn.TLabel")
            return
        out_dir = BASE / "results"
        out_dir.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save results", initialdir=str(out_dir),
            initialfile=f"results_{self.category_var.get()}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self._write_results_csv(Path(path))
            self.set_status(f"Saved {path}", "StatusOk.TLabel")
        except OSError as e:
            self.set_status(f"Error: {e}", "StatusErr.TLabel")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hazelnut Quality Control")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1280, int(sw * 0.92))
        h = min(960, int(sh * 0.90))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+"
                      f"{max(0, (sh - h) // 2)}")
        self.minsize(900, 560)
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family="Segoe UI", size=10)
            except tk.TclError:
                pass

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        # every tab lives inside a scroll wrapper: at high DPI or in a
        # small window nothing is ever cut off - it scrolls instead
        self.data_scroll = ScrollableTab(self.nb)
        self.data_tab = DataTab(self.data_scroll.inner)
        self.data_tab.pack(fill="both", expand=True)
        self.nb.add(self.data_scroll, text="1. Data")
        self.train_scroll = ScrollableTab(self.nb)
        self.train_tab = TrainTab(self.train_scroll.inner)
        self.train_tab.pack(fill="both", expand=True)
        self.nb.add(self.train_scroll, text="2. Training")
        self.check_scroll = ScrollableTab(self.nb)
        self.check_tab = CheckTab(self.check_scroll.inner)
        self.check_tab.pack(fill="both", expand=True)
        self.nb.add(self.check_scroll, text="3. Check")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_tab_changed(self, _event):
        if self.nb.select() == str(self.train_scroll):
            self.train_tab.refresh_categories()
            self.train_tab.refresh_category()
        elif self.nb.select() == str(self.check_scroll):
            # a training run may have added a new model_* file
            self.check_tab.refresh_categories()
            self.check_tab._refresh_model_block()

    def _on_close(self):
        busy = self.data_tab._build_running or self.train_tab._train_running
        if busy and not messagebox.askyesno(
                "Quit", "A script is still running. Stop it and quit?"):
            return
        self.data_tab.kill_proc_now()
        self.train_tab.kill_proc_now()
        self.destroy()

    def report_callback_exception(self, exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass
        print(text, file=sys.stderr)
        messagebox.showerror(
            "Error", f"{val}\n\nDetails were written to {ERROR_LOG}"
        )


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
