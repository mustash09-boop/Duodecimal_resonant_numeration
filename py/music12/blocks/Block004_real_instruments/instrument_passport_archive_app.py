from __future__ import annotations

import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from music12.blocks.Block004_real_instruments.instrument_passport_archive_core import (
    ArchiveBuildConfig,
    build_instrument_passport_archive,
)


class InstrumentPassportArchiveApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Block004 Instrument Passport Archive")
        self.root.geometry("980x780")

        self.source_dir = tk.StringVar()
        self.archive_root = tk.StringVar(value=r"E:\Duodecimal_resonant_numeration\Block004_data")
        self.instrument_name = tk.StringVar()
        self.library_kind = tk.StringVar(value="pitched")
        self.manifest_mode = tk.StringVar(value="auto")
        self.fix_cyrillic_abc = tk.BooleanVar(value=True)
        self.use_maxwell = tk.BooleanVar(value=True)
        self.include_archive_index = tk.BooleanVar(value=True)
        self.include_archive_audit = tk.BooleanVar(value=True)
        self.ffmpeg_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Готов к запуску.")

        self.run_button: ttk.Button | None = None
        self.log_text: tk.Text | None = None
        self.manifest_combo: ttk.Combobox | None = None
        self.index_check: ttk.Checkbutton | None = None
        self.audit_check: ttk.Checkbutton | None = None
        self.last_instrument_root: Path | None = None

        self.build_ui()
        self.apply_library_kind_ui()

    def build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text="Папка с файлами").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.source_dir).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Выбрать", command=self.choose_source_dir).grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(frame, text="Корень архива паспортов").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.archive_root).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Выбрать", command=self.choose_archive_root).grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(frame, text="Имя архива инструмента").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.instrument_name).grid(row=row, column=1, sticky="ew", padx=6)

        row += 1
        ttk.Label(frame, text="Тип библиотеки").grid(row=row, column=0, sticky="w", pady=4)
        library_combo = ttk.Combobox(
            frame,
            textvariable=self.library_kind,
            values=["pitched", "percussion"],
            state="readonly",
        )
        library_combo.grid(row=row, column=1, sticky="w", padx=6)
        library_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_library_kind_ui())

        row += 1
        ttk.Label(frame, text="Режим manifest").grid(row=row, column=0, sticky="w", pady=4)
        self.manifest_combo = ttk.Combobox(
            frame,
            textvariable=self.manifest_mode,
            values=["auto", "keyed", "token"],
            state="readonly",
        )
        self.manifest_combo.grid(row=row, column=1, sticky="w", padx=6)

        row += 1
        ttk.Label(frame, text="FFmpeg для MP3").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.ffmpeg_path).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Выбрать", command=self.choose_ffmpeg).grid(row=row, column=2, sticky="ew")

        row += 1
        self.mode_hint = ttk.Label(frame, text="", foreground="#444")
        self.mode_hint.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 8))

        row += 1
        options = ttk.Frame(frame)
        options.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Checkbutton(
            options,
            text="Исправлять кириллицу A/B/C в именах",
            variable=self.fix_cyrillic_abc,
        ).pack(anchor="w")
        ttk.Checkbutton(
            options,
            text="Оборачивать стадии через Maxwell",
            variable=self.use_maxwell,
        ).pack(anchor="w")
        self.index_check = ttk.Checkbutton(
            options,
            text="Обновлять общий индекс архива",
            variable=self.include_archive_index,
        )
        self.index_check.pack(anchor="w")
        self.audit_check = ttk.Checkbutton(
            options,
            text="Обновлять общий audit архива",
            variable=self.include_archive_audit,
        )
        self.audit_check.pack(anchor="w")

        row += 1
        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        buttons.columnconfigure(0, weight=0)
        buttons.columnconfigure(1, weight=0)
        buttons.columnconfigure(2, weight=1)

        self.run_button = ttk.Button(buttons, text="Запустить полный анализ", command=self.start_build)
        self.run_button.grid(row=0, column=0, sticky="w")

        ttk.Button(buttons, text="Очистить лог", command=self.clear_log).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Button(buttons, text="Открыть папку результата", command=self.open_last_result).grid(row=0, column=2, sticky="e")

        row += 1
        ttk.Label(frame, textvariable=self.status_text).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))

        row += 1
        self.log_text = tk.Text(frame, wrap="word", height=28)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(row, weight=1)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=row, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def apply_library_kind_ui(self) -> None:
        kind = self.library_kind.get().strip().lower()
        is_percussion = kind == "percussion"

        if self.manifest_combo is not None:
            self.manifest_combo.configure(state="disabled" if is_percussion else "readonly")

        if self.index_check is not None:
            self.index_check.configure(state="disabled" if is_percussion else "normal")

        if self.audit_check is not None:
            self.audit_check.configure(state="disabled" if is_percussion else "normal")

        if is_percussion:
            self.include_archive_index.set(False)
            self.include_archive_audit.set(False)
            self.mode_hint.configure(
                text=(
                    "Перкуссия пойдёт по event-based маршруту: manifest -> event reports -> "
                    "spiral3d -> percussion passports."
                )
            )
        else:
            self.mode_hint.configure(
                text=(
                    "Нотная библиотека пойдёт по note-based маршруту: manifest -> reports -> "
                    "box -> note box -> spiral3d -> instrument passport."
                )
            )

    def choose_source_dir(self) -> None:
        path = filedialog.askdirectory(title="Выбери папку с библиотекой инструмента")
        if not path:
            return
        self.source_dir.set(path)
        if not self.instrument_name.get().strip():
            self.instrument_name.set(Path(path).name)

    def choose_archive_root(self) -> None:
        path = filedialog.askdirectory(title="Выбери корень архива паспортов")
        if path:
            self.archive_root.set(path)

    def choose_ffmpeg(self) -> None:
        path = filedialog.askopenfilename(title="Выбери ffmpeg.exe")
        if path:
            self.ffmpeg_path.set(path)

    def append_log(self, text: str) -> None:
        def _append() -> None:
            assert self.log_text is not None
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")

        self.root.after(0, _append)

    def clear_log(self) -> None:
        if self.log_text is not None:
            self.log_text.delete("1.0", "end")

    def set_running(self, running: bool) -> None:
        def _apply() -> None:
            assert self.run_button is not None
            self.run_button.configure(state="disabled" if running else "normal")
            self.status_text.set("Выполняется..." if running else "Готов к запуску.")

        self.root.after(0, _apply)

    def start_build(self) -> None:
        source_dir = self.source_dir.get().strip()
        archive_root = self.archive_root.get().strip()
        instrument_name = self.instrument_name.get().strip()
        library_kind = self.library_kind.get().strip().lower()

        if not source_dir:
            messagebox.showerror("Ошибка", "Нужно выбрать папку с файлами.")
            return
        if not archive_root:
            messagebox.showerror("Ошибка", "Нужно выбрать корень архива.")
            return
        if not instrument_name:
            messagebox.showerror("Ошибка", "Нужно задать имя инструмента.")
            return
        if library_kind not in {"pitched", "percussion"}:
            messagebox.showerror("Ошибка", f"Неизвестный тип библиотеки: {library_kind}")
            return

        config = ArchiveBuildConfig(
            source_dir=source_dir,
            archive_root=archive_root,
            instrument_name=instrument_name,
            library_kind=library_kind,
            manifest_mode=self.manifest_mode.get().strip(),
            fix_cyrillic_abc=bool(self.fix_cyrillic_abc.get()),
            use_maxwell=bool(self.use_maxwell.get()),
            include_archive_index=bool(self.include_archive_index.get()),
            include_archive_audit=bool(self.include_archive_audit.get()),
            ffmpeg_path=self.ffmpeg_path.get().strip(),
        )

        self.clear_log()
        self.set_running(True)
        worker = threading.Thread(target=self._run_build, args=(config,), daemon=True)
        worker.start()

    def _run_build(self, config: ArchiveBuildConfig) -> None:
        try:
            summary = build_instrument_passport_archive(config, logger=self.append_log)
            self.last_instrument_root = Path(summary.instrument_root)
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Готово",
                    (
                        "Архив паспорта собран.\n\n"
                        f"Тип: {summary.library_kind}\n"
                        f"Результат:\n{summary.instrument_root}"
                    ),
                ),
            )
            self.root.after(0, lambda: self.status_text.set("Готово."))
        except Exception as exc:
            self.append_log("")
            self.append_log("ERROR:")
            self.append_log(str(exc))
            self.append_log(traceback.format_exc())
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Ошибка сборки",
                    f"{exc}",
                ),
            )
            self.root.after(0, lambda: self.status_text.set("Ошибка сборки."))
        finally:
            self.set_running(False)

    def open_last_result(self) -> None:
        if self.last_instrument_root is None or not self.last_instrument_root.exists():
            messagebox.showinfo("Нет результата", "Пока нет собранной папки результата.")
            return
        try:
            import os

            os.startfile(str(self.last_instrument_root))
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{exc}")


def main() -> None:
    root = tk.Tk()
    app = InstrumentPassportArchiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
