"""
Render PlantUML diagrams to PNG using a local plantuml.jar.
"""

import os
import json
import logging
import re
import subprocess
from typing import Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

DEFAULT_JAR_PATH = str(Path(__file__).resolve().parent.parent / "data_pipeline" / "java" / "plantuml-mit-1.2026.6.jar")

# PlantUML prints one of these per failed file to stderr when rendering a batch
# (confirmed for both a genuine syntax error and a SANDBOX-blocked !include) --
# other files in the same batch still render (successfully or as an error image),
# so file existence alone isn't a reliable pass/fail signal; this is.
RX_RENDER_ERROR_FILE = re.compile(r"^Error(?:\s+line\s+\d+)?\s+in\s+file:\s+(.+)$", re.MULTILINE)


def setup_logger():
    """Setup logging."""
    logger = logging.getLogger("render_scores")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_loggers(log_dir: str|Path = "logs"):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    combined_file = log_path / f"render_{run_id}.log"
    pass_file     = log_path / f"render_{run_id}_pass.log"
    fail_file     = log_path / f"render_{run_id}_fail.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Combined logger
    combined = logging.getLogger(f"render.combined.{run_id}")
    combined.setLevel(logging.INFO)
    combined.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(combined_file, encoding="utf-8")
    fh.setFormatter(fmt)
    combined.addHandler(ch)
    combined.addHandler(fh)

    # Pass-only logger
    passed = logging.getLogger(f"render.pass.{run_id}")
    passed.setLevel(logging.INFO)
    passed.propagate = False
    pfh = logging.FileHandler(pass_file, encoding="utf-8")
    pfh.setFormatter(fmt)
    passed.addHandler(pfh)

    # Fail-only logger
    failed = logging.getLogger(f"render.fail.{run_id}")
    failed.setLevel(logging.INFO)
    failed.propagate = False
    ffh = logging.FileHandler(fail_file, encoding="utf-8")
    ffh.setFormatter(fmt)
    failed.addHandler(ffh)

    combined.info(f"Logging to: {combined_file}")
    combined.info(f"Pass log : {pass_file}")
    combined.info(f"Fail log : {fail_file}")

    return combined, passed, failed, run_id


def ensure_wrapped(s: str) -> str:
    s = s.strip()
    if not s.lower().startswith("@startuml"):
        s = "@startuml\n" + s
    if not s.lower().endswith("@enduml"):
        s = s + "\n@enduml"
    return s + "\n"


def render_batch(puml_files: list[Path], output_dir: Path, jar_path: str, java_home: str,
                  timeout: int) -> tuple[subprocess.CompletedProcess, set[str]]:
    """
    Render a batch of .puml files to PNG in a single JVM invocation, instead of paying
    a full JVM boot per file. All files are written flat into output_dir by basename
    (confirmed: -o does not preserve subdirectory structure), which is fine for this
    module's actual (flat, non-recursive) usage but would collide on same-basename
    files from different subdirectories if ever used with recursive=True.

    Returns:
        (completed_process, failed_file_paths) -- failed_file_paths are absolute path
        strings parsed from stderr for files PlantUML reported a rendering error for.
        A file not in this set produced a real diagram; one in it still produced a PNG
        (PlantUML draws an error image on failure), just not a valid rendering of the source.
    """
    java_bin = os.path.join(java_home, "bin", "java")
    if not os.path.isfile(java_bin):
        raise FileNotFoundError(f"java executable not found at {java_bin}")
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(f"PlantUML jar not found: {jar_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Untrusted (model-generated or scraped) PlantUML source can contain !include/
    # !includeurl pointing at arbitrary external URLs; SANDBOX disables all network/
    # local-file access during rendering so this can never issue outbound requests.
    env = {**os.environ, "PLANTUML_SECURITY_PROFILE": "SANDBOX"}

    cmd = [java_bin, "-Djava.awt.headless=true", "-jar", jar_path, "-tpng",
           "-o", str(output_dir)] + [str(f) for f in puml_files]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

    failed_files = set(RX_RENDER_ERROR_FILE.findall(result.stderr))
    return result, failed_files


def batch_convert_with_logs(input_dir: str|Path, jar_path: str, java_home: str,
                            recursive: bool = True, output_dir: str|Path | None = None,
                            log_dir: str|Path = "logs", timeout: int | None = None) -> Tuple[dict, list]:
    combined, passed, failed, run_id = setup_loggers(log_dir)

    base = Path(input_dir).resolve()
    pattern = "**/*.puml" if recursive else "*.puml"
    puml_files = sorted(base.glob(pattern))

    if not puml_files:
        combined.warning(f"No .puml files found in: {base} (recursive={recursive})")
        return {"run_id": run_id, "total": 0, "ok": 0, "fail": 0}, []

    combined.info(f"Found {len(puml_files)} .puml files under {base} (recursive={recursive})")

    out_base = Path(output_dir).resolve() if output_dir else base

    # Scale the batch timeout with file count rather than using a fixed per-run value,
    # since this is now one JVM call for the whole batch instead of one per file.
    batch_timeout = timeout if timeout is not None else max(60, 5 * len(puml_files))

    combined.info(f"Rendering {len(puml_files)} diagrams with local plantuml.jar "
                  f"(single batch JVM call, timeout={batch_timeout}s)...")
    try:
        result, failed_files = render_batch(puml_files, out_base, jar_path, java_home, timeout=batch_timeout)
    except FileNotFoundError as e:
        combined.error(f"❌ {e}")
        return {"run_id": run_id, "total": len(puml_files), "ok": 0, "fail": len(puml_files), "error": True}, \
            [0] * len(puml_files)
    except subprocess.TimeoutExpired:
        combined.error(f"⏱️  Rendering timed out after {batch_timeout}s for the whole batch")
        return {"run_id": run_id, "total": len(puml_files), "ok": 0, "fail": len(puml_files), "error": True}, \
            [0] * len(puml_files)

    if result.stderr.strip():
        combined.info(f"plantuml.jar stderr:\n{result.stderr.strip()}")

    ok = 0
    fail_count = 0
    passed_list = []

    for i, puml_file in enumerate(puml_files, start=1):
        png_path = out_base / puml_file.with_suffix(".png").name

        if str(puml_file) in failed_files:
            fail_count += 1
            passed_list.append(0)
            msg = f"[{i}/{len(puml_files)}] FAIL | {puml_file} -> rendering error (see stderr above)"
            combined.error(msg)
            failed.info(msg)
        elif not png_path.is_file() or png_path.stat().st_size == 0:
            fail_count += 1
            passed_list.append(0)
            msg = f"[{i}/{len(puml_files)}] FAIL | {puml_file} -> no PNG produced at {png_path}"
            combined.error(msg)
            failed.info(msg)
        else:
            ok += 1
            passed_list.append(1)
            msg = f"[{i}/{len(puml_files)}] OK   | {puml_file} -> {png_path}"
            combined.info(msg)
            passed.info(msg)

    combined.info(f"Done. OK={ok}, FAIL={fail_count}, TOTAL={len(puml_files)}")
    assert sum(passed_list) == ok
    assert len(passed_list) == len(puml_files)
    return {"run_id": run_id, "total": len(puml_files), "ok": ok, "fail": fail_count, "error": False}, passed_list


def main(config: Dict[str, Any]) -> None:
    """
    Render PlantUML diagrams using a local plantuml.jar.

    Args:
        config: Configuration dictionary with evaluate settings
    """
    logger = setup_logger()

    eval_config = config.get("evaluate", {})
    model_name = eval_config.get("model_name")
    jar_path = eval_config.get("plantuml_jar_path") or DEFAULT_JAR_PATH
    java_home = eval_config.get("java_home")

    if not model_name:
        raise ValueError("model_name required in evaluate config")
    if not java_home:
        raise ValueError("java_home required in evaluate config (set it to your local JDK installation)")

    # Generate file paths similar to generate.py
    model_to_save = model_name.split("/")[-1]
    training_to_save = "fim"
    data_to_save = "plantuml"
    do_sample = eval_config.get("do_sample", False)
    temperature = eval_config.get("temperature", 1.0)
    seed = eval_config.get("seed", 42)
    model_name_str = f"{model_to_save}_{training_to_save}_{data_to_save}"
    if do_sample:
        model_name_str = f"{model_name_str}_temp{temperature}"
    model_name_str = f"{model_name_str}_evs{seed}"
    target_dir = eval_config.get("output_dir", "./outputs")
    source_dir = f"{target_dir}/{model_name_str}/"

    # ---------------------- DEFINE PATHS ----------------------

    puml_dir_generated = Path(f"{target_dir}/{model_name_str}/generated")
    puml_dir_generated.mkdir(parents=True, exist_ok=True)
    puml_dir_reference = Path(f"{target_dir}/{model_name_str}/reference")
    puml_dir_reference.mkdir(parents=True, exist_ok=True)
    output_dir_generated = Path(f"{target_dir}/{model_name_str}/generated_png")
    output_dir_generated.mkdir(parents=True, exist_ok=True)
    log_dir_generated = Path(f"{target_dir}/logs/{model_name_str}/generated")
    log_dir_generated.mkdir(parents=True, exist_ok=True)

    # ---------------------- LOAD DATA ----------------------

    file_path = os.path.join(source_dir, f"{model_name_str}_scored.json")
    save_path = os.path.join(source_dir, f"{model_name_str}_rendered.json")

    if os.path.exists(save_path):
        logger.info(f" > Rendered diagrams already found at {save_path}. Skipping rendering!")
        return

    if not os.path.exists(file_path):
        logger.warning(f"Scored file not found: {file_path}. Skipping rendering.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    generated_complete_snippet_list = data.get('generations', [])
    reference_puml_snippet_list = data.get('references', [])

    # ---------------------- SAVE PUML FILES ----------------------

    for i, t in enumerate(generated_complete_snippet_list, start=1):
        filename = puml_dir_generated / f"generated_diagram_{i:04d}.puml"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(ensure_wrapped(t))

    for i, t in enumerate(reference_puml_snippet_list, start=1):
        filename = puml_dir_reference / f"reference_diagram_{i:04d}.puml"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(ensure_wrapped(t))

    # ---------------------- RENDER PUML ----------------------

    logger.info(f"Rendering diagrams using local plantuml.jar at {jar_path}")
    summary_generated, passed = batch_convert_with_logs(
        puml_dir_generated,
        jar_path=jar_path,
        java_home=java_home,
        recursive=False,
        output_dir=output_dir_generated,
        log_dir=log_dir_generated,
    )
    if not summary_generated.get("error"):
        with open(save_path, "w") as jfile:
            json.dump(passed, jfile)

    # ---------------------- SUMMARY ----------------------

    logger.info(f"Model name: {model_name_str}")
    logger.info(f"Generated summary: {summary_generated['ok']}/{summary_generated['total']} rendered")


if __name__ == "__main__":
    pass
