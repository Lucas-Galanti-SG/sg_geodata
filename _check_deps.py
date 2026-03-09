"""
Verifica se todas as dependências do requirements.txt estão instaladas
e oferece instalar as ausentes antes de iniciar o SGGeoData.
"""

import ctypes
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

REQ_FILE = Path(__file__).parent / "requirements.txt"


# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------

def _setup_ansi() -> bool:
    """Habilita ANSI no console do Windows 10+. Retorna True se suportado."""
    if sys.platform != "win32":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VT_PROCESSING
        return True
    except Exception:
        return False


_ANSI = _setup_ansi() and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI else text


def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def bold(t):   return _c("1",  t)
def cyan(t):   return _c("36", t)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_requirements(path: Path) -> list[tuple[str, str | None]]:
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-\.]+)(?:>=([^\s,;]+))?", line)
        if m:
            reqs.append((m.group(1), m.group(2)))
    return reqs


def _version_ok(installed: str, minimum: str) -> bool:
    try:
        from packaging.version import Version
        return Version(installed) >= Version(minimum)
    except Exception:
        return True  # sem packaging: aceita qualquer versão instalada


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not REQ_FILE.exists():
        print(yellow(f"⚠  {REQ_FILE} não encontrado — pulando verificação."))
        return 0

    reqs = parse_requirements(REQ_FILE)
    missing: list[str] = []

    print()
    print(bold(cyan("╔══════════════════════════════════════════════════╗")))
    print(bold(cyan("║       SGGeoData — Verificação de dependências    ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════╝")))
    print()

    col_pkg = 32
    for dist_name, min_ver in reqs:
        try:
            ver = importlib.metadata.version(dist_name)
            if min_ver and not _version_ok(ver, min_ver):
                req_str = f">={min_ver}"
                print(f"  {red('✗')}  {dist_name:<{col_pkg}} {ver}  {red(f'(requer {req_str})')}")
                missing.append(dist_name)
            else:
                req_str = f">={min_ver}" if min_ver else ""
                print(f"  {green('✓')}  {dist_name:<{col_pkg}} {green(ver)}  {req_str}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {red('✗')}  {dist_name:<{col_pkg}} {red('não instalado')}")
            missing.append(dist_name)

    print()

    if not missing:
        print(green("✔  Todas as dependências estão satisfeitas."))
        print()
        return 0

    # ── Pacotes faltando ────────────────────────────────────────────────────
    print(yellow(f"⚠  {len(missing)} dependência(s) ausente(s) ou desatualizada(s):"))
    for pkg in missing:
        print(f"      • {pkg}")
    print()

    try:
        ans = input(
            bold("   Deseja instalar automaticamente via pip? [S/n] ")
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    print()

    if ans in ("", "s", "sim", "y", "yes"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)],
            check=False,
        )
        print()
        if result.returncode == 0:
            print(green("✔  Instalação concluída com sucesso."))
            print()
            return 0
        else:
            print(red("✗  Erro durante a instalação. Verifique as mensagens acima."))
            print()
            try:
                ans2 = input(
                    bold("   Continuar a iniciar mesmo assim? [s/N] ")
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans2 = "n"
            return 0 if ans2 in ("s", "sim", "y", "yes") else 1
    else:
        print(yellow("   Instalação cancelada. A aplicação pode não funcionar corretamente."))
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
