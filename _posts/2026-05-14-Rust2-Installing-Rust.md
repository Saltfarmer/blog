---
title: "Rust 2 : Installing Rust"
header :
  image : /assets/images/Rust-4-Data-Analysis.png
  teaser: https://logowik.com/content/uploads/images/rust4784.logowik.com.webp
comments : true
share : true
categories:
  - Rust
---

Alright here we go again. Now it's time to install rust and every library (not sure what it called in Rust) and focus it on data analytic, statistic, and data science environment only. If you are coming from Python, your daily workflow probably looks like this: fire up a Jupyter Notebook inside VS Code, pip install a few packages, import pandas as pd, and start exploring. The feedback loop is instant, the table rendering is pretty, and inline plots appear with a single `plt.show()`.

Rust can feel similar—once you wire up the right tooling. This guide will get you there. We will install Rust, configure VS Code for data science, and run Rust code inside Jupyter Notebook cells using the `EvCxR` kernel. By the end, you will have a live notebook reading CSVs with Polars and rendering inline plots, all while benefiting from Rust's type checking and performance.

# Snek enjoyer migration guide
| What you do in Python             | What you will do in Rust                               |
| --------------------------------- | ------------------------------------------------------ |
| `pip install pandas`              | `cargo add polars` (or `:dep polars` in Jupyter)       |
| `conda create -n myenv`           | `cargo new myproject` (isolated by default)            |
| `jupyter notebook`                | `jupyter notebook` with the **EvCxR** kernel           |
| `import pandas as pd`             | `use polars::prelude::*;`                              |
| `df.head()` renders an HTML table | `df.head()` also renders an HTML table (via EvCxR)     |
| `plt.plot(...)` inline            | `plotters` with the `evcxr` feature for inline SVG/PNG |

# Installing Rust
Visit [https://rustup.rs](https://rustup.rs) and run the installer. This gives you **rustc** (the compiler), **cargo** (the build tool and package manager), and **rustup** (the toolchain manager).

Verify the installation:
```
rustc --version
cargo --version
```

To update
```
rustup update
```

# VScode Configuration
VS Code is the de facto standard for Rust development. With two extensions, you get IDE features that rival PyCharm.

## Essential Extensions

1. rust-analyzer (by The Rust Programming Language group)
This is the language server. It provides type hints on hover, inline errors, auto-imports, and code completion. Install it from the Extensions marketplace.

2. Jupyter (by Microsoft)
Required to open and run .ipynb notebooks inside VS Code.

3. CodeLLDB (by Vadim Chugunov) — Optional but recommended
If you want to debug Rust code with breakpoints (the equivalent of pdb or PyCharm's debugger), this extension is the smoothest option.

## Recommended settings.json
Add these to your VS Code user or workspace settings for a data-science-friendly Rust experience:
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.checkOnSave.extraArgs": ["--tests"],
    "[rust]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    },
    "jupyter.askForKernelRestart": false,
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "left"
    }
}
```

What this gives you:

- Clippy on save: Clippy is Rust's flake8/pylint. It will catch style issues and common mistakes before you even run the cell.
- Format on save: rustfmt automatically cleans up indentation, just like Python's black.
- All features enabled: Ensures rust-analyzer sees optional crate features (like Polars' CSV reader) during type checking.

# Jupyter Notebook (EvCxR)
**EvCxR** is a Jupyter kernel for Rust. It compiles and executes Rust code interactively, persists variables across cells, and even displays Polars DataFrames as HTML tables and Plotters charts inline.

Install the EvCxR Kernel
You install it via Cargo:
```
cargo install evcxr_jupyter
evcxr_jupyter --install
```

Verify the kernel is registered:
```
jupyter kernelspec list
```

![](https://i.ibb.co.com/JwLGnXZQ/2026-05-14-19-30-00-Clipboard.png)

# The `:dep` Magic (Crucial for Pythonistas)
In Python, you install packages with pip in the terminal, then import them in a cell. In EvCxR, you declare dependencies inside the notebook with the `:dep` magic command. The kernel downloads and compiles the crate automatically.

```rust
:dep polars = { version = "0.39", features = ["csv", "lazy", "describe"] }
:dep plotters = { version = "0.3", default_features = false, features = ["evcxr", "bitmap_backend", "all_series"] }
:dep ndarray = "0.15"
```

# Do i have to create cargo projects ? is there any way to have global environment like python  ?

No—you do not have to create a Cargo project for every quick experiment. But you should understand why Rust pushes you toward projects, and what your "global" alternatives are.

| What you want            | Python way                                | Rust way                                                          |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------- |
| Jupyter/REPL exploration | `pip install` globally, `import` anywhere | **EvCxR** (`:dep` inside a notebook—no project needed)            |
| One-off scripts          | `python myscript.py`                      | **A single "playground" project** you reuse, or **`rust-script`** |
| CLI tools                | `pip install black`                       | `cargo install` (global binaries)                                 |
| Production/reusable code | `setup.py` / `pyproject.toml`             | `cargo new` (the standard way)                                    |

In Python, a global environment is convenient but fragile. Modern Python best practice is actually virtualenvs or conda envs for every project. Rust just formalizes this at the language level because Rust is compiled, not interpreted.
When you run cargo build, the compiler needs to know the exact versions of every crate to generate a single binary. That information lives in `Cargo.toml`. There is no "runtime" that can lazily search a global site-packages folder.
But for data exploration, you only have to use it directly on `EvCxR`:

# One Cell to Install all Dependencies
```rust
// Cell 1: Minimal setup
:dep polars = "0.40"
:dep ndarray = "0.15"

use polars::prelude::*;
println!("✅ Basic setup works!");
```

Beware, it might takes few more minutes. Basically it's try to download and install the dependencies. The Good News: It's Cached! Even if you:

- Close Jupyter and reopen it

- Restart your computer

- Create a new notebook

The compiled versions remain cached on your system!

# Quick test cell
```rust
// Test Polars
let df = df![
    "name" => ["Alice", "Bob", "Charlie"],
    "value" => [1, 2, 3]
]?;
println!("✅ Polars works!");
println!("{}", df);

// Test ndarray
use ndarray::array;
let arr = array![1, 2, 3];
println!("\n✅ ndarray works: {:?}", arr);

// Test chrono
use chrono::Utc;
println!("\n✅ chrono works: {}", Utc::now());

// Test serde
use serde_json::json;
let json = json!({"test": "works"});
println!("\n✅ serde works: {}", json);

println!("\n🎉 All dependencies loaded successfully!");
```
