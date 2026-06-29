// src/bin/qyptos_compiler.rs - The Linus Torvalds Factory
//
// Standalone executable for compiling Qyptos EBNF physics grammar into 
// hot-loadable .qmem binary cartridges.

use std::env;
use std::fs;
use std::process;

use sodals_neuro::compiler::qmem_compiler;
use sodals_neuro::compiler::qmem_linker;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("\x1b[1;31mUsage: qyptos_compiler <input_file.txt> <output_file.qmem>\x1b[0m");
        process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let source = match fs::read_to_string(input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("\x1b[1;31m[ERROR] Failed to read input file {}: {}\x1b[0m", input_path, e);
            process::exit(1);
        }
    };

    println!("[QYPTOS COMPILER] Pass 1: Parsing...");
    let cst = match qmem_compiler::compile(&source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[1;31m[SYNTAX ERROR] Compilation failed:\n{}\x1b[0m", e);
            process::exit(1);
        }
    };

    println!("[QYPTOS COMPILER] Pass 2: Linking topological physics...");
    let network = qmem_linker::compile_cst_to_network(&cst);

    println!("[QYPTOS COMPILER] Pass 3: Emitting binary...");
    match network.save_cartridge(output_path) {
        Ok(_) => {
            let file_size = match fs::metadata(output_path) {
                Ok(meta) => meta.len(),
                Err(_) => 0,
            };
            println!("\x1b[1;32m[QYPTOS COMPILER] SUCCESS: Cartridge written to disk ({} bytes)\x1b[0m", file_size);
        }
        Err(e) => {
            eprintln!("\x1b[1;31m[ERROR] Failed to write binary cartridge to {}: {}\x1b[0m", output_path, e);
            process::exit(1);
        }
    }
}
