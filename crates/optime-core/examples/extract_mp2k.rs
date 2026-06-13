use std::env;
use std::fs;
use std::process;

use optime_core::devices::gba::GbaRom;

fn main() {
    // 1. Collect argv arguments into a Vec<String>
    let args: Vec<String> = env::args().collect();

    // 2. Ensure the user actually provided a file path argument
    if args.len() < 2 {
        eprintln!("Usage: {} <GBA ROM path>", args[0]);
        process::exit(1);
    }

    let file_path = &args[1];

    // Read the file entirely into a u8 Vector
    let buffer: Vec<u8> = match fs::read(file_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to read file '{}': {}", file_path, e);
            process::exit(1);
        }
    };

    GbaRom::parse(&buffer);

    println!("Successfully read {} bytes.", buffer.len());
}
