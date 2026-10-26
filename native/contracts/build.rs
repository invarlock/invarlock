// Generated contract assets are authoritative only in src/invarlock/_data/contracts.
// An extracted native source archive uses its retained immutable schema assets.
use std::{env, fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let upstream = root.join("../../src/invarlock/_data/contracts");
    let packaged = root.join("schemas");
    println!("cargo:rerun-if-changed=schemas");
    if upstream.is_dir() {
        println!("cargo:rerun-if-changed={}", upstream.display());
        for entry in fs::read_dir(&packaged).expect("packaged schemas") {
            let path = entry.expect("schema entry").path();
            if path.file_name().unwrap() == "source-digests.json" {
                continue;
            }
            let source = upstream.join(path.file_name().unwrap());
            assert_eq!(
                fs::read(&path).expect("schema asset"),
                fs::read(&source).expect("authoritative schema"),
                "packaged schema drift: {}",
                path.display()
            );
        }
    }
}
