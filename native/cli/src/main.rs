fn main() {
    let args = std::env::args_os()
        .skip(1)
        .map(|v| v.into_string())
        .collect::<Result<Vec<_>, _>>();
    match args {
        Ok(args) => std::process::exit(invarlock_native_verifier::entry(args).into()),
        Err(_) => {
            eprintln!("command arguments must be valid UTF-8");
            std::process::exit(2);
        }
    }
}
