use super::{err, outcome, Args};
use invarlock_native_contracts as contracts;
use invarlock_native_evidence::{self as evidence, Result};
use invarlock_native_filesystem as fs;
use serde_json::{json, Value};
use std::{io::Read, path::Path};
fn setup(a: &Args) -> Result<Value> {
    let actions = ["init", "keygen", "freeze-cases"]
        .into_iter()
        .filter(|k| a.get(k).is_some())
        .collect::<Vec<_>>();
    if actions.len() != 1 || !a.positional.is_empty() {
        return Err(err("a request or exactly one setup action is required"));
    }
    let action = actions[0];
    for key in a.values.keys() {
        if !([action, "json"].contains(&key.as_str())
            || action == "init" && key == "example"
            || action == "freeze-cases" && key == "case-set-output")
        {
            return Err(err(
                "setup actions cannot be combined with evaluation options",
            ));
        }
    }
    let details = if action == "freeze-cases" {
        let raw = fs::read_file(Path::new(a.get(action).unwrap()), 128 * 1024 * 1024)
            .map_err(|e| err(e.to_string()))?;
        let mut cases = contracts::parse_json(&raw, "case set")?;
        let digest = evidence::judge::case_set_digest(&cases)?;
        cases["cases"]
            .as_array_mut()
            .unwrap()
            .sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        if let Some(p) = a.get("case-set-output") {
            fs::write_file_no_replace(
                Path::new(p),
                &contracts::canonical_json(&cases)?,
                0o600,
                true,
            )
            .map_err(|e| err(e.to_string()))?;
        }
        json!({"case_count":cases["cases"].as_array().unwrap().len(),"case_set_digest":digest,"output":a.get("case-set-output")})
    } else {
        let path = Path::new(a.get(action).unwrap());
        let absolute = std::path::absolute(path).map_err(|e| err(e.to_string()))?;
        let _parent = fs::PinnedDirectory::open(absolute.parent().unwrap(), true)
            .map_err(|e| err(e.to_string()))?;
        if std::fs::symlink_metadata(&absolute).is_ok() {
            return Err(err(format!("{action} directory must not already exist")));
        }
        let mut stage = fs::StagedDirectory::new(&absolute).map_err(|e| err(e.to_string()))?;
        let details = if action == "init" {
            let example = a.get("example").unwrap_or("classification");
            let assets=match example{"classification"=>include_str!("assets/classification.json"),"extraction"=>include_str!("assets/extraction.json"),"judge"=>include_str!("assets/judge.json"),"native-judge"=>include_str!("assets/native-judge.json"),_=>return Err(err("example must be classification, extraction, judge (recorded ratings), or native-judge"))};
            let assets: Value = serde_json::from_str(assets).unwrap();
            for (name, text) in assets.as_object().unwrap() {
                stage
                    .write(Path::new(name), text.as_str().unwrap().as_bytes(), 0o600)
                    .map_err(|e| err(e.to_string()))?;
            }
            json!({"directory":path,"request":path.join("request.yaml"),"example":example})
        } else {
            use ed25519_dalek::{pkcs8::EncodePrivateKey, SigningKey};
            let mut seed = [0u8; 32];
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(&mut seed))
                .map_err(|e| err(e.to_string()))?;
            let key = SigningKey::from_bytes(&seed);
            seed.fill(0);
            let wire = ed25519_dalek::pkcs8::KeypairBytes {
                secret_key: key.to_bytes(),
                public_key: None,
            };
            let private = wire
                .to_pkcs8_pem(Default::default())
                .map_err(|e| err(e.to_string()))?;
            stage
                .write(Path::new("private.pem"), private.as_bytes(), 0o600)
                .map_err(|e| err(e.to_string()))?;
            stage
                .write(
                    Path::new("public.pem"),
                    evidence::crypto::public_pem(&key.verifying_key())?.as_bytes(),
                    0o600,
                )
                .map_err(|e| err(e.to_string()))?;
            json!({"private_key":path.join("private.pem"),"public_key":path.join("public.pem"),"public_key_fingerprint":evidence::crypto::fingerprint(&key.verifying_key())})
        };
        stage.commit().map_err(|e| err(e.to_string()))?;
        details
    };
    Ok(
        json!({"format_version":"invarlock/evaluation-setup-v1","action":action.replace('-',"_"),"ok":true,"details":details,"errors":[]}),
    )
}
pub(super) fn run(a: &Args) -> Result<evidence::transaction::Outcome> {
    match setup(a) {
        Ok(v) => outcome(v, 0),
        Err(e) => outcome(
            json!({"format_version":"invarlock/evaluation-setup-v1","action":null,"ok":false,"details":null,"errors":[e.message]}),
            2,
        ),
    }
}
