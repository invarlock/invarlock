//! IDNA2003 ToASCII with the Unicode 3.2 Nameprep profile used by Python.
use crate::idna_tables::*;
fn combining(c: u32) -> u8 {
    COMBINING
        .binary_search_by_key(&c, |x| x.0)
        .ok()
        .map(|i| COMBINING[i].1)
        .unwrap_or(0)
}
fn contains(ranges: &[(u32, u32)], c: u32) -> bool {
    ranges
        .get(ranges.partition_point(|r| r.1 < c))
        .is_some_and(|r| r.0 <= c)
}
fn compose(a: u32, b: u32) -> Option<u32> {
    if (0x1100..0x1113).contains(&a) && (0x1161..0x1176).contains(&b) {
        return Some(0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28);
    }
    if (0xac00..=0xd7a3).contains(&a)
        && (a - 0xac00).is_multiple_of(28)
        && (0x11a8..0x11c3).contains(&b)
    {
        return Some(a + b - 0x11a7);
    }
    COMPOSITION
        .binary_search_by_key(&(a, b), |x| x.0)
        .ok()
        .map(|i| COMPOSITION[i].1)
}
fn nameprep(label: &str) -> Option<Vec<u32>> {
    let mut chars = Vec::new();
    for c in label.chars().map(u32::from) {
        if let Ok(i) = MAPPING.binary_search_by_key(&c, |x| x.0) {
            chars.extend(MAPPING[i].1)
        } else {
            chars.push(c)
        }
    }
    let mut start = 0;
    for i in 0..chars.len() {
        if combining(chars[i]) == 0 {
            chars[start..i].sort_by_key(|c| combining(*c));
            start = i + 1
        }
    }
    chars[start..].sort_by_key(|c| combining(*c));
    let mut out = Vec::new();
    let mut starter = 0;
    let mut previous = 0;
    for c in chars {
        let cls = combining(c);
        let combined = if !out.is_empty() && (previous == 0 || previous < cls) {
            compose(out[starter], c)
        } else {
            None
        };
        if let Some(combined) = combined {
            out[starter] = combined
        } else {
            if cls == 0 {
                starter = out.len()
            }
            out.push(c);
            previous = cls
        }
    }
    if out.iter().any(|c| contains(PROHIBITED, *c)) {
        return None;
    }
    if out.iter().any(|c| contains(RANDAL, *c))
        && (out.iter().any(|c| contains(LEFT, *c))
            || !out.first().is_some_and(|c| contains(RANDAL, *c))
            || !out.last().is_some_and(|c| contains(RANDAL, *c)))
    {
        return None;
    }
    Some(out)
}
fn digit(d: u64) -> char {
    if d < 26 {
        (b'a' + d as u8) as char
    } else {
        (b'0' + (d - 26) as u8) as char
    }
}
fn adapt(mut delta: u64, points: u64, first: bool) -> u64 {
    delta = if first { delta / 700 } else { delta / 2 };
    delta += delta / points;
    let mut k = 0;
    while delta > 455 {
        delta /= 35;
        k += 36
    }
    k + 36 * delta / (delta + 38)
}
fn punycode(input: &[u32]) -> String {
    let mut out = input
        .iter()
        .filter(|c| **c < 128)
        .map(|c| char::from_u32(*c).unwrap())
        .collect::<String>();
    let basic = out.len() as u64;
    let mut handled = basic;
    if basic > 0 {
        out.push('-')
    }
    let mut n = 128u64;
    let mut delta = 0u64;
    let mut bias = 72u64;
    while handled < input.len() as u64 {
        let m = input
            .iter()
            .map(|c| *c as u64)
            .filter(|c| *c >= n)
            .min()
            .unwrap();
        delta += (m - n) * (handled + 1);
        n = m;
        for &c in input {
            let c = c as u64;
            if c < n {
                delta += 1
            }
            if c == n {
                let mut q = delta;
                let mut k = 36;
                loop {
                    let t = if k <= bias {
                        1
                    } else if k >= bias + 26 {
                        26
                    } else {
                        k - bias
                    };
                    if q < t {
                        break;
                    }
                    out.push(digit(t + (q - t) % (36 - t)));
                    q = (q - t) / (36 - t);
                    k += 36
                }
                out.push(digit(q));
                bias = adapt(delta, handled + 1, handled == basic);
                delta = 0;
                handled += 1
            }
        }
        delta += 1;
        n += 1;
    }
    out
}
fn label_ascii(label: &str) -> Option<String> {
    if label.is_ascii() {
        return (1..64).contains(&label.len()).then(|| label.to_owned());
    }
    let normalized = nameprep(label)?;
    if normalized.iter().all(|c| *c < 128) {
        let s = normalized
            .iter()
            .map(|c| *c as u8 as char)
            .collect::<String>();
        return (1..64).contains(&s.len()).then_some(s);
    }
    if normalized.starts_with(&[120, 110, 45, 45]) {
        return None;
    }
    let s = format!("xn--{}", punycode(&normalized));
    (s.len() < 64).then_some(s)
}
pub(crate) fn to_ascii(host: &str) -> Option<String> {
    let host = host.replace(['\u{3002}', '\u{ff0e}', '\u{ff61}'], ".");
    let mut labels = host
        .split('.')
        .map(label_ascii)
        .collect::<Option<Vec<_>>>()?;
    for label in &mut labels {
        label.make_ascii_lowercase()
    }
    Some(labels.join("."))
}
