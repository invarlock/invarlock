//! Deterministic ISO datetime conversion with microsecond precision.
use crate::{EvidenceError, Result};
fn error(label: &str) -> EvidenceError {
    EvidenceError::contract(format!("{label} is invalid"))
}
fn leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}
fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + (153 * mp + 2) / 5 + d - 1 - 719468
}
fn date(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    y += i64::from(m <= 2);
    (y, m, d)
}
fn digits(s: &str) -> Option<i64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}
fn clock(s: &str) -> Option<i64> {
    let (s, fraction) = s.split_once(['.', ',']).unwrap_or((s, ""));
    let parts = if s.contains(':') {
        s.split(':').collect::<Vec<_>>()
    } else {
        if ![2, 4, 6].contains(&s.len()) {
            return None;
        }
        (0..s.len()).step_by(2).map(|i| &s[i..i + 2]).collect()
    };
    if parts.is_empty() || parts.len() > 3 || parts.iter().any(|p| p.len() != 2) {
        return None;
    }
    let h = digits(parts[0])?;
    let m = if parts.len() > 1 {
        digits(parts[1])?
    } else {
        0
    };
    let sec = if parts.len() > 2 {
        digits(parts[2])?
    } else {
        0
    };
    if h > 23 || m > 59 || sec > 59 || !fraction.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let micro = if fraction.is_empty() {
        0
    } else {
        digits(&format!("{fraction:0<6}")[..6])?
    };
    Some((h * 3600 + m * 60 + sec) * 1_000_000 + micro)
}
pub(crate) fn parse(s: &str, label: &str) -> Result<i64> {
    let invalid = || error(label);
    if !s.is_char_boundary(4) {
        return Err(invalid());
    }
    let y = digits(s.get(..4).ok_or_else(invalid)?)
        .filter(|y| (1..=9999).contains(y))
        .ok_or_else(invalid)?;
    let extended = s.as_bytes().get(4) == Some(&b'-');
    let mut pos = if extended { 5 } else { 4 };
    let epoch_days = if s.as_bytes().get(pos) == Some(&b'W') {
        pos += 1;
        let w = digits(s.get(pos..pos + 2).ok_or_else(invalid)?)
            .filter(|w| (1..=53).contains(w))
            .ok_or_else(invalid)?;
        pos += 2;
        let mut weekday = 1;
        if extended && s.as_bytes().get(pos) == Some(&b'-') {
            pos += 1;
            weekday = digits(s.get(pos..pos + 1).ok_or_else(invalid)?).ok_or_else(invalid)?;
            pos += 1
        } else if !extended && s.as_bytes().get(pos).is_some_and(u8::is_ascii_digit) {
            weekday = digits(s.get(pos..pos + 1).unwrap()).unwrap();
            pos += 1
        }
        if !(1..=7).contains(&weekday) {
            return Err(invalid());
        }
        let jan4 = days(y, 1, 4);
        let jan4week = (jan4 + 3).rem_euclid(7);
        let day = jan4 - jan4week + (w - 1) * 7 + weekday - 1;
        if w == 53 && date(day - ((weekday - 4).rem_euclid(7))).0 != y {
            return Err(invalid());
        }
        day
    } else {
        let m = digits(s.get(pos..pos + 2).ok_or_else(invalid)?)
            .filter(|m| (1..=12).contains(m))
            .ok_or_else(invalid)?;
        pos += 2;
        if extended {
            if s.as_bytes().get(pos) != Some(&b'-') {
                return Err(invalid());
            }
            pos += 1
        }
        let d = digits(s.get(pos..pos + 2).ok_or_else(invalid)?).ok_or_else(invalid)?;
        pos += 2;
        let max = [
            31,
            if leap(y) { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ][m as usize - 1];
        if !(1..=max).contains(&d) {
            return Err(invalid());
        }
        days(y, m, d)
    };
    let rest = s.get(pos..).ok_or_else(invalid)?;
    if rest.is_empty() {
        return Err(EvidenceError::contract(format!(
            "{label} must include a timezone"
        )));
    }
    let time = &rest[rest.chars().next().unwrap().len_utf8()..];
    let (time, offset) = if let Some(t) = time.strip_suffix('Z') {
        (t, 0)
    } else if let Some(i) = time.find(['+', '-']) {
        let sign = if time.as_bytes()[i] == b'+' { 1 } else { -1 };
        (
            &time[..i],
            sign * clock(&time[i + 1..]).ok_or_else(invalid)?,
        )
    } else {
        return Err(EvidenceError::contract(format!(
            "{label} must include a timezone"
        )));
    };
    let result = epoch_days * 86_400_000_000 + clock(time).ok_or_else(invalid)? - offset;
    if date(result.div_euclid(86_400_000_000)).0 < 1
        || date(result.div_euclid(86_400_000_000)).0 > 9999
    {
        return Err(invalid());
    }
    Ok(result)
}
pub(crate) fn format(t: i64) -> String {
    let (y, m, d) = date(t.div_euclid(86_400_000_000));
    let time = t.rem_euclid(86_400_000_000);
    let sec = time / 1_000_000;
    let micro = time % 1_000_000;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{}Z",
        sec / 3600,
        sec / 60 % 60,
        sec % 60,
        if micro == 0 {
            String::new()
        } else {
            format!(".{micro:06}")
        }
    )
}
