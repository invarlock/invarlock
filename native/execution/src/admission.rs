//! Conservative live-collection capacity calculations from the retained judge
//! collector. This layer accepts validated canonical byte counts; it does not
//! certify provider events or reinterpret reservations as measured billing.
use crate::{Error, Result};
use std::collections::BTreeMap;
pub const MAX_SOURCE_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_SOURCES: u64 = 1000;
pub const MEASUREMENTS_MAX_BYTES: u64 = 384 * 1024 * 1024;
pub const MAX_ADMISSION_GROWTH_BYTES: u64 = 20 * 1024 * 1024;
#[derive(Clone, Copy, Debug)]
pub struct CollectionReservations {
    pub max_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_cost_microusd: u64,
    pub input_tokens_per_call: u64,
    pub output_tokens_per_call: u64,
    pub cost_microusd_per_call: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct RetainedSize {
    pub header_bytes: u64,
    pub samples: u64,
    pub record_bytes: u64,
    pub retained_bytes: u64,
    pub spent_calls: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capacity {
    pub remaining_admissions: BTreeMap<&'static str, i128>,
    pub capacity: u64,
    pub limiting_resources: Vec<&'static str>,
    pub exhausted_resources: Vec<&'static str>,
    pub source_bound: u64,
    pub retained_bound: u128,
}
/// Exact integer floor formulas used by `_LiveCheckpoint._storage_bounds` and
/// `_remaining_admissions`. Negative capacity diagnostics remain negative.
pub fn collection_capacity(
    reservations: CollectionReservations,
    size: RetainedSize,
) -> Result<Capacity> {
    if reservations.input_tokens_per_call == 0
        || reservations.output_tokens_per_call == 0
        || reservations.cost_microusd_per_call == 0
        || size.header_bytes >= MAX_SOURCE_BYTES
    {
        return Err(Error::Invalid(
            "invalid collection reservation denominator or source header".into(),
        ));
    }
    let usable = (MAX_SOURCE_BYTES - size.header_bytes) as u128;
    let source_bound = (size.samples as u128).min(2 * size.record_bytes as u128 / usable + 1);
    let retained_bound =
        size.retained_bytes as u128 + source_bound * (6 * size.header_bytes as u128 + 1024) + 4096;
    let spent = size.spent_calls as i128;
    let remaining_admissions = BTreeMap::from([
        ("calls", reservations.max_calls as i128 - spent),
        (
            "input_token_reservation",
            (reservations.max_input_tokens / reservations.input_tokens_per_call) as i128 - spent,
        ),
        (
            "output_token_reservation",
            (reservations.max_output_tokens / reservations.output_tokens_per_call) as i128 - spent,
        ),
        (
            "cost_reservation",
            (reservations.max_cost_microusd / reservations.cost_microusd_per_call) as i128 - spent,
        ),
        (
            "retained_sources",
            (MAX_SOURCES as i128 - source_bound as i128).div_euclid(2),
        ),
        (
            "retained_bytes",
            (MEASUREMENTS_MAX_BYTES as i128 - retained_bound as i128)
                .div_euclid(MAX_ADMISSION_GROWTH_BYTES as i128),
        ),
    ]);
    let minimum = *remaining_admissions.values().min().unwrap();
    let capacity = minimum.max(0) as u64;
    let ordered = [
        "calls",
        "input_token_reservation",
        "output_token_reservation",
        "cost_reservation",
        "retained_sources",
        "retained_bytes",
    ];
    let limiting_resources = ordered
        .iter()
        .copied()
        .filter(|name| remaining_admissions[name] == minimum)
        .collect();
    let exhausted_resources = ordered
        .iter()
        .copied()
        .filter(|name| remaining_admissions[name] <= 0)
        .collect();
    Ok(Capacity {
        remaining_admissions,
        capacity,
        limiting_resources,
        exhausted_resources,
        source_bound: source_bound as u64,
        retained_bound,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reservations() -> CollectionReservations {
        CollectionReservations {
            max_calls: 100,
            max_input_tokens: 1000,
            max_output_tokens: 500,
            max_cost_microusd: 10000,
            input_tokens_per_call: 10,
            output_tokens_per_call: 5,
            cost_microusd_per_call: 100,
        }
    }
    #[test]
    fn storage_capacity_controls_large_output_admissions() {
        let c = collection_capacity(
            reservations(),
            RetainedSize {
                header_bytes: 500,
                samples: 100,
                record_bytes: 0,
                retained_bytes: 0,
                spent_calls: 0,
            },
        )
        .unwrap();
        assert_eq!(c.capacity, 19);
        assert_eq!(c.limiting_resources, vec!["retained_bytes"]);
        assert_eq!(c.source_bound, 1);
    }
    #[test]
    fn negative_storage_uses_python_floor_not_truncation() {
        let c = collection_capacity(
            reservations(),
            RetainedSize {
                header_bytes: 500,
                samples: 100,
                record_bytes: 0,
                retained_bytes: MEASUREMENTS_MAX_BYTES,
                spent_calls: 0,
            },
        )
        .unwrap();
        assert_eq!(c.capacity, 0);
        assert_eq!(c.remaining_admissions["retained_bytes"], -1);
        assert!(c.exhausted_resources.contains(&"retained_bytes"));
    }
    #[test]
    fn ambiguous_admissions_consume_all_call_reservations() {
        let c = collection_capacity(
            reservations(),
            RetainedSize {
                header_bytes: 500,
                samples: 100,
                record_bytes: 100,
                retained_bytes: 100,
                spent_calls: 101,
            },
        )
        .unwrap();
        assert_eq!(c.remaining_admissions["calls"], -1);
        assert_eq!(c.remaining_admissions["cost_reservation"], -1);
        assert_eq!(c.capacity, 0);
    }
}
