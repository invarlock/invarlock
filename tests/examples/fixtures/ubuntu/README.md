# Ubuntu signature fixture

`noble-backports.InRelease` is the original signed Ubuntu Noble Backports
release metadata dated 5 September 2026, fetched from
[the Ubuntu archive](https://archive.ubuntu.com/ubuntu/dists/noble-backports/InRelease).
The test preserves its bytes and creates unsigned prefix, suffix, duplicate
message, and signed-content mutations in memory.

`ubuntu-archive-keyring.gpg` contains only public signing keys from
`/usr/share/keyrings/ubuntu-archive-keyring.gpg` in the campaign's immutable
CUDA base. Its SHA256 is
`80a36b0a6de2f69f49d2df75ef473ccde121e9e190b9ea01d20a4f63778d5c31`.
No private key or generated signature is included. The fixture proves signature
and parsing behavior; it does not assert current package freshness.
