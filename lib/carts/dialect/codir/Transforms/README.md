# CODIR Transform Implementation

This directory is for CODIR-local passes only:

- `VerifyCodir.cpp`
- `CodirCodeletOpt.cpp`

Dialect-boundary conversion code belongs in
`lib/carts/dialect/codir/Conversion/`, where the source layout mirrors the
pipeline stages `sde-to-codir` and `codir-to-arts`.
