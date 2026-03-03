# Sharc.Volrus

`Sharc.Volrus` is the Rust port track for OpenVDB within the Sharc ecosystem.

## Scope

- Port core OpenVDB concepts to idiomatic Rust.
- Keep behavior and data expectations compatible where practical.
- Provide a Rust-native surface that can be consumed by other Sharc components.

## Status

- Stage: early bootstrap.
- In this workspace, `Sharc.Volrus` is tracked as a submodule.
- Canonical source repository: https://github.com/revred/Sharc.Volrus

## Relationship To Sharc.Hub

- `Sharc.Volrus` is an owned project in the Sharc.Hub umbrella.
- It sits alongside the engine, trace, and tooling projects as a focused Rust effort for volumetric data structures.

## Licensing

- Same license model as upstream OpenVDB (Apache-2.0).
- Preserve upstream attribution and notices for any directly ported material.

## Near-Term Plan

1. Establish crate structure and API boundaries.
2. Add parity tests for core OpenVDB semantics.
3. Add CI for Windows and Linux.
4. Publish an initial alpha milestone.

## References

- Sharc.Volrus: https://github.com/revred/Sharc.Volrus
- OpenVDB upstream: https://github.com/AcademySoftwareFoundation/openvdb
