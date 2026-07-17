# Results and interpretation

## State tracking

Across 2,000 electronic snapshots, the minimum singular value of all consecutive active-space overlaps is `0.999580`. The ten-state subspace therefore remains continuous even when near-degenerate fullerene orbitals rotate strongly within their manifold.

## Coherent and reduced-model dynamics

At 99.5 fs, the full ten-state coherent ensemble gives

`P(C60) = 0.279785 +/- 0.096349`.

The independently constructed 4D+3A model gives

`P(C60) = 0.246209 +/- 0.088201`.

The ensemble-curve RMSE is `0.018301`, and the mean trajectory-level RMSE is `0.025094`. The maximum difference between the fragment-projector population and the simple acceptor-subspace population is `0.006402`.

## Surface-hopping sensitivity

The approximate digitized experimental endpoint is `0.608567`. Plain CPA-FSSH gives `0.280850` with full-window RMSE `0.192214`. Boltzmann-rescaled upward hops give `0.898903` with RMSE `0.304553`.

Thus the plain treatment under-transfers, while the rescaled treatment over-transfers. The published experimental curve lies between them, showing that detailed balance is a major model choice rather than a universal correction.

## Coherence and pathways

The maximum mean donor-acceptor coherence is `0.426162` at `77.5 fs`. The most active positive-flux channel is `D2 -> A1`, with integrated positive flux `0.273903`. Its signed transfer is only `0.029675`, while its absolute transfer is `0.518130`, establishing substantial bidirectional recrossing.

The largest net-forward channels are:

- `D1 -> A3`: `0.072861`;
- `D1 -> A2`: `0.053819`;
- `D3 -> A2`: `0.049790`;
- `D2 -> A3`: `0.041704`.

The integrated current satisfies the acceptor-population continuity relation with mean RMSE `6.25e-5 fs^-1`.

## Supported conclusion

The 4D+3A Hamiltonian is a compact and quantitatively validated representation of the coherent ten-state dynamics. Charge transfer is multichannel and strongly recrossing. The main disagreement with experiment is therefore a physical-model issue - electronic structure, nuclear-path approximation, and detailed balance - rather than a state-tracking or propagation instability.

## Limitations

- Ground-state Kohn-Sham orbitals replace many-electron excited states.
- Nuclear paths are prescribed, thermostatted, and receive no electronic back-reaction.
- No explicit electronic decoherence is applied to the coherent amplitudes.
- The experimental trace is digitized from a published figure and lacks original error bars.
- The reduced model is exact only relative to the finite active-space construction used here.
