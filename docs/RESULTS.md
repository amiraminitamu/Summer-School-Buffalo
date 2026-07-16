# Results and interpretation

## State tracking

The minimum singular value of all consecutive active-space overlaps across 2,000 snapshots is 0.999580. This indicates that the ten-state subspace remains continuous and makes the matrix-log derivative-coupling construction numerically defensible.

## Libra versus experiment

At 99.5 fs, the paper-style hybrid estimator is 0.280850 for plain CPA-FSSH, compared with 0.608567 from the approximate digitized SHG trace. The full-window RMSE is 0.192214. The coherent estimator is 0.279785, demonstrating that the paper-style diagonal replacement has little effect for the plain run at the endpoint.

Boltzmann rescaling leaves the coherent amplitudes unchanged but raises the hybrid endpoint to 0.898903 and worsens RMSE to 0.304553. Thus plain dynamics under-transfer while the rescaled dynamics over-transfer. The result exposes strong sensitivity to the treatment of upward hops and the underlying electronic energy landscape.

## Reduced model

The retained 4D+3A model gives an ensemble endpoint of 0.246209 and RMSE 0.018301 relative to the full ten-state coherent propagation. This is sufficiently accurate for bath extraction while removing the upper three C60 directions.

## Bath and TENSO

Eighteen PCA components span the sampled fluctuation coordinates; twelve retain 90.764% of the variance. The within-path dynamic RMS is 0.20880 eV and the between-path static RMS is 0.14255 eV.

At 5 fs, four-mode TENSO calculations converge near 0.016691 C60 population with maximum trace errors near 10^-7 and Hermiticity errors near 10^-5 when the auxiliary-rank ceiling is 64. Dimension, initial rank, Padé order, and time-step tests change the endpoint by at most approximately 1.1e-4. A longer 12-mode production run is treated as a finite reduced-model calculation, not a full-system exact benchmark.

## Limitations

- Single-particle ground-state Kohn-Sham orbitals replace many-electron excited states.
- Nuclear paths are ground-state, thermostatted, and do not receive electronic back-reaction.
- No decoherence correction is applied to the coherent amplitudes.
- The experimental trace is digitized from a published plot and lacks original error bars.
- The PCA-diagonal bath neglects residual finite-lag cross-correlations between retained modes.
- TENSO is exact only within the reduced, truncated, fitted model.
