# Corrigendum to the VP1 report

The VP1 report, *Performance and Applicability of Physics-Informed Neural Operators for Simulating Air Flow in Porous Granular Media* ([PDF](Albertin_2026_PINO_Airflow_PorousMedia.pdf)), was produced with [GrainLegumes-PINO-Airflow](https://github.com/Rinovative/grainlegumes-pino-airflow). The report stays unchanged. This note lists what in it is wrong, imprecise or unsupported, checked against the archived code, and states, where it applies, what this project, GrainLegumes-PINO-Drying (VP2), does instead. The points are grouped by their consequence and ordered by weight within each group: A changes reported results or main conclusions, B biases individual results, comparisons or the training setup, C concerns the description only. Each point states the error and its effect; the evidence follows under Details and, where VP2 has a counterpart, what VP2 does under In VP2. All statements were checked against commit `811ba0b` (18 February 2026), the code with which the reported results were produced; [What the archive records](#what-the-archive-records) at the end states what this commit does not record. Page numbers are the printed ones. The numbers in points 2 and 5 are analytic examples computed for this note, not results of VP1 runs.

## A · Errors that change reported results or main conclusions

**1. Evaluation sets, normalisation and model selection, p. 17–19 and 22.** The ID results were computed over all 1000 ID cases, 800 of them training cases; there is no ID test partition, the normalisation and the selection of checkpoints and configurations used the validation cases, the OOD results cover all OOD cases instead of the stated 20 %, and the model analysed in detail was chosen after comparing ID and OOD results. **Effect:** The ID results of Tab. 2–5 and the ID figures are no estimate of held-out ID generalisation, and the report's statement that evaluation used the ID-validation set does not hold.

<details>
<summary>Details</summary>

What the code does: the ID data were split 80/20 into training and validation only (seed 9), so there is no independent ID test partition. The best checkpoint of each run (point 3) and the Study II configurations (p. 22) were selected on the validation subset, and the channel normalisation was fitted on all ID cases including that subset (p. 17). The artifact generation evaluates every case file of the dataset directory without restoring the subsets used in training, so the ID results of Tab. 2–5 and the ID figures were computed over all 1000 ID cases, 800 of which were training cases, and the OOD results over all OOD cases instead of the 20 % subset of p. 17. What this means: "Evaluation is carried out on the ID-validation set" (p. 18) does not hold; the ID results combine training and validation cases and are no estimate of held-out ID generalisation, the size of the resulting optimism was not measured, and validation-only results would still not be the independent "ID and OOD test sets" of p. 22. The OOD cases took no part in training or checkpoint selection, but the model analysed in detail was chosen after comparing ID and OOD results (p. 19).

</details>

<details>
<summary>In VP2</summary>

Each Dataset fixes a Train, validation and ID-test partition per ID material (80/10/10 with seed 9 in the maintained steady plan, whose OOD set is a whole held-out material). The normalisation is fitted on Train only, checkpoints and configurations are selected on validation only, and the ID-test and OOD cases never enter a selection ([Datasets and splits](08_scientific_decisions.md#datasets-and-splits), [Model selection](08_scientific_decisions.md#model-selection)).

</details>

**2. Spectral derivatives (SP), p. 14–15.** The padding of the spectral derivatives in `811ba0b` is not the smooth even extension the report describes: the periodic continuation jumps at the boundaries, so the SP physics loss does not vanish at the true solution. **Effect:** The jump distorts the SP physics loss in training and the SP residuals evaluated after training (Tab. 5, Fig. 4, 17 and 18(a)), and with them every SP run, in particular PI-FNO-divu-SP, and the conclusion that SP improves on PS.

<details>
<summary>Details</summary>

The report describes a mirror-symmetric reflection that constructs "a smooth periodic continuation". The loss classes of `811ba0b` pad each side of an axis by N−1 samples without repeating the edge sample, giving 3N−2 samples (`grad_mode` `fft_reflect`, the default and the setting of the training scripts); the periodic continuation that the FFT assumes then jumps by the full end-to-end difference of the field, for the pressure along the flow direction by the whole inlet–outlet drop. The only other mode, `fft`, applies the FFT without padding and has the same jump. For a linear pressure drop on the generator grid (251 rows × 401 columns, point 21), ∂p/∂y alternates from row to row between about 0.48 and 1.52 times the exact value (median error 53 %, maximum 55 % inside the two-sample crop), against a median of 0.9 % (maximum 23 %, next to the crop) with the even extension of period 2N−2 that the description implies; on the 250 × 400 grid stated in the report, the median error is 14 % and the maximum 52 %. With uniform velocity, this pressure field satisfies Eq. (11) exactly in a homogeneous medium, so its SP momentum residual is about half of |∇p|: the SP physics loss of `811ba0b` does not vanish at the true solution. Affected: the SP physics loss in training and the SP residuals evaluated after training (Tab. 5, Fig. 4, 17 and 18(a)), hence every SP run, in particular PI-FNO-divu-SP (Tab. 1, 4 and 5), and the conclusion that SP improves on PS (p. 27–28). Whether the extension caused the frequent divergence of SP runs (p. 27) was not tested.

</details>

<details>
<summary>In VP2</summary>

Training computes the physics residuals with finite differences on the physical grid (`physical`); `spectral` always uses Fourier continuation, so VP2 has no periodic spectral derivative ([Steady flow](04_training.md#steady-flow), [Physics diagnostics](05_evaluation.md#physics-diagnostics)).

</details>

**3. Relative L2 error and checkpoint criterion, Tab. 2 and 4.** The relative L2 error is one norm over the concatenated fields p, u, v and |u| in physical units, so the pressure dominates it; the checkpoint criterion mixes the units in the same way and sums batch RMSEs. **Effect:** Unless the velocity errors are larger in relative terms by orders of magnitude, Tab. 2 and 4 measure essentially the pressure error, not the accuracy of the whole flow field; the checkpoints were chosen by a criterion with the same unit imbalance, in which the last validation batch of 8 cases counts as much as a full one.

<details>
<summary>Details</summary>

The report describes "global, relative error measures computed over the spatial domain" (p. 18) without defining them. The evaluation computes for each case one norm ‖ŷ − y‖/‖y‖ over the concatenated fields p, u, v and |u| in physical units. With inlet pressures of order 10² Pa and superficial velocities below 1 m/s, ‖p‖ exceeds the velocity norms by orders of magnitude, and a relative velocity error enters the metric scaled down by the same factor: unless the velocity errors are larger in relative terms by orders of magnitude, Tab. 2 and 4 measure the pressure error and do not show the velocity accuracy. The checkpoint criterion mixes the units in the same way and is not one validation RMSE: it is the sum of the RMSEs of the validation batches, each over p, u and v in physical units, divided by the 200 validation cases, so with batches of 32 the last batch of 8 cases counts as much as a full one. Affected: the reading of Tab. 2 and 4, and of the percentages in point 11, as accuracy of the whole flow field, and the choice of the checkpoints; the values are computed in the same way for all models.

</details>

<details>
<summary>In VP2</summary>

RMSE is reported per field and for the velocity vector in physical units; the dimensionless `rel_l2` and `rel_h1` average ratios computed separately for each field ([Steady case metrics](05_evaluation.md#steady-case-metrics)). Checkpoints, the scheduler and the studies use `normalized_group_macro_rmse`: the mean of the pressure RMSE divided by its Train standard deviation and the velocity-vector RMSE divided by the root of the summed Train variances of u and v, accumulated over the complete validation set, so neither the units nor a partial batch shift the weights ([Model selection](08_scientific_decisions.md#model-selection)).

</details>

**4. Parameter–error correlations, Fig. 12 (p. 45) and Fig. 16 (p. 48).** The four columns labelled as channels are the relative errors of pressure rows 0 to 3, not of p, u, v and |u|. **Effect:** Neither figure shows a correlation with the velocity errors; the interpretation on p. 26 that the inlet-pressure level is the dominant driver of case difficulty would have to be recomputed.

<details>
<summary>Details</summary>

The figures show the correlations of the generator parameters with the relative L2 errors "per channel", in columns labelled rel_l2[p], rel_l2[u], rel_l2[v] and rel_l2[U]. The plotting code takes the pressure channel of each stored error field and then indexes its first four grid rows as if they were the channels, so the four columns are the relative L2 errors of pressure rows 0 to 3; this is why they are nearly identical (for the inlet-pressure level in the first panel of Fig. 12: −0.34, −0.33, −0.34 and −0.34). Neither figure shows a correlation with the velocity errors or with the global relative L2 error. Affected: the interpretation on p. 26 that the inlet-pressure level is "the dominant driver of case difficulty", with the correlations quoted there; it would have to be recomputed.

</details>

<details>
<summary>In VP2</summary>

The parameter–error views correlate each parameter with named error columns, the normalised RMSE of p, u and v (`normalized_rmse_<field>`) and, as fourth column, the normalised velocity-vector RMSE, which also counts direction errors; no channel is addressed by position ([Views](05_evaluation.md#views)).

</details>

## B · Errors that bias individual results, comparisons or the training setup

**5. Relative H1 training loss, p. 18.** The `neuralop` H1 loss uses periodic differences, so along the flow direction the stencil joins inlet and outlet, and the two wrap-around rows dominate the gradient norm of the pressure target. **Effect:** The data term of every VP1 model strongly penalises differences between the errors at inlet and outlet; how much this shapes the trained models was not measured.

<details>
<summary>Details</summary>

The report states that the losses "follow the discrete, quadrature-based definitions implemented in the neuralop library". All four training scripts use `H1Loss(d=2)` of the `neuralop` library (neuraloperator 2.0), whose finite differences are periodic by default, so along the flow direction the stencil joins inlet and outlet. For a linear pressure drop on the 251 × 401 grid, the two wrap-around rows give 124.5 times the interior derivative with opposite sign and carry 99.2 % of ‖∂p/∂y‖². Prediction and target pass through the same stencil, so an exact prediction still has zero loss, but the wrap-around rows dominate the gradient norm of the target and strongly penalise differences between the errors at inlet and outlet; how much they dominate the error term depends on the prediction. Affected: the data term of every VP1 model, supervised and physics-informed, and the H1 values logged during training and validation. The metrics of the evaluation artifacts do not use this stencil; whether the loss contributes to the boundary-localised errors (p. 26 and 28) was not tested.

</details>

<details>
<summary>In VP2</summary>

The H1 data loss uses one-sided differences at the boundaries (`periodic_in_x` and `periodic_in_y` false), so it no longer joins inlet and outlet, and it differentiates at the physical grid spacing in metres, like the evaluation `rel_h1` ([Steady flow](04_training.md#steady-flow)).

</details>

**6. Interior crop of the physics residuals, p. 15 and 19.** The momentum residuals of SP models exclude two boundary samples, those of PS models include the boundary rows and columns, in training as in evaluation; the report compares their levels without noting that they cover different parts of the grid. **Effect:** The momentum residual levels of SP and PS models in Tab. 3, 5 and Fig. 3, 4, 17 are not directly comparable, which affects the accuracy–momentum comparison of Fig. 4 and its interpretation on p. 26.

<details>
<summary>Details</summary>

The report introduces the crop of a narrow boundary margin as part of the spectral scheme (p. 15) and states that the evaluated residuals use "the same numerical differentiation strategy as employed during training" (p. 19). This holds: the evaluation builds the loss classes with their own defaults, a two-sample crop for SP and none for PS, and the reported runs used the same values (the training scripts set two samples, which the configurations of the PS runs set to zero). The momentum residuals of PS models in Tab. 3 and 5 and Fig. 3, 4 and 17 therefore include the boundary rows and columns, which those of SP models exclude; the continuity residual shown there is evaluated over the full grid for both types. Affected: the comparison of momentum residual levels between SP and PS models, including the accuracy–momentum comparison of Fig. 4 (p. 22) and its interpretation on p. 26.

</details>

<details>
<summary>In VP2</summary>

Only the momentum residual is cropped, by two samples, in training (`interior_crop: 2`) and in evaluation (`EVAL_PAD = 2`); the flux balance and the boundary terms use the full grid. The residuals of every run are evaluated with the same finite-difference operator, whatever derivatives the run trained with ([Steady case metrics](05_evaluation.md#steady-case-metrics)).

</details>

**7. Reference curves of the physics diagnostics, Fig. 3 (p. 21) and Fig. 17 (p. 49).** The ground-truth reference curves are computed differently from the model curves: drag without K_xy, the crop of the first model in the comparison, and a boundary term without the outlet. **Effect:** The remaining gaps to the reference mix these differences with discretisation effects.

<details>
<summary>Details</summary>

The report states that the reference curve shows "the same diagnostics computed from the ground-truth fields" (p. 48). They are not computed like those of the models: the momentum reference uses the drag μ(u/K_xx, v/K_yy) instead of μK⁻¹u, neglecting K_xy of the rotated tensor; the momentum and continuity references take the interior crop of the first model in the comparison (two samples for SP, none for PS) and apply it before differentiating; and the BC reference compares only the inlet, without the outlet term. The momentum and continuity references use finite differences of the COMSOL fields interpolated to the grid, which are not pointwise divergence-free (point 14). Affected: the non-zero reference level and the "remaining gaps to the GT reference" (p. 25–26, 28) mix these differences with discretisation effects; the size of the K_xy error was not quantified.

</details>

<details>
<summary>In VP2</summary>

Predictions and reference fields pass through the same residual function, so drag, crop and boundary terms are identical for both ([Physics diagnostics](05_evaluation.md#physics-diagnostics)).

</details>

**8. Reduction and balance of the loss terms, Eq. (13), (20) and (22), p. 7, 15 and 18.** The data loss is summed over the cases and fields of a batch (96 ratios at batch size 32) while the physics terms are means, and the physics terms are added unscaled in different units. **Effect:** λ_phys and λ_bc weigh the physics 96 times less than with a mean data loss, the balance depends on the batch size, and the explanation of the continuity variants on p. 24 is untested.

<details>
<summary>Details</summary>

Eq. (20) and (22) present L_data as one relative H1 norm. The scripts keep `neuralop`'s default reduction `sum`, so L_data is the sum of the relative H1 norms of every case and normalised output field of a batch, 96 ratios at the batch size of 32, while L_phys and L_bc are means over cases and grid points; compared with the mean of these ratios, the data term is 96 times larger, so λ_phys and λ_bc weigh the physics 96 times less, and the balance changes with the batch size. Relative to the single joint norm of Eq. (22) there is no fixed factor, because every field is normalised separately. The H1 derivatives use the spacings 1/H and 1/W of a unit square, not the physical grid spacing. Within L_phys, Eq. (13) adds the mean squares of R_mom in (Pa/m)² and of R_inc in 1/s² without scaling; the report allows separate weights in general (p. 7), but VP1 used none, so the balance of the two terms is set by units and characteristic scales, not by λ_phys. In the residuals evaluated after training, the continuity MSE is a small fraction of the momentum MSE for every physics-informed model: at most 7.5·10⁻⁵ at the median and 3.4·10⁻⁴ at q95 in Tab. 3 (p. 21, all PS), and at most 1.4·10⁻⁵ in Tab. 5 (p. 47), as ratios of the tabulated statistics; both tables report MSE(∇·u), also for the models trained with ∇·(εu). These statistics establish neither the share of the continuity term in the training loss nor its contribution to the gradients. The explanation of the differences between the two continuity variants on p. 24 ("a smoother, better-conditioned constraint signal") is offered as a hypothesis; the report contains no term-wise gradient analysis or ablation that would support it. Affected: the meaning of λ_phys and λ_bc, which is a limitation of the method rather than a misreported equation, and the comparison of the continuity variants.

</details>

<details>
<summary>In VP2</summary>

The data loss is averaged over cases and fields (`reduction="mean"`), so its weight does not depend on the batch size. The physics loss consists of three dimensionless terms (momentum, cross-section flux balance, boundary), each with its own weight and linear warm-up and with scales taken from the inputs ([Steady training and physics loss](08_scientific_decisions.md#steady-training-and-physics-loss)).

</details>

**9. Outlet condition pooled over the batch, Eq. (19), p. 15.** The outlet term squares the outlet mean pooled over all cases of a batch instead of the mean of each case. **Effect:** Outlet offsets of opposite sign in different cases cancel, so the training constrains the pressure reference of each case less than Eq. (19) states.

<details>
<summary>Details</summary>

Eq. (19) penalises the squared mean outlet pressure of a field. The training loss squares the mean over the outlet points of all cases in a batch (`p[outlet].mean().pow(2)`, batch size 32), so outlet offsets of opposite sign in different cases cancel. Affected: the pressure reference in the training of the physics-informed models; the BC residuals of Tab. 3 and 5 are evaluated case by case.

</details>

<details>
<summary>In VP2</summary>

The outlet term squares the outlet mean of each case and then averages over the batch ([Steady flow](04_training.md#steady-flow)).

</details>

**10. Channel-wise RMSE, Fig. 15 (p. 48).** Fig. 15 compares the domain means of each case, not the fields. **Effect:** It shows how well the domain averages agree, not the prediction error per channel.

<details>
<summary>Details</summary>

The report states that Fig. 15 "decomposes the prediction error into channel-wise RMSE". The plotting code first averages each field of a case over the domain and computes RMSE and R² across these per-case means (axes "GT mean" and "Prediction mean"). Spatial errors of opposite sign cancel in the mean, so the figure measures how well the domain averages agree, not the field accuracy per channel. Affected: the reading of Fig. 15 as a decomposition of the prediction error.

</details>

<details>
<summary>In VP2</summary>

The comparison of case means remains as a view titled "GT vs prediction mean"; the error per field is a separate metric (`normalized_rmse_<field>`).

</details>

**11. Percentages, p. 24.** The 12.6 % (ID) and 9.3 % (OOD) advantage of divu is relative to divu, not "relative to the porosity-weighted variant"; relative to that variant it is 11.2 % and 8.5 %.

## C · Errors in the description that change no computation

**12. Reproducibility, Abstract and p. 2, 9 and 28.** The report calls its pipeline fully reproducible (Abstract; p. 9 and 28) and its description sufficient to ensure reproducibility (p. 2), but the configurations of the reported runs and the COMSOL template that produced the data are missing from the archive. **Effect:** the results cannot be reproduced exactly from the repository ([What the archive records](#what-the-archive-records)).

<details>
<summary>In VP2</summary>

Every run persists its fully resolved configuration with digests of its training and model code. Every Source binds its resolved physical configuration, seeds and COMSOL template; the templates are part of the repository (`simulation/steady_flow/`, `simulation/transient_drying/`) with a SHA-256 checksum ([Identity](07_reproducibility_and_validation.md#identity)).

</details>

**13. Brinkman viscous term, Eq. (3), p. 4.** Eq. (3) writes the viscous term as μ∇²u with the fluid viscosity, whereas Eq. (10)–(11) and the code use the stress (μ/ε)(∇u + ∇uᵀ − ⅔(∇·u)I); the stated reason for keeping its trace term is unsupported. **Effect:** Description only: every training loss and evaluated residual uses the stress of Eq. (10).

<details>
<summary>Details</summary>

Eq. (3) writes the viscous term as μ∇²u, and the sentence below it sets the effective viscosity of the Brinkman term equal to the fluid viscosity μ. Both contradict Eq. (10)–(11) of the same report and the code: the training losses of `811ba0b` evaluate ∇·τ with τ = (μ/ε)(∇u + ∇uᵀ − ⅔(∇·u)I) of Eq. (10), with the heterogeneous porosity ε(x) inside the divergence; this is the viscous stress of COMSOL's Brinkman interface (checked on a solved VP2 model). For constant ε and ∇·u = 0, ∇·τ reduces to (μ/ε)∇²u, so the Brinkman term has the coefficient μ/ε, not μ; for heterogeneous ε, ∇·τ also contains terms in ∇(1/ε). The residuals of p. 14, said to be derived from Eq. (3), are those of Eq. (11). Separately, below Eq. (10) (p. 7), the trace term "formally vanishes" only for an exactly divergence-free field, not for predictions or for reference fields sampled on the grid, and the stated reason for retaining it, "to improve numerical stability", is supported by no experiment in the report and no switch in the code; COMSOL's Brinkman interface keeps the term also for incompressible flow (checked on the same model). Both errors lie in the description only: the training losses and all evaluated momentum residuals, including the reference of point 7, use the stress of Eq. (10), so correcting Eq. (3) changes no computation.

</details>

<details>
<summary>In VP2</summary>

VP2 uses the same stress, written as such and with the trace term ([Steady flow](04_training.md#steady-flow)).

</details>

**14. Continuity forms, Eq. (15)–(16), p. 14.** εu is called the mass flux, and the report does not state that ∇·(εu) penalises the exact solution wherever u·∇ε ≠ 0. **Effect:** This bears on the interpretation of the ∇·(εu) models; their description as a well-conditioned regularisation is untested.

<details>
<summary>Details</summary>

The report calls εu "the mass flux", although u is the superficial velocity: at constant density the conserved mass flux is the density times u, and mass conservation is ∇·u = 0. The report correctly presents ∇·(εu) as a regularising surrogate that "does not impose pointwise divergence-free velocity", but does not state its consequence: ∇·(εu) = ε∇·u + u·∇ε (Eq. 16) penalises the exact solution wherever u·∇ε ≠ 0. Its description as "a numerically well-conditioned regularisation of the momentum balance" is untested (point 8). Pointwise ∇·u is no exact target on the grid either: the COMSOL fields interpolated to the grid are not pointwise divergence-free under finite differences, as the non-zero continuity reference in Fig. 3 and 17 shows (point 7).

</details>

<details>
<summary>In VP2</summary>

Continuity enters as the flux balance through each row: the deviation of each row's volume flux from the mean row flux, scaled by a reference flow. It is an integral condition that the COMSOL solution nearly satisfies on the grid ([Steady flow](04_training.md#steady-flow)).

</details>

**15. Permeability–porosity coupling, p. 11 and 38–39, Eq. (39).** The coupling between permeability scale and porosity stated on p. 11 does not exist in the generator: κ̄ cancels, and the porosity level depends only on A_rel. **Effect:** Description of the input distribution, not of the solutions.

<details>
<summary>Details</summary>

On p. 11, the report states "a physically meaningful global coupling between permeability scale and porosity level". The generator sets A_mat = A_rel·κ̄ and solves κ̄ = A_mat·ε0³/(1−ε0)² for ε0, so κ̄ cancels: ε0 depends only on the sampled A_rel (ε0 = 0.5 at its base value 2), not on the permeability level. Affected: the description of the input distribution, not the solutions for the given coefficient fields. Corrected after submission in commit [`98ba05a`](https://github.com/Rinovative/grainlegumes-pino-airflow/commit/98ba05a) of the VP1 repository.

</details>

<details>
<summary>In VP2</summary>

Porosity follows a fixed Kozeny–Carman trend of each material from the sampled permeability scale, with a bounded packing scatter per case ([Model](02_physical_model.md#model)).

</details>

**16. Permeability construction, Eq. (30)–(35), p. 37–38.** κ̄ is a nominal scale, not an enforced mean of each case, and ρ is a stretch factor whose principal-permeability ratio is ρ², not the anisotropy ratio itself. **Effect:** Description of the input distribution.

<details>
<summary>Details</summary>

κ̄ in Eq. (30)–(32) is a nominal scale, not an enforced spatial mean: the subtraction of s_log²/2 preserves the expected value for a standard Gaussian structure field, but the generated field is not renormalised, so "the prescribed mean permeability remains unchanged" (p. 37) does not hold for a single case. ρ in Eq. (33)–(35) is called "a local anisotropy ratio" (p. 38), but it is a stretch factor: Eq. (35) and the code give κ₁ = ρκ and κ₂ = κ/ρ, so κ₁/κ₂ = ρ² (ρ = 2 gives a principal-permeability ratio of 4). Affected: the description of the input distribution.

</details>

<details>
<summary>In VP2</summary>

The sampled parameter `max_ratio` is the principal-permeability ratio itself: the generator sets κ₁ = κ√r and κ₂ = κ/√r, so κ₁/κ₂ = r, and its ranges rest on measured directional airflow resistance of grain beds, with the sources in the provenance. The register calls `kappa_mean` a nominal scale whose realised case mean differs ([Physical model and data generation](08_scientific_decisions.md#physical-model-and-data-generation)).

</details>

**17. Tensor orientation, Eq. (34), p. 38.** The generator swaps the outputs of MATLAB's `gradient`, so, up to the random perturbation, the principal axes are reflected across the direction x = y relative to Eq. (34). **Effect:** Description of the input distribution; the models were trained on the tensor actually generated.

<details>
<summary>Details</summary>

The generator takes the outputs of MATLAB's `gradient` in swapped order (`[dzdy, dzdx] = gradient(z_bg, dy, dx)`, whose first output is the x-derivative), so θ_raw = atan2(∂z_bg/∂x, ∂z_bg/∂y) ≡ π/2 − θ_raw of Eq. (34) (mod π): up to the random perturbation, the principal-axis directions are reflected across the direction x = y relative to the description. Affected: the description of the input distribution; the models were trained on the tensor actually generated.

</details>

<details>
<summary>In VP2</summary>

The Python generator takes the gradient components in the right order (`np.gradient` returns ∂/∂y, then ∂/∂x), so θ_raw = atan2(∂b/∂y, ∂b/∂x) as in Eq. (34).

</details>

**18. Parameter transformations, A.1, p. 35.** A.1 omits the linear, phase, integer and softmax transformations of the generator; with δ = 1.2 in the OOD set, each of the four linearly scaled parameters changes sign in about 8.3 % of the cases. **Effect:** Description of the input distribution.

<details>
<summary>Details</summary>

A.1 describes the logarithmic (Eq. 23) and logit (Eq. 24) transformations only. The generator also scales a_sin, f_sin, a_gauss and a_lin linearly as α₀(1 + δξ) with ξ uniform in [−1, 1], draws the phase φ_sin as mod(φ₀ + δπξ, 2π), rounds the number of Gaussian bumps to 1–5 and derives the two structure weights by a softmax of log-weights. With δ = 1.2 in the OOD set, each of the four linearly scaled parameters changes sign in about 8.3 % of the cases (ξ < −1/1.2), so, for example, the inlet trend reverses and Gaussian bumps become dips. Affected: the description of the input distribution (Fig. 8 shows the resulting distributions).

</details>

<details>
<summary>In VP2</summary>

Every interval parameter declares its transform (linear, log, logit or phase) in `registry.yaml`, and the other sampling kinds, such as the integer number of Gaussian bumps, are declared there too. A parameter-OOD case moves one eligible parameter, or one complete coupled record, into an explicit range outside the natural support instead of widening a δ-scaling ([Sampling](02_physical_model.md#sampling)).

</details>

**19. Porosity bounds, p. 39.** The porosity field is clipped to [0.3, 0.8], not to [0.2, 0.8]. **Effect:** Description of the input distribution.

<details>
<summary>Details</summary>

The porosity field is clipped to [0.3, 0.8], not to "ε_min = 0.2 and ε_max = 0.8": `gen_porosity_field.m` sets `phi_min_global = 0.30`, and the batch run does not override it. Affected: the description of the input distribution.

</details>

<details>
<summary>In VP2</summary>

VP2 clips the porosity to [0.2, 0.8] (`eps_min_global`, `eps_max_global` in `common.yaml`) and records the clipped fraction of each case ([Model](02_physical_model.md#model)).

</details>

**20. Inlet pressure profile, Eq. (41), p. 39.** Eq. (41) omits the sampled phase of the sine term, and p_inlet,mean scales the profile without being its mean. **Effect:** Description only.

<details>
<summary>Details</summary>

Eq. (41) omits the phase φ_sin of the sine term, which the generator samples per case (sin(2πf_sin x̂ + φ_sin)). p_inlet,mean scales the profile without being its mean, because the sine and Gaussian terms are not mean-free, so the profiles do not preserve "a common mean pressure level" (p. 39) exactly. Affected: the description only.

</details>

<details>
<summary>In VP2</summary>

The sine phase is sampled and applied. The base level `pressure_bc.mean` is still not the realised inlet mean, and the register says so ([Physical model and data generation](08_scientific_decisions.md#physical-model-and-data-generation)).

</details>

**21. Grid size, p. 13 and 41, Eq. (45), p. 42.** The grid has 251 × 401 points with both ends included, not 250 × 400. **Effect:** Description only.

<details>
<summary>Details</summary>

The generator includes both ends of each axis (`round(L/Δx) + 1` points), so the 1200 mm × 750 mm domain is sampled on 251 rows (y) × 401 columns (x) at 3 mm spacing, not on 250 × 400 points, and the dataset tensors are (1000, 7, 251, 401) and (1000, 3, 251, 401), not those of Eq. (45). Affected: the description only.

</details>

<details>
<summary>In VP2</summary>

The grid is boundary-inclusive, 401 × 251 (X × Y) at 3 mm, and Generation checks that (points − 1) × spacing equals the domain length ([Domain and resolution](02_physical_model.md#domain-and-resolution)).

</details>

**22. Fourier modes and mode ratio, p. 16 and Tab. 1 (p. 17).** The mode counts act in the other axis order (128 modes along y, 160 along x), and the U-NO mode ratio is 0.4951, which keeps 31 instead of 32 modes. **Effect:** Description of the architectures.

<details>
<summary>Details</summary>

The report assigns the mode counts m_x × m_y to the horizontal and vertical directions. The scripts pass the pair (modes_x, modes_y) = (128, 160) as `n_modes`, which `neuralop` applies in tensor order (rows y, columns x), so the FNO keeps 128 modes along y (vertical) and 160 along x (horizontal). Tab. 1 gives r_mode = 0.5 for U-NO; the code uses 0.4951318201313778 (run names `mr0p495`), which keeps 31 instead of 32 modes in the reduced layers. Affected: the description of the architectures.

</details>

<details>
<summary>In VP2</summary>

The configs name the axis order of the mode counts ([Y, X]) and store the full mode ratio.

</details>

**23. FNO composition, Eq. (9), p. 6.** Eq. (9) composes the integral operators K_t alone and omits W_t and σ of each Fourier layer. **Effect:** Notation only.

<details>
<summary>Details</summary>

Eq. (9) composes the Fourier integral operators K_t of Eq. (7) alone, although each Fourier layer of Eq. (8) also contains the pointwise linear transformation W_t and the activation σ. The composition runs over the full layer maps v_t ↦ σ(W_t v_t + K_t v_t) of Eq. (8). Affected: the notation only.

</details>

**24. Citation, p. 7.** Physics-informed neural operators are cited as [8], the FNO paper; the introduction (p. 1) cites them correctly as [9].

## What the archive records

Commit [`811ba0b`](https://github.com/Rinovative/grainlegumes-pino-airflow/commit/811ba0b) (18 February 2026) holds the code with which the reported results were produced: the MATLAB generator and the COMSOL runner, the training scripts with their loss classes, and the evaluation and figure code. All points refer to this code. The ID and OOD datasets are named by their sampling settings, `lhs_var80_seed3001` (Latin hypercube, δ = 0.8, seed 3001) and `lhs_var120_seed4001` (δ = 1.2, seed 4001), which the scripts load; the batch script of the commit is set to another dataset (Sobol sampling, seed 13001). Two things the commit does not record:

**Run configurations.** Each training script holds its configuration as defaults that a run overrides, and the configurations of the reported runs were not saved in the repository; the author holds them. The defaults are not those of every reported run: the PI-FNO script defaults to four layers, the reported PI-FNO-divu-SP run has three, and the learning-rate defaults of the FNO, U-NO and PI-U-NO scripts (6.2·10⁻³, 5.3·10⁻⁴ and 8.5·10⁻³) differ from the learning rate of 10⁻² that the runs used as the report states. The run configurations also set the interior crop of the PS runs to zero (point 6).

**COMSOL template.** The templates in `811ba0b` cannot have produced the data: they have a constant porosity of 0.5, a constant inlet pressure of 500 Pa and at most three interpolation functions, whereas the runner re-points the interpolation functions `int1`–`int4` to each case file and exports `int4(x,y)` and `int5(x,y)` as porosity and inlet pressure; it changes only the data of these functions, not how the physics uses them. The template in the later VP1 history (commit [`904956d`](https://github.com/Rinovative/grainlegumes-pino-airflow/commit/904956d), 27 July 2026) binds the porosity field and the inlet profile in COMSOL's Brinkman interface (incompressible, Stokes flow), but its interpolation functions and porous-medium node were modified on 27 July 2026, so it does not prove the exact model used.

Later commits of the VP1 repository changed the evaluation code (from `801865b`, April 2026, among other things to a relative L2 error averaged over p, u and v and one spectral residual calculator for all models), so its current state does not reproduce the evaluation of the report; points 3, 6 and 7 do not describe that later code. The PDF was added in commit `f6b26f4` (23 April 2026); its text is that of the submitted report.
