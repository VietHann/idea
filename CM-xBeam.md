# CM-xBeam: Covariance-Aligned xLSTM Memory for Physics-Guided Neural Beamforming in Multichannel Speech Separation

## Executive summary

The strongest version of an **xLSTM + neural beamforming paper for multichannel speech separation** is not “replace an LSTM/Transformer with xLSTM and predict beamforming weights.” That would be incremental and exposed to a straightforward reviewer objection: xLSTM is merely another sequence backbone.

The stronger research thesis is:

$$\boxed{\text{Recursive spatial covariance estimation and mLSTM matrix memory are structurally related outer-product memories.}}$$

A conventional online spatial covariance matrix (SCM) can be written as

$$\Phi_{t,f}^{(\nu)} \propto \lambda_{t,f}^{(\nu)}\Phi_{t - 1,f}^{(\nu)} + a_{t,f}^{(\nu)}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H},$$

where $\lambda$ determines how rapidly old spatial statistics are forgotten. Moving-speaker beamforming work has already shown why a fixed forgetting factor is problematic: the optimal tracking speed changes with source motion, and learned temporal weighting of instantaneous SCMs substantially improves adaptation. [\[1\]](https://arxiv.org/pdf/2205.03568)

Meanwhile, mLSTM updates a matrix-valued memory through a gated outer product,

$$C_{t} = f_{t}C_{t - 1} + i_{t}v_{t}k_{t}^{\top},$$

and xLSTM was explicitly introduced with a matrix memory and a covariance-style update rule. [\[2\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html)

That correspondence creates a credible research hypothesis:

> **mLSTM can serve as a learned controller of spatial-statistics memory, adapting the effective covariance-memory horizon to the current acoustic scene.**

However, I would **not** directly interpret the native mLSTM state $C_{t}$ as a physical acoustic covariance matrix. Standard mLSTM does not guarantee that $C_{t}$ is complex Hermitian positive semidefinite (PSD), whereas a valid microphone SCM should have those properties. The publishable design is therefore a **covariance-aligned architecture**: mLSTM learns *how to update* an explicit PSD SCM while the beamforming layer retains its physical structure.

The proposed model, **CM-xBeam**, is:

$$\boxed{\begin{aligned}
 & \text{STFT} \rightarrow \text{complex spatial encoder} \rightarrow \text{target/interference masks} \\
 & \rightarrow \text{dual mLSTM spatial memories} \rightarrow \text{adaptive PSD SCM accumulation} \\
 & \rightarrow \text{MVDR} \rightarrow \text{distortionless neural residual} \\
 & \rightarrow \text{uncertainty gating} \rightarrow \text{small TF-xLSTM refiner} \rightarrow \text{iSTFT}.
\end{aligned}}$$

The **highest-value technical contribution** beyond covariance-aligned mLSTM is a constraint-preserving neural correction to MVDR. Rather than allowing a neural network to arbitrarily replace the beamformer, predict a residual $\Delta w$, project it into the nullspace of the target steering vector,

$$r = \left( I - \frac{dd^{H}}{d^{H}d + \epsilon} \right)\Delta w,$$

and use

$$w_{\text{final}} = w_{\text{MVDR}} + \alpha r.$$

Then

$$r^{H}d = 0$$

and consequently

$$w_{\text{final}}^{H}d = w_{\text{MVDR}}^{H}d = 1.$$

Thus the learned correction can suppress interference beyond classical MVDR **without breaking the estimated distortionless constraint**. An uncertainty head controls $\alpha$, so neural correction is small when the physical statistics are reliable and stronger when covariance estimation becomes uncertain.

This distinction matters because learned SCM aggregation for moving speakers already exists through attention, including array-robust extensions. [\[1\]](https://arxiv.org/pdf/2205.03568) Therefore the paper must **not claim novelty simply from “learned covariance weighting.”** The defensible novelty is the combination of:

1.  a theoretical correspondence between mLSTM outer-product matrix memory and recursive spatial covariance accumulation;
2.  a **PSD-preserving explicit SCM controlled by mLSTM**, rather than treating arbitrary hidden matrices as covariance;
3.  dual target/interference spatial memories;
4.  a **distortionless-preserving neural residual beamformer**;
5.  calibrated uncertainty to govern neural-vs-physical correction;
6.  experiments designed around the hypothesis—especially moving sources, abrupt spatial changes, source crossings, DoA perturbations, and parameter-matched LSTM controls.

A targeted literature search through **August 30, 2026** found xLSTM applied to single-channel speech enhancement and multiple lines of learned/adaptive SCM aggregation and neural beamforming, but did not surface a primary publication explicitly framing **mLSTM matrix memory as covariance-aligned memory for multichannel neural beamforming**. This is encouraging but should not be treated as a formal novelty guarantee; a final literature sweep is mandatory before submission. [\[3\]](https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com)

For a speech venue, I would target **ICASSP/Interspeech first**, with a longer, more extensively validated version suitable for **IEEE/ACM TASLP**. The idea could interest a broader ML venue only if the covariance-memory formulation becomes a genuinely general methodological contribution rather than remaining a speech-specific module.

## Research thesis and problem formulation

### What the paper should actually claim

A weak framing would be:

> “We introduce xLSTM into neural beamforming for better long-context modeling.”

A much stronger framing is:

> **“We reinterpret adaptive beamforming as a matrix-memory problem. Motivated by the structural correspondence between recursive SCM accumulation and mLSTM covariance-style memory, we introduce a covariance-aligned mLSTM that learns the temporal memory horizon of physically valid target and interference SCMs. The learned statistics drive an MVDR beamformer whose residual neural correction is explicitly projected to preserve the distortionless constraint.”**

That sentence already tells a reviewer:

- what problem you discovered;
- why xLSTM is not arbitrary;
- where the physics enters;
- what mathematical constraint is preserved;
- what must be proven experimentally.

This is a much stronger research story than module stacking.

### Multichannel speech separation

For $M$ microphones and $K$ speakers,

$$y_{m}\lbrack n\rbrack = \sum_{k = 1}^{K}s_{k,m}\lbrack n\rbrack + v_{m}\lbrack n\rbrack,\quad\quad m = 1,\ldots,M.$$

The multichannel STFT is

$$\mathbf{y}_{t,f} = \lbrack Y_{t,f,1},\ldots,Y_{t,f,M}\rbrack^{\top} \in \mathbb{C}^{M}.$$

The system estimates $K$ separated signals at a chosen reference microphone,

$${\widehat{s}}_{k}\lbrack n\rbrack,\quad\quad k = 1,\ldots,K.$$

For **blind separation**, output ordering is unknown, so training and evaluation use permutation-invariant assignment.

For **target extraction**, the system additionally receives a target cue $q_{k}$. For this paper, I would make **direction/DoA the main target cue**, because it directly matches the spatial beamforming story. A dynamic cue can be written

$$q_{k,t} = E_{DoA}(\theta_{k,t}).$$

A circular representation is natural:

$$E_{DoA}(\theta) = \lbrack\sin\theta,\cos\theta,\ldots,\sin P\theta,\cos P\theta\rbrack.$$

This gives the same backbone two operating modes:

$$\left\{ \begin{matrix}
q_{k} = q_{k}^{\text{slot}}, & \text{blind separation} \\
q_{k,t} = E_{DoA}(\theta_{k,t}), & \text{target extraction}.
\end{matrix} \right.\ $$

Do **not** make speaker embeddings, text prompts, visual cues, and DoA all part of the main paper. That dilutes the contribution. Design a generic query interface in code, but evaluate DoA-guided extraction as the primary TSE case.

### Recommended primary metrics

The primary separation measure should be SI-SDR. Le Roux et al. introduced SI-SDR specifically to avoid several misleading behaviors of older SDR implementations. [\[4\]](https://arxiv.org/abs/1811.02508?utm_source=chatgpt.com) For zero-mean target $s$ and estimate $\widehat{s}$,

$$\alpha = \frac{\langle\widehat{s},s\rangle}{\parallel s \parallel_{2}^{2}},$$

$$s_{target} = \alpha s,\quad\quad e = \widehat{s} - s_{target},$$

$$SI - SDR = 10\log_{10}\frac{\parallel s_{target} \parallel_{2}^{2}}{\parallel e \parallel_{2}^{2}}.$$

Report improvement relative to the designated reference-microphone mixture:

$$SI - SDRi = SI - SDR(\widehat{s},s) - SI - SDR(y_{ref},s).$$

For blind separation, choose the source permutation using SI-SDR and **reuse that same assignment** for STOI and PESQ rather than independently optimizing every metric.

STOI is an intrusive intelligibility metric developed to predict intelligibility of degraded speech and remains useful as a complementary measure. [\[5\]](https://dl.acm.org/doi/abs/10.1109/TASL.2011.2114881?utm_source=chatgpt.com)

The user requested PESQ, so it is reasonable to report wideband PESQ for comparability with older enhancement/separation literature. However, ITU-T deleted the P.862/PESQ Recommendations on January 5, 2024 and now points users to P.863/POLQA; that status should be stated explicitly rather than presenting PESQ as the current ITU standard. [\[6\]](https://www.itu.int/rec/t-rec-p.862?utm_source=chatgpt.com)

A strong results table should therefore report:

| Category            | Metric                | Reporting                           |
|---------------------|-----------------------|-------------------------------------|
| Separation fidelity | SI-SDR                | dB, higher better                   |
| Improvement         | SI-SDRi               | dB over reference-mic mixture       |
| Intelligibility     | STOI                  | mean score                          |
| Perceptual quality  | WB-PESQ               | legacy comparability; label clearly |
| Model size          | Parameters            | millions                            |
| Arithmetic cost     | MAC/s or GFLOP/s      | fixed-duration input                |
| Speed               | Real-time factor      | CPU and one specified GPU           |
| Streaming           | Algorithmic latency   | milliseconds                        |
| Memory              | Peak inference memory | MB/GB                               |
| Training cost       | GPU-hours             | per seed and total                  |

Do not report “real time” without specifying hardware, batch size, utterance length, precision, and whether STFT/iSTFT are included.

### Datasets that support the research question

**SMS-WSJ should be the main controlled benchmark.** It was explicitly designed as a multichannel overlapping-speech database and randomizes room and spatial configurations; generation code and a baseline recipe were released with it. [\[7\]](https://arxiv.org/abs/1910.13934?utm_source=chatgpt.com) SpatialNet also uses SMS-WSJ-style multichannel evaluation and has an official open-source repository, which makes it a practical strong neural baseline. [\[8\]](https://arxiv.org/html/2307.16516v2?utm_source=chatgpt.com)

But static SMS-WSJ alone cannot validate the central claim about adaptive spatial memory. Create and release a **Dynamic-SMS** extension containing moving trajectories. This is well motivated by prior moving-speaker beamforming research, which shows that static SCM assumptions and fixed forgetting behavior become problematic when the transfer function changes over time. [\[9\]](https://arxiv.org/pdf/2205.03568) gpuRIR provides an open GPU-accelerated image-source implementation suitable for generating large numbers of synthetic RIRs. [\[10\]](https://arxiv.org/abs/1810.11359?utm_source=chatgpt.com)

For cross-domain generalization, spatialize **Libri2Mix/Libri3Mix** with a held-out family of rooms and array configurations. LibriMix was deliberately introduced for generalizable two- and three-speaker separation, making it useful as the clean-speech source corpus for such a test. [\[11\]](https://arxiv.org/html/2005.11262v1?utm_source=chatgpt.com)

Finally, use **LibriCSS** as an optional real-recorded continuous stress test. LibriCSS consists of multichannel recordings of simulated conversations and was created specifically to bridge conventional separation experiments and continuous real-world processing. [\[12\]](https://arxiv.org/pdf/2001.11482?utm_source=chatgpt.com) I would emphasize recognition-oriented or continuous-separation metrics there rather than forcing every LibriCSS condition into a conventional utterance SI-SDR table.

## Covariance memory as the theoretical core

### Classical SCM accumulation

A mask-based instantaneous target SCM is

$$\Psi_{t,f}^{(s)} = m_{t,f}^{(s)}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H}.$$

Similarly,

$$\Psi_{t,f}^{(i)} = m_{t,f}^{(i)}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H}.$$

Classical online beamforming recursively integrates these statistics. A simple form is

$$\Phi_{t,f}^{(\nu)} = \lambda\Phi_{t - 1,f}^{(\nu)} + \Psi_{t,f}^{(\nu)},\quad\quad\nu \in \{ s,i\}.$$

Unrolling gives

$$\Phi_{t,f}^{(\nu)} = \sum_{\tau = 1}^{t}\lambda^{t - \tau}\Psi_{\tau,f}^{(\nu)}$$

up to initialization. This exact interpretation—older instantaneous SCMs receiving exponentially decreasing weights—is described in moving-source neural-beamforming work, along with the central limitation: a fixed forgetting factor forces one tracking speed across different source-motion conditions. [\[9\]](https://arxiv.org/pdf/2205.03568)

A more useful normalized version for our architecture is

$$S_{t,f}^{(\nu)} = \lambda_{t,f}^{(\nu)}S_{t - 1,f}^{(\nu)} + a_{t,f}^{(\nu)}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H},$$

$$z_{t,f}^{(\nu)} = \lambda_{t,f}^{(\nu)}z_{t - 1,f}^{(\nu)} + a_{t,f}^{(\nu)},$$

$$\boxed{\Phi_{t,f}^{(\nu)} = \frac{S_{t,f}^{(\nu)}}{z_{t,f}^{(\nu)} + \epsilon}}.$$

Here both

$$\lambda_{t,f}^{(\nu)} \in \lbrack 0,1)$$

and

$$a_{t,f}^{(\nu)} \geq 0$$

may be learned.

Unrolling the numerator gives

$$S_{t,f}^{(\nu)} = \left( \prod_{j = 1}^{t}\lambda_{j,f}^{(\nu)} \right)S_{0,f}^{(\nu)} + \sum_{\tau = 1}^{t}\left\lbrack a_{\tau,f}^{(\nu)}\prod_{j = \tau + 1}^{t}\lambda_{j,f}^{(\nu)} \right\rbrack\mathbf{y}_{\tau,f}\mathbf{y}_{\tau,f}^{H}.$$

Define

$$c_{t,\tau,f}^{(\nu)} = a_{\tau,f}^{(\nu)}\prod_{j = \tau + 1}^{t}\lambda_{j,f}^{(\nu)}.$$

Then

$$S_{t,f}^{(\nu)} = \sum_{\tau \leq t}^{}c_{t,\tau,f}^{(\nu)}\mathbf{y}_{\tau,f}\mathbf{y}_{\tau,f}^{H} + \text{initial term}.$$

So the system is learning an **adaptive causal weighting kernel over instantaneous spatial covariance observations**.

This is closely related to the generalized learned SCM-weighting view developed in prior moving-speaker beamforming: Ochiai et al. express time-varying SCM computation as a weighted sum of instantaneous SCMs and learn those weights with self-attention. [\[9\]](https://arxiv.org/pdf/2205.03568) Therefore *that alone is not your novelty*.

### Why mLSTM is structurally interesting

The xLSTM paper introduced mLSTM as an LSTM extension with matrix memory and a covariance update rule. [\[13\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) In the speech-enhancement implementation, the matrix memory follows the form

$$C_{t} = f_{t}C_{t - 1} + i_{t}v_{t}k_{t}^{\top},$$

with query/key/value projections and gated memory access. [\[14\]](https://arxiv.org/html/2501.06146v2)

Ignoring normalization terms for the structural argument,

$$C_{t} = \left( \prod_{j = 1}^{t}f_{j} \right)C_{0} + \sum_{\tau = 1}^{t}\left\lbrack i_{\tau}\prod_{j = \tau + 1}^{t}f_{j} \right\rbrack v_{\tau}k_{\tau}^{\top}.$$

Compare this directly with adaptive SCM memory:

$$S_{t,f} = \sum_{\tau = 1}^{t}\left\lbrack a_{\tau,f}\prod_{j = \tau + 1}^{t}\lambda_{j,f} \right\rbrack\mathbf{y}_{\tau,f}\mathbf{y}_{\tau,f}^{H}.$$

The correspondence is:

| mLSTM                | Adaptive spatial memory               |
|----------------------|---------------------------------------|
| $C_{t}$              | accumulated spatial statistics        |
| $f_{t}$              | forgetting factor $\lambda_{t}$       |
| $i_{t}$              | update strength $a_{t}$               |
| $v_{t}k_{t}^{\top}$  | instantaneous outer-product statistic |
| gated memory horizon | adaptive SCM integration horizon      |

This is the paper's central theoretical observation.

### The critical caveat that actually strengthens the paper

Do **not** claim

$$C_{t} = \Phi_{t}.$$

That is generally false.

A physical complex SCM has

$$\Phi = \Phi^{H}$$

and

$$x^{H}\Phi x \geq 0\quad\quad\forall x,$$

so it is Hermitian PSD.

Standard real-valued mLSTM uses general

$$v_{t}k_{t}^{\top},$$

which is not necessarily symmetric, let alone complex Hermitian PSD. Its native exponential gating should also not simply be interpreted as a physically valid forgetting coefficient.

That gives a principled reason for the proposed **Covariance-Aligned mLSTM**:

$$h_{t,f}^{mLSTM} \rightarrow \left\{ \begin{matrix}
\lambda_{t,f} = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min})\sigma(g_{\lambda}(h)) \\
a_{t,f} = softplus(g_{a}(h))
\end{matrix} \right.\ $$

and then update the **explicit physical memory**

$$S_{t} = \lambda_{t}S_{t - 1} + a_{t}yy^{H}.$$

This separation between:

$$\boxed{\text{learned contextual memory}}$$

and

$$\boxed{\text{physical covariance memory}}$$

is both safer numerically and much easier to defend theoretically.

### PSD-preservation proposition

Assume:

$$S_{0,f} \succcurlyeq 0,\quad\quad\lambda_{t,f} \geq 0,\quad\quad a_{t,f} \geq 0.$$

Since

$$\mathbf{y}\mathbf{y}^{H} \succcurlyeq 0,$$

then recursively,

$$S_{t,f} = \lambda_{t,f}S_{t - 1,f} + a_{t,f}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H} \succcurlyeq 0.$$

Because $z_{t,f} > 0$,

$$\Phi_{t,f} = \frac{S_{t,f}}{z_{t,f} + \epsilon} \succcurlyeq 0.$$

Thus the mLSTM may freely learn temporal adaptation while the SCM remains physically valid by construction.

That small proposition is worth putting in the Method section. It turns what could look like “xLSTM plugged into MVDR” into a constrained architecture motivated by matrix structure.

### Testable expected behavior

This theory generates predictions before experiments:

$$\text{stationary source} \Rightarrow \lambda_{t,f} \uparrow$$

because long averaging reduces noisy covariance estimates.

$$\text{rapidly moving source} \Rightarrow \lambda_{t,f} \downarrow$$

because stale spatial statistics should be forgotten faster.

$$\text{abrupt direction change} \Rightarrow \lambda_{t,f} \downarrow \text{ near the transition}.$$

$$\text{stable post-transition scene} \Rightarrow \lambda_{t,f} \uparrow \text{ again}.$$

The moving-speaker beamforming literature specifically establishes the conflict between reliable long-term averaging and rapid motion tracking, making these hypotheses well grounded. [\[9\]](https://arxiv.org/pdf/2205.03568)

A paper becomes much stronger if you show a figure where source angle, learned $\lambda_{t}$, covariance innovation, uncertainty, and framewise SI-SDR are aligned over time.

### Why this is preferable to attention

Self-attention SCM aggregation already learns temporal weights and has demonstrated improved tracking of moving speakers; subsequent work has examined array-geometry robustness. [\[1\]](https://arxiv.org/pdf/2205.03568)

The xLSTM argument should therefore be:

> Attention learns arbitrary pairwise temporal weighting. CM-xLSTM instead parameterizes **recursive stateful covariance memory**, making the update directly interpretable as adaptive forgetting and naturally appropriate for streaming inference.

Do not claim the mLSTM solution is automatically faster until measured. xLSTM-SENet shows xLSTM is viable for speech enhancement, but also reports that a well-configured ordinary LSTM can be highly competitive, making a parameter-matched LSTM control essential. [\[14\]](https://arxiv.org/html/2501.06146v2)

## Architecture and implementation specification

### Complete proposed pipeline

|                                                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------|
| <img src="data:image/png;base64,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" style="width:5.83333in;height:1.11625in" alt="Rendered Mermaid diagram 1" /> |

The architecture intentionally keeps the major roles separated:

$$\text{mLSTM} = \text{memory/control},$$

$$\text{SCM recurrence} = \text{valid spatial statistics},$$

$$\text{MVDR} = \text{physics},$$

$$\text{neural residual} = \text{controlled model mismatch correction},$$

$$\text{TF-xLSTM} = \text{post-beamforming spectral repair}.$$

That decomposition makes ablation and interpretation straightforward.

### Recommended base configuration

Use a **16 kHz** main system. A practical starting frontend is:

$$N_{FFT} = 512,\quad\quad F = 257,$$

$$\text{window} = 512\quad(32\text{ ms}),$$

$$\text{hop} = 128\quad(8\text{ ms}).$$

For a causal model:

- `center=False`;
- left-only temporal padding;
- unidirectional temporal mLSTM;
- no future SCM frames;
- no temporal bidirectionality in the refiner.

Frequency-axis bidirectionality is compatible with streaming because all frequency bins from the current STFT frame are already available.

### Spatial feature construction

For each TF bin,

$$\mathbf{y}_{t,f} \in \mathbb{C}^{M}.$$

First compute a normalized instantaneous spatial outer product:

$$G_{t,f} = \frac{\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H}}{\parallel \mathbf{y}_{t,f} \parallel_{2}^{2} + \epsilon}.$$

For $M$ microphones, an $M \times M$ Hermitian matrix has $M^{2}$ real degrees of freedom. Rather than feeding the redundant full complex matrix to a real-valued network, pack:

$$Re\{ G_{mm}\},$$

and, for $m < n$,

$$Re\{ G_{mn}\},\quad\quad Im\{ G_{mn}\}.$$

For $M = 6$, this is

$$6 + 2\binom{6}{2} = 36$$

real spatial features per TF bin.

Append:

- reference-channel log power;
- normalized multichannel energy;
- optionally $\cos(IPD)$ and $\sin(IPD)$ relative to the reference microphone.

Then use a small depthwise-separable convolutional spatial encoder:

$$X_{sp} \in \mathbb{R}^{B \times T \times F \times C_{in}} \rightarrow H \in \mathbb{R}^{B \times T \times F^{\prime} \times D}.$$

Recommended first implementation:

$$D = 96,\quad\quad F^{\prime} = \lceil F/2\rceil = 129.$$

Frequency downsampling roughly halves the recurrent token count while retaining frequency-specific spatial memory.

### Source assignment and target-query interface

Let

$$K = 2$$

for the main two-speaker experiments.

The mask head returns

$$M \in \lbrack 0,1\rbrack^{B \times K \times T \times F}.$$

For separation, use $K$ learnable source-slot embeddings.

For directional target extraction,

$$q_{t} = W_{q}E_{DoA}(\theta_{t}).$$

Use query-conditioned mask logits and query-conditioned spatial-memory tokens:

$$H_{k,t,f} = H_{t,f} + W_{slot}q_{k,t}.$$

This keeps separation and TSE in a common code path rather than maintaining two model families.

### Dual spatial mLSTM memories

For each source $k$, construct target and interference tokens:

$$H_{k}^{(s)} = Cond(H,M_{k},q_{k}),$$

$$H_{k}^{(i)} = Cond(H,1 - M_{k},q_{k}).$$

Reshape:

$$B \times T \times F^{\prime} \times D \rightarrow (BF^{\prime}) \times T \times D$$

and apply temporal mLSTM.

Recommended:

$$2\text{ mLSTM blocks},\quad D = 96\text{ or }128.$$

I would **share mLSTM weights between speech and interference streams but maintain separate recurrent states**, while adding a learned stream-type embedding. This gives two physical memories without nearly doubling model parameters.

The official xLSTM repository is actively maintained and supplies the native implementation and optimized kernels; use it rather than reimplementing mLSTM from a blog or unofficial port. [\[15\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com) xLSTM-SENet also provides a speech-domain implementation reference. [\[16\]](https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com)

Outputs:

$$H_{k}^{(s)},H_{k}^{(i)} \in \mathbb{R}^{B \times T \times F^{\prime} \times D}.$$

Heads predict

$$\lambda_{k,t,f}^{(s)},\quad\lambda_{k,t,f}^{(i)},$$

and

$$a_{k,t,f}^{(s)},\quad a_{k,t,f}^{(i)}.$$

Interpolate only these scalar control quantities from $F^{\prime}$ back to $F$; do not interpolate complex SCM matrices.

### Explicit target and interference memories

For target source $k$,

$$S_{k,t,f}^{(s)} = \lambda_{k,t,f}^{(s)}S_{k,t - 1,f}^{(s)} + a_{k,t,f}^{(s)}M_{k,t,f}\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H}.$$

For interference,

$$S_{k,t,f}^{(i)} = \lambda_{k,t,f}^{(i)}S_{k,t - 1,f}^{(i)} + a_{k,t,f}^{(i)}(1 - M_{k,t,f})\mathbf{y}_{t,f}\mathbf{y}_{t,f}^{H}.$$

Maintain corresponding normalizers $z^{(s)},z^{(i)}$.

Then

$$\Phi^{(s)} = \frac{S^{(s)}}{z^{(s)} + \epsilon},\quad\quad\Phi^{(i)} = \frac{S^{(i)}}{z^{(i)} + \epsilon}.$$

Symmetrize numerically:

$$\Phi \leftarrow \frac{1}{2}(\Phi + \Phi^{H}).$$

### Steering / relative transfer function

Rather than an expensive eigendecomposition at every TF point, use a reference-vector estimate:

$$u = \lbrack 1,0,\ldots,0\rbrack^{\top}.$$

Under an approximately rank-one target SCM, estimate

$$d = \frac{\Phi^{(s)}u}{u^{H}\Phi^{(s)}u + \epsilon}.$$

Then

$$u^{H}d \approx 1.$$

The general mask-based MVDR framework relies on target and interference SCMs, and moving-source work computes time-varying beamformers from exactly such SCM estimates. [\[9\]](https://arxiv.org/pdf/2205.03568)

### Physics-guided MVDR

Apply diagonal loading:

$$R_{i} = \Phi^{(i)} + \delta I,$$

with

$$\delta = \epsilon_{load}\frac{tr\Phi^{(i)}}{M} + \epsilon_{floor}.$$

Never explicitly compute

$$R_{i}^{- 1}.$$

Instead solve

$$R_{i}v = d$$

with Cholesky or `torch.linalg.solve`, then

$$w_{MVDR} = \frac{v}{d^{H}v + \epsilon}.$$

By construction,

$$w_{MVDR}^{H}d \approx 1.$$

### Distortionless-preserving neural residual

Let a small MLP predict

$$\Delta w \in \mathbb{C}^{M}.$$

Predict real and imaginary components with a real network:

$$\lbrack\Delta w_{\Re},\Delta w_{\Im}\rbrack \in \mathbb{R}^{2M}.$$

Project into the subspace orthogonal to $d$:

$$P_{\perp}(d) = I - \frac{dd^{H}}{d^{H}d + \epsilon},$$

$$r = P_{\perp}(d)\Delta w.$$

Then

$$r^{H}d = 0.$$

Use

$$\boxed{w_{final} = w_{MVDR} + \alpha r}.$$

Therefore,

$$w_{final}^{H}d = 1$$

whenever the original MVDR weight satisfies its distortionless constraint.

This is, in my view, the **second most publishable idea in the pipeline after covariance-aligned mLSTM**.

### Uncertainty gating

Construct physical reliability features:

$$u_{t,f} = \lbrack u_{cond},u_{rank},u_{mask},u_{innovation},h_{mLSTM}\rbrack.$$

Useful components are:

$$u_{rank} = 1 - \frac{\lambda_{1}(\Phi_{s})}{tr(\Phi_{s}) + \epsilon},$$

$$u_{mask} = - \sum_{k}^{}p_{k}\log(p_{k} + \epsilon),$$

$$u_{innovation} = \frac{\parallel \Psi_{t} - \Phi_{t - 1} \parallel_{F}}{\parallel \Phi_{t - 1} \parallel_{F} + \epsilon},$$

plus a clipped log condition estimate for the loaded interference SCM.

Predict

$$\alpha = \alpha_{\max}\sigma(g_{u}(u_{t,f})).$$

During synthetic training, calibrate uncertainty against an oracle SCM error:

$$u^{*} = \log\left( \left. \parallel{\widetilde{\Phi}}_{est} - {\widetilde{\Phi}}_{oracle} \right.\parallel_{F} + \epsilon \right).$$

The model then learns not merely “how much neural beamforming to use,” but how to associate physical covariance unreliability with correction strength.

### TF-xLSTM refiner

After beamforming,

$$Z_{k,t,f} = w_{k,t,f}^{H}\mathbf{y}_{t,f}.$$

Feed

$$\lbrack\Re Z,\Im Z,\log(|Z| + \epsilon)\rbrack$$

into a small 2–4 block TF-xLSTM module predicting a complex residual or bounded complex ratio mask.

xLSTM-SENet establishes that time-frequency xLSTM modeling is practical for spectral speech enhancement, although its task is single-channel rather than multichannel beamforming. [\[14\]](https://arxiv.org/html/2501.06146v2)

Crucially, **always report performance before and after this refiner**. Otherwise a reviewer can reasonably argue that the downstream enhancement network—not the covariance-memory beamformer—is responsible for the gains.

### Exact interfaces

| Module                 | Input                         | Output                            |
|------------------------|-------------------------------|-----------------------------------|
| STFT                   | `[B,M,N]` real                | `[B,T,F,M]` complex               |
| Spatial feature packer | `[B,T,F,M]` complex           | `[B,T,F,Csp]` real                |
| Spatial encoder        | `[B,T,F,Csp]`                 | `[B,T,F',D]`                      |
| Query encoder          | DoA `[B,K,T]` or slots        | `[B,K,T,D]`                       |
| Mask head              | spatial embedding + query     | `[B,K,T,F]`                       |
| Target mLSTM           | `[B*K*F',T,D]`                | same                              |
| Interference mLSTM     | `[B*K*F',T,D]`                | same                              |
| Memory-control heads   | mLSTM outputs                 | $\lambda,a$: `[B,K,T,F]`          |
| SCM accumulator        | $\lambda,a,M,Y$               | `[B,K,T,F,M,M]` complex logically |
| RTF estimator          | target SCM                    | `[B,K,T,F,M]` complex             |
| MVDR                   | target/interference SCM + RTF | `[B,K,T,F,M]`                     |
| Residual head          | memories + physical features  | `Δw [B,K,T,F,M]` complex          |
| Uncertainty head       | physical/memory features      | `α [B,K,T,F,1]`                   |
| Beamform               | weights + mixture             | `[B,K,T,F]` complex               |
| TF-xLSTM refiner       | beamformed spectrum           | `[B,K,T,F]` complex               |
| iSTFT                  | complex spectrum              | `[B,K,N]`                         |

Do not actually materialize every

$$\lbrack B,K,T,F,M,M\rbrack$$

SCM tensor for long utterances. In streaming or memory-constrained training, keep only the current

$$\lbrack B,K,F,M,M\rbrack$$

state and optionally checkpoint selected frames.

### Forward-pass pseudocode

    def forward(
        wav: Tensor,                 # [B, M, N]
        doa: Tensor | None = None,   # [B, K, T] for TSE; None for blind SS
    ) -> Tensor:
        # ----- Front end -----
        Y = stft(wav)  # complex [B, T, F, M]

        spatial_feat = build_spatial_features(Y)       # [B,T,F,Csp]
        H = spatial_encoder(spatial_feat)              # [B,T,F',D]

        query = query_encoder(doa, num_sources=K)      # [B,K,T,D]
        masks = mask_head(H, query, out_freq=F)        # [B,K,T,F]

        outputs = []

        for k in range(K):
            target_tokens = condition_tokens(H, masks[:, k], query[:, k])
            interf_tokens = condition_tokens(H, 1.0 - masks[:, k], query[:, k])

            # [B,F',T,D] -> [B*F',T,D]
            hs = target_mLSTM(rearrange_for_time(target_tokens))
            hi = interf_mLSTM(rearrange_for_time(interf_tokens))

            lambda_s, update_s = target_memory_head(hs)
            lambda_i, update_i = interf_memory_head(hi)

            # Scalar gates returned to full F resolution.
            lambda_s, update_s = restore_frequency(lambda_s, update_s, F)
            lambda_i, update_i = restore_frequency(lambda_i, update_i, F)

            scm_state_s = init_psd_state(B, F, M)
            scm_state_i = init_psd_state(B, F, M)

            Z_frames = []

            for t in range(T):
                yyH = outer(Y[:, t], Y[:, t].conj())   # [B,F,M,M]

                scm_state_s = update_psd_scm(
                    scm_state_s,
                    forgetting=lambda_s[:, t],
                    weight=update_s[:, t] * masks[:, k, t],
                    outer_product=yyH,
                )
                scm_state_i = update_psd_scm(
                    scm_state_i,
                    forgetting=lambda_i[:, t],
                    weight=update_i[:, t] * (1.0 - masks[:, k, t]),
                    outer_product=yyH,
                )

                Phi_s = scm_state_s.normalized()
                Phi_i = scm_state_i.normalized()

                Phi_s = hermitian_symmetrize(Phi_s)
                Phi_i = diagonal_load(hermitian_symmetrize(Phi_i))

                d = reference_rtf(Phi_s)                # [B,F,M]

                w_mvdr = mvdr_solve(Phi_i, d)           # no explicit inverse

                delta_w = residual_head(
                    hs_at(hs, t), hi_at(hi, t),
                    physical_features(Phi_s, Phi_i, masks[:, k, t]),
                )

                residual = project_orthogonal(delta_w, d)

                alpha = uncertainty_gate(
                    hs_at(hs, t), hi_at(hi, t),
                    physical_features(Phi_s, Phi_i, masks[:, k, t]),
                )

                w = w_mvdr + alpha * residual

                # Important unit test: |wᴴd - 1| must remain tiny.
                Z_t = einsum_conj_weights(w, Y[:, t])   # [B,F]
                Z_frames.append(Z_t)

            Z = stack(Z_frames, dim=1)                  # [B,T,F]
            Z_refined = tf_xlstm_refiner(Z)            # complex [B,T,F]
            outputs.append(istft(Z_refined))

        return stack(outputs, dim=1)                    # [B,K,N]

For speed, the production implementation should fuse or scan the SCM recurrence rather than leave a Python `for` loop over $T$, but the explicit loop is the correct reference implementation and is much easier to test.

## Training data, objectives, hyperparameters, and compute

### Training curriculum

I recommend a staged schedule because directly training masks, adaptive SCMs, matrix memory, a differentiable solver, neural residuals, uncertainty, and a postfilter from random initialization creates too many coupled failure modes.

**Stage A — spatial estimation warm-up**

Train:

$$\text{encoder} + \text{mask head} + \text{mLSTM memory controller}$$

using mask/source-image and oracle-SCM auxiliary losses.

Do not enable neural residual correction yet.

**Stage B — differentiable MVDR**

Enable:

$$\text{adaptive SCM} + \text{MVDR}$$

and optimize end-to-end SI-SDR.

**Stage C — constrained neural correction**

Enable:

$$\Delta w,\quad P_{\perp},\quad\alpha.$$

Initialize the residual head close to zero so that training begins near pure MVDR.

**Stage D — TF-xLSTM refiner**

Add the post-beamforming refiner last. This prevents it from learning to compensate for a poorly trained spatial frontend.

### Proposed losses

Use

$$\mathcal{L} = \mathcal{L}_{SI - SDR} + \beta_{c}\mathcal{L}_{complex} + \beta_{\Phi}\mathcal{L}_{SCM} + \beta_{u}\mathcal{L}_{unc} + \beta_{r}\mathcal{L}_{res} + \beta_{m}\mathcal{L}_{mask}.$$

A reasonable **initial**, not sacred, weighting is

$$\beta_{c} = 0.2,\quad\beta_{\Phi} = 0.05,\quad\beta_{u} = 0.02,\quad\beta_{r} = 10^{- 3},\quad\beta_{m} = 0.05.$$

The primary waveform term:

$$\mathcal{L}_{SI - SDR} = - \frac{1}{K}\sum_{k}^{}{SI - SDR}({\widehat{s}}_{k},s_{\pi(k)}),$$

with PIT permutation $\pi$ in blind separation.

The complex STFT auxiliary loss:

$$\mathcal{L}_{complex} = \parallel \Re\widehat{S} - \Re S \parallel_{1} + \parallel \Im\widehat{S} - \Im S \parallel_{1} + 0.5 \parallel |\widehat{S}| - |S| \parallel_{1}.$$

Normalized SCM supervision:

$$\widetilde{\Phi} = \frac{\Phi}{tr(\Phi) + \epsilon},$$

$$\mathcal{L}_{SCM} = \parallel {\widetilde{\Phi}}_{est} - {\widetilde{\Phi}}_{oracle} \parallel_{F}.$$

Uncertainty calibration:

$$\mathcal{L}_{unc} = Huber(\widehat{u},u^{*}).$$

Residual regularization:

$$\mathcal{L}_{res} = \frac{\parallel \alpha r \parallel_{2}^{2}}{\parallel w_{MVDR} \parallel_{2}^{2} + \epsilon}.$$

This discourages the neural branch from needlessly overpowering the physical beamformer.

### Recommended data program

Use three levels.

| Data                  | Role                          | Why                                                                   |
|-----------------------|-------------------------------|-----------------------------------------------------------------------|
| SMS-WSJ               | Main controlled benchmark     | Native multichannel separation benchmark with reproducible generation |
| Dynamic-SMS           | Core novelty benchmark        | Directly tests adaptive spatial memory                                |
| Spatialized Libri2Mix | Cross-domain test             | Different speech corpus/rooms/array distribution                      |
| LibriCSS              | Optional recorded stress test | Continuous real-room multichannel mixtures                            |

SMS-WSJ was designed specifically for multichannel source separation and provides generation/baseline infrastructure. [\[7\]](https://arxiv.org/abs/1910.13934?utm_source=chatgpt.com) LibriMix provides two- and three-speaker material for generalization experiments. [\[11\]](https://arxiv.org/html/2005.11262v1?utm_source=chatgpt.com) LibriCSS provides recorded continuous overlap conditions. [\[12\]](https://arxiv.org/pdf/2001.11482?utm_source=chatgpt.com)

### Dynamic-SMS generation

The moving dataset is not optional if the main paper claims adaptive covariance memory.

Generate:

**Stationary subset**

$$\omega = 0^{\circ}/s.$$

**Continuous motion**

$$\omega \in \{ 5,15,30\}^{\circ}/s.$$

**Abrupt switch**

At the midpoint:

$$\Delta\theta \in \{ 30^{\circ},60^{\circ},90^{\circ}\}.$$

**Crossing trajectories**

Two speakers cross angularly with minimum separation around:

$$10^{\circ},\ 20^{\circ},\ 40^{\circ}.$$

**Dual movement**

Both sources move.

**Target silence**

Include pauses surrounding a movement event; this tests whether memory persists sensibly when the target provides little instantaneous evidence.

Use continuously varying or piecewise-interpolated RIRs generated from spatial trajectories. gpuRIR is a practical open-source starting point for large-scale image-source RIR generation. [\[10\]](https://arxiv.org/abs/1810.11359?utm_source=chatgpt.com) Prior moving-speaker neural-beamforming work likewise evaluates explicitly simulated motion trajectories because time-varying acoustic transfer functions are central to the task. [\[9\]](https://arxiv.org/pdf/2205.03568)

### Spatial robustness augmentation

Recommended training randomization:

- random room dimensions;
- $T_{60}$ spanning roughly $0.1$–$0.8$ s;
- SNR roughly $- 5$–20 dB;
- random source distance;
- angular separation stratification;
- microphone gain mismatch;
- small channel phase/calibration errors;
- channel dropout;
- microphone-position jitter;
- random reference-microphone selection in a dedicated robustness variant.

These are proposed engineering ranges, not claims about the standard SMS-WSJ distribution.

Do **not** make arbitrary-array invariance part of the first main contribution unless resources are abundant. Array-robust learned SCM aggregation is already a dedicated research problem in its own right. [\[17\]](https://arxiv.org/html/2402.03058v2) Keep the primary paper on a fixed nominal array and include geometry mismatch as a robustness study.

### Optimizer and schedule

Recommended starting recipe:

$$\text{AdamW},\quad\eta_{\max} = 3 \times 10^{- 4},$$

$$\text{weight decay} = 10^{- 2},$$

$$\text{gradient clipping} = 5.$$

Use:

- linear warm-up: roughly 10k steps;
- cosine decay;
- BF16 for neural layers;
- FP32/complex64 for SCM construction, Cholesky/solve and eigendiagnostics;
- 4 s training crops;
- effective batch size around 16 mixtures;
- activation checkpointing if required.

The xLSTM speech-enhancement paper trained 2 s, 16 kHz examples with multiple L40S GPUs and explicitly evaluated five random seeds, showing that xLSTM speech models are practical but that proper seed replication matters. [\[14\]](https://arxiv.org/html/2501.06146v2) Treat that only as a useful engineering reference; CM-xBeam's multichannel solver and SCM states make its cost materially different.

### Compute planning

These are **planning budgets**, not measured benchmarks.

| Experiment tier          | Configuration                       | Estimated planning budget |
|--------------------------|-------------------------------------|---------------------------|
| Smoke test               | 1k–5k mixtures, short model         | 10–30 GPU-h               |
| Main prototype           | CM-xBeam without TF refiner         | 80–150 GPU-h/seed         |
| Full model               | \+ residual + uncertainty + refiner | 150–250 GPU-h/seed        |
| Five-seed final          | Full model                          | 750–1,250 GPU-h           |
| Short ablation           | 50–70% schedule                     | 30–80 GPU-h/variant       |
| Dynamic robustness suite | Inference only after training       | comparatively small       |

A sensible parameter target is:

| Component                          | Design target |
|------------------------------------|---------------|
| Spatial encoder + masks            | 1–2 M         |
| Shared dual-state mLSTM controller | 1.5–3 M       |
| Residual + uncertainty heads       | \<1 M         |
| TF-xLSTM refiner                   | 2–4 M         |
| **Full system**                    | **~6–10 M**   |

Do **not** put those parameter values in a paper until `count_parameters.py` produces the actual numbers.

The design objective should be:

$$\boxed{\text{full model} < 10\text{ M parameters}}$$

unless performance clearly justifies a larger model.

## Ablations, evaluation protocol, and statistical proof

This is the section that can turn the idea from “interesting” into “hard to reject.”

### Mandatory baselines

Use both classical/physics-guided and strong learned baselines.

The official SpatialNet repository provides implementations for SpatialNet, online SpatialNet and related narrow-band architectures for multichannel separation/enhancement. [\[18\]](https://github.com/Audio-WestlakeU/NBSS?utm_source=chatgpt.com) ADL-MVDR has public code for an all-deep-learning MVDR approach to multichannel target separation. [\[19\]](https://github.com/zzhang68/adlmvdr?utm_source=chatgpt.com) The original moving-source SCM-attention beamformer is especially important conceptually because it is the closest prior art to learned temporal spatial-statistics aggregation. [\[9\]](https://arxiv.org/pdf/2205.03568)

At minimum compare against:

- fixed-EMA mask-MVDR;
- learned-attention SCM/MVDR if reproducibly available;
- LSTM-controlled SCM;
- parameter-matched LSTM controller;
- SpatialNet/online SpatialNet;
- a strong TF-domain neural separation baseline;
- direct-neural-beamformer variant of your own architecture.

### Main ablation matrix

| ID  | Memory                 | SCM update                | Beamformer       | Residual      | Uncertainty    | Refiner  | Purpose                      |
|-----|------------------------|---------------------------|------------------|---------------|----------------|----------|------------------------------|
| A0  | None                   | Fixed EMA                 | MVDR             | –             | –              | –        | Classical baseline           |
| A1  | MLP                    | Learned $\lambda$         | MVDR             | –             | –              | –        | Is recurrence necessary?     |
| A2  | LSTM vector            | Learned $\lambda$         | MVDR             | –             | –              | –        | Standard recurrent memory    |
| A3  | mLSTM matrix           | Learned $\lambda$         | MVDR             | –             | –              | –        | Core covariance-memory claim |
| A4  | mLSTM                  | Learned                   | Direct neural BF | –             | –              | –        | Is physical MVDR useful?     |
| A5  | mLSTM                  | Learned                   | MVDR             | unconstrained | fixed $\alpha$ | –        | Generic neural correction    |
| A6  | mLSTM                  | Learned                   | MVDR             | nullspace     | fixed $\alpha$ | –        | Constraint contribution      |
| A7  | mLSTM                  | Learned                   | MVDR             | nullspace     | learned        | –        | Uncertainty contribution     |
| A8  | mLSTM                  | Learned                   | MVDR             | nullspace     | learned        | TF-xLSTM | Full model                   |
| A9  | raw $C_{t}$            | mLSTM state as pseudo-SCM | MVDR             | –             | –              | –        | Prove PSD alignment matters  |
| A10 | parameter-matched LSTM | Learned                   | MVDR             | nullspace     | learned        | refiner  | Backbone control             |

**A9 is especially valuable.** If directly treating mLSTM memory as the covariance matrix is unstable or inferior while the constrained explicit SCM works, you have experimentally validated the theoretical caveat—not merely stated it.

### The essential factorial experiment

The most important $2 \times 2$ experiment should be:

|                                    | Fixed forgetting | Learned forgetting |
|------------------------------------|------------------|--------------------|
| **LSTM/vector contextual memory**  | B00              | B01                |
| **mLSTM/matrix contextual memory** | B10              | B11                |

Parameter-match B01 and B11 as closely as possible.

This separates:

$$\text{effect of learned forgetting}$$

from

$$\text{effect of matrix memory}.$$

The key statistical question becomes whether there is an interaction:

$$\Delta_{interaction} = (B_{11} - B_{10}) - (B_{01} - B_{00}).$$

If the mLSTM advantage becomes largest specifically under learned dynamic covariance updating—especially for moving speakers—that is far more convincing than “xLSTM score \> LSTM score.”

This control is mandatory because xLSTM-SENet itself found that carefully configured conventional LSTMs can be highly competitive in speech enhancement. [\[14\]](https://arxiv.org/html/2501.06146v2)

### Moving-source experiments that prove the theory

Report SI-SDRi as a function of angular velocity:

$$SI - SDRi(\omega),\quad\quad\omega \in \{ 0,5,15,30\}^{\circ}/s.$$

Report it as a function of jump size:

$$SI - SDRi(\Delta\theta).$$

Report it as a function of angular separation:

$$\Delta\theta_{speaker} \in \{ 10^{\circ},20^{\circ},40^{\circ},80^{\circ}\}.$$

Add crossing-speaker and both-moving conditions.

Prior neural beamforming work already establishes moving-source SCM tracking as a meaningful challenge and specifically explains why fixed temporal aggregation can become suboptimal. [\[9\]](https://arxiv.org/pdf/2205.03568) Your paper needs to show that the mLSTM memory controller is better precisely where its theory predicts.

### DoA-error robustness for target extraction

Use both Gaussian and systematic error:

$${\widetilde{\theta}}_{t} = \theta_{t} + \epsilon_{t}.$$

Evaluate:

$$|\epsilon| \in \{ 0^{\circ},5^{\circ},10^{\circ},20^{\circ},30^{\circ}\}.$$

Also test:

**Latency error**

$${\widetilde{\theta}}_{t} = \theta_{t - \tau},$$

for several tracking delays.

**Bias**

$${\widetilde{\theta}}_{t} = \theta_{t} + b.$$

**Jitter**

frame-to-frame noisy DoA.

**Dropout**

temporarily remove the cue and require the spatial memory to bridge the missing interval.

This is substantially more informative than just reporting clean oracle-DoA target extraction.

### Interpretable diagnostics

The paper should contain plots of:

$${\overline{\lambda}}_{t} = \frac{1}{F}\sum_{f}^{}\lambda_{t,f},$$

$${\overline{\alpha}}_{t} = \frac{1}{F}\sum_{f}^{}\alpha_{t,f},$$

covariance error,

$$E_{\Phi}(t) = \parallel {\widetilde{\Phi}}_{t} - {\widetilde{\Phi}}_{t}^{*} \parallel_{F},$$

and source angle.

The strongest possible qualitative result is:

    source stable      → λ high → long spatial memory
    speaker starts moving → λ decreases
    abrupt direction jump → λ sharply decreases
    new direction stabilizes → λ increases again

    SCM unreliable → uncertainty rises → α increases
    SCM reliable   → α decreases → model stays close to MVDR

If that behavior emerges naturally, it gives the paper unusually strong interpretability.

### Parameter-matched controls

Every backbone comparison should match one of:

$$\pm 5\%$$

parameters, or preferably both parameters and approximate MAC/s.

If the mLSTM model has 8 M parameters and the LSTM baseline has 4 M, a reviewer cannot tell whether the gain comes from matrix memory.

Report:

| Variant       | Params | MAC/s | State memory | SI-SDRi |
|---------------|--------|-------|--------------|---------|
| LSTM matched  |        |       |              |         |
| mLSTM         |        |       |              |         |
| Attention SCM |        |       |              |         |
| CM-xBeam      |        |       |              |         |

### Seed policy

Use:

$$\boxed{5\text{ independent seeds}}$$

for the headline models:

- fixed-EMA MVDR;
- parameter-matched LSTM;
- core mLSTM-MVDR;
- full CM-xBeam.

xLSTM-SENet itself reports five-seed evaluation, so using multiple seeds is particularly appropriate when making a claim that xLSTM/mLSTM itself matters. [\[14\]](https://arxiv.org/html/2501.06146v2)

For expensive secondary ablations:

- one seed during development;
- three seeds for promising variants;
- five seeds for final headline comparisons.

Do not cherry-pick the best seed.

### Statistical protocol

Predefine three primary comparisons:

$$H_{1}:\text{CM-xBeam} > \text{fixed-EMA MVDR},$$

$$H_{2}:\text{mLSTM controller} > \text{parameter-matched LSTM controller},$$

$$H_{3}:\text{nullspace+uncertainty residual} > \text{unconstrained residual}.$$

For each model pair, outputs exist on the **same test mixtures**, so use paired inference.

Recommended reporting:

$$\Delta_{SI - SDRi} = \frac{1}{N}\sum_{n = 1}^{N}\left( x_{n}^{(A)} - x_{n}^{(B)} \right).$$

Report:

- mean difference;
- 95% confidence interval;
- seed mean ± seed standard deviation;
- paired utterance-level significance.

Use a **hierarchical bootstrap** that resamples seeds and then utterances within seeds, or report both seed-level variation and a paired 10,000-resample utterance bootstrap.

For confirmatory testing, use a paired permutation/sign-flip test on utterance-level differences. Correct the three primary comparisons with Holm-Bonferroni.

Do not write merely:

> $p < 0.05$.

Write:

> $+ \Delta$ dB SI-SDRi, 95% CI $\lbrack a,b\rbrack$, corrected $p = \cdots$.

The confidence interval is more informative than the p-value alone.

### Evaluation split discipline

Freeze:

- data-generation seeds;
- room partitions;
- speaker partitions;
- trajectory-generation seeds;
- DoA-error conditions;

before final training.

For PIT systems, select the permutation using SI-SDR and use that same permutation for all other source-wise metrics.

For runtime, run at least:

- batch $= 1$;
- fixed 10 s utterances;
- warm-up iterations;
- at least 100 timed repetitions;
- CPU and one named GPU;
- FP32 and deployment precision if different.

Report median and high-percentile latency, not only average throughput.

## Reproducible implementation and failure-resistant engineering

### Suggested repository structure

    cm_xbeam/
    ├── README.md
    ├── LICENSE
    ├── environment.yml
    ├── pyproject.toml
    ├── configs/
    │   ├── base.yaml
    │   ├── sms_wsj.yaml
    │   ├── dynamic_sms.yaml
    │   ├── ablations/
    │   └── evaluation/
    │
    ├── data/
    │   ├── sms_wsj.py
    │   ├── dynamic_sms.py
    │   ├── spatial_librimix.py
    │   ├── libricss.py
    │   ├── rir_generation.py
    │   └── augmentation.py
    │
    ├── models/
    │   ├── stft.py
    │   ├── spatial_features.py
    │   ├── spatial_encoder.py
    │   ├── query_encoder.py
    │   ├── mask_head.py
    │   ├── covariance_xlstm.py
    │   ├── scm_accumulator.py
    │   ├── steering.py
    │   ├── mvdr.py
    │   ├── residual_beamformer.py
    │   ├── uncertainty.py
    │   ├── tf_xlstm_refiner.py
    │   └── cm_xbeam.py
    │
    ├── losses/
    │   ├── sisdr.py
    │   ├── complex_loss.py
    │   ├── scm_loss.py
    │   └── uncertainty_loss.py
    │
    ├── metrics/
    │   ├── separation.py
    │   ├── perceptual.py
    │   ├── statistics.py
    │   └── runtime.py
    │
    ├── scripts/
    │   ├── generate_dynamic_sms.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── evaluate_motion.py
    │   ├── evaluate_doa_error.py
    │   ├── profile.py
    │   └── aggregate_seeds.py
    │
    └── tests/
        ├── test_scm_psd.py
        ├── test_scm_streaming.py
        ├── test_mvdr_constraint.py
        ├── test_nullspace_residual.py
        ├── test_no_future_leakage.py
        ├── test_streaming_equivalence.py
        ├── test_complex_gradients.py
        └── test_pit_metrics.py

### Primary external implementations worth reusing

Do not recreate all baselines.

The **official NX-AI xLSTM repository** should be the starting point for mLSTM blocks and optimized kernels. [\[20\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com)

The **NBSS/SpatialNet repository** explicitly provides SpatialNet, online SpatialNet and related multichannel models. [\[18\]](https://github.com/Audio-WestlakeU/NBSS?utm_source=chatgpt.com)

**ADL-MVDR** has a public implementation associated with its all-deep-learning MVDR target-separation work. [\[19\]](https://github.com/zzhang68/adlmvdr?utm_source=chatgpt.com)

Use these as external baselines or adapters, preserving their upstream commits and configuration files.

### Complex numerical operations

Neural feature layers may use BF16, but beamforming linear algebra should run in at least:

$$\text{complex64}.$$

Disable autocast around:

- SCM accumulation;
- Hermitian symmetrization;
- Cholesky;
- `torch.linalg.solve`;
- eigenvalue diagnostics.

Implement:

    phi = 0.5 * (phi + phi.mH)

    scale = phi.diagonal(dim1=-2, dim2=-1).real.sum(-1) / M
    phi_loaded = phi + eps_load * scale[..., None, None] * eye

Prefer:

    torch.linalg.cholesky_ex
    torch.cholesky_solve

when numerically valid, with a robust fallback to:

    torch.linalg.solve

Never train with an explicit:

    torch.inverse(phi)

unless it exists solely in a correctness test.

### Essential unit tests

**PSD test**

$$\lambda_{\min}(\Phi) \geq - \epsilon_{num}.$$

**Hermitian test**

$$\frac{\parallel \Phi - \Phi^{H} \parallel_{F}}{\parallel \Phi \parallel_{F} + \epsilon} < 10^{- 5}$$

in FP32.

**MVDR distortionless test**

$$|w_{MVDR}^{H}d - 1| < 10^{- 4}.$$

**Residual constraint test**

$$|r^{H}d| < 10^{- 4}.$$

**Final beamformer test**

$$|w_{final}^{H}d - 1| < 10^{- 4}.$$

**Causality test**

Take an input $Y$, perturb every frame after $t_{0}$, rerun the model, and verify

$${\widehat{S}}_{t \leq t_{0}}$$

does not change beyond numerical tolerance.

**Streaming-equivalence test**

For causal mode:

$$model(Y_{1:T}) \approx concat\lbrack stream(Y_{1}),\ldots,stream(Y_{T})\rbrack.$$

These tests are far more convincing for reproducibility than a README saying “the model is causal.”

### Depthwise convolutions and causality

Use depthwise-separable Conv2D blocks for the feature encoder:

$$\text{DWConv} \rightarrow 1 \times 1\text{ pointwise} \rightarrow \text{normalization} \rightarrow \text{activation}.$$

Time padding must be left-only:

    past | current
    ███████████│x

not:

    past | current | future
    █████│x│█████

Frequency padding can be symmetric because no future time information is introduced.

### Major risks and mitigation

| Risk                                             | Why it matters                                      | Mitigation                                                           |
|--------------------------------------------------|-----------------------------------------------------|----------------------------------------------------------------------|
| Raw mLSTM $C_{t}$ is not a valid SCM             | May be non-Hermitian/non-PSD                        | Use mLSTM only to control explicit PSD recurrence                    |
| Native exponential gates unsuitable as $\lambda$ | Forget factor needs bounded physical interpretation | Separate sigmoid head for $\lambda$                                  |
| SCM nearly singular                              | MVDR solver instability                             | Trace-scaled diagonal loading + FP32 complex solve                   |
| Residual learns to ignore MVDR                   | Physics contribution becomes cosmetic               | Nullspace projection, $\alpha$ cap, residual norm loss               |
| TF refiner explains all gains                    | Weakens beamforming claim                           | Report pre-refiner output and refiner ablation                       |
| mLSTM does not beat LSTM                         | Plausible; prior SE work shows strong LSTM          | Mandatory parameter-matched LSTM control                             |
| Motion results improve but static degrades       | Learned forgetting too reactive                     | Range-constrain $\lambda$, regularize gate variation                 |
| Masks dominate system quality                    | Memory contribution obscured                        | Oracle-mask and fixed-mask diagnostic experiments                    |
| DoA cue is inaccurate                            | TSE collapses under realistic tracking error        | Train with cue corruption; report systematic robustness              |
| Geometry overfitting                             | Learned spatial encoder tied to array               | Channel perturbation + held-out geometry test                        |
| Training is too costly                           | Many ablations/seeds                                | Frequency downsample, shared dual memory, staged search              |
| Novelty overlap with attention SCM               | Learned SCM weighting already exists                | Frame novelty around covariance-aligned mLSTM + constrained residual |

The LSTM risk is especially important. xLSTM-SENet found xLSTM effective for speech enhancement but also found conventional LSTM surprisingly competitive. [\[14\]](https://arxiv.org/html/2501.06146v2) If CM-xBeam beats LSTM only because it is larger, the central claim weakens considerably.

### Novelty boundary to state honestly

Existing literature already supports:

$$\text{learned SCM aggregation}$$

for moving sources, including attention-based weighting. [\[1\]](https://arxiv.org/pdf/2205.03568)

xLSTM already supports:

$$\text{matrix memory + covariance-style updates}.$$

[\[13\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html)

xLSTM has already been demonstrated for:

$$\text{single-channel speech enhancement}.$$

[\[16\]](https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com)

Therefore do **not** write:

> “We are the first to learn temporal covariance weights.”

or:

> “We introduce matrix memories.”

The defensible contribution is:

$$\boxed{\text{covariance-aligned mLSTM control of explicit physical SCMs}}$$

plus

$$\boxed{\text{distortionless-preserving uncertainty-gated neural residual beamforming}.}$$

That is a much more precise novelty claim.

## Publication package, figures, tables, and milestones

### Recommended paper title

My preferred title is:

> **CM-xBeam: Covariance-Aligned xLSTM Memory for Physics-Guided Neural Beamforming in Multichannel Speech Separation**

For an explicitly moving-speaker paper:

> **CM-xBeam: Adaptive Matrix-Memory Neural Beamforming for Moving-Speaker Separation**

For a paper equally emphasizing blind separation and directional extraction:

> **Covariance-Aligned xLSTM Beamforming for Multichannel Speech Separation and Target Extraction**

The first is strongest because it communicates both the theoretical idea and the physics-guided design.

### Contributions section

A strong final Introduction should converge on approximately these claims:

> **First**, we establish a structural correspondence between recursive spatial covariance accumulation and mLSTM's gated outer-product matrix memory, revealing adaptive beamforming as a matrix-memory problem.
>
> **Second**, we propose covariance-aligned mLSTM memories that predict adaptive forgetting and update gates for explicitly Hermitian PSD target and interference SCMs, retaining physically meaningful covariance structure.
>
> **Third**, we introduce a physics-guided neural beamformer that augments MVDR with an uncertainty-gated residual constrained to the steering-vector nullspace, preserving the estimated distortionless response by construction.
>
> **Fourth**, we evaluate the resulting system under static and dynamic multichannel separation and direction-guided target extraction, using moving-source trajectories, DoA perturbations, parameter-matched LSTM controls, factorial memory ablations, multiple seeds, and significance testing.

That is much stronger than:

> “We combine xLSTM, MVDR and a postfilter.”

### Essential figures

**Figure: Architecture**

Caption:

> **Overview of CM-xBeam.** Multichannel complex spectra are encoded into spatial features and source masks. Target- and interference-conditioned mLSTM memories predict adaptive forgetting and update coefficients for explicit PSD spatial covariance recurrences. The resulting SCMs parameterize an MVDR beamformer. A learned correction is projected onto the subspace orthogonal to the estimated target steering vector and modulated by covariance uncertainty, thereby augmenting MVDR while preserving its estimated distortionless constraint. A lightweight TF-xLSTM module performs final spectral refinement.

**Figure: Theoretical correspondence**

Draw side by side:

$$C_{t} = f_{t}C_{t - 1} + i_{t}v_{t}k_{t}^{\top}$$

and

$$S_{t} = \lambda_{t}S_{t - 1} + a_{t}y_{t}y_{t}^{H}.$$

Caption:

> **Matrix-memory correspondence motivating covariance-aligned mLSTM.** Both mLSTM and recursive SCM estimation accumulate gated outer-product statistics. Unlike unconstrained mLSTM memory, an acoustic SCM must be Hermitian PSD; CM-xBeam therefore lets mLSTM control a separate physically valid covariance recurrence rather than directly identifying its hidden matrix with the SCM.

This should be one of the main figures. It immediately explains why xLSTM is in the paper.

**Figure: Dynamic-memory interpretation**

Plot:

- target DoA;
- interferer DoA;
- ${\bar{\lambda}}_{s}(t)$;
- ${\bar{\lambda}}_{i}(t)$;
- uncertainty $\bar{u}(t)$;
- correction gate $\bar{\alpha}(t)$;
- framewise SI-SDR.

Caption:

> **Learned adaptation to a moving acoustic scene.** The spatial-memory controller shortens its effective memory around source motion and abrupt spatial changes, while the uncertainty gate increases neural correction when covariance estimates become unreliable.

If the model behaves this way, this is probably the most memorable figure in the paper.

**Figure: Robustness curves**

Four curves:

$$SI - SDRi\text{ vs angular velocity},$$

$$SI - SDRi\text{ vs DoA error}.$$

This directly demonstrates that the gain is not limited to one mean benchmark score.

### Essential tables

**Main performance table**

| Method               | Static SI-SDRi | Moving SI-SDRi | STOI | PESQ | Params | RTF |
|----------------------|----------------|----------------|------|------|--------|-----|
| Mask-MVDR fixed EMA  |                |                |      |      |        |     |
| Attention-SCM MVDR   |                |                |      |      |        |     |
| LSTM-SCM             |                |                |      |      |        |     |
| SpatialNet           |                |                |      |      |        |     |
| CM-xBeam, no refiner |                |                |      |      |        |     |
| **CM-xBeam full**    |                |                |      |      |        |     |

**Memory-factorial table**

| Memory    | Forgetting  | Static | Slow motion | Fast motion | Abrupt jump |
|-----------|-------------|--------|-------------|-------------|-------------|
| LSTM      | fixed       |        |             |             |             |
| LSTM      | learned     |        |             |             |             |
| mLSTM     | fixed       |        |             |             |             |
| **mLSTM** | **learned** |        |             |             |             |

**Beamformer-ablation table**

| MVDR | Residual | Nullspace | Uncertainty | Refiner | SI-SDRi |
|------|----------|-----------|-------------|---------|---------|
| ✓    | –        | –         | –           | –       |         |
| ✓    | ✓        | –         | –           | –       |         |
| ✓    | ✓        | ✓         | –           | –       |         |
| ✓    | ✓        | ✓         | ✓           | –       |         |
| ✓    | ✓        | ✓         | ✓           | ✓       |         |

**Complexity table**

| Model           | Params | MAC/s | Peak GPU memory | RTF GPU | RTF CPU | Algorithmic latency |
|-----------------|--------|-------|-----------------|---------|---------|---------------------|
| Fixed MVDR      |        |       |                 |         |         |                     |
| LSTM controller |        |       |                 |         |         |                     |
| Attention SCM   |        |       |                 |         |         |                     |
| CM-xBeam        |        |       |                 |         |         |                     |

### Expected deliverables

| Deliverable        | Minimum publishable         | Strong version                               |
|--------------------|-----------------------------|----------------------------------------------|
| Core model         | mLSTM-controlled SCM + MVDR | \+ constrained residual + uncertainty        |
| Speech separation  | SMS-WSJ                     | SMS-WSJ + cross-domain                       |
| Moving sources     | one moving test             | full speed/jump/crossing suite               |
| Target extraction  | oracle/noisy DoA            | dynamic DoA + latency/dropout                |
| Seeds              | 3                           | 5                                            |
| LSTM control       | yes                         | parameter + compute matched                  |
| Statistical test   | paired CI                   | hierarchical bootstrap + corrected test      |
| Interpretability   | $\lambda_{t}$ plot          | $\lambda_{t},\alpha_{t},\Phi$-error, motion  |
| Reproducibility    | source + configs            | generation scripts + checkpoints + manifests |
| Runtime            | Params + GPU RTF            | CPU/GPU RTF + latency + memory + MAC/s       |
| Real-recorded test | optional                    | LibriCSS/meeting stress test                 |

### Development timeline

A realistic research timeline starting in September 2026 is:

|                                                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------|
| <img src="data:image/png;base64,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" style="width:5.83333in;height:1.94444in" alt="Rendered Mermaid diagram 2" /> |

### Go/no-go checkpoints

Before spending five-seed compute, require the following:

**Checkpoint A — the theoretical mechanism works**

On dynamic validation mixtures,

$$\lambda_{\text{learned}}$$

must outperform the best tuned fixed $\lambda$.

Otherwise adaptive covariance memory is not yet justified.

**Checkpoint B — mLSTM matters**

Parameter-matched:

$$\text{mLSTM controller} > \text{LSTM controller}$$

on moving conditions, or mLSTM must provide a meaningful complexity/robustness advantage.

If it does not, do not force the xLSTM claim. A paper about adaptive covariance memory can still be viable, but it becomes a different paper.

**Checkpoint C — the physics constraint matters**

Require the nullspace residual to match or beat unconstrained correction while producing much smaller distortionless error:

$$|w^{H}d - 1|.$$

If that succeeds, it is a very clean contribution.

**Checkpoint D — the refiner is not carrying the paper**

The **pre-refiner** CM-xBeam should already outperform the principal MVDR/LSTM alternatives.

### Final assessment of publishability

The pipeline I would actually implement is therefore:

$$\boxed{\begin{aligned}
 & \text{Complex spatial encoder} \\
 & \quad \downarrow \\
 & \text{shared-weight dual-state mLSTM spatial memories} \\
 & \quad \downarrow \\
 & \text{adaptive, explicitly PSD target/interference SCM recurrence} \\
 & \quad \downarrow \\
 & \text{differentiable MVDR} \\
 & \quad \downarrow \\
 & \text{distortionless-preserving nullspace neural residual} \\
 & \quad \downarrow \\
 & \text{calibrated uncertainty gate} \\
 & \quad \downarrow \\
 & \text{lightweight TF-xLSTM spectral refiner}.
\end{aligned}}$$

The central paper message should remain even shorter:

$$\boxed{\text{Beamforming covariance estimation is an adaptive matrix-memory problem.}}$$

and the architectural answer:

$$\boxed{\text{Use mLSTM to learn the memory dynamics, but keep the covariance and beamforming physics explicitly constrained.}}$$

That is the difference between an incremental **“xLSTM + neural beamforming”** submission and a potentially strong paper. The individual primitives already have prior art: xLSTM has matrix/covariance-style memory, xLSTM has been used for single-channel enhancement, and learned SCM tracking has already been developed for moving-speaker beamforming. [\[21\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) The publishable insight is the **structural alignment and resulting constrained design**, with experiments constructed specifically to test that hypothesis.

If the parameter-matched ablations show that mLSTM-controlled covariance memory produces the largest benefit exactly under **fast motion, abrupt spatial changes, source crossings and imperfect target direction cues**, while the learned gates exhibit the predicted adaptive-memory behavior, the paper has a substantially stronger scientific story than a conventional backbone-replacement study. The open-source ecosystem is also favorable: official xLSTM code, public SpatialNet implementations, ADL-MVDR code, SMS-WSJ generation infrastructure and GPU RIR simulation all reduce implementation risk and improve reproducibility. [\[22\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com)

[\[1\]](https://arxiv.org/pdf/2205.03568) [\[9\]](https://arxiv.org/pdf/2205.03568) arxiv.org

<https://arxiv.org/pdf/2205.03568>

[\[2\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) [\[13\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) [\[21\]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) xLSTM: Extended Long Short-Term Memory

<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html>

[\[3\]](https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com) [\[16\]](https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com) xLSTM for Single-Channel Speech Enhancement

<https://arxiv.org/abs/2501.06146?utm_source=chatgpt.com>

[\[4\]](https://arxiv.org/abs/1811.02508?utm_source=chatgpt.com) SDR - half-baked or well done?

<https://arxiv.org/abs/1811.02508?utm_source=chatgpt.com>

[\[5\]](https://dl.acm.org/doi/abs/10.1109/TASL.2011.2114881?utm_source=chatgpt.com) An Algorithm for Intelligibility Prediction of Time–Frequency ...

<https://dl.acm.org/doi/abs/10.1109/TASL.2011.2114881?utm_source=chatgpt.com>

[\[6\]](https://www.itu.int/rec/t-rec-p.862?utm_source=chatgpt.com) P.862 : Perceptual evaluation of speech quality (PESQ)

<https://www.itu.int/rec/t-rec-p.862?utm_source=chatgpt.com>

[\[7\]](https://arxiv.org/abs/1910.13934?utm_source=chatgpt.com) SMS-WSJ: Database, performance measures, and baseline recipe for multi-channel source separation and recognition

<https://arxiv.org/abs/1910.13934?utm_source=chatgpt.com>

[\[8\]](https://arxiv.org/html/2307.16516v2?utm_source=chatgpt.com) SpatialNet: Extensively Learning Spatial Information for ...

<https://arxiv.org/html/2307.16516v2?utm_source=chatgpt.com>

[\[10\]](https://arxiv.org/abs/1810.11359?utm_source=chatgpt.com) gpuRIR: A Python Library for Room Impulse Response Simulation with GPU Acceleration

<https://arxiv.org/abs/1810.11359?utm_source=chatgpt.com>

[\[11\]](https://arxiv.org/html/2005.11262v1?utm_source=chatgpt.com) LibriMix: An Open-Source Dataset for Generalizable ...

<https://arxiv.org/html/2005.11262v1?utm_source=chatgpt.com>

[\[12\]](https://arxiv.org/pdf/2001.11482?utm_source=chatgpt.com) arXiv:2001.11482v3 \[cs.SD\] 7 May 2020

<https://arxiv.org/pdf/2001.11482?utm_source=chatgpt.com>

[\[14\]](https://arxiv.org/html/2501.06146v2) xLSTM-SENet: xLSTM for Single-Channel Speech Enhancement

<https://arxiv.org/html/2501.06146v2>

[\[15\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com) [\[20\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com) [\[22\]](https://github.com/nx-ai/xlstm?utm_source=chatgpt.com) NX-AI/xlstm: Official repository of the ...

<https://github.com/nx-ai/xlstm?utm_source=chatgpt.com>

[\[17\]](https://arxiv.org/html/2402.03058v2) Array Geometry-Robust Attention-Based Neural Beamformer for Moving Speakers

<https://arxiv.org/html/2402.03058v2>

[\[18\]](https://github.com/Audio-WestlakeU/NBSS?utm_source=chatgpt.com) Audio-WestlakeU/NBSS: The official repo of NBC & ...

<https://github.com/Audio-WestlakeU/NBSS?utm_source=chatgpt.com>

[\[19\]](https://github.com/zzhang68/adlmvdr?utm_source=chatgpt.com) zzhang68/adlmvdr

<https://github.com/zzhang68/adlmvdr?utm_source=chatgpt.com>
