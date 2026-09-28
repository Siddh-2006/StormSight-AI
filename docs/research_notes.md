# Research Notes — SOTA & Datasets for Thunderstorm/Lightning Nowcasting

Gathered for SIH PS 26072. Feeds `docs/design.md` and justifies the architecture/data choices there.

## 1. State-of-the-art architectures (0–6h nowcasting)

**Recurrent/convolutional sequence models** (hackathon-feasible):
- **ConvLSTM** (Shi et al., NeurIPS 2015) — convolutional gates instead of fully-connected LSTM gates, preserves spatial structure. Easy-to-train baseline; this is what we implement first.
- **TrajGRU** (Shi et al., NeurIPS 2017) — introduced the **HKO-7** benchmark (Hong Kong radar CAPPI); learns location-variant motion fields vs. ConvLSTM's location-invariant convolution.
- **PredRNN/PredRNN++** (Wang et al.) — dual memory pathway, stronger long-horizon accuracy, heavier to tune.
- **SmaAt-UNet** (Trebing, Stańczyk, Mehrkanoon, *Pattern Recognition Letters* 2021, arXiv:2007.04417) — attention U-Net with depthwise-separable convs + CBAM attention, ~1/4 the parameters of comparable models, matches larger models. GitHub: `HansBambel/SmaAt-UNet`. **Best fit for our compute budget** — our `ml/model.py` follows this family (depthwise-separable convs + ConvLSTM bottleneck). Extensions: SmaAt-fUsion/SmaAt-Krige-GNet (arXiv:2502.16116) fuse point station data via kriging/GNN.

**Generative/probabilistic SOTA** (reference only, not reimplementable in hackathon time):
- **DGMR** (Ravuri et al., *Nature* 2021, DOI 10.1038/s41586-021-03854-z) — DeepMind + UK Met Office conditional GAN; 56 Met Office forecasters preferred it over pySTEPS/PredRNN/UNet 93% of the time. Needs large radar archives + adversarial training — cite as SOTA, don't attempt to retrain.
- **NowcastNet** (Zhang, Long et al., *Nature* 2023, Tsinghua) — physics-conditioned kernel + generative decoder, preferred by 71% of 62 Chinese meteorologists on extreme rain.
- **MetNet/MetNet-2/MetNet-3** (Google/DeepMind) — axial-attention transformer over radar+satellite+NWP+surface obs directly; powers Google's Pixel Weather nowcast. Requires huge compute; architectural reference for attention-based fusion only.

**Operational classical baseline**:
- **pySTEPS** (Pulkkinen et al. 2019, *GMD*) — Lucas-Kanade/semi-Lagrangian optical-flow extrapolation + STEPS ensemble blending. No training data needed, standard worldwide operational baseline. **This is our baseline model** (reimplemented as a small C++ advection routine, `cpp_module/`, mirroring the same operational principle).

**Verdict**: ConvLSTM/SmaAt-UNet trained on multi-channel grids, benchmarked against an optical-flow baseline, is the realistic sweet spot — matches `docs/design.md` §2.2 exactly.

## 2. Lightning-specific nowcasting

- **NOAA LightningCast** (Cintineo et al., *Weather and Forecasting* 2022) — operational CNN on GOES-16/17 ABI predicting P(GLM flash within 60 min); used operationally at NWS. Confirms "predict flash probability from satellite+radar features" is the right framing, not raw point extrapolation.
- **Multi-task U-Net for VIL + lightning** (*Adv. Atmos. Sci.* 2022) — joint VIL regression + lightning occurrence from combined satellite+radar, 90 min horizon. Direct template for our dual-head model.
- **MeteoSwiss multi-hazard fusion** (Leinonen et al., *GRL* 2023, arXiv:2211.01001) — fuses radar + satellite + lightning + NWP + DEM into one network jointly nowcasting lightning/hail/heavy rain; notes satellite retains skill longer than radar at longer lead times. **This is the closest published analogue to our exact problem statement** (multi-radar + satellite + lightning + model data → thunderstorm/lightning nowcast).
- **CAPE/shear as predictors**: standard to derive CAPE/CIN/bulk shear from reanalysis soundings as features correlating with charge separation; 2025 *GMD* paper trains DNNs on ERA5 profiles against ALDIS ground truth. Radar echo-top height/VIL is the classic "first-flash" proxy.
- **India-specific**: IITM's **Indian Lightning Location Network (ILLN)** — 83+ ground sensors, <500m accuracy, feeds IMD forecast offices and the **Damini** app (IITM, 2020; 40-min lead time, 40 sq.km radius, 500k+ downloads) — closest existing product analogue to our deliverable. Global alternatives if ILLN data isn't accessible: **WWLLN** (VLF, archival access via wwlln.net) and **ENTLN** (commercial API).

## 3. Public datasets usable for the prototype

| Dataset | Content | Access |
|---|---|---|
| **SEVIR** (MIT Lincoln Lab, NeurIPS 2020) | 10k+ storm events, 384×384km, 4h @ 5min; aligned GOES-16 ABI + NEXRAD VIL + **GOES-16 GLM lightning** | Free, AWS Open Data, `MIT-AI-Accelerator/neurips-2020-sevir` — **best real-data proxy for this exact problem, recommended drop-in if time permits** |
| GOES-16/17/18 GLM | Native lightning flash/group/event data | Free, NOAA CLASS/AWS/GCS |
| HKO-7 | Hong Kong radar CAPPI 2009–2015 | Via TrajGRU paper / mirrors |
| NASA GPM IMERG | Global 0.1°, 30-min precip | Free API/AWS/GEE |
| ERA5 | 0.25° hourly CAPE/shear/profiles | Free, Copernicus CDS API (`cdsapi`) |
| MOSDAC (ISRO) | INSAT-3D/3DR IR/WV/cloud products, India | Free for non-commercial, portal access |
| IMD DWR network | 51 radars (5 C-band/22 S-band/12 X-band) | Viewable at mausam.imd.gov.in; **bulk raw access not openly documented** — real obstacle, justifies proxy-data strategy |
| NCMRWF IMDAA/NCUM | 12km India regional reanalysis/NWP | Registration required, not fully open |

**Decision**: no time to onboard SEVIR/ERA5 downloads plus build the full stack in the hackathon window, so the prototype uses the synthetic generator (`backend/app/data/synthetic_generator.py`) as primary, explicitly designed to be structurally identical to SEVIR's channel layout (radar/VIL-like, satellite-IR-like, lightning-flash-density) so swapping in real SEVIR/ERA5 data later requires no architecture change — only a new data loader behind the same grid contract in `docs/technical_flow.md`.

## 4. Multi-modal fusion techniques

- **Early fusion / channel stacking** (what we use): stack all channels into one `(T,C,H,W)` tensor — simplest, used by SEVIR-based models.
- **Dual-stream + cross-attention**: MAG-Net (arXiv:2604.02818), MeteoSwiss model — separate per-modality encoders merged via attention; better when modalities differ a lot in resolution/noise. Noted as a post-hackathon upgrade path in `design.md` §5.
- **Graph-based fusion for sparse points**: SmaAt-Krige-GNet (arXiv:2502.16116) — kriging/GNN to interpolate sparse lightning-strike points onto the grid before fusion. Our synthetic generator sidesteps this by rasterizing directly, but this is the right technique for real ILLN/WWLLN point data.
- **Multi-task decoding**: joint reflectivity/VIL regression + lightning probability classification head sharing one backbone — directly what our model does (`design.md` §2.2), matches the multi-task U-Net and MeteoSwiss designs.

## 5. Existing operational systems / prior art

- **IMD Nowcast Warnings**: 3-hourly station/district nowcasts (~1019 stations/732 districts) via State Meteorological Centres, mausam.imd.gov.in.
- **Meghdoot** app (IMD+IITM+ICAR+ICRISAT): 3.5-day forecast + 3h nowcast + crop advisories, 668 districts.
- **Damini** (IITM): the closest existing product to our deliverable — see §2 above.
- Note: GAGAN (ISRO/AAI satellite-based aviation augmentation) is unrelated to lightning — do not cite it as a lightning system despite surface similarity in some briefs.
- IITM: INSAT-3D MCS-detection nowcasting (Kumar et al. 2017); GeoConvNet (deep-learning convective cloud segmentation from INSAT-3D/3DR, NE India); new X-band dual-pol DWR at Mahabaleshwar (2026).
- **NCMRWF** IMDAA/NCUM ~12km NWP — the "model data" component named in the PS title.

## 6. Evaluation convention (adopted in `ml/metrics.py`)

Standard triplet at multiple thresholds, multiple lead times — same convention DGMR/NowcastNet papers use:
- **POD** = Hits/(Hits+Misses)
- **FAR** = FalseAlarms/(Hits+FalseAlarms)
- **CSI** = Hits/(Hits+Misses+FalseAlarms)
- **HSS** (Heidke Skill Score) — also reported, standard in Indian/IMD nowcasting verification literature.

Report as skill-vs-lead-time curves, ML model vs. optical-flow (pySTEPS-style) baseline — this is the headline "does ML actually help" evidence for judges.

## 7. Framing for judges

State explicitly (already in `design.md`): no dataset was issued with this PS; IMD DWR / MOSDAC INSAT / ILLN are the intended production inputs; synthetic data (structured like SEVIR) is a transparent, documented stand-in; the architecture is deliberately built to swap onto real feeds without changing the model or API contract. This "proxy-data-now, real-data-later" framing is standard and expected for SIH prototypes lacking official data access.

---
*Sources: Nature (Ravuri et al. 2021; Zhang et al. 2023), arXiv (2007.04417, 1706.03458, 2604.02818, 2502.16116, 2211.01001), Weather and Forecasting (2022 LightningCast), npj Climate and Atmospheric Science (2023), GMD (2025, 2019), MIT SEVIR, NASA GPM/GES DISC, MOSDAC.gov.in, mausam.imd.gov.in, IITM/Damini/Meghdoot coverage.*
