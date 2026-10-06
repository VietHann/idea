# CM-xBeam

Triển khai theo `CM-xBeam.md`: bộ nhớ mLSTM điều khiển ma trận hiệp phương sai âm học tường minh, MVDR khả vi, residual giữ ràng buộc distortionless, uncertainty được hiệu chuẩn bằng SCM oracle và refiner TF-xLSTM. Code, môi trường cục bộ, manifest, checkpoint và kết quả đầu ra đều nằm trong thư mục dự án. Dữ liệu gốc chỉ được mở để đọc.

## Môi trường

Python >=3.10, PyTorch >=2.2, `xlstm==2.0.5` chính thức của NX-AI, h5py, soundfile, scipy và PyYAML. STOI/WB-PESQ cần pystoi/pesq; mô phỏng RIR cần pyroomacoustics; vẽ diagnostics cần matplotlib.

Bản xLSTM chính thức đã được cài trong `.vendor/` tại lần triển khai đầu tiên, không thay đổi package toàn cục. Wrapper chỉ bổ sung clone cache convolution trước `step()` để tránh cập nhật in-place làm hỏng backpropagation theo thời gian. Không thay mLSTM bằng LSTM thông thường.

Môi trường hệ thống được quan sát có PyTorch `2.11.0+cu130`, driver NVIDIA `550.163.01`, CUDA driver 12.4; tổ hợp này không khởi tạo được CUDA. Có thể chạy CPU ngay hoặc tạo môi trường PyTorch CUDA 12.4 riêng trong dự án:

```bash
cd /project/anhlt/0607/research/work/idea
bash scripts/setup_environment.sh cu124
source .venv/bin/activate
```

Môi trường `.venv` đã được setup sẵn với Python 3.12.7 và PyTorch `2.6.0+cu124`; CUDA nhận đủ 3 GPU NVIDIA A30. Forward/backward BF16 của cả hai bản capacity 7.180.289 tham số đã chạy trên GPU với dữ liệu thật. `pip check` không phát hiện dependency lỗi. Phiên bản dependency được cố định trong `requirements-cu124.lock` để tái tạo môi trường.

Script này không cài driver và không sửa môi trường Python bên ngoài dự án. Nó cần tải các dependency khi tạo lại môi trường; không được tự chạy như một phần huấn luyện. Chọn `cpu` thay cho `cu124` nếu không cần GPU. Sau khi cài, kiểm tra:

```bash
python -m scripts.doctor --config configs/lrs2.yaml
```

## Hai recipe dữ liệu

`configs/lrs2.yaml` đọc:

`/project/anhlt/0607/lrs2_paper_simulated_dataset_separate_v2_20260811`

- HDF5 riêng cho `2speakers/{train,val,test}/audio.hdf5`.
- `mixture`: `[N,3,L]`; `reverberant_sources`: `[N,2,3,L]`.
- 16 kHz, FFT/window 512, hop 128, 3 microphone.
- Target chính là source image reverberant tại microphone tham chiếu. Direct-path/aligned-clean có trong dữ liệu nhưng không bị dùng nhầm làm source image SCM.
- Dữ liệu hiện có đoạn 2 giây. Crop tối đa 4 giây không tự lặp hoặc kéo dài một đoạn 2 giây thành 4 giây.

`configs/librispeech.yaml` đọc:

`/project/anhlt/0607/jnf_mamba_experiment/paper_reproduction/data_original_var_pos_parallel_sp1/8k`

- `prep_mix_ch3_sp1_var_target.hdf5`, mỗi split `[N,3,3,96000]`.
- Trục thứ hai có thứ tự: target reverberant, tổng interferer, target direct-path.
- Mixture được tính đúng bằng **component 0 + component 1**. Component 2 không phải speaker thứ ba.
- Recipe có đúng một interferer nên component 0/1 là hai source image tách biệt.
- 8 kHz, FFT/window 256, hop 64, 3 microphone; cùng độ dài cửa sổ/hop theo thời gian với bản LRS2.
- `configs/librispeech_16k.yaml` resample tất cả microphone và nguồn bằng cùng bộ lọc để chạy frontend 16 kHz.

Mỗi bộ hiện có 54.000 train, 2.700 validation và 1.800 test. Manifest sử dụng `valid` để trỏ tới split gốc `val`, không chia lại dữ liệu ngẫu nhiên. Chuẩn bị manifest:

```bash
python -m scripts.prepare_data --config configs/lrs2.yaml --strict-speaker-splits
python -m scripts.prepare_data --config configs/librispeech.yaml --strict-speaker-splits
```

Manifest đầy đủ đã được tạo trong `manifests/lrs2/` và `manifests/librispeech/`. Lệnh cố ý từ chối ghi đè; dùng `--overwrite` nếu muốn tạo lại. `--limit` chỉ dành cho thao tác giới hạn dữ liệu, không dùng cho kết quả nghiên cứu cuối cùng. Kiểm tra split hiện có không tìm thấy overlap ID mixture hoặc ID speaker giữa train/validation/test.

Audit phòng cho thấy LRS2 không trùng geometry/T60 giữa split. **LibriSpeech hiện có phòng trùng**: 2.700 geometry train–validation, 1.800 train–test và 1.800 validation–test. Dữ liệu gốc được giữ nguyên; recipe này không được trình bày như benchmark held-out-room. Dùng `--audit-only` để kiểm tra manifest sẵn có và `--strict-room-splits` để từ chối các split trùng phòng. Bộ động sinh mới dùng seed/phòng phân tách theo split.

`configs/lrs2_3speakers.yaml` hỗ trợ nhánh 3 speaker có sẵn; chuẩn bị bằng cùng lệnh với cấu hình này. Số microphone/source được kiểm tra nghiêm ngặt, không tự bỏ microphone hoặc broadcast một target mono thành image đa kênh.

Để chỉ giữ nhóm hướng theo chỉ số `0,5,10,...,175` trong 180 nhóm, dùng `configs/lrs2_capacity_stride5.yaml` hoặc `configs/librispeech_capacity_stride5.yaml`. Đây là stride **nhóm hướng**, không phải 5 độ: trên lưới gốc 2 độ, 36 nhóm được giữ có góc `-180,-170,...,170` độ. Tập train còn 10.800 mixture, 5.400 batch/epoch với batch size 2, 675 optimizer step/epoch với accumulation 8. Chọn theo hướng của source target/source 0 trong metadata mô phỏng; không lọc riêng microphone hay hướng ngẫu nhiên của interferer. Validation/test giữ nguyên 2.700/1.800 mixture để đánh giá cả những nhóm hướng không giữ trong train.

Lọc được thực hiện trong bộ nhớ từ manifest hiện có; không tạo lại RIR/audio và không thay đổi góc thật của mẫu. Các khóa cấu hình là `data.direction_stride: 5`, `direction_offset: 0`, `direction_subset_splits: [train]`. **Tổng lịch optimizer vẫn là 120.000 bước**: giảm dữ liệu làm một epoch ngắn hơn, nhưng không tự rút ngắn tổng huấn luyện khi lịch được định nghĩa theo step. Dùng output run riêng; resume từ run dùng toàn bộ hướng sang run subset bị từ chối để tránh tiếp tục sai sampler/cursor.

Bản mới chọn chỉ số `0,10,20,...,170` dùng `configs/lrs2_capacity_stride10.yaml` và `configs/librispeech_capacity_stride10.yaml`: 18 nhóm hướng, 5.400 mixture train, góc thật `-180,-160,...,160` độ. Validation/test vẫn giữ nguyên. Chạy hai GPU vật lý **0 và 1** bằng launcher đã cấu hình sẵn:

```bash
bash scripts/launch_2gpu.sh lrs2
# Hoặc:
bash scripts/launch_2gpu.sh librispeech
```

Launcher dùng Python của `.venv`, `CUDA_VISIBLE_DEVICES=0,1`, `torchrun` với 2 process và backend chunkwise. Default batch **2/GPU**, accumulation **8**, nên effective batch toàn bộ là **32**. Mỗi rank đọc 2.700 mixture/epoch và thanh progress rank 0 hiển thị **1.350 batch/epoch**; mỗi batch trên thanh tương ứng 4 mixture trên cả hai GPU. Phần accumulation chưa đủ 8 batch được giữ sang epoch kế tiếp. Output mặc định là `runs/lrs2_stride10_2gpu/` hoặc `runs/librispeech_stride10_2gpu/`; chỉ rank 0 ghi file, training metrics được trung bình qua cả hai rank. Checkpoint lưu world size và RNG từng rank; resume phải giữ cùng số process và tập hướng.

Dừng lượt training một GPU trước khi chuyển sang launcher này để tránh hai job tranh cùng GPU. Launcher không tự khởi động khi sửa cấu hình. Các tùy chọn training có thể truyền sau tên dataset, ví dụ `bash scripts/launch_2gpu.sh lrs2 --resume runs/lrs2_stride10_2gpu/latest.pt`. Tổng schedule 120.000 optimizer step chưa bị thay đổi.

## Các mô-đun

- `cm_xbeam/models/stft.py`: Hann STFT `center=False`, padding biên causal tường minh và weighted overlap-add; giữ đủ mẫu đầu/cuối, không gặp lỗi NOLA tại đầu Hann.
- `spatial_features.py`: pack `M²` bậc tự do thực của outer product Hermitian chuẩn hóa, năng lượng và IPD.
- `spatial_encoder.py`: convolution depthwise-separable, padding thời gian bên trái, LayerNorm trên feature tại từng frame; downsample tần số hệ số 2.
- `query_encoder.py`, `mask_head.py`: source-slot cho blind separation hoặc biểu diễn vòng tròn DoA theo độ. Source mask separation có thêm noise slot. NaN ở cue biểu thị mất DoA.
- `covariance_xlstm.py`: 2 block mLSTM chính thức, dùng chung weight giữa target/interference nhưng có trạng thái và stream embedding riêng. Sigmoid giới hạn forgetting, softplus giữ update không âm.
- `scm_accumulator.py`: cập nhật numerator/normalizer tường minh, chỉ giữ SCM frame hiện tại, symmetrize Hermitian. Nội suy các scalar gate, không nội suy SCM complex.
- `steering.py`, `mvdr.py`: RTF từ cột microphone tham chiếu; diagonal loading theo trace; Cholesky/solve thay vì inverse; fallback silent bin và chuẩn hóa đáp ứng distortionless.
- `residual_beamformer.py`: dự đoán residual complex và chiếu trực giao steering vector. Neural head khởi tạo gần zero.
- `uncertainty.py`: condition, rank impurity, mask entropy, innovation và contextual memory; hiệu chuẩn log lỗi SCM oracle bằng Huber; gate tăng đơn điệu theo uncertainty dự đoán.
- `tf_xlstm_refiner.py`: 2 block TF với mLSTM thời gian causal, mLSTM tần số hai chiều và complex ratio correction bị chặn biên, khởi tạo identity.
- `cm_xbeam/streaming.py`: trạng thái analysis, encoder, mLSTM, SCM, refiner và synthesis tồn tại xuyên chunk. `push()`/`flush()` không chạy lại cả lịch sử.
- `cm_xbeam/losses/`: SI-SDR/PIT, complex STFT, SCM, uncertainty, residual và mask objective. Cùng permutation SI-SDR được dùng cho mọi loss source-wise.
- `cm_xbeam/data/`: HDF5 đọc theo mẫu, handle riêng theo process, resampling đồng bộ, channel gain/dropout/fractional calibration delay và DoA corruption. Random reference là biến thể riêng, không mặc định.

Blind separation không nhận DoA từ metadata lúc train/evaluate; điều này tránh làm biến thể blind thành hệ thống oracle-direction không được khai báo. Source images chỉ phục vụ supervision/diagnostics, không tham gia tính waveform suy luận.

## Huấn luyện

```bash
python -m scripts.train --config configs/lrs2.yaml --output runs/lrs2 --device cuda
python -m scripts.train --config configs/librispeech.yaml --output runs/librispeech --device cuda
```

Nếu dùng môi trường hệ thống chưa tương thích driver, chọn `--device cpu`.

Curriculum mặc định tính theo **optimizer steps**, không phải microbatch:

- A: 10.000 bước warm-up spatial/mask/oracle-SCM, waveform từ masked reference.
- B: 40.000 bước bật MVDR khả vi.
- C: 30.000 bước bật residual + uncertainty.
- D: 40.000 bước bật TF-xLSTM refiner.

AdamW `3e-4`, weight decay `1e-2`, warm-up 10.000 bước, cosine decay, clip norm 5. Batch 2, accumulation 8 tạo effective batch 16 trên một process. Neural autocast BF16 chỉ bật trên CUDA; SCM/solve/eigendiagnostics luôn FP32/complex64. Checkpoint theo chunk 32 frame giữ native matrix memory tại các biên chunk, thay vì checkpoint từng frame vẫn giữ toàn bộ trạng thái `O(TD²)`.

Có thể điều chỉnh `--batch-size`, `--accumulation-steps`, `--crop-seconds`, `--max-steps`. Resume phải giữ cùng cấu hình training và ablation:

```bash
python -m scripts.train --config configs/lrs2.yaml --output runs/lrs2 --device cuda --resume runs/lrs2/latest.pt
```

Checkpoint có model/optimizer/scheduler, stage, RNG, epoch, vị trí batch và thời gian huấn luyện. `best.pt` chọn bằng SI-SDR validation, không chọn seed tốt nhất. `latest.pt` dùng để tiếp tục. `torchrun` được hỗ trợ qua DistributedDataParallel; effective batch nhân thêm số process.

Thanh progress console bật mặc định, cập nhật sau **mỗi microbatch**, không chờ đủ 8 batch accumulation hoặc 10 optimizer step. Thanh hiển thị epoch/stage, số batch, elapsed/ETA, loss, `si_sdr_db`, `si_sdri_db`, learning rate, `acc` (batch đang tích lũy) và `step` (optimizer update). Trong một batch dài, bộ đếm thời gian tiếp tục cập nhật mỗi giây, kèm phase `loading`, `transfer`, `forward`, `backward`, `optimizer`, `validation` hoặc lưu checkpoint. Validation có thanh riêng. Progress đi ra stderr; JSON log vẫn ở stdout và `train.jsonl`. Dùng `--no-progress` khi muốn tắt thanh cho chạy batch/job logs. Khi sửa code progress, cần chạy lại tiến trình Python hiện tại để nạp phiên bản mới.

Các JSON log mới tách rõ `si_sdr_loss` (loss âm SI-SDR, càng thấp càng tốt), `si_sdr_db` (điểm đầu ra), `mixture_si_sdr_db` (điểm mixture tại microphone tham chiếu) và `si_sdri_db` (điểm đầu ra trừ mixture). Các trường `*_db` càng cao càng tốt; metric không tham gia gradient và không thay đổi loss huấn luyện. Training dùng đúng permutation PIT đã chọn cho waveform loss, không ghép nguồn lại cho baseline. Dùng chiều dài thật để loại padding, loại reference im lặng/constant khỏi độ đo và báo `metric_valid_sources`/`metric_excluded_sources`; không có nguồn hợp lệ thì JSON là `null`, thanh hiện `n/a`.

Training metric trong JSON được tính từ tổng dB và số nguồn hợp lệ của cả accumulation và cả hai rank, không trung bình các batch khác số nguồn hợp lệ. Thanh training hiển thị batch hiện tại ở rank 0; validation hiển thị trung bình tích lũy trên các utterance đã chạy. Mỗi lần validation ghi thêm `valid.jsonl` với SI-SDR, SI-SDRi, baseline mixture, số nguồn và số utterance thực sự đánh giá. Tần suất validation vẫn 1.000 optimizer step và mặc định tối đa 32 utterance, không tự đổi thành mỗi epoch. Best checkpoint vẫn chọn bằng cùng `si_sdr_loss` validation như trước.

Các dòng log cũ dùng `si_sdr` cho loss âm; chúng được giữ nguyên, không suy ra SI-SDRi cho kết quả quá khứ khi không có waveform/baseline tương ứng. Sau khi cập nhật code, cần khởi động lại training; chỉ dùng `--resume` nếu `latest.pt` của chính run đó đã tồn tại và giữ nguyên cấu hình.

`runs/lrs2_initial/` và `runs/librispeech_initial/` có checkpoint sau các bước khởi chạy stage A trên crop ngắn (LRS2 đã resume đến bước 2; LibriSpeech bước 1), dùng để xác nhận đường đọc dữ liệu/huấn luyện/lưu checkpoint. Chúng không phải model đã train đủ và không thể dùng làm kết quả paper.

## Blind separation và target extraction

Cấu hình `configs/extraction.yaml` dùng LRS2, `configs/librispeech_extraction.yaml` dùng LibriSpeech. Chúng chọn target index 0, một output query và DoA thật trong metadata. Huấn luyện extraction không dùng PIT đổi target.

```bash
python -m scripts.train --config configs/extraction.yaml --output runs/lrs2_extraction --device cuda
python -m scripts.infer --checkpoint runs/lrs2/best.pt --input recording_3channels.wav --output runs/separated --device cuda
python -m scripts.infer --checkpoint runs/lrs2_extraction/best.pt --input recording_3channels.wav --doa 35 --chunk-samples 128 --output runs/extracted --device cuda
```

Có thể truyền ba WAV mono sau `--input` theo đúng thứ tự microphone. DoA động được hỗ trợ ở spectral API `[B,K,T]` và dataset; waveform streaming CLI nhận DoA tĩnh, còn ứng dụng có thể cập nhật cue từng frame qua `step_spectrum`.

## Đánh giá

```bash
python -m scripts.evaluate --checkpoint runs/lrs2/best.pt --output runs/lrs2/evaluation.jsonl --device cuda
python -m scripts.evaluate --checkpoint runs/librispeech/best.pt --output runs/librispeech/evaluation.jsonl --device cuda
```

Mặc định giữ stage đã lưu trong checkpoint; `--stage D` chỉ nên dùng khi các branch tương ứng đã được train. Kết quả lưu từng utterance/source, cả trước và sau refiner; permutation final SI-SDR được tái sử dụng cho STOI, WB-PESQ và output trước refiner. Source ID trong kết quả là source target, không phải thứ tự output slot, để ghép cặp hai hệ thống đúng.

WB-PESQ luôn chạy ở 16 kHz; dữ liệu 8 kHz được resample rõ ràng. PESQ là phép đo legacy để so sánh, không được mô tả là khuyến nghị ITU hiện hành. Reference im lặng/metric không hợp lệ trả về `null` cùng lý do; không biến lỗi metric thành điểm tốt. `--no-perceptual` là opt-out tường minh. `--config` cho phép đổi bộ dữ liệu cross-domain nhưng frontend/model vẫn lấy từ checkpoint.

`--diagnostics-dir runs/.../diagnostics` lưu DoA, forgetting, alpha, uncertainty, covariance error oracle, innovation và distortionless error theo cùng trục thời gian. `--audio-dir` ghi WAV output. Vẽ diagnostics đã lưu:

```bash
python -m scripts.plot_diagnostics --input runs/lrs2/diagnostics/utterance.npz --output runs/lrs2/memory_diagnostics.png
```

## Ablation và seed

`--ablation A0` đến `A10` và `B00/B01/B10/B11` triển khai đúng các trục memory/forgetting/physics/residual/uncertainty/refiner trong tài liệu. A9 cố ý sử dụng phần native matrix memory làm pseudo-SCM, không sửa nó thành PSD để che giấu tính không hợp lệ. A10 và factorial LSTM tự tìm hidden width để khớp tham số controller mLSTM trong sai lệch tối đa 5%; receipt chứa số đo thực tế.

```bash
python -m scripts.run_experiments --config configs/lrs2.yaml --output runs/research --variants A0 A3 A8 A10 --seeds 0 1 2 3 4
```

Lệnh mặc định chỉ in lệnh cần chạy; thêm `--execute` để thực sự huấn luyện tuần tự. Dùng `--variants B00 B01 B10 B11` cho factorial memory/forgetting. Số tham số được đo, không thay bằng budget 6–10 M dự kiến trong tài liệu:

```bash
python -m scripts.count_parameters --config configs/lrs2.yaml --device cpu
```

Recipe 3 microphone, `D=96`, 2 memory block và 2 TF block có 1.168.369 tham số; vẫn giữ toàn bộ các nhánh kiến trúc. Các con số component trong đặc tả là budget, không phải số tham số suy ra bắt buộc từ các kích thước này.

Hai cấu hình `configs/lrs2_capacity.yaml` và `configs/librispeech_capacity.yaml` đáp ứng budget paper: **7.180.289 tham số**, gồm encoder 1.200.128, controller 2.326.036 và refiner 3.553.634. Chúng dùng `D=128`, 2 memory block, 4 TF block và FFN lớn hơn; có thể chọn cho thí nghiệm chính khi đủ tài nguyên, vẫn giữ nguyên pipeline/constraints:

```bash
python -m scripts.train --config configs/lrs2_capacity.yaml --output runs/lrs2_capacity --device cuda
python -m scripts.train --config configs/librispeech_capacity.yaml --output runs/librispeech_capacity --device cuda
```

Factorial giữ learned update strength trong cả bốn ô; chỉ thay loại contextual memory và việc học forgetting. Nhờ vậy các ô fixed-forgetting vẫn thực sự sử dụng contextual memory để điều khiển update, thay vì có một backbone không ảnh hưởng waveform.

SpatialNet, online SpatialNet, ADL-MVDR và các implementation attention-SCM bên ngoài phải dùng upstream thật và checkpoint thật. `cm_xbeam/baselines/external.py` cung cấp factory adapter, kiểm tra exact Git SHA và checkout sạch; `scripts/evaluate_baseline.py` đánh giá bằng cùng source metrics và format kết quả. Cấu hình adapter phải khai báo `factory` dạng `module:function`, `source_directory`, `expected_commit` đủ 40 ký tự, `parameters`, `seed`, `name`; factory tải model/checkpoint upstream và trả về module nhận waveform `[B,M,N]`, trả `[B,K,N]`. Không gán nhãn một model tự viết là bản tái tạo các baseline đó. Chúng chưa được tải/huấn luyện/đánh giá trong lần triển khai này.

## Mô phỏng chuyển động

```bash
python -m scripts.generate_dynamic_sms --config configs/lrs2.yaml --split train --output generated/dynamic_lrs2 --num-examples 1000
python -m scripts.generate_dynamic_sms --config configs/lrs2.yaml --split valid --output generated/dynamic_lrs2 --num-examples 200
python -m scripts.generate_dynamic_sms --config configs/lrs2.yaml --split test --output generated/dynamic_lrs2 --num-examples 200
```

Tên script giữ theo đặc tả, nhưng khi dùng config LRS2 thì corpus sinh ra là **Dynamic-LRS2**, không bị gọi nhầm là SMS-WSJ. Có stationary, motion 5/15/30°/s, jump 30/60/90°, crossing, cả hai nguồn chuyển động, target silence và approach với khoảng cách góc tối thiểu 10/20/40°. Crossing thật có minimum azimuth gần 0; `approach` giữ minimum separation dương. Metadata ghi cả giá trị nominal lẫn minimum thực tế, không giả định crossing thật vẫn có góc tối thiểu dương.

Dùng image-source RIR thật của pyroomacoustics, nội suy theo thời điểm phát của source; tail phản xạ tiếp tục sau từng block. Jump dùng RIR piecewise constant ở đúng điểm nhảy. Random room/T60/SNR/distance/microphone jitter, seed và split nguồn được lưu lại. Không dùng delay-and-gain làm giả RIR reverberant. `--max-order` là truncation tường minh để người chạy lựa chọn; mặc định dùng gợi ý inverse-Sabine. Sinh đủ benchmark động và train nhiều seed cần thời gian/tài nguyên riêng, không diễn ra tự động.

```bash
python -m scripts.evaluate_motion --inputs runs/dynamic/evaluation.jsonl --group-by angular_velocity --output runs/dynamic/motion_results.json
python -m scripts.evaluate_doa_error --checkpoint runs/lrs2_extraction/best.pt --output runs/doa --condition error --values 0 5 10 20 30 --execute
```

Các sweep khác: `bias`, `jitter`, `delay` (giây), `dropout` (tỷ lệ frame). Corruption seed được cố định theo utterance ID và seed thí nghiệm, không phụ thuộc thứ tự DataLoader.

## Thống kê và runtime

```bash
python -m scripts.aggregate_seeds --inputs runs/seed0/evaluation.jsonl runs/seed1/evaluation.jsonl --output runs/seed_statistics.json
python -m scripts.compare_systems --pair H1 runs/model/evaluation.jsonl runs/ema/evaluation.jsonl --pair H2 runs/mlstm/evaluation.jsonl runs/lstm/evaluation.jsonl --pair H3 runs/constrained/evaluation.jsonl runs/unconstrained/evaluation.jsonl --output runs/paired_comparisons.json
python -m scripts.profile --checkpoint runs/lrs2/best.pt --device cuda --duration 10 --warmup 5 --repetitions 100 --output runs/lrs2/runtime.json
```

Danh sách file trong mỗi hệ thống của `--pair` có thể phân tách bằng dấu phẩy để gộp đủ seed. Statistical API giữ paired IDs/source/seed, trung bình source trong utterance trước khi bootstrap, có hierarchical bootstrap, paired utterance bootstrap 10.000 mẫu, sign-flip và Holm-Bonferroni. Thiếu cặp hoặc metric bị thiếu bất đối xứng làm lệnh báo lỗi.

Runtime báo hardware, precision, batch size, median/p95, RTF, GPU peak allocation/CPU process RSS và đã bao gồm STFT/iSTFT. WOLA lookahead là 24–32 ms, maximum packet latency 32 ms cho cả recipe 16 kHz và 8 kHz. Có thể đo `--precision bf16` trên CUDA. `--flops` chỉ báo operator FLOPs profiler hỗ trợ, không tự gọi phần đếm thiếu FFT/complex solver là MAC/s đầy đủ.

Mặc định dùng `execution_backend=chunkwise`: convolution/FFN tính cả chunk causal, mLSTM dùng backend chunkwise chính thức của NX-AI với state được quy đổi đúng giữa key-scaling/query-scaling, SCM dùng associative affine scan không chia cho cumulative product, MVDR và uncertainty tính theo batch. Chỉ materialize SCM của chunk tối đa 32 frame, không giữ toàn bộ SCM của utterance dài. Các nhánh residual/uncertainty chưa được dùng trong loss stage A/B được bỏ qua khi không yêu cầu diagnostics. Stage D có checkpoint thêm từng TF block để giảm peak VRAM khi recompute; số layer, dim, FFT, crop và batch không bị giảm.

Đường `--execution-backend framewise` giữ lại phép tính từng frame làm reference; A9 raw-memory tự dùng đường này. Backend thực thi được lưu riêng trong checkpoint, có thể chuyển khi resume mà không đổi kiến trúc hoặc parameter keys. Do đổi thứ tự phép tính và tích lũy matrix memory ở FP32, không tuyên bố bit-identical ở FP32 hay BF16; waveform/loss/gradient đã được đối chiếu cùng trọng số và dữ liệu, cùng kiểm tra causal/PSD/distortionless. Trên mẫu GPU full-capacity đối chiếu cả A–D ở FP32, sai số tương đối waveform tối đa khoảng 0,013%, gradient khoảng 0,13%; MVDR gần singular có thể khuếch đại sai số làm tròn rất nhỏ ở mask/SCM. Đây là kiểm chứng số học, không phải kết quả chất lượng sau huấn luyện.

Benchmark stage A trên NVIDIA A30, PyTorch 2.6.0+cu124, BF16, batch 2, LRS2 2 giây/250 frame, 1 warm-up và 3 lần đo: median forward+loss+backward giảm từ 12,09 s xuống 1,73 s (khoảng 7 lần), peak allocated VRAM từ khoảng 10,9 GB xuống 5,3 GB. Đây là phép đo batch cố định, chưa gồm optimizer/accumulation và máy đang chia sẻ GPU; tốc độ thực tế có thể dao động. Stage D toàn bộ 4 TF block cũng đã chạy batch 2 đoạn 2 giây với gradient hữu hạn; một lần đo cold khoảng 11,2 s và peak khoảng 10,7 GB, không được gán tốc độ stage A cho stage D.

Đo lại trên máy và dữ liệu hiện tại bằng:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.benchmark_training --config configs/lrs2_capacity.yaml --device cuda --stage A --warmup 1 --repetitions 3
```

Đây là implementation batched/chunkwise bằng PyTorch và backend mLSTM chính thức, chưa được tuyên bố là một fused CUDA kernel toàn pipeline hoặc real-time trước khi đo trên hardware đích.
