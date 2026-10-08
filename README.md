# Dự án: Hệ thống AI trên FPGA hỗ trợ người mù nhận diện đối tượng trên đường

> Kiến trúc phần cứng lấy cảm hứng từ **openTPU** (FeSens) — một repo gộp RTL, ISA, simulator, compiler và profiler, chạy LLM trên card PCIe Kintex-7. Dự án này giữ nguyên triết lý "một repo, một chuỗi công cụ" nhưng thu gọn cho bài toán **phân loại ảnh INT8** trên thiết bị đeo.

## 1. Tổng quan

### 1.1 Mục tiêu
Xây dựng một thiết bị đeo (gắn ngực hoặc kính) dùng camera + FPGA để **phân loại vật thể/tình huống phía trước** người khiếm thị và phản hồi bằng âm thanh hoặc rung, với độ trễ thấp và hoạt động hoàn toàn offline.

### 1.2 Phạm vi
- **Bài toán:** phân loại ảnh (image classification) theo vùng nhìn, *không* làm phát hiện/khoanh vùng đầy đủ như mô hình gốc.
- **Mô hình thầy (teacher):** NVIDIA LocateAnything-3B — chỉ dùng **offline trên máy chủ** để gán nhãn và chưng cất tri thức, không chạy trên FPGA.
- **Mô hình trò (student):** CNN nhỏ (khoảng 1–3 triệu tham số), lượng tử hóa INT8, chạy trên bộ tăng tốc kiểu openTPU.

### 1.3 Tập lớp mục tiêu (phiên bản 1)

| ID | Lớp | Mức ưu tiên cảnh báo |
|---|---|---|
| 0 | Đường trống / an toàn | Thấp |
| 1 | Người đi bộ | Trung bình |
| 2 | Xe máy | Cao |
| 3 | Ô tô / xe tải | Cao |
| 4 | Xe đạp | Trung bình |
| 5 | Vạch qua đường | Thông tin |
| 6 | Đèn tín hiệu — đỏ | Cao |
| 7 | Đèn tín hiệu — xanh | Thông tin |
| 8 | Bậc thềm / bậc thang | Cao |
| 9 | Cột / cây / vật cản đứng | Cao |
| 10 | Hố / mặt đường hỏng | Cao |
| 11 | Xe đỗ trên vỉa hè | Trung bình |

Ảnh đầu vào được chia thành **3 vùng (trái – giữa – phải)**, mỗi vùng phân loại riêng, nhờ đó thiết bị vẫn nói được *hướng* của vật cản mà không cần bộ phát hiện đầy đủ.

### 1.4 Phần cứng đích
- **Bo phát triển chính:** AMD/Xilinx **Kria KV260** (Zynq UltraScale+ MPSoC) — có lõi ARM để chạy Linux, camera MIPI và phản hồi âm thanh; phần PL chứa bộ tăng tốc.
- **Phương án giá rẻ:** **Zynq-7020** (PYNQ-Z2) với mảng tính toán thu nhỏ.
- **Lý do không dùng Kintex-7 như openTPU:** card PCIe cần máy chủ, không phù hợp thiết bị đeo; ta chỉ kế thừa kiến trúc và chuỗi công cụ.

### 1.5 Chỉ tiêu kỹ thuật

| Chỉ tiêu | Mục tiêu |
|---|---|
| Độ trễ từ khung hình đến cảnh báo | ≤ 100 ms |
| Tốc độ suy luận | ≥ 15 khung/giây (3 vùng/khung) |
| Công suất toàn hệ | ≤ 5 W |
| Độ chính xác top-1 (INT8) | ≥ 90% và giảm ≤ 1 điểm so với FP32 |
| Recall các lớp ưu tiên Cao | ≥ 95% (ưu tiên không bỏ sót hơn là báo nhầm) |

### 1.6 Lưu ý giấy phép
LocateAnything-3B dùng giấy phép **phi thương mại** của NVIDIA (chỉ nghiên cứu học thuật/phi lợi nhuận). Nhãn giả và mô hình trò sinh ra từ nó nên được coi là chịu cùng ràng buộc; nếu muốn thương mại hóa, cần gán nhãn lại bằng mô hình/dữ liệu có giấy phép mở.

## 2. Cấu trúc repo

Theo triết lý của openTPU: một repo chứa toàn bộ chuỗi từ mô hình đến bitstream, có mô hình tham chiếu (golden model) để kiểm tra khớp từng bit ở mọi tầng.

blind-assist-tpu/
├── README.md
├── LICENSE
├── docs/
│   ├── isa.md                  # Đặc tả tập lệnh
│   ├── memory_map.md           # Bản đồ thanh ghi AXI-Lite, bố cục bộ nhớ
│   └── quantization.md         # Quy ước lượng tử hóa INT8
│
├── model/                      # Giai đoạn 1 — Python / PyTorch
│   ├── data/
│   │   ├── collect/            # Script thu video đường phố, cắt khung
│   │   ├── pseudo_label.py     # Gọi LocateAnything-3B → nhãn giả
│   │   └── build_dataset.py    # Chia vùng trái/giữa/phải, cân bằng lớp
│   ├── teacher/
│   │   └── locate_anything_client.py
│   ├── student/
│   │   ├── arch.py             # Kiến trúc CNN thân thiện phần cứng
│   │   ├── train_kd.py         # Huấn luyện chưng cất
│   │   └── qat.py              # Huấn luyện nhận biết lượng tử hóa
│   ├── export/
│   │   ├── to_onnx.py
│   │   └── to_npz.py           # Trọng số INT8 + scale cho compiler
│   └── eval/
│       └── metrics.py          # Accuracy, recall theo lớp, confusion matrix
│
├── compiler/                   # Giai đoạn 2 — Python
│   ├── frontend_onnx.py        # ONNX → IR
│   ├── passes/                 # Gộp BN, gộp ReLU, chia tile
│   ├── scheduler.py            # Lập lịch tile vào buffer on-chip
│   └── codegen.py              # IR → chuỗi lệnh nhị phân
│
├── sim/                        # Giai đoạn 2 — C++
│   ├── golden/                 # Mô hình tham chiếu số nguyên, khớp bit với RTL
│   ├── isa_sim/                # Simulator mức lệnh, đếm chu kỳ
│   └── profiler/               # Báo cáo sử dụng MAC, băng thông DDR
│
├── rtl/                        # Giai đoạn 3 — SystemVerilog
│   ├── pkg/tpu_pkg.sv
│   ├── core/
│   │   ├── systolic_array.sv
│   │   ├── pe.sv
│   │   ├── requant.sv          # Nhân scale + dịch + bão hòa INT8
│   │   ├── activation.sv
│   │   ├── pool.sv
│   │   └── controller.sv       # Giải mã lệnh, điều phối
│   ├── mem/
│   │   ├── weight_buf.sv
│   │   ├── act_buf.sv          # Ping-pong
│   │   └── axi_dma.sv
│   ├── top/tpu_top.sv          # AXI-Lite (cấu hình) + AXI4 (dữ liệu)
│   └── tb/                     # Testbench, cocotb hoặc Verilator
│
├── fpga/
│   ├── kv260/                  # Block design Vivado, ràng buộc, script build
│   └── pynq_z2/
│
├── firmware/                   # C++ chạy trên ARM (Linux)
│   ├── camera/                 # V4L2 / MIPI
│   ├── driver/                 # Điều khiển TPU qua UIO/XRT
│   ├── app/                    # Vòng lặp chính, lọc thời gian, ưu tiên cảnh báo
│   └── feedback/               # TTS tiếng Việt, rung
│
└── tests/
    ├── test_model_vs_golden.py
    ├── test_sim_vs_rtl.py
    └── test_e2e_board.py

## 3. Giai đoạn 1 — Dữ liệu và mô hình (Python)

### 3.1 Thu thập dữ liệu
- Quay video từ camera đặt ở **độ cao ngực/mắt người đi bộ**, đúng góc của thiết bị thật; đa dạng giờ (sáng, trưa, tối, mưa) và loại đường (vỉa hè, ngõ, giao lộ).
- Bổ sung dữ liệu công khai: Cityscapes, Mapillary Vistas, BDD100K (lấy khung nhìn phù hợp người đi bộ).
- Cắt khung ở 2–5 fps để tránh trùng lặp; giữ lại metadata thời gian để chia train/val **theo video**, không theo khung, tránh rò rỉ dữ liệu.

### 3.2 Gán nhãn giả bằng LocateAnything-3B
1. Chạy mô hình thầy (bản PyTorch trên GPU, hoặc `locate-anything.cpp` trên máy CPU) với các câu truy vấn mở, ví dụ: *"motorbike"*, *"red traffic light"*, *"stairs"*, *"pothole"*, *"pole or tree trunk"*.
2. Nhận về các hộp giới hạn kèm độ tin cậy.
3. Chiếu hộp lên lưới 3 vùng: một vùng mang nhãn của vật thể **ưu tiên cao nhất và chiếm diện tích đủ lớn** (ví dụ ≥ 8% diện tích vùng); không có gì → lớp 0.
4. Lưu cả **phân phối mềm** trên 12 lớp (chuẩn hóa từ độ tin cậy) để dùng cho chưng cất.
5. Kiểm tra thủ công ngẫu nhiên khoảng 5% mẫu, và **gán nhãn tay toàn bộ tập test** để đánh giá không bị thiên lệch theo mô hình thầy.

### 3.3 Kiến trúc mô hình trò
Thiết kế cho phần cứng ngay từ đầu:
- Nền tảng: MobileNetV2/V3-Small rút gọn hoặc CNN tùy biến; **chỉ dùng conv 3×3, conv 1×1, depthwise 3×3, ReLU/ReLU6, max/avg pool, FC** — đúng tập phép toán mà RTL hỗ trợ.
- Tránh: Swish/h-swish, Squeeze-Excitation, LayerNorm, attention — tốn tài nguyên hoặc khó lượng tử hóa.
- Đầu vào **128×128** mỗi vùng (3 vùng ghép batch), số kênh là bội số của 16 để khớp mảng systolic 16×16.
- Ngân sách: ≤ 3 triệu tham số (≤ 3 MB INT8, vừa DDR và có thể cache một phần trên BRAM/URAM), ≤ 150 MMAC/vùng.

### 3.4 Chưng cất tri thức
Hàm mất mát kết hợp nhãn cứng $$y$$ và phân phối mềm của thầy $$p_T$$:

$$
\mathcal{L} = \alpha \, \mathrm{CE}(y, p_S) + (1-\alpha)\, \tau^2 \, \mathrm{KL}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right)
$$

với $$\alpha \approx 0.5$$, nhiệt độ $$\tau \approx 4$$. Thêm **trọng số lớp** cho các lớp ưu tiên Cao để đẩy recall, và tăng cường dữ liệu: thay đổi độ sáng, mờ chuyển động, mưa giả, lệch góc camera ±15°.

### 3.5 Lượng tử hóa INT8
Quy ước (ghi trong `docs/quantization.md` và phải giống hệt ở golden model và RTL):
- **Trọng số:** INT8 đối xứng, **theo từng kênh ra** (per-channel), zero-point = 0.
- **Activation:** INT8 theo từng tensor; sau ReLU dùng dải không dấu [0, 255] (UINT8) để không phí một bit.
- **Bias:** INT32.
- **Tích lũy:** INT32 trong PE.
- **Requantize:** thay scale thực bằng số nguyên nhân + dịch phải, $$M = M_0 \cdot 2^{-n}$$ với $$M_0$$ là INT32, làm tròn round-half-up, sau đó bão hòa.
- BatchNorm được **gộp vào conv** trước khi lượng tử hóa.

Quy trình: huấn luyện FP32 → lượng tử hóa sau huấn luyện (PTQ) để có mốc → **QAT** 10–20 epoch nếu tụt quá 1 điểm. 4-bit để dành cho phiên bản sau, khi INT8 đã chạy ổn.

### 3.6 Xuất mô hình
- `to_onnx.py`: xuất ONNX dạng QDQ để kiểm tra bằng ONNX Runtime.
- `to_npz.py`: xuất trọng số INT8, bias INT32, cặp $$(M_0, n)$$ cho từng lớp — đây là đầu vào của compiler.
- **Tiêu chí hoàn thành:** mô hình số nguyên thuần (NumPy) cho kết quả khớp ONNX Runtime trên toàn bộ tập test.

## 4. Giai đoạn 2 — ISA, compiler và simulator

### 4.1 Tập lệnh (ISA) tối giản
Lệnh 64 bit, thực thi tuần tự, đồng bộ bằng cờ phụ thuộc giữa các khối DMA và tính toán (giống cách openTPU tách nạp dữ liệu và tính toán).

| Lệnh | Chức năng |
|---|---|
| `LOAD_W` | DMA trọng số từ DDR → weight buffer |
| `LOAD_A` | DMA activation từ DDR → activation buffer |
| `CONV` | Conv 1×1 / 3×3 trên một tile, tích lũy INT32 |
| `DWCONV` | Depthwise 3×3 |
| `FC` | Lớp kết nối đầy đủ (conv 1×1 trên đầu vào 1×1) |
| `POOL` | Max hoặc average pool |
| `REQUANT` | Cộng bias, nhân $$M_0$$, dịch $$n$$, ReLU, bão hòa |
| `STORE` | Activation buffer → DDR |
| `SYNC` | Chờ các thao tác trước hoàn tất |
| `HALT` | Kết thúc, phát ngắt cho ARM |

### 4.2 Compiler (Python)
1. **Frontend:** đọc ONNX/NPZ → đồ thị IR.
2. **Pass tối ưu:** gộp conv + bias + ReLU + requant thành một nút; loại bỏ reshape thừa.
3. **Chia tile:** chọn kích thước tile theo dung lượng buffer, ưu tiên giữ trọng số trên chip (weight-stationary) vì mô hình nhỏ.
4. **Lập lịch:** xen kẽ `LOAD` của tile kế tiếp với `CONV` của tile hiện tại (ping-pong).
5. **Codegen:** sinh file nhị phân lệnh + ảnh bộ nhớ trọng số.

### 4.3 Golden model và simulator (C++)
- **Golden model:** cài đặt các phép toán bằng số nguyên thuần, cùng quy tắc làm tròn/bão hòa. Đây là "nguồn sự thật" cho mọi tầng.
- **ISA simulator:** chạy chương trình nhị phân, mô phỏng buffer và đếm chu kỳ để dự báo độ trễ trước khi có RTL.
- **Profiler:** báo cáo tỷ lệ sử dụng MAC, thời gian chờ DMA, băng thông DDR theo từng lớp — dùng để điều chỉnh kiến trúc mô hình ở Giai đoạn 1 nếu có lớp nghẽn.

### 4.4 Chuỗi kiểm tra
PyTorch (QAT) ──► NumPy số nguyên ──► Golden C++ ──► ISA sim ──► RTL sim ──► Bo mạch
                  khớp argmax          khớp bit       khớp bit    khớp bit    khớp bit
Mỗi mũi tên là một test tự động trong `tests/`; không chuyển tầng khi tầng trước chưa đạt.

## 5. Giai đoạn 3 — Phần cứng FPGA và tích hợp hệ thống

### 5.1 Kiến trúc bộ tăng tốc (SystemVerilog)
- **Mảng systolic 16×16 PE** (256 MAC INT8) trên KV260; bản Zynq-7020 dùng 8×8. Mỗi PE: nhân 8×8 bit, cộng dồn 32 bit; ánh xạ vào DSP48, có thể ghép **2 phép nhân INT8 vào một DSP** để tăng mật độ.
- **Khối depthwise riêng** (16 làn × 9 MAC), vì depthwise chạy kém hiệu quả trên mảng systolic.
- **Requant/activation pipeline:** bias → nhân $$M_0$$ (32×32) → dịch làm tròn → ReLU → bão hòa UINT8.
- **Bộ nhớ trên chip:** weight buffer (URAM), activation buffer ping-pong (BRAM), accumulator buffer INT32.
- **Giao tiếp:** AXI4-Lite để ARM ghi địa chỉ chương trình và khởi chạy; AXI4 master (HP port) để DMA đọc/ghi DDR; một đường ngắt báo hoàn tất.
- **Tần số mục tiêu:** 200–250 MHz trên KV260, 100–150 MHz trên Zynq-7020.

Ước tính thông lượng KV260: $$256 \times 200\,\text{MHz} = 51.2$$ GMAC/s đỉnh; với 150 MMAC/vùng × 3 vùng × 15 fps ≈ 6.75 GMAC/s, chỉ cần hiệu suất khoảng 15% là đạt chỉ tiêu — còn dư cho tiền xử lý và mở rộng mô hình.

### 5.2 Kiểm thử RTL
- Testbench từng khối (PE, mảng, requant) bằng **cocotb + Verilator**, so với golden C++.
- Test toàn mạng: chạy chương trình do compiler sinh trong mô phỏng, so từng byte activation đầu ra với ISA simulator.
- Mô phỏng sau tổng hợp trong Vivado cho đường dẫn AXI.

### 5.3 Tích hợp trên bo mạch
- **Vivado block design:** Zynq MPSoC + `tpu_top` + AXI interconnect; build bằng script Tcl trong `fpga/kv260/` để tái lập được.
- **Linux trên ARM** (PetaLinux hoặc Ubuntu cho Kria): nạp bitstream, driver qua UIO, cấp phát bộ đệm liên tục (CMA) cho DMA.
- **Pipeline firmware (C++):**
  1. Camera MIPI/USB → khung 640×480.
  2. Tiền xử lý trên ARM: chia 3 vùng, resize 128×128, lượng tử hóa đầu vào.
  3. Gọi TPU, đọc 3 vector logit.
  4. **Lọc thời gian:** chỉ cảnh báo khi một lớp xuất hiện ổn định ≥ 3/5 khung gần nhất, tránh nói liên tục.
  5. **Phản hồi:** câu nói ngắn tiếng Việt qua tai nghe dẫn truyền xương (ví dụ "Xe máy, bên trái"), hoặc rung theo hướng; lớp ưu tiên Cao được ngắt lời cảnh báo đang phát.

### 5.4 Đo đạc
Đo độ trễ đầu-cuối (camera → âm thanh), fps, công suất (qua cảm biến trên bo hoặc đồng hồ ngoài), và recall theo lớp trên video thử nghiệm thực địa.

Hướng mở rộng
- Lượng tử hóa 4-bit trọng số để tăng gấp đôi thông lượng.
- Thêm ước lượng khoảng cách (camera stereo hoặc cảm biến ToF) để cảnh báo "gần/xa".
- Chuyển từ phân loại vùng sang phát hiện nhẹ (kiểu SSD/YOLO-nano) trên cùng bộ tăng tốc.
- Thu nhỏ sang ASIC hoặc FPGA công suất thấp (Lattice, Efinix) cho sản phẩm đeo thực sự.
