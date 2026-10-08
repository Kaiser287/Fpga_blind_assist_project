# Dự án: Hệ thống AI trên FPGA hỗ trợ người mù nhận diện đối tượng trên đường

> Kiến trúc phần cứng lấy cảm hứng từ **openTPU** (FeSens) — một repo gộp RTL, ISA, simulator, compiler và profiler, chạy LLM trên card PCIe Kintex-7. Dự án này giữ nguyên triết lý "một repo, một chuỗi công cụ" nhưng thu gọn cho bài toán **phân loại ảnh INT8** trên thiết bị đeo.

## 1. Tổng quan

### 1.1 Mục tiêu
Xây dựng một thiết bị đeo (gắn ngực hoặc kính) dùng camera + FPGA để **phân loại vật thể/tình huống phía trước** người khiếm thị và phản hồi bằng âm thanh hoặc rung, với độ trễ thấp và hoạt động hoàn toàn offline.

Thiết bị có **ba chế độ**, điều khiển bằng một nút bấm vật lý (bấm ngắn = Nhận diện vật, bấm giữ 1 giây = Đọc chữ, tự quay về Đi đường):
- **Chế độ Đi đường (mặc định):** liên tục phân loại 3 vùng trái/giữa/phải; cảnh báo vật cản, phương tiện, con vật, công trình, đèn tín hiệu; đồng thời báo thông tin hữu ích như gạch dẫn hướng, nhà chờ xe buýt, cửa ra vào.
- **Chế độ Nhận diện vật:** người dùng đưa vật ra trước camera, thiết bị đọc tên **loại quả** (ví dụ "Quả cam") hoặc **mệnh giá tiền giấy** (ví dụ "Hai mươi nghìn"). Đây là bài toán phân loại một đối tượng trên cả khung hình.
- **Chế độ Đọc chữ:** đọc to chữ trước camera: biển tên đường, số tuyến trên biển nhà chờ, biển hiệu và biển quảng cáo, vé xe buýt, hạn sử dụng trên bao bì. Đây là bài toán **nhận dạng văn bản (OCR)**, khác hẳn phân loại, nên được thiết kế thành một nhánh riêng (mục 1.3d).

### 1.2 Phạm vi
- **Bài toán:** phân loại ảnh (image classification), *không* làm phát hiện/khoanh vùng đầy đủ như mô hình gốc. Một **backbone dùng chung** với **ba đầu phân loại**: đầu Đi đường (3 vùng × 22 lớp, đa nhãn), đầu Trái cây (21 lớp) và đầu Tiền giấy (10 lớp).
- **Ngoại lệ — Đọc chữ:** OCR cần phát hiện dòng chữ rồi nhận dạng ký tự, không gói được vào một đầu phân loại. Bản v1 chạy mô hình OCR nhẹ trên lõi ARM; bản v2 chuyển phần tích chập của bộ phát hiện chữ sang bộ tăng tốc.
- **Mô hình thầy (teacher):** NVIDIA LocateAnything-3B — chỉ dùng **offline trên máy chủ** để gán nhãn và chưng cất tri thức, không chạy trên FPGA.
- **Mô hình trò (student):** CNN nhỏ (khoảng 1–3 triệu tham số), lượng tử hóa INT8, chạy trên bộ tăng tốc kiểu openTPU.

### 1.3 Tập lớp Đi đường (phiên bản 2 — bổ sung tình huống đô thị)

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
| 8 | Bậc thềm / bậc thang đi lên | Cao |
| 9 | Cột / cây / vật cản đứng | Cao |
| 10 | Hố / mặt đường hỏng | Cao |
| 11 | Xe đỗ trên vỉa hè | Trung bình |
| 12 | **Con vật** (chó, mèo, trâu bò) | Trung bình |
| 13 | **Vật cản ngang đầu** (cành cây thấp, mái bạt, biển treo, gương xe tải) | Cao |
| 14 | **Bậc / mép đi xuống** (mép vỉa hè, cầu thang xuống, rìa mương) | Cao |
| 15 | **Rào chắn / công trình thi công** | Cao |
| 16 | **Hàng quán lấn vỉa hè** (bàn ghế nhựa, xe đẩy, gánh hàng rong) | Trung bình |
| 17 | **Vũng nước / đường ngập** | Trung bình |
| 18 | **Xe buýt** | Cao |
| 19 | **Nhà chờ / biển dừng xe buýt** | Thông tin |
| 20 | **Gạch dẫn hướng** (gạch nổi cho người khiếm thị) | Thông tin |
| 21 | **Cửa ra vào / lối vào tòa nhà, ga** | Thông tin |

Ảnh đầu vào được chia thành **3 vùng (trái – giữa – phải)**, mỗi vùng phân loại riêng, nhờ đó thiết bị vẫn nói được *hướng* của vật cản mà không cần bộ phát hiện đầy đủ.

**Lý do chọn các lớp mới:** đây là những thứ gậy dò đường không phát hiện được hoặc phát hiện quá muộn (vật cản ngang đầu, mép đi xuống, vũng nước, con vật di chuyển), những thứ đặc thù vỉa hè Việt Nam (hàng quán, công trình), và những mốc giúp người dùng tự đi xe buýt (nhà chờ, xe buýt đến, gạch dẫn hướng dẫn ra điểm đón). Chó, mèo và trâu bò gộp một lớp vì dữ liệu gia súc trong đô thị rất ít; tách lớp khi có đủ dữ liệu. Bậc *lên* và bậc *xuống* tách riêng vì bậc xuống nguy hiểm hơn nhiều.

**Chuyển NavHead sang đa nhãn:** một vùng có thể đồng thời có gạch dẫn hướng và xe máy. Với softmax, lớp Thông tin luôn bị lớp nguy hiểm che mất; vì vậy NavHead dùng **sigmoid độc lập cho từng lớp**, mỗi lớp có ngưỡng riêng, và firmware quyết định câu nói theo mức ưu tiên. Lớp 0 nghĩa là không lớp nào vượt ngưỡng.

### 1.3b Tập lớp Trái cây (chế độ Nhận diện vật)
Chọn các loại quả phổ biến ở chợ/siêu thị Việt Nam, ưu tiên những cặp dễ nhầm khi sờ bằng tay (cam – quýt – bưởi, nhãn – vải – chôm chôm) vì đây là chỗ người khiếm thị cần hỗ trợ nhất.

| ID | Lớp | ID | Lớp |
|---|---|---|---|
| 0 | **Không phải trái cây / không chắc chắn** | 11 | Thanh long |
| 1 | Chuối | 12 | Dứa (thơm) |
| 2 | Cam | 13 | Ổi |
| 3 | Quýt | 14 | Nho |
| 4 | Bưởi | 15 | Măng cụt |
| 5 | Chanh | 16 | Chôm chôm |
| 6 | Táo | 17 | Nhãn |
| 7 | Lê | 18 | Vải |
| 8 | Xoài | 19 | Mít |
| 9 | Đu đủ | 20 | Sầu riêng |
| 10 | Dưa hấu | | |

Lớp 0 rất quan trọng: thiết bị phải nói "không nhận ra" thay vì đoán bừa khi người dùng đưa cốc, điện thoại hay rau củ vào camera.

### 1.3c Tập lớp Tiền giấy (chế độ Nhận diện vật)
Trả tiền vé xe buýt, mua hàng ở chợ là nhu cầu hằng ngày, và tiền polymer Việt Nam có kích thước, chất liệu gần giống nhau nên sờ tay rất khó phân biệt.

| ID | Lớp | ID | Lớp |
|---|---|---|---|
| 0 | **Không phải tiền / không chắc chắn** | 5 | 20.000 đ |
| 1 | 1.000 đ | 6 | 50.000 đ |
| 2 | 2.000 đ | 7 | 100.000 đ |
| 3 | 5.000 đ | 8 | 200.000 đ |
| 4 | 10.000 đ | 9 | 500.000 đ |

Cặp dễ nhầm theo màu: **10.000 – 200.000** (nâu vàng – nâu đỏ) và **20.000 – 500.000** (xanh lam – xanh ngọc). Gọi sai mệnh giá gây thiệt hại thật, nên ngưỡng từ chối của đầu này đặt chặt hơn đầu Trái cây.

Trong chế độ Nhận diện vật, backbone chạy **một lần**, rồi cả FruitHead và MoneyHead cùng tính trên một vector đặc trưng (chi phí FC không đáng kể). Firmware đọc kết quả của đầu có lớp 0 thấp hơn và top-1 vượt ngưỡng; nếu cả hai đều từ chối thì nói "Không nhận ra".

### 1.3d Đọc chữ (chế độ Đọc chữ)
Biển quảng cáo, biển tên đường, vé xe buýt, số tuyến xe là **văn bản tùy ý**, không thể liệt kê thành lớp. Nhánh này gồm hai mô hình:
1. **Phát hiện dòng chữ:** mạng kiểu DBNet với backbone MobileNetV3 nhỏ, đầu vào 640×640, trả về các vùng chứa chữ.
2. **Nhận dạng dòng chữ:** mạng thuần tích chập + CTC, bộ ký tự **tiếng Việt có dấu** (khoảng 230 ký tự gồm chữ, số, dấu câu).

Hành vi khi đọc:
- **Đọc theo thứ tự ưu tiên:** dòng chữ to nhất và gần giữa khung hình trước (thường là tên cửa hàng, tên đường, số tuyến), sau đó hỏi "Đọc tiếp?" thay vì đọc hết một biển quảng cáo dài.
- **Hướng dẫn ngắm:** nếu chữ bị cắt ở mép, nói "Có chữ phía trên bên trái, xoay sang trái"; nếu chữ quá nhỏ, nói "Lại gần hơn".
- **Mẫu nhận biết:** với vé xe buýt và biển nhà chờ, tìm các mẫu như "Tuyến 08", giá vé, giờ chạy để đọc gọn: "Vé tuyến 08, bảy nghìn đồng".

Ngoài phạm vi: đọc chữ trên xe buýt **đang chạy** từ xa (v1 chỉ hỗ trợ khi xe đã dừng, cách ≤ 5 m), chữ viết tay, và nhận diện khuôn mặt (vấn đề quyền riêng tư).

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
| Top-1 trái cây (INT8, ảnh cầm tay thực tế) | ≥ 92% |
| Tỷ lệ gọi sai tên khi vật không phải trái cây | ≤ 3% |
| Độ trễ chế độ Nhận diện vật (bấm nút → đọc tên) | ≤ 300 ms |
| Top-1 tiền giấy (INT8, cầm tay, gấp/nhàu) | ≥ 98% |
| Tỷ lệ đọc sai mệnh giá | ≤ 1% (thà nói "không chắc" còn hơn) |
| Recall lớp Thông tin (gạch dẫn hướng, nhà chờ, cửa) | ≥ 85% |
| Độ chính xác ký tự OCR trên biển hiệu chụp thực tế | ≥ 90% |
| Độ trễ Đọc chữ v1 trên ARM (bấm giữ → bắt đầu đọc) | ≤ 2,5 s (cần đo thực tế) |

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
│   │   ├── build_dataset.py    # Chia vùng trái/giữa/phải, cân bằng lớp
│   │   ├── fruit/
│   │   │   ├── capture_guide.md    # Quy trình chụp quả cầm tay, nhiều nền/ánh sáng
│   │   │   ├── crop_with_teacher.py# LocateAnything-3B khoanh quả → crop trung tâm
│   │   │   ├── negatives/          # Ảnh "không phải trái cây" (cốc, điện thoại, rau...)
│   │   │   └── build_fruit_set.py
│   │   ├── money/
│   │   │   ├── capture_guide.md    # Chụp tờ tiền gấp, nhàu, mặt trước/sau, ánh sáng yếu
│   │   │   ├── negatives/          # Vé xe, hóa đơn, tờ rơi (dễ nhầm với tiền)
│   │   │   └── build_money_set.py
│   │   └── ocr/
│   │       ├── synth_vi_text.py    # Sinh ảnh chữ tiếng Việt có dấu (biển, nhãn, vé)
│   │       └── real_samples/       # Ảnh thật: biển hiệu, vé xe buýt, nhãn hàng
│   ├── teacher/
│   │   ├── locate_anything_client.py
│   │   └── fruit_teacher.py    # Bộ phân loại lớn fine-tune trên tập trái cây (thầy cho KD)
│   ├── student/
│   │   ├── arch.py             # Backbone CNN dùng chung, thân thiện phần cứng
│   │   ├── heads.py            # NavHead (3×22, sigmoid), FruitHead (21), MoneyHead (10)
│   │   ├── ocr/                # Phát hiện dòng chữ + nhận dạng CRNN/CTC tiếng Việt
│   │   ├── train_kd.py         # Huấn luyện chưng cất đa nhiệm
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
│   ├── app/                    # Vòng lặp chính, chuyển chế độ, lọc thời gian, ưu tiên cảnh báo
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

### 3.2b Dữ liệu trái cây
- **Nhãn gần như miễn phí:** chụp theo từng loại quả (mua một túi cam → chụp 300–500 ảnh cam), nên nhãn lớp đã biết sẵn; không cần mô hình thầy để đặt tên.
- **Chụp đúng điều kiện sử dụng:** quả **cầm trên tay**, trên bàn, trong túi nilon, trên mâm ở chợ; ánh sáng trong nhà, đèn vàng, ngoài trời; khoảng cách 20–60 cm; góc lệch, mờ nhẹ do tay run. Mỗi lớp nên có ≥ 5 quả khác nhau và ≥ 3 độ chín.
- **Dữ liệu công khai** như Fruits-360 chỉ dùng tiền huấn luyện: ảnh nền trắng studio khác xa thực tế, nếu dựa vào đó mô hình sẽ hỏng khi gặp ảnh thật.
- **Vai trò LocateAnything-3B:** chạy truy vấn "fruit in hand" để khoanh hộp quả, rồi crop sát quả làm tăng cường dữ liệu và kiểm tra ảnh có thực sự chứa quả.
- **Ảnh âm (lớp 0):** ≥ 20% tập, gồm đồ vật thường gặp (cốc, chai, điện thoại, bóng, bàn tay trống) và **rau củ hình dạng giống quả** (cà chua, khoai tây, hành tây) để mô hình học cách từ chối.
- Chia train/val/test **theo từng quả vật lý**, không theo ảnh, để tránh mô hình học thuộc một quả cụ thể.

### 3.3 Kiến trúc mô hình trò
Thiết kế cho phần cứng ngay từ đầu:
- Nền tảng: MobileNetV2/V3-Small rút gọn hoặc CNN tùy biến; **chỉ dùng conv 3×3, conv 1×1, depthwise 3×3, ReLU/ReLU6, max/avg pool, FC** — đúng tập phép toán mà RTL hỗ trợ.
- Tránh: Swish/h-swish, Squeeze-Excitation, LayerNorm, attention — tốn tài nguyên hoặc khó lượng tử hóa.
- Đầu vào **128×128** mỗi vùng (3 vùng ghép batch), số kênh là bội số của 16 để khớp mảng systolic 16×16.
- Ngân sách: ≤ 3 triệu tham số (≤ 3 MB INT8, vừa DDR và có thể cache một phần trên BRAM/URAM), ≤ 150 MMAC/vùng.

**Ba đầu phân loại trên một backbone (OCR là mô hình riêng):**
                         ┌─► NavHead:   GAP → FC(C→22), sigmoid × 3 vùng (128×128 mỗi vùng)
Ảnh → Backbone CNN chung ┼─► FruitHead: GAP → FC(C→21), softmax (crop giữa 160×160)
                         └─► MoneyHead: GAP → FC(C→10), softmax (cùng crop giữa)
- Backbone hoàn toàn tích chập nên nhận được cả 128×128 và 160×160; chế độ Trái cây dùng **crop giữa 160×160** để giữ chi tiết vỏ (cam vs quýt, nhãn vs vải). Chi phí ≈ 235 MMAC/lần, chỉ chạy khi bấm nút.
- FruitHead chỉ thêm khoảng $$C \times 21$$ tham số (vài chục KB), gần như không tốn bộ nhớ.
- Có thể chèn thêm một khối conv 1×1 riêng trước FruitHead nếu độ chính xác chưa đạt, vì đặc trưng màu/vân vỏ quả khác với đặc trưng cảnh đường phố.

### 3.4 Chưng cất tri thức
Hàm mất mát kết hợp nhãn cứng $$y$$ và phân phối mềm của thầy $$p_T$$:

$$
\mathcal{L} = \alpha \, \mathrm{CE}(y, p_S) + (1-\alpha)\, \tau^2 \, \mathrm{KL}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right)
$$

với $$\alpha \approx 0.5$$, nhiệt độ $$\tau \approx 4$$. Thêm **trọng số lớp** cho các lớp ưu tiên Cao để đẩy recall, và tăng cường dữ liệu: thay đổi độ sáng, mờ chuyển động, mưa giả, lệch góc camera ±15°.

**Thầy cho nhánh Trái cây:** LocateAnything-3B là mô hình định vị, không phải bộ phân loại tinh, nên không dùng làm nguồn phân phối mềm cho trái cây. Thay vào đó fine-tune một bộ phân loại lớn (ví dụ ConvNeXt-Base hoặc ViT-B tiền huấn luyện ImageNet) trên tập trái cây, rồi chưng cất sang FruitHead bằng cùng công thức trên. Phân phối mềm của thầy dạy cho trò biết "quýt giống cam hơn giống chuối", giúp trò ít nhầm lẫn thô.

**Huấn luyện đa nhiệm:** mỗi ảnh chỉ có nhãn của một nhiệm vụ, nên xen kẽ batch Đi đường và batch Trái cây, tổng mất mát:

$$
\mathcal{L}_{tổng} = \mathcal{L}_{nav}^{BCE} + \lambda_1 \, \mathcal{L}_{fruit} + \lambda_2 \, \mathcal{L}_{money}, \quad \lambda_1 \approx 1, \; \lambda_2 \approx 0.5
$$

NavHead dùng binary cross-entropy có trọng số lớp (lớp hiếm như bậc xuống, vật cản ngang đầu được tăng trọng số); ngưỡng từng lớp được chọn trên tập val sao cho recall lớp Nguy hiểm ≥ 95% và lớp Thông tin ≥ 85%.

Huấn luyện theo 3 bước: (1) backbone + NavHead, (2) đóng băng phần đầu backbone, huấn luyện FruitHead, (3) mở băng toàn bộ, tinh chỉnh chung với learning rate nhỏ và kiểm tra độ chính xác Đi đường **không giảm** quá 0.5 điểm.

**Từ chối khi không chắc:** ngoài lớp 0, áp ngưỡng xác suất top-1 (ví dụ < 0.6 → "không chắc chắn"), hiệu chỉnh ngưỡng trên tập val để đạt chỉ tiêu ≤ 3% gọi sai.

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
6. **Hai chương trình, một ảnh trọng số:** compiler sinh `nav.bin` (3 vùng 128×128 → NavHead) và `object.bin` (160×160 → FruitHead + MoneyHead chạy nối tiếp trên cùng đặc trưng GAP). Trọng số backbone nằm một chỗ trong DDR, hai chương trình chỉ khác kích thước tile và lớp FC cuối; sigmoid của NavHead tính trên ARM. **ISA không cần lệnh mới.** OCR v1 chạy hoàn toàn trên ARM; v2 có thể biên dịch phần tích chập của bộ nhận dạng thành `ocr.bin`.

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
- **Chế độ Nhận diện vật (khi bấm nút):**
  1. ARM nạp con trỏ chương trình `fruit.bin` (không cần nạp lại bitstream hay trọng số).
  2. Chụp 5 khung liên tiếp, crop giữa 160×160, chạy suy luận từng khung.
  3. **Bỏ phiếu đa số + trung bình xác suất** trên 5 khung để chống rung tay.
  4. Đọc kết quả: "Quả cam", hoặc khi xác suất trung bình thấp: "Không chắc chắn, có thể là cam hoặc quýt, hãy xoay quả và thử lại".
  5. Nếu khung hình quá tối/mờ (đo độ sáng và độ nét trên ARM) thì nhắc "Đưa quả lại gần hơn" thay vì đoán.
  6. Tự động quay về chế độ Đi đường sau 10 giây, kèm tiếng bíp báo hiệu; chế độ Đi đường **vẫn chạy cảnh báo ưu tiên Cao xen kẽ** nếu người dùng đang đứng ngoài đường.

### 5.4 Đo đạc
Đo độ trễ đầu-cuối (camera → âm thanh), fps, công suất (qua cảm biến trên bo hoặc đồng hồ ngoài), và recall theo lớp trên video thử nghiệm thực địa.

## 6. Lộ trình và rủi ro

### 6.1 Rủi ro chính

| Rủi ro | Giảm thiểu |
|---|---|
| Nhãn giả từ mô hình thầy sai với vật thể đặc thù Việt Nam (xe máy dày đặc, hàng quán lấn vỉa hè) | Kiểm tra thủ công, bổ sung nhãn tay cho các lớp yếu |
| Phân loại theo 3 vùng bỏ sót vật thể nhỏ/xa | Chấp nhận ở v1 (vật thể xa ít nguy hiểm tức thời); v2 nâng lên lưới 3×2 hoặc bộ phát hiện nhẹ |
| Lệch số học giữa các tầng | Một tài liệu quy ước lượng tử hóa duy nhất, test khớp bit tự động |
| Không đạt timing ở 200 MHz | Thêm tầng pipeline trong PE/requant, hoặc hạ xuống 150 MHz (vẫn đủ thông lượng) |
| Báo nhầm quá nhiều làm người dùng mệt mỏi | Lọc thời gian, ngưỡng tin cậy theo lớp, thử nghiệm với người dùng thật sớm |
| Ràng buộc giấy phép phi thương mại | Giữ dự án ở phạm vi nghiên cứu; kế hoạch gán nhãn lại nếu thương mại hóa |
| Nhầm các cặp quả giống nhau (cam/quýt, nhãn/vải) | Crop 160×160, thêm ảnh cận cảnh vỏ, theo dõi riêng ma trận nhầm lẫn các cặp này; khi không chắc thì đọc cả hai khả năng |
| Gọi tên quả cho vật không phải quả | Lớp 0 với ≥ 20% ảnh âm, ngưỡng từ chối hiệu chỉnh trên val |
| Thêm nhiệm vụ trái cây làm giảm độ chính xác Đi đường | Huấn luyện 3 bước, kiểm tra hồi quy Nav sau mỗi lần huấn luyện |
| Quá nhiều lớp Thông tin làm người dùng bị "ngập" thông báo | Lớp Nguy hiểm luôn ưu tiên; lớp Thông tin chỉ nói khi đổi trạng thái, tối đa 1 câu/3 giây, người dùng có thể tắt từng nhóm |
| Vật cản ngang đầu, bậc xuống, vũng nước khó thấy với camera đeo ngực | Góc camera nghiêng xuống 15–20°, tăng trọng số lớp hiếm, thu dữ liệu riêng từ góc đeo thực tế |
| Đọc sai mệnh giá tiền (hậu quả tài chính) | Chỉ nói khi 5/5 khung trùng kết quả và xác suất ≥ 0.9; ngược lại nói "lật hoặc trải phẳng tờ tiền" |
| OCR tiếng Việt sai dấu, chậm trên ARM | Dữ liệu tổng hợp có dấu, từ điển hậu xử lý; giới hạn vùng đọc ở giữa khung, mục tiêu ≤ 2,5 s |

### 6.2 Hướng mở rộng
- Lượng tử hóa 4-bit trọng số để tăng gấp đôi thông lượng.
- Mở rộng chế độ Nhận diện vật: **độ chín** (chuối xanh/chín/nẫu, xoài xanh/chín) như một đầu phân loại phụ, rồi đến **rau củ**, hộp sữa, màu quần áo — chỉ cần thêm đầu FC mới trên backbone chung.
- Tách lớp con vật thành chó/mèo và gia súc; thêm đèn tín hiệu cho người đi bộ (xanh/đỏ) khi có đủ dữ liệu.
- Đọc số tuyến trên đầu xe buýt: khi NavHead báo "xe buýt" ở vùng giữa, tự động gọi OCR cho vùng phía trên.
- Thêm ước lượng khoảng cách (camera stereo hoặc cảm biến ToF) để cảnh báo "gần/xa".
- Chuyển từ phân loại vùng sang phát hiện nhẹ (kiểu SSD/YOLO-nano) trên cùng bộ tăng tốc.
- Thu nhỏ sang ASIC hoặc FPGA công suất thấp (Lattice, Efinix) cho sản phẩm đeo thực sự.
