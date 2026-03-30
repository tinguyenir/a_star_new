2. Kiểm tra Môi trường & GPU

Để quá trình train diễn ra suôn sẻ, hãy đảm bảo máy tính của bạn đã cài đặt đầy đủ Python, PyTorch và nhận diện được GPU.

Kiểm tra Python và GPU NVIDIA:
Bash

python3 --version
nvidia-smi

(Lệnh nvidia-smi sẽ hiển thị thông tin card nếu bạn có GPU và driver đang hoạt động).

Kiểm tra PyTorch có nhận CUDA không:
Bash

python3 - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

    Lưu ý: Nếu lệnh import torch bị lỗi, nghĩa là máy chưa có PyTorch. Bạn cần cài đặt môi trường (pip/conda) và cài đặt PyTorch chuẩn theo hướng dẫn chính thức (chọn đúng phiên bản Linux, pip/conda và CUDA phù hợp với máy). Sau đó chạy lại đoạn kiểm tra trên.

3. Kiểm tra Mã nguồn và Dữ liệu (Dataset)

Đảm bảo bạn có đủ file chạy và bộ dữ liệu trước khi train.

Kiểm tra thư mục repo:
Bash

cd ~/a_star_new
ls

Yêu cầu: Bạn phải thấy ít nhất file train_policy_v3.py và thư mục policy_dataset_v3.

Kiểm tra chi tiết dataset:
Bash

find policy_dataset_v3 -maxdepth 2 | head -50

Yêu cầu: Bạn phải thấy các thư mục con phân chia dữ liệu:

    policy_dataset_v3/train

    policy_dataset_v3/val

    policy_dataset_v3/test

Nếu muốn chắc chắn dữ liệu không bị trống:
Bash

ls policy_dataset_v3/train | head
ls policy_dataset_v3/val | head
ls policy_dataset_v3/test | head

(Nếu các thư mục này không có file .npz, dataset của bạn chưa đầy đủ và cần được bổ sung).
4. Chạy Huấn luyện (Training)

Tạo thư mục để lưu mô hình (model weights) và bắt đầu train.
Lệnh Train Chuẩn (Chạy ngay)
Bash

cd ~/a_star_new
mkdir -p model_v3

python3 train_policy_v3.py \
  --dataset_root "$(pwd)/policy_dataset_v3" \
  --save_dir "$(pwd)/model_v3" \
  --batch_size 64 \
  --epochs 40 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --base_ch 32 \
  --dropout_p 0.10 \
  --device cuda \
  --amp \
  --enable_augmentation \
  --aug_rot90 \
  --aug_hflip \
  --no-aug_vflip \
  --eval_rollout_max_factor 6 \
  --loop_fail_visit_count 2 \
  --eval_rollout_samples_per_batch 4 \
  --early_stop_patience 6 \
  --min_epochs_before_stop 10

5. Kiểm tra Kết quả sau khi Train

Sau khi quá trình train hoàn tất, hãy kiểm tra xem mô hình đã được lưu thành công chưa:
Bash

cd ~/a_star_new
ls -lh model_v3
python3 - <<'PY'
import os
print("best.pt exists:", os.path.exists("model_v3/best.pt"))
print("last.pt exists:", os.path.exists("model_v3/last.pt"))
PY

(Nếu best.pt exists: True thì quá trình train đã thành công tốt đẹp).
6. Xử lý Lỗi Thường Gặp (Troubleshooting)
Lỗi: CUDA Out Of Memory (Tràn RAM GPU)

Nguyên nhân do GPU không đủ VRAM để chứa batch size lớn. Hãy giảm tham số --batch_size xuống 32 (thay vì 64):
Bash

cd ~/a_star_new
python3 train_policy_v3.py \
  --dataset_root "$(pwd)/policy_dataset_v3" \
  --save_dir "$(pwd)/model_v3" \
  --batch_size 32 \
  --epochs 40 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --base_ch 32 \
  --dropout_p 0.10 \
  --device cuda \
  --amp \
  --enable_augmentation \
  --aug_rot90 \
  --aug_hflip \
  --no-aug_vflip \
  --eval_rollout_max_factor 6 \
  --loop_fail_visit_count 2 \
  --eval_rollout_samples_per_batch 4 \
  --early_stop_patience 6 \
  --min_epochs_before_stop 10

Nếu vẫn tiếp tục thiếu VRAM, hãy thử giảm tiếp --batch_size 16.
Muốn Train lại từ đầu

Nếu bạn muốn hủy bỏ kết quả cũ và train lại một model hoàn toàn mới, hãy xóa thư mục model cũ:
Bash

cd ~/a_star_new
rm -rf model_v3
mkdir -p model_v3

Sau đó chạy lại lệnh train ở mục 4.


