# Cho người bảo trì

```bash
pip install -e ".[test,all]"
pytest                                   # chạy test (test hồi quy chạy nếu có cài hm3denv)
python scripts/smoke_wheel.py            # build dist/navsim-<phiên bản>-py3-none-any.whl, thử trong venv sạch
python scripts/make_drive_folder.py      # GGDrive/navsim/ = wheel + README.md + README.vi.md, để tải lên Drive
```

Sau mỗi thay đổi hãy chạy `scripts/make_drive_folder.py` để `GGDrive/navsim/` khớp với project.
Dataset được build bằng [hm3denv](https://github.com/TrKimHieu/2D_Navigation_Sim) (cùng định dạng).

Giấy phép MIT. Các dataset build từ HM3D tuân theo điều khoản sử dụng HM3D của Matterport.
