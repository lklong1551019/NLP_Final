# Báo cáo ACL

Khung báo cáo theo [acl-style-files](https://github.com/acl-org/acl-style-files) chính thức.
`acl.sty` và `acl_natbib.bst` tải trực tiếp từ repo đó.

## Quy định

- **4 trang** nội dung cho short paper.
- **Không giới hạn** trang cho tài liệu tham khảo.
- Mục **Limitations bắt buộc**, đặt sau Conclusion, **không tính** vào giới hạn trang.
- Phụ lục cũng không tính vào giới hạn trang.

## Số liệu tự sinh, không gõ tay

Mọi bảng nằm trong `generated/` và do script tạo ra từ file kết quả JSON:

```bash
python scripts/export_paper_tables.py                 # sau khi có kết quả
python scripts/export_paper_tables.py --placeholders  # khi chưa chạy thực nghiệm
```

Chạy chưa xong vẫn build được — bảng hiện dấu `--`, bố cục vẫn kiểm tra được.
**Đừng sửa tay file trong `generated/`**, chạy lại script thay vì vậy; như thế
số trong báo cáo không bao giờ lệch khỏi kết quả thật.

## Build

```bash
bash paper/build.sh
```

Script tự sinh lại bảng trước khi build, rồi in số trang và số TODO còn lại.

Chưa cài LaTeX thì dùng **Overleaf**: upload cả thư mục `paper/`, chọn `main.tex`
làm file chính, compiler pdfLaTeX. Không cần cài gì trên máy.

## Việc cần làm

25 chỗ đánh dấu `TODO` trong `main.tex`, mỗi chỗ có ghi chú gợi ý nội dung.
Thứ tự nên viết:

1. **Dataset**, **Experimental Setup** — viết được ngay, không cần chờ kết quả.
2. **Method** — phần cốt lõi; mục 4.2 (hai biến thể metric) và 4.3 (chấm điểm
   log-prob) chính là đóng góp của nhóm, viết kỹ.
3. **Related Work** — cần đọc và **kiểm chứng từng trích dẫn**.
4. **Results**, **Error Analysis** — sau khi có số.
5. **Abstract**, **Conclusion** — viết cuối cùng.
6. **Limitations**, **Ethical Considerations** — đã liệt kê sẵn ý trong file.

## Trích dẫn

`custom.bib` có sẵn khung cho các mục cần thiết, nhưng **mọi entry đều cần kiểm
chứng** trên [ACL Anthology](https://aclanthology.org). Đừng để tôi hay bất kỳ
mô hình nào bịa thông tin trích dẫn — sai trích dẫn nặng hơn thiếu trích dẫn.

Muốn dùng toàn bộ ACL Anthology thì tải `anthology.bib` về rồi đổi thành
`\bibliography{anthology,custom}`.

## Nộp bài

Đổi `\usepackage[review]{acl}` thành `\usepackage[final]{acl}` để hiện tên tác giả,
rồi điền tên và MSHV của cả 4 thành viên.

## Trạng thái hiện tại

Đã kiểm tra cấu trúc: đúng template, đủ 8 mục theo đề bài, môi trường LaTeX cân
bằng, 6 file bảng đều tồn tại.

**Chưa compile ra PDF** — máy dựng khung này không cài LaTeX. Lần build đầu nên
chạy sớm để bắt lỗi cú pháp, đừng để sát hạn nộp.
