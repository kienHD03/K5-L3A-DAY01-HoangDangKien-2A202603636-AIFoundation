# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603636
- **Họ và tên:** Hoàng Đăng Kiên
- **SEED** (in ở ô đầu notebook): 9006

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.830      0.907        0.857   
2. BoW + MLP (deep learning)         0.990   0.943      0.966        0.974   
3a. Encoder (MiniLM) + kNN           0.863   0.953      0.906        0.950   
3b. Encoder (MiniLM) + LogReg        0.990   0.906      0.946        0.974   
3c. Encoder (MiniLM) + MLP head      0.990   0.943      0.966        0.974   
4a. LLM nhỏ zero-shot                0.952   1.000      0.976        0.976   
4c. LLM nhỏ few-shot                 1.000   1.000      1.000        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.319      15159.551       0.064   
2. BoW + MLP (deep learning)         8.384      28734.490       0.259   
3a. Encoder (MiniLM) + kNN          90.417         27.713      45.995   
3b. Encoder (MiniLM) + LogReg       90.872         30.556      29.337   
3c. Encoder (MiniLM) + MLP head     91.324         25.676      63.402   
4a. LLM nhỏ zero-shot                0.000          1.280     717.559   
4c. LLM nhỏ few-shot                 0.000          1.394     705.448   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.101         0.000  
2. BoW + MLP (deep learning)          0.342         0.000  
3a. Encoder (MiniLM) + kNN           72.280         0.000  
3b. Encoder (MiniLM) + LogReg        37.730         0.000  
3c. Encoder (MiniLM) + MLP head     116.760         0.000  
4a. LLM nhỏ zero-shot              1168.484         0.019  
4c. LLM nhỏ few-shot               1020.908         0.050  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. Dự đoán về hiệu năng của MLP so với Logistic Regression (Từ dl_curve)
- Dự đoán: Mô hình Deep Learning (MLP) sẽ có F1-score kém hơn Logistic Regression (LogReg) do mô hình quá phức tạp, dễ bị học vẹt (overfitting) trên dữ liệu BoW thưa thớt.
- Thực tế: Trái ngược hoàn toàn, MLP có điểm số vượt trội với F1 = 0.966, trong khi LogReg tự code chỉ đạt F1 = 0.907.
- Vì sao: Dù BoW thưa thớt, nhưng việc thêm các lớp ẩn (hidden layers) giúp MLP học được sự kết hợp phi tuyến tính giữa các từ. LogReg chỉ đơn thuần cộng/trừ trọng số của từng từ một cách độc lập, trong khi MLP có thể nhận ra các "cụm" từ khóa thường đi chung với nhau để tạo thành dấu hiệu spam mạnh hơn, giúp nó phân loại chính xác hơn hẳn.
2. Dự đoán về Learning Rate cực lớn (Từ ml_lr)
- Dự đoán: Với learning rate rất lớn (lr=50), đường loss sẽ đi xuống rất mạnh, thậm chí là một đường kéo thẳng tuột xuống dưới.
- Thực tế: Đường loss không đi thẳng xuống ngay mà có một nhịp "giật" (spike) vọt lên cực kỳ cao (vượt mức 2.5) ở những epoch đầu tiên, sau đó mới rớt thê thảm xuống 0.006.
- Vì sao: Đã dự đoán đúng kết quả cuối cùng là loss rất nhỏ, nhưng bỏ qua hiện tượng Overshooting (vượt quá đích). Khi bước nhảy (learning rate) quá khổng lồ, thuật toán Gradient Descent sẽ nhảy vọt lố qua đáy của hàm loss, văng lên vách núi bên kia khiến loss tăng vọt đột biến. Sau vài nhịp văng lố, nó mới bị dội ngược trở lại đáy.
3. Dự đoán về việc loại bỏ hàm kích hoạt ReLU (Từ dl_relu)
- Dự đoán: Nếu xoá ReLU, 2 lớp Linear nối tiếp sẽ gộp thành 1 phép biến đổi tuyến tính duy nhất (tương đương LogReg). Do đó, F1-score sẽ tụt xuống và quay về xấp xỉ mức của LogReg (~0.907).
- Thực tế: Mô hình không ReLU vẫn đạt điểm rất cao là F1 = 0.961 (chỉ kém một chút so với MLP gốc 0.966, và vẫn cao hơn hẳn LogReg 0.907).
- Vì sao: Về mặt toán học, suy luận hoàn toàn đúng: 2 lớp tuyến tính nhân với nhau bằng 1 lớp tuyến tính. Tuy nhiên, trong quá trình huấn luyện (training dynamics), việc phân rã 1 ma trận trọng số thành 2 ma trận nhân với nhau đóng vai trò như một cơ chế điều chuẩn ngầm (implicit regularization). Nó giúp thuật toán tối ưu (Adam/SGD) tìm được một điểm cực tiểu cục bộ (local minima) tốt hơn so với LogReg thuần túy. Hơn nữa, bản thân bộ dữ liệu SMS Spam cũng khá dễ phân tách tuyến tính, nên dù bị biến thành mạng tuyến tính, mô hình vẫn đạt kết quả rất ấn tượng.

## 4. Phân tích 50 case khó
- Model tụt nhiều nhất: Khối các mô hình ML truyền thống (đặc biệt là BoW + LogReg) tụt điểm mạnh nhất. F1 giảm sâu từ 0.857 (trên tập test thường) xuống chỉ còn 0.286 trên 50 case khó, với Accuracy vỏn vẹn 50%.
- Phân tích bối cảnh: Điểm số trên tập test thường trước đó thực chất "quá lạc quan" do chia sẻ cùng phân phối dữ liệu với tập train, cho phép mô hình đạt điểm cao chỉ nhờ việc học vẹt các từ khóa bề mặt (như "FREE", "WIN") thay vì thực sự hiểu nghĩa của văn bản. Bản chất 50 case khó này là một bài "stress test" nhằm tìm ra điểm mù bảo mật, chứ không đại diện cho luồng spam hàng loạt ngoài đời thực.
- Nhóm case bị lỗi nặng: BoW sập nặng nhất ở 3 nhóm: Spam giọng tự nhiên (chỉ đúng 0%), Thao túng / lừa đảo (đúng 12%), và Né từ khoá (đúng 17%).

2 case cụ thể và cơ chế khiến model sai:

Case 1: Nhóm Né từ khoá (Obfuscation)
- Nội dung: C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4r... (Nhãn thật: Spam)
- Cơ chế khiến model sai: BoW dễ bị đánh lừa bởi nhóm này vì nó chỉ nhìn văn bản như tập hợp các token rời rạc có trong từ điển. Kẻ gian cố tình viết sai chính tả (đổi chữ o thành số 0, i thành 1), khiến BoW không thể khớp được từ khóa rác quen thuộc và gán nhãn sai thành Ham.
(Lưu ý thêm: Để cứu BoW, ta có thể dùng Regex chuẩn hóa chữ số ngụy trang về lại chữ cái, nhưng rủi ro lớn nhất là dễ bắt nhầm, vô tình làm biến dạng các đoạn mã hợp lệ như mã OTP, mã đơn hàng, biển số xe).

Case 2: Nhóm Spam giọng tự nhiên (Natural spam)
- Nội dung: Long time no see! I've started a little side b... (Nhãn thật: Spam)
- Cơ chế khiến model sai: Kẻ lừa đảo sử dụng văn phong giao tiếp cực kỳ tự nhiên, không hề chứa các từ khóa rác rập khuôn bề mặt. Do BoW không có khả năng thấu hiểu ngữ cảnh liền mạch mà chỉ đếm từ, nó phân tích thấy toàn từ ngữ an toàn nên quyết định đây là Ham. (Trong khi đó, LLM khắc phục được yếu điểm này nhờ khả năng đọc hiểu ngữ cảnh liền mạch giống con người, nhận diện được kịch bản lôi kéo dù không có "từ khóa rác" nào).

## 5. Thử thách cá nhân

Đề bài: Đổi ngưỡng quyết định (threshold) 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?

Cách làm: Sử dụng hàm tính xác suất predict_proba(X_te, w, b), sau đó duyệt qua vòng lặp với các ngưỡng th = [0.2, 0.5, 0.8]. Với mỗi ngưỡng, phân loại lại nhãn (probs >= th).astype(int) và đưa vào hàm metrics để tính 3 chỉ số.

Kết quả:
Ngưỡng 0.2: Precision=0.961 | Recall=0.934 | F1=0.947
Ngưỡng 0.5: Precision=1.000 | Recall=0.830 | F1=0.907
Ngưỡng 0.8: Precision=1.000 | Recall=0.604 | F1=0.753

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày chọn phương án nâng ngưỡng quyết định lên 0.8

## 7. AI log và điều đã học

Mỗi lần dùng AI hỏi về các định nghĩa chưa biết, từ đó hiểu sâu hơn để trả lời các câu hỏi đề bài đưa ra.
