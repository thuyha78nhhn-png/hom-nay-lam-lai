import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io

# ==========================================
# 0. CẤU HÌNH TRANG STREAMLIT ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# 1. HÀM CACHE NẠP DỮ LIỆU DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash của cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# ==========================================
# 2. SIDEBAR - VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các thuộc tính từ X_1 đến X_14 và biến mục tiêu 'default'"
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: RandomForestClassifier")
    
    # Các siêu tham số cấu hình dựa trên mô hình trong notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng độc lập."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, max_value=30, value=10, step=1,
        help="Độ sâu tối đa của mỗi cây quyết định (None nếu không giới hạn)."
    )
    
    min_samples_split = st.slider(
        "Mẫu tối thiểu để tách nút (min_samples_split)",
        min_value=2, max_value=20, value=2, step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )

    with st.expander("⚙️ Tham số nâng cao"):
        criterion = st.selectbox(
            "Hàm đo lường chất lượng phân tách (criterion)",
            options=["gini", "entropy", "log_loss"],
            index=0,
            help="Tiêu chí đo lường chất lượng của việc phân tách nút."
        )
        random_state = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=42, step=1,
            help="Hạt giống tạo số ngẫu nhiên để đảm bảo tính tái lập kết quả."
        )
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm thử (test_size)", 
            min_value=0.1, max_value=0.5, value=0.3, step=0.05,
            help="Tỷ lệ tập dữ liệu được tách ra để đánh giá mô hình."
        )

    st.divider()
    # Nút bấm kích hoạt huấn luyện duy nhất
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu trích xuất đặc trưng, tiền xử lý và huấn luyện mô hình."
    )

# ==========================================
# 3. HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tín dụng và phát hiện giao dịch bất thường dựa trên mô hình học máy Học có giám sát.")

if uploaded_file is None:
    st.info("👋 Chào mừng bạn! Vui lòng tải tập dữ liệu mẫu (`.csv` hoặc `.xlsx`) ở thanh điều hướng bên trái để bắt đầu.")
    st.stop()
else:
    # Đọc dữ liệu đã tải lên
    file_bytes = uploaded_file.read()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Không thể đọc định dạng tệp. Vui lòng kiểm tra lại.")
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Kích thước dữ liệu gốc: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột.")
    st.divider()

# Xác định danh sách biến đầu vào cần thiết theo thiết kế trong notebook
expected_features = [f'X_{i}' for i in range(1, 15)]
target_col = 'default'

# Kiểm tra tính hợp lệ của schema dữ liệu huấn luyện
if not all(col in df_raw.columns for col in expected_features + [target_col]):
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc. Yêu cầu đầy đủ các biến từ X_1 đến X_14 và cột mục tiêu '{target_col}'.")
    st.stop()

# ==========================================
# 4. KHỐI XỬ LÝ HUẤN LUYỆN (Lưu vào session_state)
# ==========================================
if train_clicked:
    with st.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_raw[expected_features]
        y = df_raw[target_col]
        
        # Phân tách tập dữ liệu train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Thực hiện chuẩn hóa đặc trưng (Pipeline tái hiện từ notebook)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Khởi tạo và huấn luyện mô hình ngẫu nhiên (Random Forest)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else None,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Dự đoán và tính toán các chỉ số kiểm định
        y_pred = model.predict(X_test_scaled)
        y_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ các cấu phần quan trọng vào session_state nhằm tránh train lại khi chuyển tab
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'cm': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
        st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các thẻ tab bên dưới để xem chi tiết kết quả.")

# ==========================================
# 5. GIAO DIỆN CHÍNH - CHIA TABS NỘI DUNG
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Dự báo thực tế"
])

# ------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    st.subheader("Phân tích Thống kê Dữ liệu Thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng (Samples)", f"{df_raw.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số cột (Features)", f"{df_raw.shape[1]}")
    with col_m3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
        
    st.markdown("##### Hiển thị 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.markdown("##### Bảng mô tả thống kê các biến đặc trưng (X & y):")
    st.dataframe(df_raw[expected_features + [target_col]].describe(), use_container_width=True)

# ------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab2:
    st.subheader("Biểu đồ Phân tích Phân phối Biến Dữ liệu")
    
    # Tạo layout lưới 2x2 cho 4 biểu đồ trực quan hóa chính
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Biểu đồ biến mục tiêu (y) - Phân phối lớp rủi ro/gian lận
    with row1_col1:
        st.markdown("**Phân phối của biến mục tiêu (default)**")
        target_counts = df_raw[target_col].value_counts().reset_index()
        target_counts.columns = ['Trạng thái (default)', 'Số lượng']
        target_counts['Trạng thái (default)'] = target_counts['Trạng thái (default)'].map({0: '0: Bình thường', 1: '1: Gian lận/Rủi ro'})
        fig_target = px.bar(target_counts, x='Trạng thái (default)', y='Số lượng', color='Trạng thái (default)',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_target.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_target, use_container_width=True)

    # 2. Biểu đồ phân phối biến đầu vào quan trọng X_1
    with row1_col2:
        st.markdown("**Phân phối mật độ biến X_1 theo trạng thái mặc định**")
        fig_x1 = px.histogram(df_raw, x="X_1", color=target_col, marginal="box", barmode="overlay",
                              color_discrete_sequence=px.colors.qualitative.Safe)
        fig_x1.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_x1, use_container_width=True)

    # 3. Biểu đồ tương quan phân tán giữa X_1 và X_2
    with row2_col1:
        st.markdown("**Mối quan hệ phân tán giữa X_1 và X_2**")
        fig_scatter = px.scatter(df_raw, x="X_1", y="X_2", color=target_col, opacity=0.6,
                                 color_continuous_scale=px.colors.sequential.Bluered)
        fig_scatter.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # 4. Biểu đồ hộp tương quan tổng thể các thuộc tính đặc trưng tiêu biểu (X_1 đến X_6)
    with row2_col2:
        st.markdown("**Biểu đồ hộp (Boxplot) các thuộc tính đặc trưng tiêu biểu**")
        df_melt = pd.melt(df_raw, id_vars=[target_col], value_vars=['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6'])
        fig_box = px.box(df_melt, x="variable", y="value", color=target_col, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab3:
    st.subheader("Đánh giá Chất lượng Mô hình Học máy")
    
    # Kiểm tra trạng thái lưu trữ của mô hình trong session_state
    if 'trained_model' not in st.session_state:
        st.info("💡 Chưa tìm thấy mô hình đã huấn luyện. Vui lòng thiết lập cấu hình tham số và bấm nút **[🚀 Huấn luyện mô hình]** ở thanh Sidebar trái để xem kết quả đánh giá.")
    else:
        metrics = st.session_state['metrics']
        
        # Hiển thị các chỉ số đo lường cốt lõi
        c_acc, c_pre, c_rec, c_f1 = st.columns(4)
        with c_acc:
            st.metric("Accuracy (Độ chính xác tổng thể)", f"{metrics['accuracy']:.4f}")
        with c_pre:
            st.metric("Precision (Độ chính xác dự báo đúng)", f"{metrics['precision']:.4f}")
        with c_rec:
            st.metric("Recall (Tỷ lệ bỏ sót lỗi)", f"{metrics['recall']:.4f}")
        with c_f1:
            st.metric("F1-Score (Điểm cân bằng)", f"{metrics['f1']:.4f}")
            
        st.divider()
        
        col_g1, col_g2 = st.columns(2)
        
        # Trực quan hóa ma trận nhầm lẫn (Confusion Matrix)
        with col_g1:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm_data = metrics['cm']
            x_labels = ['Dự báo Sạch (0)', 'Dự báo Gian lận (1)']
            y_labels = ['Thực tế Sạch (0)', 'Thực tế Gian lận (1)']
            
            fig_cm = ff.create_annotated_heatmap(
                z=cm_data, x=x_labels, y=y_labels, 
                colorscale='Blues', showscale=True
            )
            fig_cm.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        # Hiển thị báo cáo phân loại chi tiết (Classification Report dạng bảng)
        with col_g2:
            st.markdown("**Báo cáo chi tiết theo từng lớp phân loại (Classification Report)**")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='YlGnBu', axis=0), use_container_width=True)
            
        # Biểu đồ mức độ quan trọng của các đặc trưng đầu vào (Feature Importance)
        st.divider()
        st.markdown("**Độ quan trọng của các thuộc tính đặc trưng (Feature Importance)**")
        importance = st.session_state['trained_model'].feature_importances_
        feat_imp_df = pd.DataFrame({
            'Thuộc tính': expected_features,
            'Độ quan trọng': importance
        }).sort_values(by='Độ quan trọng', ascending=True)
        
        fig_imp = px.bar(feat_imp_df, x='Độ quan trọng', y='Thuộc tính', orientation='h',
                         color='Độ quan trọng', color_continuous_scale='Viridis')
        fig_imp.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO THỰC TẾ)
# ------------------------------------------
with tab4:
    st.subheader("Kiểm thử & Khai thác Mô hình")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng hoàn tất bước huấn luyện mô hình tại Sidebar trước khi thực hiện dự báo.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào để dự báo:",
            options=["Chế độ 1: Nhập trực tiếp thủ công", "Chế độ 2: Tải file dữ liệu danh sách mới"],
            horizontal=True
        )
        
        st.divider()
        
        # ---------------------------------------------------
        # CHẾ ĐỘ 1: NHẬP THỦ CÔNG QUA FORM
        # ---------------------------------------------------
        if "Chế độ 1" in mode:
            st.markdown("##### Nhập thông số các thuộc tính đặc trưng giao dịch:")
            
            # Khởi tạo form nhập liệu
            with st.form("prediction_form"):
                # Tạo lưới nhập liệu 3 cột để giao diện gọn gàng
                f_cols = st.columns(3)
                input_values = {}
                
                # Trích xuất giá trị min, max, median làm mặc định từ tập dữ liệu gốc ban đầu
                for index, feat in enumerate(expected_features):
                    col_idx = index % 3
                    with f_cols[col_idx]:
                        min_val = float(df_raw[feat].min())
                        max_val = float(df_raw[feat].max())
                        mean_val = float(df_raw[feat].median())
                        
                        input_values[feat] = st.number_input(
                            f"Giá trị {feat}",
                            min_value=min_val - (abs(min_val)*0.5),
                            max_value=max_val + (abs(max_val)*0.5),
                            value=mean_val,
                            format="%.6f",
                            help=f"Khoảng dữ liệu mẫu thực tế: [{min_val:.4f} đến {max_val:.4f}]"
                        )
                
                submit_pred = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary", use_container_width=True)
                
            if submit_pred:
                # Chuyển đổi dữ liệu input sang cấu trúc DataFrame chuẩn đặc trưng
                input_df = pd.DataFrame([input_values])
                
                # Áp dụng bộ tiền xử lý chuẩn hóa scaler từ session_state
                input_scaled = scaler.transform(input_df)
                
                # Thực hiện dự báo
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                # Hiển thị kết quả ra màn hình trực quan
                st.markdown("### Kết quả đánh giá hệ thống:")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 1:
                        st.error("🚨 CẢNH BÁO: Giao dịch được hệ thống phân loại thuộc nhóm GIAN LẬN / RỦI RO CAO!")
                    else:
                        st.success("🟢 AN TOÀN: Giao dịch được hệ thống phân loại thuộc nhóm BÌNH THƯỜNG.")
                        
                with res_col2:
                    st.metric(label="Xác suất rủi ro gian lận (Lớp 1)", value=f"{probabilities[1]*100:.2f}%")
                    st.progress(float(probabilities[1]))

        # ---------------------------------------------------
        # CHẾ ĐỘ 2: TẢI TỆP TIN KIỂM THỬ HÀNG LOẠT
        # ---------------------------------------------------
        else:
            st.markdown("##### Tải lên tệp danh sách các bản ghi mới cần chấm điểm rủi ro:")
            bulk_file = st.file_uploader(
                "Chọn file dữ liệu cần dự báo hàng loạt (.csv, .xlsx)",
                type=["csv", "xlsx"],
                key="bulk_uploader",
                help="File tải lên phải cấu trúc chứa đầy đủ các cột thuộc tính từ X_1 đến X_14 giống file mẫu."
            )
            
            if bulk_file is not None:
                bulk_bytes = bulk_file.read()
                df_bulk = load_data(bulk_bytes, bulk_file.name)
                
                if df_bulk is not None:
                    # Kiểm tra tính đồng bộ cấu trúc đặc trưng đầu vào
                    if not all(col in df_bulk.columns for col in expected_features):
                        st.error("Cấu trúc file lỗi! File tải lên bắt buộc phải chứa toàn bộ các cột đặc trưng từ X_1 đến X_14.")
                    else:
                        with st.spinner("Đang xử lý chấm điểm chuỗi hàng loạt dữ liệu..."):
                            X_bulk = df_bulk[expected_features]
                            
                            # Tiền xử lý dữ liệu hàng loạt bằng scaler có sẵn
                            X_bulk_scaled = scaler.transform(X_bulk)
                            
                            # Thực hiện dự đoán hàng loạt
                            bulk_preds = model.predict(X_bulk_scaled)
                            bulk_probs = model.predict_proba(X_bulk_scaled)[:, 1]
                            
                            # Tạo DataFrame đầu ra kết hợp kết quả dự báo
                            df_result = df_bulk.copy()
                            df_result['Dự báo (Prediction)'] = bulk_preds
                            df_result['Xác suất rủi ro (Fraud Probability)'] = bulk_probs
                            
                            st.success(f"Chấm điểm hoàn tất cho {df_result.shape[0]} bản ghi giao dịch mới!")
                            
                            # Thống kê nhanh kết quả dự báo hàng loạt vừa thực hiện
                            cnt_fraud = int(np.sum(bulk_preds == 1))
                            st.warning(f"⚠️ Phát hiện hệ thống: Có **{cnt_fraud}** trên tổng số **{df_result.shape[0]}** giao dịch mang dấu hiệu rủi ro gian lận.")
                            
                            # Xem trước bảng kết quả sau phân tích
                            st.dataframe(df_result, use_container_width=True)
                            
                            # Xuất file kết quả phục vụ download
                            csv_buffer = io.StringIO()
                            df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Tải xuống kết quả phân tích (.CSV)",
                                data=csv_data,
                                file_name="ket_qua_du_bao_gian_lan.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
