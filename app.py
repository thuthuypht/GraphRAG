import streamlit as st
import pandas as pd
import os
import random
import re

from modules.data_handler import load_ontology, load_authors_list
from modules.feature_extraction import extract_features_for_pair
from modules.model_handler import load_model, predict_link_prob, suggest_collaborators

st.set_page_config(layout="wide")
# --- CUSTOM CSS INJECTION FOR SIDEBAR WIDTH (Đã khôi phục) ---
# Đặt chiều rộng sidebar bằng 25% độ rộng màn hình
st.markdown("""
<style>
/* Selector targeting the main sidebar container */
section[data-testid="stSidebar"] {
    /* Đặt chiều rộng sidebar bằng 25% độ rộng màn hình */
    width: 30% !important; 
    transition: width 0.3s ease-in-out;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------

# --- Hàm sử dụng Caching để tối ưu hiệu suất ---

@st.cache_resource(show_spinner="Đang tải Ontology...")
def load_cached_ontology(path):
    """
    Nạp file ontology và lưu vào bộ nhớ cache.
    """
    print(f"Tải ontology từ: {path}")
    if path:
        try:
            return load_ontology(path)
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi nạp Ontology: {e}. Vui lòng kiểm tra lại file.")
            return None
    return None

@st.cache_data(show_spinner="Đang trích xuất danh sách tác giả...")
def get_cached_author_list(onto_data):
    """
    Trích xuất danh sách tác giả từ ontology và lưu vào bộ nhớ cache.
    """
    print("Trích xuất danh sách tác giả từ ontology.")
    if onto_data:
        try:
            authors_full_list = load_authors_list(onto_data)
            return authors_full_list
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi trích xuất danh sách tác giả: {e}.")
            return []
    return []

# --- Cấu hình Thanh bên (Sidebar) ---
st.sidebar.title("Quản lý dữ liệu học thuật ⚙️")
st.sidebar.header("Chọn Cơ sở dữ liệu")

# Lấy danh sách các cơ sở dữ liệu (thư mục con trong data)
try:
    databases = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
    selected_db = st.sidebar.selectbox("Vui lòng chọn Cơ sở dữ liệu:", options=databases)
    print(f"Thư mục cơ sở dữ liệu đã chọn: {selected_db}")
except FileNotFoundError:
    st.sidebar.error("Không tìm thấy thư mục 'data/'. Vui lòng tạo thư mục này.")
    st.stop()

# Lựa chọn Ontology dựa trên cơ sở dữ liệu đã chọn
ontology_path = None
selected_ontology = None
review_year = None
if selected_db:
    st.sidebar.header("Chọn Ontology")
    ONTOLOGIES_DIR = os.path.join("data", selected_db, "ontologies")
    print(f"Đường dẫn đầy đủ tới thư mục ontologies: {ONTOLOGIES_DIR}")
    try:
        owl_files = [f for f in os.listdir(ONTOLOGIES_DIR) if f.endswith('.owl')]
        if not owl_files:
            st.sidebar.warning(f"Không tìm thấy file Ontology nào trong thư mục '{ONTOLOGIES_DIR}'.")
            
        selected_ontology = st.sidebar.selectbox("Vui lòng chọn một file Ontology:", options=owl_files)
        print(f"File ontology đã chọn: {selected_ontology}")
        if selected_ontology:
            ontology_path = os.path.join(ONTOLOGIES_DIR, selected_ontology)
            print(f"Đường dẫn đầy đủ tới file ontology: {ontology_path}")

            # Trích xuất năm từ tên file ontology
            year_match = re.search(r'\b\d{4}\b', selected_ontology)
            if year_match:
                review_year = int(year_match.group(0))
                st.sidebar.info(f"Năm xem xét: **{review_year}**")
            else:
                st.sidebar.warning("Không thể tìm thấy năm trong tên file ontology.")
    except FileNotFoundError:
        st.sidebar.error(f"Thư mục '{ONTOLOGIES_DIR}' không tồn tại.")
        selected_ontology = None
        ontology_path = None

def load_initial_ontology(path):
    """Nạp file ontology và xử lý lỗi."""
    print(f"Tải ontology từ: {path}")
    if path:
        try:
            return load_ontology(path)
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi nạp Ontology: {e}. Vui lòng kiểm tra lại file.")
            return None
    return None

onto_data = load_initial_ontology(ontology_path)
author_names = []
# Đảm bảo danh sách tác giả luôn được tải và có sẵn
if onto_data:
    try:
        authors_full_list = load_authors_list(onto_data)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi trích xuất danh sách tác giả: {e}.")
        authors_full_list = []
else:
    authors_full_list = []
    st.info("Chưa có Ontology nào được nạp. Một số chức năng có thể không hoạt động.")
if authors_full_list:
    author_names = [author.name for author in authors_full_list]
# Tạo một danh sách tác giả ngẫu nhiên để sử dụng trong session state
if 'authorslist' not in st.session_state and authors_full_list:
    num_to_choose = min(20, len(author_names))
    st.session_state.authorslist = random.choices(author_names, k=num_to_choose)

# Hàm để làm mới danh sách tác giả
def reset_authors_list():
    if authors_full_list:
        num_to_choose = min(20, len(authors_full_list))
        st.session_state.authorslist = random.choices(author_names, k=num_to_choose)

authors_list = st.session_state.get('authorslist', [])
# --- Centralized Model Selection ---
st.sidebar.header("Chọn Mô hình")
model_choice = st.sidebar.radio("Chọn mô hình", [
    "Logistic Regression (LR)",
    "Decision Tree (DT)",
    "Random Forest (RF)",
    "Multi-Layer Perceptron (MLP)",
    "Graph Convolutional Netwwork (GCN)",
])
    # --- Tham số chung cho dự đoán ---
st.sidebar.header("Tham số dự đoán")
prob_threshold = st.sidebar.slider("Ngưỡng xác suất", 0.0, 1.0, 0.5)

# --- Tab 1: Dự đoán Liên kết ---
tab1, tab2, tab3, tab4 = st.tabs(["Dự đoán liên kết", "Tra cứu và gợi ý", "Quản trị", "Xuất báo cáo"])

with tab1:
    st.header("1. Dự đoán Liên kết 🤝")
    # --- Logic quản lý danh sách ngẫu nhiên ---
    if 'authorslist' not in st.session_state:
        if authors_full_list:
            num_to_choose = min(20, len(authors_full_list))
            authors_random = random.choices(authors_full_list, k=num_to_choose)
            st.session_state.authorslist = [author.name for author in authors_random]
        else:
            st.session_state.authorslist = []

    authors_list = st.session_state.authorslist
    def reset_authors_list():
        if 'authorslist' in st.session_state:
            del st.session_state.authorslist

    
    # --- Chọn phương thức nhập liệu ---
    input_method = st.radio(
        "Chọn phương thức nhập liệu",
        ("Thủ công", "Ngẫu nhiên", "Hàng loạt từ file")
    )

    author_A = None
    author_B = None
    uploaded_file = None

    if input_method == "Thủ công":
        st.subheader("Nhập liệu thủ công")
        author_A = st.text_input("Nhập tên Tác giả 1", key="author_A_manual_val")
        author_B = st.text_input("Nhập tên Tác giả 2", key="author_B_manual_val")
       
    elif input_method == "Ngẫu nhiên":
        st.subheader("Nhập liệu theo danh sách ngẫu nhiên")
        
        st.button("Làm mới danh sách tác giả", on_click=reset_authors_list)
        
        col_a, col_b = st.columns(2)
        with col_a:
            author_A = st.selectbox(
                "Chọn Tác giả 1 (Ngẫu nhiên)", 
                options=authors_list,
                key="author_A_random_val"
            )
        with col_b:
            author_B = st.selectbox(
                "Chọn Tác giả 2 (Ngẫu nhiên)", 
                options=authors_list, 
                key="author_B_random_val"
            )

    elif input_method == "Hàng loạt từ file":
        st.subheader("Dự đoán hàng loạt từ file")
        uploaded_file = st.file_uploader("Tải lên danh sách cặp tác giả (.csv hoặc .xlsx)", type=["csv", "xlsx"])
    name1 = author_A.hasName if hasattr(author_A, 'hasName') else author_A
    name2 = author_B.hasName if hasattr(author_B, 'hasName') else author_B

    if st.button("Dự đoán"):
        if not onto_data or not review_year:
            st.error("Vui lòng đảm bảo ontology đã được nạp và có năm xem xét.")
        else:
            if input_method == "Ngẫu nhiên" or input_method == "Thủ công":
                if author_A and author_B:
                    if author_A == author_B:
                        st.error("Vui lòng chọn hai tác giả khác nhau.")
                    elif author_A not in author_names or author_B not in author_names:
                        st.error(f"Một hoặc cả hai tác giả không tồn tại trong ontology. Vui lòng kiểm tra lại.")
                    else:
                        with st.spinner('Đang trích xuất đặc trưng và dự đoán...'):
                            # Xác định model_type cho hàm extract_features_for_pair
                            if model_choice == "Graph Convolutional Netwwork (GCN)":
                                model_type = "GCN"
                                # Khi dùng GCN cần truyền thêm onto_data và review_year vào load_model
                                model = load_model(model_choice, selected_db, _onto_data=onto_data, _review_year=review_year)
                            else:
                                model_type = "MLP"
                                model = load_model(model_choice, selected_db)
                            if model:
                                st.info(f"Đang sử dụng mô hình: **{model_choice}**")
                                features_df = extract_features_for_pair(onto_data, author_A, author_B, review_year, model_type=model_type)
                                # name1 = author_A.hasName if hasattr(author_A, 'hasName') else author_A
                                # name2 = author_B.hasName if hasattr(author_B, 'hasName') else author_B
                                commonAff = float(features_df.loc[0, 'hasCommonAffiliation']) if 'hasCommonAffiliation' in features_df.columns else 0
                                commonInt = float(features_df.loc[0, 'hasCommonInterest']) if 'hasCommonInterest' in features_df.columns else 0
                                commonPast1 = features_df.loc[0, 'hasPastStatus'] if 'hasPastStatus' in features_df.columns else 0
                                commonPast2 = features_df.loc[0, 'hasPast2'] if 'hasPast2' in features_df.columns else 0
                                commonPast3 = features_df.loc[0, 'hasPast3'] if 'hasPast3' in features_df.columns else 0
                                commonPastTotal = features_df.loc[0, 'hasPastTotal'] if 'hasPastTotal' in features_df.columns else 0
                                explanation = f"Hai tác giả "
                                if commonAff > 0: 
                                    explanation += f" có **{commonAff*100:.2f}%** độ tương đồng về cơ quan; "
                                else:
                                    explanation += f"không có liên quan  nơi làm việc hoặc cộng tác; "
                                if commonInt > 0:
                                    explanation += f"**{commonInt*100:.2f}%** độ tương đồng về lĩnh vực quan tâm; "
                                else:
                                    explanation += f"không có lĩnh vực quan tâm chung; "
                                if commonPast1 > 0:
                                    explanation += f"đã cùng nhau viết **{commonPast1}** bài báo trong khoảng thời gian cách đây 01 năm; "
                                if commonPast2 > 0:
                                    explanation += f"đã cùng nhau viết **{commonPast2}** bài báo trong khoảng thời gian cách đây 02 năm; "
                                if commonPast3 > 0:
                                    explanation += f"đã cùng nhau viết **{commonPast3}** bài báo trong khoảng thời gian cách đây 03 năm; "
                                explanation += f"tổng cộng đã cùng nhau viết **{commonPastTotal}** bài báo trong quá khứ."
                                # Gắn tên cặp vào DataFrame để GCNWrapper dùng
                                if model_type == "GCN":
                                    features_df.author_names_pair = (author_A, author_B)
                                if features_df is not None:
                                    st.subheader("Đặc trưng đã trích xuất")
                                    st.dataframe(features_df, hide_index=True)
                                    prob = predict_link_prob(features_df, model)
                                    if prob >= prob_threshold:
                                        explanation += f" Với xác suất **{prob*100:.2f}%** cao hơn ngưỡng dự đoán, hai tác giả có nhiều khả năng cộng tác trong tương lai."
                                        st.info(f"**Giải thích:** {explanation}")
                                        st.success(f"Hai tác giả **{author_A}** và **{author_B}** CÓ nhiều khả năng cộng tác trong tương lai")
                                        # st.success(f"Hai tác giả **{author_A}** - (**{name1}**) và **{author_B}** - (**{name2}**) có nhiều khả năng cộng tác trong tương lai")

                                    else:
                                        explanation += f" Với xác suất **{prob*100:.2f}%** thấp hơn ngưỡng dự đoán, hai tác giả có thể không cộng tác trong tương lai."
                                        st.info(f"**Giải thích:** {explanation}")
                                        st.warning(f"Hai tác giả **{author_A}** và **{author_B}** có thể KHÔNG cộng tác trong tương lai")
                                else:
                                    st.error("Không thể trích xuất đặc trưng. Vui lòng kiểm tra lại tên tác giả và dữ liệu ontology.")
                            else:
                                st.error("Không thể tải mô hình. Vui lòng kiểm tra file mô hình.")
                else:
                    st.error("Vui lòng nhập hoặc chọn ít nhất hai tác giả.")
            
            elif input_method == "Hàng loạt từ file":
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)
                        else:
                            st.error("Định dạng file không được hỗ trợ.")
                            df = None
                        
                        if df is not None:
                            st.dataframe(df)
                            st.info("Đang xử lý dự đoán hàng loạt...")
                            
                            required_cols = ['author_A', 'author_B']
                            if not all(col in df.columns for col in required_cols):
                                st.error("File tải lên phải chứa các cột 'author_A' và 'author_B'.")
                            else:
                                with st.spinner("Đang xử lý dự đoán..."):
                                    model = load_model(model_choice, selected_db)
                                    if model:
                                        predictions = []
                                        for index, row in df.iterrows():
                                            author_A_batch = row['author_A']
                                            author_B_batch = row['author_B']
                                            features = extract_features_for_pair(onto_data, author_A_batch, author_B_batch, review_year)
                                            if features is not None:
                                                prob = predict_link_prob(features, model)
                                                predictions.append(prob)
                                            else:
                                                predictions.append(None)
                                        
                                        df['probability'] = predictions
                                        
                                        st.subheader("Kết quả dự đoán hàng loạt")
                                        st.dataframe(df)

                                        csv_output = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="Tải xuống kết quả dự đoán (.csv)",
                                            data=csv_output,
                                            file_name='ket_qua_du_doan_hang_loat.csv',
                                            mime='text/csv'
                                        )

                                    else:
                                        st.error("Không thể tải mô hình. Vui lòng kiểm tra file mô hình.")
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi khi đọc file: {e}")
                else:
                    st.warning("Vui lòng tải lên một file để dự đoán hàng loạt.")

# Tab 2: Tra cứu Thông tin
with tab2:
    st.header("2. Tra cứu thông tin & Gợi ý cộng tác 🔍")
    if not onto_data:
        st.warning("Chưa có Ontology nào được nạp. Vui lòng chọn Ontology ở thanh bên để sử dụng chức năng này.")
    else:
        # reset_authors_list()
        # Chọn tác giả cần tra cứu
        search_author_name = st.selectbox("Chọn tác giả cần tra cứu:", options=authors_list)
        
        with st.spinner("Đang tìm kiếm..."):
            # Lấy thông tin tác giả
            selected_author = next((a for a in authors_full_list if a.name == search_author_name), None)

        # Hiển thị thông tin
        if selected_author:
            st.subheader("Thông tin chi tiết")
            st.write(f"**Năm xem xét:** **{review_year}**")
            st.write(f"**Tên:** {selected_author.hasName}")
            
            # 1. Tên (hasName) và đơn vị (hasAffiliation)
            if selected_author.hasAffiliation:
                st.write(f"**Đơn vị:** {selected_author.hasAffiliation}")
            else:
                st.write("**Đơn vị:** Không có thông tin.")

            # 2. Lĩnh vực quan tâm (hasInterests)
            if selected_author.hasInterestArea:
                st.write(f"**Lĩnh vực quan tâm:** {', '.join(selected_author.hasInterestArea)}")
            else:
                st.write("**Lĩnh vực quan tâm:** Không có thông tin.")

            # 3. Số bài báo trước năm xem xét
            publications_before_review = []
            if selected_author.authored and review_year:
                publications_before_review = [
                    pub for pub in selected_author.authored if int(pub.hasPublicationYear) < review_year
                ]
                st.write(f"**Số bài báo trước năm {review_year}:** {len(publications_before_review)}")
            else:
                st.write("**Số bài báo:** Không có thông tin hoặc thiếu năm xem xét.")

            # 4. Cộng tác viên trước năm xem xét (hasCoAuthors)
            co_authors_before_review = set()
            if selected_author.authored and review_year:
                for pub in publications_before_review:
                    if pub.authored:
                        for co_author_iri in pub.authored:
                            if co_author_iri != selected_author.iri:
                                co_author_name = co_author_iri.name
                                co_authors_before_review.add(co_author_name)
            
            if co_authors_before_review:
                st.write(f"**Đồng tác giả trước năm {review_year}:**")
                st.write(f"Số lượng đồng tác giả: {len(co_authors_before_review)}")
                coAuthorString = "; ".join(co_authors_before_review)
                st.write(coAuthorString)
            else:
                st.write("**Đồng tác giả trước năm xem xét:** Không có.")
        else:
            st.info("Vui lòng chọn một tác giả để xem thông tin chi tiết.")
        N = min(100,len(author_names))
        author_namesN = random.choices(author_names,k= N)
        print(f"Số tác giả quan tâm: {len(author_namesN)}")
        st.subheader("Gợi ý cộng tác")
        if st.button("Gợi ý"):
            if not onto_data or not review_year:
                st.error("Vui lòng đảm bảo rằng ontology đã được tải lên tương ứng với năm xem xét.")
            else:
                with st.spinner("Tìm kiếm cộng tác viên tiềm năng..."):
                    # Truyền thêm onto_data và review_year nếu là GCN
                    if model_choice == "Graph Convolutional Netwwork (GCN)":
                        suggestion_model = load_model(model_choice, selected_db, _onto_data=onto_data, _review_year=review_year)
                    else:
                        suggestion_model = load_model(model_choice, selected_db)
                    if suggestion_model:
                        suggestions_df = suggest_collaborators(search_author_name, suggestion_model, author_namesN, onto_data, review_year)
                        if not suggestions_df.empty:
                            st.write("Đề xuất cộng với:")
                            suggestions_df.insert(0, "STT", suggestions_df.index + 1)
                            suggestions_df['Xác suất cộng tác (float)'] = suggestions_df['Xác suất cộng tác'].str.rstrip('%').astype('float') / 100
                            suggestions_df['Khả năng cộng tác tương lai'] = suggestions_df['Xác suất cộng tác (float)'].apply(
                                lambda x: "Có" if x >= prob_threshold else "Không"
                            )
                            st.dataframe(suggestions_df[['STT', 'Tên Tác giả', 'Khả năng cộng tác tương lai', 'Xác suất cộng tác']], hide_index=True)
                        else:
                            st.write("Không tìm thấy đề xuất.")
                    else:
                        st.error("Không thể tải mô hình. Vui lòng kiểm tra lại file mô hình.")
# Tab 3: Quản trị
with tab3:
    st.header("3. Quản trị Dữ liệu & Mô hình 🛠️")
    st.info("Đây là khu vực dành cho quản trị viên")
    
    st.subheader("Quản lý Ontology")
    if st.button("Nạp lại Ontology"):
        st.cache_data.clear()
        st.success("Đã xóa cache. Vui lòng làm mới trang để nạp lại ontology.")
    
    st.subheader("Quản lý Mô hình")
    uploaded_model = st.file_uploader("Tải lên Mô hình mới (.pkl)", type="pkl")
    if uploaded_model:
        st.success("Đã tải lên mô hình mới.")

# Tab 4: Xuất Báo cáo
with tab4:
    st.header("4. Xuất Báo cáo 📄")
    st.subheader("Xuất kết quả dự đoán")
    st.write("Nội dung báo cáo sẽ được hiển thị ở đây.")
    
    csv_data = "author_a,author_b,probability\nJohn,Jane,0.85"
    st.download_button(
        label="Tải xuống dữ liệu CSV",
        data=csv_data,
        file_name='ket_qua_du_doan.csv',
        mime='text/csv'
    )
    
    st.button("Tạo báo cáo PDF/Word (Chức năng nâng cao)")
