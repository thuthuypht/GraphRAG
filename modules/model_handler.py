import joblib
import os
import streamlit as st
import pandas as pd
from modules.feature_extraction import extract_features_for_pair
from modules.data_handler import load_ontology, load_authors_list

@st.cache_resource
def load_model(model_choice, selected_db, _onto_data=None, _review_year=None):
    print("Bắt đầu load_model")
    # Import torch và nn bên trong hàm để tránh lỗi scope với cache
    try:
        import torch
        import torch.nn as nn
        torch_ok = True
    except ImportError:
        torch = None
        nn = None
        torch_ok = False

    """
    Tải mô hình học máy đã được huấn luyện dựa trên lựa chọn của người dùng
    và bộ dữ liệu đã chọn.
    Bổ sung logic tải riêng cho mô hình MLP (PyTorch).
    """
    # Ánh xạ tên mô hình trên giao diện với tên file tương ứng, tùy thuộc vào bộ dữ liệu

    MODEL_FILES = {
        "Aminer": {
            "Logistic Regression (LR)": "logistic_regression_model11989A465-prob.pkl",
            "Decision Tree (DT)": "DecisionTreeClassifier_model1989A465-prob.pkl",
            "Random Forest (RF)": "RandomForestClassifier1-1989A465-prob.pkl",
            "Multi-Layer Perceptron (MLP)": "mlp_model_aminer.pth",
            "Graph Convolutional Netwwork (GCN)": "gcn_model_aminer.pkl",
            #"Mô hình học máy đơn giản": "simple_ml_model_aminer.pkl",
        },
        "Mendeley": {
            "Logistic Regression (LR)": "logistic_regression_model2019A5-prob.pkl",
            "Decision Tree (DT)": "DecisionTreeClassifier_model2019A5-prob.pkl",
            "Random Forest (RF)": "RandomForestClassifier-2019A5-prob.pkl",
            "Multi-Layer Perceptron (MLP)": "mlp_model_mendeley.pth", # Giả định đây là .pth, đổi tên trong file mapping nếu cần
            "Graph Convolutional Netwwork (GCN)": "gcn_model_mendeley.pkl",
            #"Mô hình học máy đơn giản": "simple_ml_model_mendeley.pkl",
        },
    }
    model_name = MODEL_FILES.get(selected_db, {}).get(model_choice)
    if not model_name:
        st.warning(f"Mô hình '{model_choice}' chưa được hỗ trợ cho bộ dữ liệu '{selected_db}'.")
        return None

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', model_name))

    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô hình tại đường dẫn: {model_path}")
        return None

    if model_choice == "Multi-Layer Perceptron (MLP)":
        if not torch_ok:
            st.error("Không thể tải MLP. PyTorch không được cài đặt.")
            return None

        # Định nghĩa class MLP và PyTorchMLPWrapper bên trong hàm
        class MLP(nn.Module):
            def __init__(self, input_size):
                super(MLP, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2)
                )
            def forward(self, x):
                return self.model(x)

        class PyTorchMLPWrapper:
            def __init__(self, model_instance, input_size):
                self.model = model_instance
                self.model.eval()
                self.input_size = input_size
            def predict_proba(self, features_df):
                if features_df.shape[1] != self.input_size:
                    st.error(f"Kích thước đặc trưng đầu vào không khớp: {features_df.shape[1]} (thực tế) vs {self.input_size} (yêu cầu)")
                    return torch.tensor([[0.0, 0.0]]).numpy()
                features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
                with torch.no_grad():
                    output = self.model(features_tensor)
                    probabilities = torch.softmax(output, dim=1).numpy()
                return probabilities

        try:
            INPUT_FEATURE_SIZE = 6
            # Allowlist class MLP for safe loading
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([MLP])
            except Exception as e:
                st.warning(f"Không thể allowlist class MLP: {e}")
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            except Exception as e:
                st.warning(f"Thử tải mô hình với weights_only=False thất bại: {e}. Đang thử lại với weights_only=True.")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            model_instance = None
            if isinstance(checkpoint, nn.Module):
                model_instance = checkpoint
                # st.info("Đã tải mô hình MLP (PyTorch) đầy đủ.")
            else:
                model_instance = MLP(input_size=INPUT_FEATURE_SIZE)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model_instance.load_state_dict(checkpoint['state_dict'])
                    # st.info("Đã tải state_dict của MLP (PyTorch) từ dict.")
                else:
                    model_instance.load_state_dict(checkpoint)
                    # st.info("Đã tải state_dict của MLP (PyTorch).")
            model = PyTorchMLPWrapper(model_instance, INPUT_FEATURE_SIZE)
            # st.success("Tải mô hình MLP hoàn tất.")
            return model
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải mô hình PyTorch: {e}. Vui lòng kiểm tra lại cấu trúc lớp MLP và cách lưu file .pth.")
            return None
    elif model_choice == "Graph Convolutional Netwwork (GCN)":
        if not torch_ok:
            st.error("Không thể tải GCN. PyTorch không được cài đặt.")
            return None

        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                from torch_geometric.nn import GCNConv
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x

        class GCNWrapper:
            def __init__(self, model, in_channels, hidden_channels, out_channels, onto_data, review_year):
                self.model = model
                self.in_channels = in_channels
                self.hidden_channels = hidden_channels
                self.out_channels = out_channels
                self.model.eval()
                self.onto_data = onto_data
                self.review_year = review_year

            def predict_proba(self, features_df):
                # features_df phải chứa thông tin tên hai tác giả
                if hasattr(features_df, "author_names_pair"):
                    author_A, author_B = features_df.author_names_pair
                else:
                    st.error("GCNWrapper: Không truyền được tên hai tác giả cho dự đoán.")
                    return torch.tensor([[0.0, 0.0]]).numpy()
                # Tính đặc trưng cho từng node (tác giả)
                from modules.feature_extraction import extract_features_for_pair
                node_A = extract_features_for_pair(self.onto_data, author_A, author_A, self.review_year, model_type="MLP")
                node_B = extract_features_for_pair(self.onto_data, author_B, author_B, self.review_year, model_type="MLP")
                if node_A is None or node_B is None:
                    st.error("GCNWrapper: Không thể trích xuất đặc trưng cho hai tác giả.")
                    return torch.tensor([[0.0, 0.0]]).numpy()
                x = torch.tensor(pd.concat([node_A, node_B]).values, dtype=torch.float32)
                # Tạo edge_index cho 2 node
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                with torch.no_grad():
                    out = self.model(x, edge_index)
                    combined = out[0] + out[1]
                    prob = torch.softmax(combined, dim=0).numpy()
                    return prob.reshape(1, -1)

        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            in_channels = checkpoint.get("in_channels", 6)
            hidden_channels = checkpoint.get("hidden_channels", 32)
            out_channels = checkpoint.get("out_channels", 2)
            model = GCN(in_channels, hidden_channels, out_channels)
            model.load_state_dict(checkpoint["state_dict"])
            # Sử dụng _onto_data và _review_year truyền từ app.py
            if _onto_data is None or _review_year is None:
                st.error("Cần truyền _onto_data và _review_year vào load_model khi dùng GCN.")
                return None
            wrapper = GCNWrapper(model, in_channels, hidden_channels, out_channels, _onto_data, _review_year)
            # st.success("Tải mô hình GCN hoàn tất.")
            return wrapper
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải mô hình GCN: {e}. Vui lòng kiểm tra lại file mô hình GCN.")
            return None
    else:
        try:
            model = joblib.load(model_path)
            # st.success(f"Đã tải mô hình {model_choice} (Joblib) thành công.")
            return model
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải mô hình (joblib): {e}")
            return None

# Lưu ý: Hàm suggest_collaborators không cần thay đổi vì nó gọi predict_link_prob,
# và predict_link_prob giờ đã hoạt động với cả hai loại mô hình.

def predict_link_prob(features_df, model):
    """
    Sử dụng mô hình đã nạp để dự đoán xác suất cộng tác.
    """
    try:
        if not hasattr(model, 'predict_proba'):
            st.error("Mô hình được chọn không hỗ trợ dự đoán xác suất (predict_proba).")
            return 0.0
        prob = model.predict_proba(features_df)[:, 1][0]
        return float(prob)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")
        return 0.0

def suggest_collaborators(search_author_name, model, author_names, onto_data, review_year, num_suggestions=20):
    """
    Sử dụng mô hình học máy đã chọn để gợi ý các cộng tác viên tiềm năng.
    """
    if not author_names or not model:
        return pd.DataFrame()
    
    suggestions = []
    model_type = "GCN" if hasattr(model, "in_channels") else ""
    for other_author in author_names:
        if other_author == search_author_name:
            continue
        # Trích xuất đặc trưng cho cặp tác giả
        features_df = extract_features_for_pair(onto_data, search_author_name, other_author, review_year, model_type=model_type)
        if features_df is not None:
            # Gắn tên cặp vào DataFrame để GCNWrapper dùng (dùng setattr để tránh lỗi gán thuộc tính động)
            if model_type == "GCN":
                setattr(features_df, "author_names_pair", (search_author_name, other_author))
            prob = predict_link_prob(features_df, model)
            suggestions.append({
                "Tên Tác giả": other_author,
                "Xác suất cộng tác": prob
            })

    # Tạo DataFrame, sắp xếp và lấy top N
    suggestions_df = pd.DataFrame(suggestions)
    if not suggestions_df.empty:
        suggestions_df = suggestions_df.sort_values(
            by="Xác suất cộng tác",
            ascending=False
        ).head(num_suggestions).reset_index(drop=True)
        # Định dạng xác suất dưới dạng phần trăm
        suggestions_df['Xác suất cộng tác'] = suggestions_df['Xác suất cộng tác'].apply(lambda x: f"{x*100:.2f}%")
    return suggestions_df
