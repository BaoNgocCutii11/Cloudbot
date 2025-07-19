import streamlit as st
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- DI CHUYỂN DÒNG NÀY LÊN ĐẦU TIÊN SAU CÁC LỆNH IMPORTS ---
st.set_page_config(page_title="CloudBot: Phân Loại Rác & Môi Trường Xanh ♻️", page_icon="🌳", layout="wide")

# --- Thêm CSS tùy chỉnh để thay đổi màu nền ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6F7E6; /* Một tông màu xanh lá cây nhạt, bạn có thể thay đổi mã màu này */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Kết thúc CSS tùy chỉnh ---

# --- 1. SETUP API & CONFIG ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Cấu hình Gemini API
genai.configure(api_key=google_api_key)

# KHỞI TẠO MÔ HÌNH GEMINI CÓ KHẢ NĂNG HIỂU HÌNH ẢNH
# Đây là model gemini-1.5-flash hỗ trợ multi-modal (text và image)
generative_model_vision = genai.GenerativeModel("gemini-1.5-flash")


# Load config
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
initial_bot_message_config = config.get("initial_bot_message", "Xin chào! Tôi là CloudBot, sẵn sàng giúp bạn tìm hiểu về rác thải và bảo vệ môi trường xanh!")

# --- 2. KHỞI TẠO SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "parts": [initial_bot_message_config]}]
if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0
if "chat_locked" not in st.session_state:
    st.session_state.chat_locked = False
if "chat_locked_time" not in st.session_state:
    st.session_state.chat_locked_time = None
if "show_chat_history_section" not in st.session_state:
    st.session_state.show_chat_history_section = False

# --- 3. CHATBOT HELPER FUNCTIONS (Gemini API & Negative Detection) ---

def call_gemini_api(prompt):
    """Calls the Gemini API for text-only responses."""
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    response = model_gemini.generate_content(prompt)
    return response.text

def detect_negative_language(text):
    """
    Basic function to detect negative words.
    """
    negative_words = ["tiêu cực", "tồi tệ", "ghét", "xấu", "phá hoại", "ô nhiễm", "thảm họa", "đáng sợ", "rác rưởi", "vô ích", "chán", "buồn", "tệ","ghê","kinh"]
    text_lower = text.lower()
    for word in negative_words:
        if word in text_lower:
            return True
    return False

# --- Hàm cho phần "Giải Pháp & Nhận Thức Xanh" ---
def get_environmental_solution(waste_type="chung"):
    """Generates environmental solutions based on waste type."""
    prompt = f"""
Là CloudBot, một trợ lý bảo vệ môi trường, hãy đưa ra một vài giải pháp cụ thể và thực tế để ngăn chặn con người phá hủy môi trường, đặc biệt liên quan đến loại rác thải **{waste_type}**. Đồng thời, hãy lồng ghép thông điệp nâng cao nhận thức về môi trường xanh.
Trả lời bằng tiếng Việt, khoảng 3-5 gạch đầu dòng.
"""
    return call_gemini_api(prompt)

def get_green_environment_awareness():
    """Generates general awareness messages about green environment."""
    prompt = """
Là CloudBot, một trợ lý bảo vệ môi trường, hãy đưa ra 3-5 thông điệp truyền cảm hứng và nâng cao nhận thức về việc bảo vệ môi trường xanh, tầm quan trọng của việc giảm rác thải, tái chế và sống bền vững.
Trả lời bằng tiếng Việt, khoảng 3-5 gạch đầu dòng.
"""
    return call_gemini_api(prompt)

# --- Chức năng phân loại rác thải ---
@st.cache_resource
def load_classification_model():
    """Tải mô hình phân loại rác đã huấn luyện."""
    model_path = 'model.keras' # Đảm bảo đường dẫn này đúng
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Không tìm thấy mô hình tại đường dẫn: {model_path}. Vui lòng đảm bảo bạn đã huấn luyện và lưu mô hình.")
        return None

# Tải mô hình
classification_model = load_classification_model()

# Lấy tên các lớp từ train_generator
data_dir = "./dataset/garbage_classification"
input_size = (128, 128)
val_frac = 0.2
batch_size = 32

data_augmentor = ImageDataGenerator(samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir,
                                                     target_size=input_size,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     subset="training")
CLASS_NAMES = list(train_generator.class_indices.keys())

def classify_waste_image(image):
    """
    Phân loại hình ảnh rác thải sử dụng mô hình đã huấn luyện.
    Args:
        image (PIL.Image.Image): Hình ảnh đầu vào.
    Returns:
        str: Tên loại rác được dự đoán.
    """
    if classification_model is None:
        return "Không thể phân loại. Mô hình chưa được tải."

    img_array = np.array(image.resize(input_size))
    img_array = img_array.astype('float32')
    img_array -= np.mean(img_array)
    img_array /= np.std(img_array) + 1e-7 # Tránh chia cho 0

    img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch

    predictions = classification_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    return predicted_class_name

def get_detailed_waste_info(image, predicted_class):
    """
    Sử dụng Gemini Vision để đưa ra giải thích chi tiết về loại rác,
    lý do phân loại và cách xử lý.
    """
    # Chuyển đổi PIL Image sang bytes để gửi qua API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG') # Hoặc JPEG
    img_byte_arr = img_byte_arr.getvalue()

    # Tạo đối tượng Gemini Image
    gemini_image = {'mime_type': 'image/png', 'data': img_byte_arr} # Hoặc image/jpeg

    prompt_parts = [
        gemini_image,
        f"""Dựa trên hình ảnh và thông tin đã được phân loại sơ bộ là **{predicted_class}**, hãy cung cấp thông tin chi tiết về loại rác này theo định dạng sau:

**Loại rác:** [Tên loại rác bạn phân loại]
**Giải thích:** [Lý do bạn phân loại như vậy, dựa trên đặc điểm hình ảnh và kiến thức chung về loại rác này. Ví dụ: vật liệu, hình dáng, kết cấu, v.v.]
**Cách xử lý:** [Hướng dẫn ngắn gọn và cụ thể về cách xử lý loại rác này để bảo vệ môi trường, ví dụ: rửa sạch, gấp gọn, bỏ vào thùng riêng, mang đến điểm thu gom đặc biệt, có tái chế được không, v.v.]

Nếu không thể xác định rõ ràng, hãy trả lời 'Không xác định' và giải thích lý do không thể xác định. Đảm bảo câu trả lời hoàn toàn bằng tiếng Việt và thân thiện.
"""
    ]

    try:
        response = generative_model_vision.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi gọi Gemini Vision API: {e}")
        return "Không thể lấy thông tin chi tiết về rác thải lúc này. Vui lòng thử lại sau."


# --- 4. STREAMLIT APP LAYOUT ---
st.sidebar.title("Điều Hướng 🧭")

app_mode = st.sidebar.radio(
    "Chọn Chức Năng:",
    [
        "Giới Thiệu CloudBot ✨",
        "Phân Loại Rác Thải 🗑️",
        "Giải Pháp & Nhận Thức Xanh 🌿",
        "Trò Chuyện Với CloudBot 🤖"
    ]
)

# --- App content based on selected mode ---

# Phần 1: Giới Thiệu CloudBot
if app_mode == "Giới Thiệu CloudBot ✨":
    st.title("Chào Mừng Bạn Đến Với CloudBot 👋")
    st.markdown("---")
    st.markdown("""
    **CloudBot** là người bạn đồng hành AI được thiết kế để nâng cao nhận thức của cộng đồng về **rác thải** và **bảo vệ môi trường**. Với giao diện thân thiện và các tính năng thông minh, CloudBot giúp bạn dễ dàng hơn trong việc tìm hiểu và thực hành lối sống xanh.

    **Các tính năng chính của CloudBot:**

    * **Phân Loại Rác Thải 🗑️:** Chụp hoặc tải ảnh rác thải lên, CloudBot sẽ giúp bạn nhận diện loại rác đó (nhựa, giấy, kim loại, thủy tinh, hữu cơ, rác tổng hợp) và cung cấp thông tin hữu ích về cách phân loại đúng.
    * **Giải Pháp & Nhận Thức Xanh 🌿:** Khám phá các giải pháp thiết thực để giảm thiểu tác động tiêu cực đến môi trường, cũng như những thông điệp truyền cảm hứng về lối sống xanh, bền vững.
    * **Trò Chuyện Với CloudBot 🤖:** Đặt câu hỏi và nhận câu trả lời về mọi thứ liên quan đến môi trường, rác thải, tái chế, và các vấn đề môi trường khác. CloudBot luôn sẵn lòng chia sẻ kiến thức và cùng bạn thảo luận!

    Hãy cùng CloudBot bắt đầu hành trình tạo nên sự thay đổi tích cực cho hành tinh của chúng ta!
    """)
    st.markdown("---")
    st.info("Sử dụng thanh điều hướng bên trái để khám phá các tính năng của CloudBot!")

# Phần 2: Phân Loại Rác Thải (Bây giờ đã có chức năng phân loại!)
elif app_mode == "Phân Loại Rác Thải 🗑️":
    st.title("🗑️ Phân Loại Rác Thải Bằng Hình Ảnh")
    st.markdown("---")
    st.write("Tải lên một hình ảnh rác thải để CloudBot giúp bạn xác định loại rác và cách xử lý.")

    uploaded_file = st.file_uploader("Chọn một tệp hình ảnh (.jpg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Hình ảnh tải lên.', use_column_width=True)
        st.write("")
        st.write("Đang phân loại và tạo thông tin chi tiết...")

        # Bước 1: Phân loại rác bằng mô hình TensorFlow
        predicted_class_tf = classify_waste_image(image)
        st.success(f"Mô hình cục bộ dự đoán đây là: **{predicted_class_tf.replace('_', ' ').capitalize()}**")

        # Bước 2: Sử dụng Gemini Vision để lấy thông tin chi tiết
        with st.spinner("Đang yêu cầu CloudBot giải thích chi tiết..."):
            detailed_info = get_detailed_waste_info(image, predicted_class_tf)
            st.markdown(detailed_info)


# Phần 3: Giải Pháp & Nhận Thức Xanh
elif app_mode == "Giải Pháp & Nhận Thức Xanh 🌿":
    st.title("🌿 Giải Pháp & Nhận Thức Môi Trường Xanh")
    st.markdown("---")

    st.write("Cùng CloudBot khám phá các giải pháp thiết thực và những thông điệp ý nghĩa để chung tay bảo vệ hành tinh của chúng ta!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Giải pháp chung để ngăn chặn phá hủy môi trường:")
        if st.button("Nhận giải pháp chung🍃"):
            with st.spinner("Đang tạo giải pháp..."):
                solution = get_environmental_solution("chung")
                if solution:
                    st.markdown(solution)
                else:
                    st.warning("Không thể tạo giải pháp môi trường lúc này.")

    with col2:
        st.subheader("Nâng cao nhận thức về môi trường xanh:")
        if st.button("Nhận thông điệp nhận thức🍀"):
            with st.spinner("Đang tạo thông điệp..."):
                awareness_message = get_green_environment_awareness()
                if awareness_message:
                    st.markdown(awareness_message)
                else:
                    st.warning("Không thể tạo thông điệp nhận thức lúc này.")

    st.markdown("---")
    st.info("Hãy cùng thực hiện những hành động nhỏ mỗi ngày để tạo nên sự khác biệt lớn!")

# Phần 4: Trò Chuyện Với CloudBot
elif app_mode == "Trò Chuyện Với CloudBot 🤖":
    st.title("🤖 Trò Chuyện Với CloudBot")
    st.markdown("---")
    st.write(f"*{initial_bot_message_config}*")

    # Phần hiển thị lịch sử trò chuyện chính
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(" ".join(str(part) for part in message["parts"]))

    # Logic quản lý trạng thái khóa chat và input người dùng
    user_query = None
    LOCK_DURATION_SECONDS = 60 # 1 phút khóa

    if st.session_state.chat_locked:
        if st.session_state.chat_locked_time is None:
            st.session_state.chat_locked_time = datetime.datetime.now()

        time_elapsed = (datetime.datetime.now() - st.session_state.chat_locked_time).total_seconds()
        
        if time_elapsed >= LOCK_DURATION_SECONDS:
            st.session_state.chat_locked = False
            st.session_state.negative_count = 0
            st.session_state.chat_locked_time = None
            st.session_state.chat_history.append({"role": "assistant", "parts": ["CloudBot: Bạn có thể tiếp tục trò chuyện rồi đó! Hãy cùng tìm hiểu về môi trường nhé!"]})
            st.rerun()
        else:
            remaining_seconds = int(LOCK_DURATION_SECONDS - time_elapsed)
            minutes = remaining_seconds // 60
            seconds = remaining_seconds % 60
            st.warning(f"Bạn đã sử dụng quá nhiều từ ngữ tiêu cực. Vui lòng ngừng trò chuyện {minutes} phút {seconds} giây nữa.")
            user_query = None
    else:
        user_query = st.chat_input("Hỏi CloudBot về môi trường hoặc rác thải...")

    # Phần logic xử lý tin nhắn của người dùng
    if user_query:
        if detect_negative_language(user_query):
            st.session_state.negative_count += 1
            if st.session_state.negative_count >= 3:
                st.session_state.chat_locked = True
                st.session_state.chat_locked_time = datetime.datetime.now()
                st.session_state.chat_history.append({"role": "assistant", "parts": ["CloudBot: Tôi nhận thấy bạn đang sử dụng nhiều từ ngữ tiêu cực. Để giữ cho cuộc trò chuyện thân thiện, tôi sẽ tạm dừng phản hồi. Vui lòng thử lại sau khi bình tĩnh lại nhé!"]})
                with st.chat_message("assistant"):
                    st.markdown("Tôi nhận thấy bạn đang sử dụng nhiều từ ngữ tiêu cực. Để giữ cho cuộc trò chuyện thân thiện, tôi sẽ tạm dừng phản hồi. Vui lòng thử lại sau khi bình tĩnh lại nhé!")
                st.rerun()
            else:
                st.session_state.chat_history.append({"role": "user", "parts": [user_query]})
                with st.chat_message("user"):
                    st.markdown(user_query)
                warning_message = f"CloudBot: Vui lòng sử dụng ngôn ngữ tích cực. Bạn đã sử dụng từ tiêu cực {st.session_state.negative_count} lần.🚫"
                st.session_state.chat_history.append({"role": "assistant", "parts": [warning_message]})
                with st.chat_message("assistant"):
                    st.markdown(warning_message)
        else:
            st.session_state.negative_count = 0
            
            st.session_state.chat_history.append({"role": "user", "parts": [user_query]})
            with st.chat_message("user"):
                st.markdown(user_query)

            prompt_for_chat = f"""
Bạn là CloudBot, một trợ lý AI chuyên về phân loại rác thải, bảo vệ môi trường và nâng cao nhận thức về môi trường xanh. Hãy trả lời câu hỏi của người dùng một cách thân thiện, hữu ích và mang tính xây dựng.

Lịch sử trò chuyện trước đó (để giữ ngữ cảnh):
{json.dumps(st.session_state.chat_history[:-1], ensure_ascii=False)}

Câu hỏi mới của người dùng: "{user_query}"
"""
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = call_gemini_api(prompt_for_chat)
                message_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "parts": [full_response]})

    # --- Phần Lịch sử cuộc trò chuyện (có thể ẩn/hiện) và Nút xóa ---
    st.sidebar.markdown("---")
    st.sidebar.header("Lịch Sử Cuộc Trò Chuyện 📜")
    
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("Hiện/Ẩn Chi Tiết"):
            st.session_state.show_chat_history_section = not st.session_state.show_chat_history_section
    with col2:
        if st.button("Xóa Lịch Sử"):
            st.session_state.chat_history = [{"role": "assistant", "parts": [initial_bot_message_config]}]
            st.session_state.negative_count = 0
            st.session_state.chat_locked = False
            st.session_state.chat_locked_time = None
            st.session_state.show_chat_history_section = False
            st.rerun()

    if st.session_state.show_chat_history_section:
        st.sidebar.subheader("Chi tiết:")
        for i, message in enumerate(st.session_state.chat_history):
            role = "Bạn" if message["role"] == "user" else "CloudBot"
            content = " ".join(str(part) for part in message["parts"])
            st.sidebar.markdown(f"**{role}:** {content[:100]}...")
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("Nhấn 'Hiện/Ẩn Chi Tiết' để xem toàn bộ lịch sử.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("CloudBot 💬 - Trợ lý môi trường AI")