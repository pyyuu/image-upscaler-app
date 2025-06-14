# ------------------------------------------------------------------
# 1. ライブラリのインポート
# ------------------------------------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from realesrgan import RealESRGAN
from gfpgan import GFPGANer

# ------------------------------------------------------------------
# 2. アプリケーションの基本設定
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Upscaler & Enhancer",
    page_icon="✨",
    layout="wide"
)

st.title("🖼️ AI 画像アップスケーラー & 顔高画質化ツール")
st.write("Real-ESRGANによる高解像度化と、GFPGANによる顔の補正を同時に行います。")
st.markdown("---")

# ------------------------------------------------------------------
# 3. AIモデルをロードする関数（キャッシュで高速化）
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Real-ESRGANとGFPGANのモデルをロードして返す"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Real-ESRGANモデル (汎用的なアップスケーラー)
    model_realesrgan = RealESRGAN(device, scale=4)
    model_realesrgan.load_weights('weights/RealESRGAN_x4plus.pth', download=True)

    # GFPGANモデル (顔に特化した高画質化)
    # model_pathは自動でダウンロードされる
    model_gfpgan = GFPGANer(
        model_path='weights/GFPGANv1.4.pth',
        upscale=4, # Real-ESRGANのスケールと合わせる
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=model_realesrgan # 背景のアップスケールにはReal-ESRGANを利用
    )
    return model_realesrgan, model_gfpgan, device

st.info("AIモデルの準備をしています...初回は数分かかることがあります。")
try:
    realesrgan_model, gfpgan_model, device = load_models()
    st.success(f"AIモデルの準備が完了しました (実行デバイス: {device})")
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.error("しばらくしてからページを再読み込みしてください。")
    st.stop()

# ------------------------------------------------------------------
# 4. メインの処理
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("高画質化したい画像をアップロードしてください", type=['png', 'jpg', 'jpeg'])
enhance_face = st.checkbox('顔を特に綺麗に補正する (GFPGAN)', value=True)

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("元画像")
            st.image(input_image, caption=f"Original ({input_image.width}x{input_image.height})", use_column_width=True)

        if st.button('✨ 高画質化スタート！', type="primary"):
            with st.spinner('AIが高画質化処理を実行中です...'):
                
                # Pillow ImageをNumpy配列に変換
                input_image_np = np.array(input_image)

                if enhance_face:
                    # GFPGANで顔を補正し、同時に背景もアップスケール
                    _, _, output_image_np = gfpgan_model.enhance(
                        input_image_np,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                else:
                    # Real-ESRGANのみで画像全体をアップスケール
                    output_image_np = realesrgan_model.predict(input_image_np)

                # Numpy配列をPillow Imageに変換
                output_image = Image.fromarray(output_image_np)

            with col2:
                st.subheader("高画質化後の画像")
                st.image(output_image, caption=f"Enhanced ({output_image.width}x{output_image.height})", use_column_width=True)

            # ダウンロードボタンの準備
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="画像をダウンロード",
                data=byte_im,
                file_name=f"enhanced_{uploaded_file.name}.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
