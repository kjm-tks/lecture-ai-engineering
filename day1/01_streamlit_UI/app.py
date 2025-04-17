import streamlit as st
import pandas as pd
import numpy as np
import time

# ============================================
# ページ設定
# ============================================
st.set_page_config(
    page_title="Streamlit デモ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: 600;
        color: #0D47A1;
        border-bottom: 1px solid #90CAF9;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .description {
        font-size: 18px !important;
        color: #424242;
    }
    .sidebar-content {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
    }
    .highlight {
        background-color: #E1F5FE;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #03A9F4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# タイトルと説明
# ============================================
st.markdown('<p class="main-header">Streamlit 初心者向けデモ</p>', unsafe_allow_html=True)
st.markdown('<p class="description">コメントを解除しながらStreamlitの機能を学びましょう</p>', unsafe_allow_html=True)
st.markdown('<div class="highlight">このデモコードでは、コメントアウトされた部分を順番に解除しながらUIの変化を確認できます。</div>', unsafe_allow_html=True)

# ============================================
# サイドバー 
# ============================================
st.sidebar.markdown('<p class="sub-header">デモのガイド</p>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="sidebar-content">コードのコメントを解除して、Streamlitのさまざまな機能を確認しましょう。</div>', unsafe_allow_html=True)
    
    # サイドバーに目次を追加
    st.markdown('### 目次')
    st.markdown('- [基本的なUI要素](#基本的なui要素)')
    st.markdown('- [レイアウト](#レイアウト)')
    st.markdown('- [データ表示](#データの表示)')
    st.markdown('- [グラフ表示](#グラフの表示)')
    st.markdown('- [インタラクティブ機能](#インタラクティブ機能)')

# ============================================
# 基本的なUI要素
# ============================================
st.markdown('<p class="sub-header" id="基本的なui要素">基本的なUI要素</p>', unsafe_allow_html=True)

# テキスト入力
st.subheader("テキスト入力")
name = st.text_input("あなたの名前", "ゲスト")
st.write(f"こんにちは、{name}さん！")

# ボタン
st.subheader("ボタン")
if st.button("クリックしてください"):
    st.success("ボタンがクリックされました！")

# チェックボックス
st.subheader("チェックボックス")
if st.checkbox("チェックを入れると追加コンテンツが表示されます"):
    st.info("これは隠れたコンテンツです！")

# スライダー
st.subheader("スライダー")
age = st.slider("年齢", 0, 100, 25)
st.write(f"あなたの年齢: {age}")

# セレクトボックス
st.subheader("セレクトボックス")
option = st.selectbox(
    "好きなプログラミング言語は?",
    ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
)
st.write(f"あなたは{option}を選びました")

# ============================================
# レイアウト
# ============================================
st.markdown('<p class="sub-header" id="レイアウト">レイアウト</p>', unsafe_allow_html=True)

# カラム
st.subheader("カラムレイアウト")
col1, col2 = st.columns(2)
with col1:
    st.write("これは左カラムです")
    st.number_input("数値を入力", value=10)
with col2:
    st.write("これは右カラムです")
    st.metric("メトリクス", "42", "2%")

# タブ
st.subheader("タブ")
tab1, tab2 = st.tabs(["第1タブ", "第2タブ"])
with tab1:
    st.write("これは第1タブの内容です")
with tab2:
    st.write("これは第2タブの内容です")

# エクスパンダー
st.subheader("エクスパンダー")
with st.expander("詳細を表示"):
    st.write("これはエクスパンダー内の隠れたコンテンツです")
    st.code("print('Hello, Streamlit！')")

# ============================================
# データ表示
# ============================================
st.markdown('<p class="sub-header" id="データの表示">データの表示</p>', unsafe_allow_html=True)

# サンプルデータフレームを作成
df = pd.DataFrame({
    '名前': ['田中', '鈴木', '佐藤', '高橋', '伊藤'],
    '年齢': [25, 30, 22, 28, 33],
    '都市': ['東京', '大阪', '福岡', '札幌', '名古屋']
})

# データフレーム表示
st.subheader("データフレーム")
st.dataframe(df, use_container_width=True)

# テーブル表示
st.subheader("テーブル")
st.table(df)

# メトリクス表示
st.subheader("メトリクス")
col1, col2, col3 = st.columns(3)
col1.metric("温度", "23°C", "1.5°C")
col2.metric("湿度", "45%", "-5%")
col3.metric("気圧", "1013hPa", "0.1hPa")

# ============================================
# グラフ表示
# ============================================
st.markdown('<p class="sub-header" id="グラフの表示">グラフの表示</p>', unsafe_allow_html=True)

# ラインチャート
st.subheader("ラインチャート")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C'])
st.line_chart(chart_data)

# バーチャート
st.subheader("バーチャート")
chart_data = pd.DataFrame({
    'カテゴリ': ['A', 'B', 'C', 'D'],
    '値': [10, 25, 15, 30]
}).set_index('カテゴリ')
st.bar_chart(chart_data)

# ============================================
# インタラクティブ機能
# ============================================
st.markdown('<p class="sub-header" id="インタラクティブ機能">インタラクティブ機能</p>', unsafe_allow_html=True)

# プログレスバー
st.subheader("プログレスバー")
progress = st.progress(0)
if st.button("進捗をシミュレート"):
    for i in range(101):
        time.sleep(0.01)
        progress.progress(i / 100)
    st.balloons()

# ファイルアップロード
st.subheader("ファイルアップロード")
uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "txt"])
if uploaded_file is not None:
    # ファイルのデータを表示
    bytes_data = uploaded_file.getvalue()
    st.write(f"ファイルサイズ: {len(bytes_data)} bytes")
    
    # CSVの場合はデータフレームとして読み込む
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("CSVデータのプレビュー:")
        st.dataframe(df.head())

# ============================================
# デモの使用方法
# ============================================
st.divider()
st.markdown('<p class="sub-header">このデモの使い方</p>', unsafe_allow_html=True)
st.markdown("""
1. コードエディタでコメントアウトされた部分を見つけます（#で始まる行）
2. 確認したい機能のコメントを解除します（先頭の#を削除）
3. 変更を保存して、ブラウザで結果を確認します
4. 様々な組み合わせを試して、UIがどのように変化するか確認しましょう
""")

st.code("""
# コメントアウトされた例:
# if st.button("クリックしてください"):
#     st.success("ボタンがクリックされました！")

# コメントを解除した例:
if st.button("クリックしてください"):
    st.success("ボタンがクリックされました！")
""")
