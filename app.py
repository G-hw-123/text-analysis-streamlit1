import streamlit as st
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 设置页面配置
st.set_page_config(page_title="文本分析工具", page_icon="📝", layout="wide")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 页面标题
st.title("📝 简易文本分析 Web 应用")
st.divider()

# 1. 文本输入区域
st.subheader("1. 请输入待分析文本")
content = st.text_area("在此粘贴文本内容", height=200)
analyze_btn = st.button("开始分析", type="primary")

# 2. 文本分析逻辑
if analyze_btn and content.strip():
    st.success("分析完成，结果如下：")
    st.divider()

    # 基础统计
    total_len = len(content)
    char_count = len(content.replace(" ", "").replace("\n", ""))
    st.metric(label="总字符数(含空格)", value=total_len)
    st.metric(label="纯文本字符数(不含空格/换行)", value=char_count)

    # 分词 + 词频统计
    st.subheader("2. 分词 & 高频词统计")
    words = jieba.lcut(content)
    # 过滤单字、空格
    filter_words = [w for w in words if len(w) > 1 and w.strip()]
    word_freq = Counter(filter_words)
    top10_words = word_freq.most_common(10)

    st.write("Top10 高频词汇：")
    for word, count in top10_words:
        st.text(f"{word} ：{count} 次")

    # 3. 生成词云
    st.subheader("3. 文本词云图")
    word_str = " ".join(filter_words)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        font_path="simhei.ttf",  # 无本地字体可注释此行
        max_words=100
    ).generate(word_str)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

elif analyze_btn and not content.strip():
    st.warning("请先输入文本内容！")
