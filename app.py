import streamlit as st
import jieba
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
import numpy as np
from io import StringIO
import math

# 页面配置
st.set_page_config(page_title="多功能文本分析工具", page_icon="📊", layout="wide")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化session_state
if "text" not in st.session_state:
    st.session_state.text = ""
if "compare_text" not in st.session_state:
    st.session_state.compare_text = ""

# 标题
st.title("📊 多功能中文文本分析工具")
st.divider()

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 分析设置")
    stopwords_file = st.file_uploader("上传自定义停用词表（可选）", type=["txt"])
    top_n = st.slider("高频词Top N", 5, 50, 10)
    st.divider()
    st.subheader("功能选择")
    show_basic = st.checkbox("基础统计", value=True)
    show_wordfreq = st.checkbox("词频分析", value=True)
    show_wordcloud = st.checkbox("词云图", value=True)
    show_sentiment = st.checkbox("情感分析", value=True)
    show_keywords = st.checkbox("关键词提取", value=True)
    show_compare = st.checkbox("文本对比", value=False)
    show_readability = st.checkbox("可读性分析", value=True)

# 停用词处理
stopwords = set()
if stopwords_file:
    stopwords = set([line.strip() for line in stopwords_file.read().decode("utf-8").splitlines()])
else:
    # 内置基础停用词
    default_stopwords = {"的", "了", "和", "是", "在", "我", "有", "就", "也", "都", "要", "这", "那", "与", "或", "但", "而", "所以", "因为", "如果", "可以", "能", "会", "可能", "应该", "不", "没", "无", "很", "非常", "太", "更", "最", "比较", "还", "又", "再", "已", "已", "曾", "曾", "经", "将", "会", "是", "有", "为", "以", "于", "对", "向", "从", "到", "及", "等", "这些", "那些", "什么", "怎么", "如何", "为什么", "多少", "几", "哪", "哪里", "谁", "什么", "时候", "年", "月", "日", "今天", "明天", "昨天", "现在", "过去", "未来", "可以", "能够", "可能", "必须", "应该", "应当", "要", "得", "使", "让", "叫", "令", "把", "被", "给", "由", "被", "受", "所", "被", "对", "于", "关于", "至于", "对于", "除了", "除开", "除非", "如果", "只要", "只有", "因为", "由于", "所以", "因此", "于是", "从而", "以致", "以免", "以便", "为了", "为着", "为的是", "使得", "致使", "导致", "引起", "造成", "产生", "发生", "出现", "形成", "变成", "成为", "是", "等于", "相当于", "即", "就是", "也就是", "换句话说", "也就是说", "即", "即", "即", "即"}
    stopwords = default_stopwords

# 文本输入区域
tab1, tab2 = st.tabs(["直接输入文本", "上传文本文件"])
with tab1:
    st.session_state.text = st.text_area("请输入或粘贴文本内容", height=200, value=st.session_state.text)
with tab2:
    uploaded_file = st.file_uploader("上传TXT文件", type=["txt"])
    if uploaded_file:
        st.session_state.text = uploaded_file.read().decode("utf-8")
        st.success("文件上传成功！")

# 文本对比区域
if show_compare:
    st.divider()
    st.subheader("📝 文本对比功能")
    st.session_state.compare_text = st.text_area("请输入对比文本", height=150, value=st.session_state.compare_text)

# 分析按钮
analyze_btn = st.button("🚀 开始全面分析", type="primary", use_container_width=True)

if analyze_btn and st.session_state.text.strip():
    text = st.session_state.text
    compare_text = st.session_state.compare_text

    st.divider()
    st.header("📈 分析结果")

    # --------------------------
    # 1. 基础统计
    # --------------------------
    if show_basic:
        st.subheader("1. 基础文本统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总字符数", len(text))
        col2.metric("纯文本字符数（不含空格/换行）", len(re.sub(r"\s", "", text)))
        col3.metric("中文字符数", len(re.findall(r"[\u4e00-\u9fff]", text)))
        col4.metric("英文字符数", len(re.findall(r"[a-zA-Z]", text)))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("数字字符数", len(re.findall(r"[0-9]", text)))
        col6.metric("标点符号数", len(re.findall(r"[^\u4e00-\u9fff\w\s]", text)))
        col7.metric("段落数", len(text.split("\n\n")))
        col8.metric("句子数", len(re.split(r"[。！？.!?]", text)) - 1)

        # 文本预览
        st.expander("查看原始文本预览").write(text[:500] + ("..." if len(text) > 500 else ""))

    # --------------------------
    # 2. 词频分析
    # --------------------------
    if show_wordfreq:
        st.divider()
        st.subheader("2. 分词与词频分析")
        words = jieba.lcut(text)
        filtered_words = [w for w in words if len(w) > 1 and w.strip() and w not in stopwords]
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(top_n)

        # 词频表格
        freq_df = pd.DataFrame(top_words, columns=["词语", "出现次数"])
        st.dataframe(freq_df, use_container_width=True)

        # 词频柱状图
        if len(top_words) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            words_list = [w[0] for w in top_words]
            counts_list = [w[1] for w in top_words]
            ax.bar(words_list, counts_list, color="skyblue")
            plt.xticks(rotation=45, ha="right")
            ax.set_title(f"Top {top_n} 高频词分布")
            ax.set_ylabel("出现次数")
            st.pyplot(fig)

    # --------------------------
    # 3. 词云图
    # --------------------------
    if show_wordcloud and len(filtered_words) > 0:
        st.divider()
        st.subheader("3. 文本词云图")
        word_str = " ".join(filtered_words)
        wc = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            max_words=200,
            max_font_size=100,
            random_state=42
        ).generate(word_str)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    # --------------------------
    # 4. 情感分析
    # --------------------------
    if show_sentiment:
        st.divider()
        st.subheader("4. 情感倾向分析")
        s = SnowNLP(text)
        sentiment_score = s.sentiments

        col1, col2 = st.columns(2)
        with col1:
            st.metric("情感倾向分数", f"{sentiment_score:.2f}")
            if sentiment_score > 0.7:
                st.success("整体情感倾向：积极 😊")
            elif sentiment_score < 0.3:
                st.error("整体情感倾向：消极 😔")
            else:
                st.info("整体情感倾向：中性 😐")

        with col2:
            # 情感分布饼图
            fig, ax = plt.subplots()
            labels = ["消极", "中性", "积极"]
            sizes = [1 - sentiment_score, 0.1, sentiment_score]
            colors = ["#ff9999", "#66b3ff", "#99ff99"]
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("情感分布")
            st.pyplot(fig)

    # --------------------------
    # 5. 关键词提取（基于TF-IDF简化实现）
    # --------------------------
    if show_keywords and len(filtered_words) > 0:
        st.divider()
        st.subheader("5. 关键词提取")
        # 简化TF-IDF实现
        doc_words = filtered_words
        word_tf = Counter(doc_words)
        total_words = len(doc_words)
        # 简化IDF（假设文档数为1）
        keywords = [(word, count / total_words) for word, count in word_tf.items()]
        keywords.sort(key=lambda x: x[1], reverse=True)
        top_keywords = keywords[:top_n]

        kw_df = pd.DataFrame(top_keywords, columns=["关键词", "权重"])
        st.dataframe(kw_df, use_container_width=True)

    # --------------------------
    # 6. 文本对比
    # --------------------------
    if show_compare and compare_text.strip():
        st.divider()
        st.subheader("6. 文本相似度对比")
        # Jaccard相似度计算
        words1 = set(jieba.lcut(text))
        words2 = set(jieba.lcut(compare_text))
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_sim = intersection / union if union != 0 else 0

        st.metric("Jaccard相似度", f"{jaccard_sim:.2%}")
        if jaccard_sim > 0.7:
            st.success("文本相似度较高")
        elif jaccard_sim > 0.3:
            st.info("文本有一定相似度")
        else:
            st.warning("文本相似度较低")

    # --------------------------
    # 7. 可读性分析（中文字体难度估算）
    # --------------------------
    if show_readability:
        st.divider()
        st.subheader("7. 文本可读性分析")
        # 中文字符平均笔画数（简化估算）
        common_chars = set("的一是了有不人在这中大为会上个国就与说年也要出得时用道那说子自和他地")
        total_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        common_count = len([c for c in text if c in common_chars])
        common_ratio = common_count / total_chars if total_chars > 0 else 0

        col1, col2 = st.columns(2)
        col1.metric("常用汉字占比", f"{common_ratio:.2%}")
        if common_ratio > 0.8:
            col1.success("文本难度：简单（常用字占比高）")
        elif common_ratio > 0.6:
            col1.info("文本难度：中等")
        else:
            col1.warning("文本难度：偏难（生僻字较多）")

        # 平均句长
        sentences = re.split(r"[。！？.!?]", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = np.mean([len(s) for s in sentences]) if sentences else 0
        col2.metric("平均句长（字符数）", f"{avg_sentence_len:.1f}")
        if avg_sentence_len < 15:
            col2.success("句子长度：简短易懂")
        elif avg_sentence_len < 30:
            col2.info("句子长度：适中")
        else:
            col2.warning("句子长度：偏长，阅读难度较高")

    st.divider()
    st.success("✅ 分析完成！以上结果仅供参考。")

elif analyze_btn and not st.session_state.text.strip():
    st.warning("⚠️ 请先输入或上传文本内容！")

# 页脚
st.divider()
st.caption("💡 提示：支持中文文本分析、词频统计、词云生成、情感分析、关键词提取等功能，可上传自定义停用词表优化分析结果。")
