import streamlit as st
import jieba
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
import numpy as np
import requests
from io import BytesIO

# ===================== 页面全局配置 =====================
st.set_page_config(page_title="多功能文本分析工具", page_icon="📊", layout="wide")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化会话状态
if "text" not in st.session_state:
    st.session_state.text = ""
if "compare_text" not in st.session_state:
    st.session_state.compare_text = ""

# 标题
st.title("📊 多功能中文文本分析工具")
st.divider()

# ===================== 侧边栏配置 =====================
with st.sidebar:
    st.header("⚙️ 分析设置")
    stopwords_file = st.file_uploader("上传自定义停用词表（可选）", type=["txt"])
    top_n = st.slider("高频词 Top N", min_value=5, max_value=50, value=10)
    st.divider()
    st.subheader("功能选择")
    show_basic = st.checkbox("基础统计", value=True)
    show_wordfreq = st.checkbox("词频分析", value=True)
    show_wordcloud = st.checkbox("词云图", value=True)
    show_sentiment = st.checkbox("情感分析", value=True)
    show_keywords = st.checkbox("关键词提取", value=True)
    show_compare = st.checkbox("文本对比", value=False)
    show_readability = st.checkbox("可读性分析", value=True)

# ===================== 加载停用词 =====================
stopwords = set()
if stopwords_file:
    try:
        stopwords = set([line.strip() for line in stopwords_file.read().decode("utf-8").splitlines()])
    except:
        st.warning("停用词文件读取失败，将使用默认停用词")

# 内置通用中文停用词
default_stopwords = {
    "的", "了", "和", "是", "在", "我", "有", "就", "也", "都", "要", "这", "那",
    "与", "或", "但", "而", "所以", "因为", "如果", "可以", "能", "会", "可能",
    "应该", "不", "没", "无", "很", "非常", "太", "更", "最", "比较", "还", "又",
    "再", "已", "曾", "经", "将", "为", "以", "于", "对", "向", "从", "到", "及",
    "等", "这些", "那些", "什么", "怎么", "如何", "为什么", "多少", "几", "哪",
    "哪里", "谁", "时候", "年", "月", "日", "今天", "明天", "昨天", "现在",
    "过去", "未来", "必须", "应当", "得", "使", "让", "把", "被", "给", "由"
}
if not stopwords:
    stopwords = default_stopwords

# ===================== 文本输入区域 =====================
tab1, tab2 = st.tabs(["直接输入文本", "上传文本文件"])
with tab1:
    st.session_state.text = st.text_area("请输入或粘贴文本内容", height=200, value=st.session_state.text)
with tab2:
    uploaded_file = st.file_uploader("上传 TXT 文本文件", type=["txt"])
    if uploaded_file:
        try:
            st.session_state.text = uploaded_file.read().decode("utf-8")
            st.success("✅ 文件上传成功！")
        except:
            st.error("❌ 文件编码错误，请使用 UTF-8 编码的 TXT 文件")

# 文本对比输入框
if show_compare:
    st.divider()
    st.subheader("📝 文本对比功能")
    st.session_state.compare_text = st.text_area("请输入对比文本", height=150, value=st.session_state.compare_text)

# 分析按钮
analyze_btn = st.button("🚀 开始全面分析", type="primary", use_container_width=True)

# ===================== 核心分析逻辑 =====================
if analyze_btn:
    raw_text = st.session_state.text.strip()
    if not raw_text:
        st.warning("⚠️ 请先输入或上传文本内容！")
    else:
        compare_text = st.session_state.compare_text.strip()
        st.divider()
        st.header("📈 分析结果")

        # 全局分词 + 过滤（一次分词，多处复用，提升效率）
        words = jieba.lcut(raw_text)
        filtered_words = [w for w in words if len(w) > 1 and w.strip() and w not in stopwords]

        # 1. 基础统计
        if show_basic:
            st.subheader("1. 基础文本统计")
            total_char = len(raw_text)
            pure_char = len(re.sub(r"\s", "", raw_text))
            cn_char = len(re.findall(r"[\u4e00-\u9fff]", raw_text))
            en_char = len(re.findall(r"[a-zA-Z]", raw_text))
            num_char = len(re.findall(r"[0-9]", raw_text))
            punct_char = len(re.findall(r"[^\u4e00-\u9fff\w\s]", raw_text))
            para_count = len(raw_text.split("\n\n"))
            sent_count = max(len(re.split(r"[。！？.!?]", raw_text)) - 1, 0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总字符数", total_char)
            col2.metric("纯文本字符数", pure_char)
            col3.metric("中文字符数", cn_char)
            col4.metric("英文字符数", en_char)

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("数字字符数", num_char)
            col6.metric("标点符号数", punct_char)
            col7.metric("段落数", para_count)
            col8.metric("句子数", sent_count)

            # 文本预览
            with st.expander("查看原始文本预览"):
                preview = raw_text[:500] + ("..." if len(raw_text) > 500 else "")
                st.write(preview)

        # 2. 词频分析
        if show_wordfreq:
            st.divider()
            st.subheader("2. 分词与词频分析")
            if filtered_words:
                word_freq = Counter(filtered_words)
                top_words = word_freq.most_common(top_n)
                freq_df = pd.DataFrame(top_words, columns=["词语", "出现次数"])
                st.dataframe(freq_df, use_container_width=True)

                # 词频柱状图
                fig, ax = plt.subplots(figsize=(10, 5))
                word_list = [item[0] for item in top_words]
                count_list = [item[1] for item in top_words]
                ax.bar(word_list, count_list, color="#5B9BD5")
                plt.xticks(rotation=45, ha="right")
                ax.set_title(f"Top {top_n} 高频词分布")
                ax.set_ylabel("出现次数")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("当前文本过滤后无有效词汇，无法生成词频统计")

        # 3. 词云图（【已修复：云端/本地通用】）
        if show_wordcloud:
            st.divider()
            st.subheader("3. 文本词云图")
            if filtered_words:
                word_content = " ".join(filtered_words)
                # 云端自动下载思源黑体，解决字体找不到的问题
                try:
                    font_url = "https://github.com/lixianlin/simfang/raw/master/SourceHanSansCN-Regular.ttf"
                    response = requests.get(font_url, timeout=10)
                    font_path = BytesIO(response.content)
                except Exception as e:
                    st.warning("词云字体下载失败，使用默认字体，中文可能显示为方框")
                    font_path = None

                wc = WordCloud(
                    font_path=font_path,
                    width=1000,
                    height=500,
                    background_color="white",
                    max_words=200,
                    max_font_size=100,
                    random_state=42
                ).generate(word_content)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc)
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("当前文本过滤后无有效词汇，无法生成词云")

        # 4. 情感分析
        if show_sentiment:
            st.divider()
            st.subheader("4. 情感倾向分析")
            try:
                s = SnowNLP(raw_text)
                sentiment_score = s.sentiments

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("情感倾向分数", f"{sentiment_score:.2f}")
                    if sentiment_score > 0.7:
                        st.success("整体情感：积极 😊")
                    elif sentiment_score < 0.3:
                        st.error("整体情感：消极 😔")
                    else:
                        st.info("整体情感：中性 😐")

                with col2:
                    fig, ax = plt.subplots()
                    labels = ["消极", "中性", "积极"]
                    sizes = [1 - sentiment_score, 0.1, sentiment_score]
                    colors = ["#ff9999", "#66b3ff", "#99ff99"]
                    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
                    ax.set_title("情感占比分布")
                    st.pyplot(fig)
            except:
                st.warning("文本内容过少，无法进行情感分析")

        # 5. 关键词提取
        if show_keywords:
            st.divider()
            st.subheader("5. 关键词提取")
            if filtered_words:
                word_tf = Counter(filtered_words)
                total_words = len(filtered_words)
                keywords = [(w, cnt / total_words) for w, cnt in word_tf.items()]
                keywords.sort(key=lambda x: x[1], reverse=True)
                top_keywords = keywords[:top_n]

                kw_df = pd.DataFrame(top_keywords, columns=["关键词", "权重"])
                st.dataframe(kw_df, use_container_width=True)
            else:
                st.info("当前文本过滤后无有效词汇，无法提取关键词")

        # 6. 文本相似度对比
        if show_compare:
            st.divider()
            st.subheader("6. 文本相似度对比")
            if compare_text:
                words1 = set(jieba.lcut(raw_text))
                words2 = set(jieba.lcut(compare_text))
                inter = len(words1 & words2)
                union = len(words1 | words2)
                jaccard = inter / union if union != 0 else 0

                st.metric("Jaccard 相似度", f"{jaccard:.2%}")
                if jaccard > 0.7:
                    st.success("两篇文本相似度较高")
                elif jaccard > 0.3:
                    st.info("两篇文本存在一定相似度")
                else:
                    st.warning("两篇文本相似度较低")
            else:
                st.info("请输入对比文本后再执行相似度分析")

        # 7. 可读性分析
        if show_readability:
            st.divider()
            st.subheader("7. 文本可读性分析")
            common_chars = set("的一是了有不人在这中大为会上个国就与说年也要出得时用道那说子自和他地")
            cn_total = len(re.findall(r"[\u4e00-\u9fff]", raw_text))
            common_count = sum(1 for c in raw_text if c in common_chars)
            common_ratio = common_count / cn_total if cn_total > 0 else 0

            sentences = re.split(r"[。！？.!?]", raw_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0

            col1, col2 = st.columns(2)
            col1.metric("常用汉字占比", f"{common_ratio:.2%}")
            if common_ratio > 0.8:
                col1.success("文本难度：简单")
            elif common_ratio > 0.6:
                col1.info("文本难度：中等")
            else:
                col1.warning("文本难度：偏难")

            col2.metric("平均句长", f"{avg_sent_len:.1f} 字符")
            if avg_sent_len < 15:
                col2.success("句子简短，易懂")
            elif avg_sent_len < 30:
                col2.info("句子长度适中")
            else:
                col2.warning("句子偏长，阅读难度较高")

        st.divider()
        st.success("✅ 全部分析完成！")

# 页脚说明
st.divider()
st.caption("💡 提示：支持中文分词、词云、情感分析、文本对比、可读性评估，可上传自定义停用词优化结果。")
