import streamlit as st
import pypdf
import re
import numpy as np
import requests
from openai import OpenAI

st.set_page_config(page_title="导论 AI 科研助手", layout="wide")
st.title("📚 导论 AI 科研助手")
st.caption("语义检索增强版（RAG）")

# ================= 核心组件函数 =================

# 文本分块（把一本厚厚的书切成小块）
def split_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap # 保留一定的重叠，防止上下文被切断
    return chunks

# 调用智谱 API 获取文本向量（Embedding）
def get_embeddings(texts, api_key):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"input": texts, "model": "embedding-2"}
    try:
        resp = requests.post(url, headers=headers, json=data)
        return [item["embedding"] for item in resp.json()["data"]]
    except Exception as e:
        st.error(f"向量化失败，请检查智谱 Key 是否正确: {e}")
        return []

# 计算余弦相似度
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 尝试从云端 Secrets 读取 Key
    secrets_deepseek = st.secrets.get("DEEPSEEK_API_KEY", "")
    secrets_zhipu = st.secrets.get("ZHIPU_API_KEY", "")
    
    # 如果 Secrets 里配置了 Key，就直接使用并隐藏输入框
    if secrets_deepseek and secrets_zhipu:
        api_key = secrets_deepseek
        zhipu_key = secrets_zhipu
        st.success("✅ 接口已就绪 (云端安全配置已加载)")
    else:
        # 如果没配置（比如别人把你的代码下载到本地跑），才显示输入框
        st.warning("⚠️ 未检测到云端配置，请手动输入 Key")
        api_key = st.text_input("大模型 API Key (DeepSeek)", type="password")
        zhipu_key = st.text_input("向量模型 API Key (智谱)", type="password")
        
    st.divider()
    
    st.header("📂 资料上传")
    uploaded_file = st.file_uploader("上传 PDF 文献", type=["pdf"])
    scope_label = st.text_input("书籍 / 领域标签", placeholder="例如：应用统计")
    
    if st.button("解析并建立向量库") and uploaded_file:
        if not zhipu_key:
            st.warning("请先填写智谱的 API Key！")
        else:
            with st.spinner("正在提取文字并建立 AI 向量库..."):
                reader = pypdf.PdfReader(uploaded_file)
                full_text = "".join([page.extract_text() or "" for page in reader.pages])
                
                # 清理文本
                full_text = re.sub(r'\s+', '', full_text) 
                
                if not full_text:
                    st.error("提取文字为空，请换一本文字版 PDF。")
                else:
                    # 1. 切块
                    chunks = split_text(full_text)
                    
                    # 2. 批量获取向量 (智谱 API 一次最多支持 64 个输入，这里简单分批)
                    embeddings = []
                    progress_bar = st.progress(0)
                    for i in range(0, len(chunks), 64):
                        batch = chunks[i:i+64]
                        batch_emb = get_embeddings(batch, zhipu_key)
                        embeddings.extend(batch_emb)
                        progress_bar.progress(min((i+64)/len(chunks), 1.0))
                    
                    if embeddings:
                        st.session_state.docs = [{
                            "name": uploaded_file.name,
                            "scope": scope_label,
                            "chunks": chunks,
                            "embeddings": embeddings
                        }]
                        st.success(f"建立成功！已切分为 {len(chunks)} 个语义块。")

# ================= 主界面问答 =================
st.subheader("💬 检索与问答")
query = st.text_input("输入你想查询的问题（支持大白话）：")

if st.button("检索并回答") and query:
    if not api_key or not zhipu_key:
        st.warning("请先填写两个 API Key！")
    elif "docs" not in st.session_state or not st.session_state.docs:
        st.warning("请先上传文献并建立向量库！")
    else:
        with st.spinner("正在理解你的问题并搜索相关段落..."):
            # 1. 把用户的提问也变成向量
            query_emb = get_embeddings([query], zhipu_key)[0]
            
            # 2. 计算用户提问和每一个文本块的相似度
            doc = st.session_state.docs[0]
            scores = [cosine_similarity(query_emb, emb) for emb in doc["embeddings"]]
            
            # 3. 找出最相似的 3 个段落
            top_indices = np.argsort(scores)[-3:][::-1]
            matched_contexts = [doc["chunks"][idx] for idx in top_indices]
            
            # 4. 拼装发给大模型
            context_str = "\n\n---\n\n".join(matched_contexts)
            system_prompt = f"你是一个科研助手。请严格根据以下参考资料回答用户的问题。\n\n参考资料：\n{context_str}"
            
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    stream=True
                )
                def generate():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                st.write_stream(generate)
            except Exception as e:
                st.error(f"调用模型失败：{e}")