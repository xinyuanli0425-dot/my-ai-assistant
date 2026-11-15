import streamlit as st
import pypdf
import re
from openai import OpenAI

st.set_page_config(page_title="导论 AI 科研助手", layout="wide")
st.title("📚 导论 AI 科研助手")
st.caption("第一步：在左侧输入 API Key 并上传 PDF，第二步：在下方提问")

with st.sidebar:
    st.header("⚙️ 设置")
    
    # 自动读取云端配置的 API Key（本地测试时如果没配置，会自动变为空）
    default_key = ""
    try:
        default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except:
        pass
        
    api_key = st.text_input("输入大模型 API Key", value=default_key, type="password", help="去 DeepSeek 官网申请")
    st.divider()
    
    st.header("📂 资料上传")
    uploaded_file = st.file_uploader("上传 PDF 文献", type=["pdf"])
    scope_label = st.text_input("书籍 / 领域标签", placeholder="例如：应用统计")
    
    if st.button("解析并加入空间") and uploaded_file:
        with st.spinner("正在解析 PDF..."):
            reader = pypdf.PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            if not full_text.strip():
                st.error("⚠️ 警告：提取出的文字为空！它可能是扫描版，请换一本文字版 PDF。")
            else:
                if "docs" not in st.session_state:
                    st.session_state.docs = []
                st.session_state.docs.append({
                    "name": uploaded_file.name,
                    "scope": scope_label,
                    "text": full_text
                })
                st.success(f"《{uploaded_file.name}》已加入！共 {len(full_text)} 个字符。")
                
                with st.expander("👀 点我查看提取出的文字样本"):
                    st.write(full_text[:200] + "...")

st.subheader("💬 检索与问答")
query = st.text_input("输入你想查询的问题或关键词：")

if st.button("检索并回答") and query:
    if not api_key:
        st.warning("请先在左侧填写 API Key！")
    elif "docs" not in st.session_state or not st.session_state.docs:
        st.warning("请先上传文献！")
    else:
        matched_contexts = []
        
        # 模糊搜索逻辑
        stop_words = ["什么是", "的", "了", "吗", "如何", "请问", "是", "和", "定义"]
        clean_q = query
        for w in stop_words:
            clean_q = clean_q.replace(w, " ")
        
        keywords = [k for k in clean_q.split() if len(k) > 1]
        if not keywords:
            keywords = [query]

        for doc in st.session_state.docs:
            clean_text = re.sub(r'\s+', '', doc["text"])
            best_idx = -1
            for kw in keywords:
                if kw in clean_text:
                    best_idx = clean_text.find(kw)
                    break
            
            if best_idx != -1:
                snippet = clean_text[max(0, best_idx-200): min(len(clean_text), best_idx+500)]
                matched_contexts.append(f"【来源：{doc['name']}】\n...{snippet}...")
        
        if not matched_contexts:
            st.info("在资料中没有找到相关上下文。")
        else:
            context_str = "\n\n".join(matched_contexts)
            system_prompt = f"你是一个科研助手。请严格根据以下参考资料回答用户的问题。如果资料中没有答案，请直接说不知道。\n\n参考资料：\n{context_str}"
            
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            with st.spinner("AI 正在阅读资料并生成回答..."):
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