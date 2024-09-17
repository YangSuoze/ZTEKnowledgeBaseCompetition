## 一、基于 es 数据库进行关键词检索
linux 环境上下载配置 es 包，默认端口为 9200
- es 数据库安装部署方法：
https://www.cnblogs.com/weibanggang/p/11589464.html——注意最新版本的 es 需要设
计 xss
https://www.cnblogs.com/hxlasky/p/17514728.html
第 2 步：Kibana 部署
注意：Kibana 版本与 ES 保持一致
linux 环境上下载配置 Kibana 包，默认端口为 5601
数据库前端显示路径：http://10.13.14.16:5601
kibana 安装部署方法：https://www.cnblogs.com/kiko2014551511/p/16195605.html
第 3 步：核心代码
代码路径在：es_kb_service.py
主要进行数据库的增删改查工作，支持关键字检索、向量检索、混合检索。
## 二、重排序
使用向量化模型对检索的 chunks 进行重排序
模型选择使用 bge-reranker-base
## 三、PDF 转换和切分模块
第 1 步：PDF 转换为 txt 格式
pdf 分割模块支持单栏、双栏布局
主要借用 kimi 对 pdf 统一转换为模型可读取的 txt 文件
第 2 步：PDF 切分为 chunks
from langchain.text_splitter import
RecursiveCharacterTextSplitter
使用 langchain 中的 RecursiveCharacterTextSplitter 切分。chunk 大小值设定为 1000，
chunk_overlap 为 50
## 四、输入问题增强
考虑到输入参考资料有中文英文两种语言，因此在检索前需要对用户输入的问题进行增强，
使的输入问题能够参考多种语言的知识。
具体方法：
翻译 prompt：
input_query = f"""请将以下句子进行翻译，如果用户输入是中文就翻译成英文，如果用户输
入是英文就翻译成中文。
【注意】直接输出翻译结果，不要输出其他多余的内容
用户输入：{query} """ 将原问题增强为：query+query_trans
## 五、prompt 设计和答案提取
input_query = f"""你是一个做判断题的能手，请结合给定的知识，判断用户输入的语句是否
正确，如果参考文档中未提及则不算错误
【注意】回答格式请统一输出，如：
如果选项正确，则输出：正确。原因是*** 如果选项错误，则输出：错误。原因是*** 【注意】回答请精简准确，不超过 50 字，注意一定要谨慎小心可能错误的选项，并结合已
有的常识综合进行判断。
<参考消息>
{context}
<参考消息>
用户输入：{query} """ 注意点：
1. 指明任务，让模型明确做判断题的任务和做题技巧；
2. 约束回答格式，方便调用脚本转换为最终答案 T or F；
3. 给定从数据库中检索出来的参考内容。
4. 根据回答正确 or 错误，转换为 T or F，并保存在 result.csv 文件中。
## 六、自一致性提高模型回复准确性
采用 self-consistence（SC）的思想。由于模型具有概率随机性，因为，我们打开采样后，
对每个问题重复五次，取频率最高的答案作为最终答案，以提升大模型回复的稳定性和准确
性。七、模型使用 vllm 本地部署的 glm-9b 模型
1. 下载 hugging-face 的 glm-9b 模型到本地服务器
2. VLLM 部署模型方法：
```python
pip install vllm modelscope transformers
# 这也是模型读取路径，如果不设置，就回去 hugging face 下载！
export VLLM_USE_MODELSCOPE=True
nohup python -m vllm.entrypoints.openai.api_server --model
/home/llmapi/finreport_test/models/ZhipuAI/glm-4-9b-chat --served-model-name
glm-4-9b-chat --max-model-len=32000 --trust-remote-code --tensor-parallel-size 4 --gpu-memory-utilization=0.5 --port=7863 >
logs_glm-9b.txt &
# 另一种形式
python -m vllm.entrypoints.openai.api_server \ --model ZhipuAI/glm-4-9b-chat \ --tokenizer ZhipuAI/glm-4-9b-chat \ --served-model-name glm-4-9b-chat \ --max-model-len 8192 \ --gpu-memory-utilization 1 \ --tensor-parallel-size 1 \ --max-parallel-loading-workers 2 \ --trust-remote-code \ --enforce-eager
# 清理 GPU 内存函数
def torch_gc(): if torch.cuda.is_available(): # 检查是否可用 CUDA
with torch.cuda.device(CUDA_DEVICE): # 指定 CUDA 设备
torch.cuda.empty_cache() # 清空 CUDA 缓存
torch.cuda.ipc_collect() # 收集 CUDA 内存碎片
torch_gc() # 执行 GPU 内存清理
3. 部署成功后的调用代码
# 使用 openai
from openai import OpenAI
client = OpenAI(
base_url="http://IP:PORT/v1", api_key="token-abc123", # 随便设，只是为了通过接口参数校验
)
completion = client.chat.completions.create(
model="glm-4-9b-chat", messages=[
{"role": "user", "content": "你好"}
],# 设置额外参数
extra_body={ "stop_token_ids": [151329, 151336, 151338]
}
)
print(completion.choices[0].message)
# 使用 langchain
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(
streaming=False, verbose=True, openai_api_key="lingyue-123", openai_api_base='http://IP:PORT/v1', model_name='glm-4-9b-chat', temperature=0.8, model_kwargs={"stop": ["<|endoftext|>", "<|user|>", "<|observation|>"], "extra_body": {"skip_special_tokens": False}}
)
chat.invoke('你好').content
```