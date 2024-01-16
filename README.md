# LLMs_api_service

### 一、介绍

---

实现了一个简单的 Openai 接口形式的大模型对话服务，支持如下模型：

ChaGLM3、ChatGLM2、Codegeex2、Baichuan2、Qwen

### 二、环境

---

python 3.10

torch 2.0.0+cu118

transformers 4.33.2

peft 0.6.0 （如果使用peft加载模型）

### 二、使用说明

1. 配置参数：
    
    使用前请配置：`api_service.py` 中的 `root_model_hub_path`，配置为自己的大模型根目录地址
    
2. 启动服务
    
    ```jsx
    uvicorn api.api_service:app --host 0.0.0.0 --port 3101
    ```
    
3. 接口文档可查看 https://0.0.0.0:3101/redoc
4. 请求服务
    - api：http://192.168.11.249:3101/v1/chat/completions
    - 请求方式：POST
    
    示例：
    
    ```json
       {
          "model": "THUDM/chatglm3-6b",
          "messages": [
            {"role": "user", "content": "你好"}
            ],
          "stream": false,
          "max_tokens": 512,
          "temperature": 0.1
       }
    ```
    

### 三、参考

---

- [https://github.com/THUDM/ChatGLM3/blob/main/openai_api_demo/api_server.py](https://github.com/THUDM/ChatGLM3/blob/main/openai_api_demo/api_server.py)
