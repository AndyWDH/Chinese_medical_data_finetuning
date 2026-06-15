
# Qwen2.5-7B 医疗领域微调项目

基于 Unsloth 框架对 Qwen2.5-7B-Instruct 模型进行医疗领域的参数高效微调（LoRA）。

## 项目概述

本项目实现了一个完整的大语言模型微调流程，将通用的 Qwen2.5-7B 模型适配到医疗领域，使其能够回答患者的医疗相关问题。

### 技术栈

| 组件 | 版本/说明 |
|------|-----------|
| **框架** | Unsloth |
| **模型** | Qwen2.5-7B-Instruct |
| **微调方法** | LoRA (Low-Rank Adaptation) |
| **量化策略** | 4-bit 量化 |
| **训练器** | SFTTrainer |

## 核心功能

- ✅ **高效微调**：使用 LoRA 技术，仅训练 1-10% 的模型参数
- ✅ **显存优化**：4-bit 量化 + 梯度检查点，大幅降低显存占用
- ✅ **多科室数据支持**：涵盖内科、外科、儿科、肿瘤科、妇产科、男科
- ✅ **完整流程**：数据准备 → 模型训练 → 推理测试 → 模型保存

## 目录结构

```
├── Qwen2_5_(7B)_医疗微调.py    # 主微调脚本
├── Data_数据/                   # 医疗对话数据集
│   ├── IM_内科/                 # 内科数据
│   ├── Surgical_外科/           # 外科数据
│   ├── Pediatric_儿科/          # 儿科数据
│   ├── Oncology_肿瘤科/         # 肿瘤科数据
│   ├── OAGD_妇产科/            # 妇产科数据
│   └── Andriatria_男科/         # 男科数据
├── outputs/                     # 训练输出目录
├── lora_model_medical/          # 保存的LoRA模型
└── README.md                    # 项目说明文档
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（推荐）

### 安装依赖

```bash
pip install unsloth torch transformers datasets trl accelerate
```

### 数据准备(需要数据的联系我)

将医疗对话数据按科室分类放置在 `Data_数据/` 目录下，每个CSV文件需包含以下列之一：
- 问题列：`question`、`问题`、`ask`
- 回答列：`answer`、`回答`、`response`

### 运行微调

```bash
python Qwen2_5_(7B)_医疗微调.py
```

## 代码结构详解

### 1. 模型加载与配置

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### 2. LoRA 适配器配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` | 16 | LoRA 秩 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 应用 LoRA 的模块 |
| `lora_alpha` | 16 | LoRA 缩放因子 |
| `use_gradient_checkpointing` | "unsloth" | 减少 30% 显存使用 |

### 3. 训练参数

| 参数 | 值 |
|------|-----|
| 批次大小 | 2 |
| 梯度累积步数 | 4 |
| 学习率 | 2e-4 |
| 训练轮数 | 3 |
| 优化器 | adamw_8bit |

### 4. 推理示例

```python
def generate_medical_response(question):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([medical_prompt.format(question, "")], return_tensors="pt").to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
```

## 使用示例

```python
# 测试医疗问答
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    generate_medical_response(question)
```

## 模型保存与加载

### 保存模型

```python
model.save_pretrained("lora_model_medical")
tokenizer.save_pretrained("lora_model_medical")
```

### 加载模型

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    adapter_name="lora_model_medical",
    load_in_4bit=True,
)
```

## 注意事项

1. **数据格式**：确保CSV文件编码正确（支持GBK、GB2312、GB18030、UTF-8）
2. **显存要求**：推荐使用至少 16GB 显存的GPU
3. **数据安全**：医疗数据包含敏感信息，请妥善处理
4. **法律合规**：本项目仅用于学习研究，不可用于实际医疗诊断

## 许可证

本项目仅供学习和研究使用。

---

*使用 Unsloth 框架实现高效的大模型微调*
