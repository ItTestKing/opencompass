# -*- coding: utf-8 -*-
import json
import re
import argparse
import time

from openai import OpenAI
from tqdm import tqdm  # 导入tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run the model testing script.")
parser.add_argument("model", type=str, help="The model ID to use.")
parser.add_argument("url", type=str, help="The IP address of the API server.")

args = parser.parse_args()

# 使用命令行参数设置 API key 和 API base
openai_api_key = "EMPTY"
openai_api_base = f"{args.url}"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = args.model  # 使用命令行参数中的模型ID
print(model)

def get_json_time(day):
    if day == 0:
        return int(round(time.time()))  # 如果是零返回今天时间戳
    else:
        now = int(round(time.time()))
        return now + 3600 * 24 * day
def clean_answer(answer):
    pattern = re.compile(r'[ABCD对错]')
    matches = pattern.finditer(answer)
    cleaned_answer = ""
    for match in matches:
        if match.group(0) not in cleaned_answer:
            cleaned_answer += match.group(0)
    return cleaned_answer

def ask_model(prompt, content, pp=False, temperature=0.1, top_p=1):
    ret = ""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content + "回答内容以下格式：A"}
            ],
            model=model,
            stream=True,
            max_tokens=4000,
            temperature=temperature,
            top_p=top_p,
        )
        for chunk in chat_completion:
            i = chunk.choices[0].delta.content or ""
            ret += i
            if pp:
                print(i, end="")
    except Exception as e:
        print(f"Error in ask_model: {e}")
        ret = None
    return ret

if __name__ == '__main__':
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset(r"cseval/cs-eval")
    total_questions = len(dataset['test'])  # 获取测试集中的问题总数
    json_data = []  # 初始化一个列表来存储所有问题的 JSON 数据

    pbar = tqdm(total=4670, desc='Processing questions')
    for i in range(4670):
        text = dataset['test'][i]["prompt"]
        id = dataset['test'][i]["id"]
        prompt = "你是一个考生，会有中文英文问题，你要做出回答，只帮我选出正确选项的序号如：以这种格式回答：A"
        ret = ask_model(prompt, text)
        if ret is not None:
            ret = clean_answer(ret)
            json_data.append({"question_id": id, "answer": ret})  # 将数据添加到列表
        pbar.update(1)  # 更新进度条
    pbar.close()  # 确保进度条正确关闭

    # 将 JSON 数据写入文件
    with open(f"{model}_{get_json_time(0)}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print("Script executed successfully.")
