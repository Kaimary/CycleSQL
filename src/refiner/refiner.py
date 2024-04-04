import time
import openai

class Refiner:
    def __init__(self):
        # 设置OpenAI API访问密钥
        openai.api_key = "****"
        self.prompt = open("prompt.txt").read()

    def refine(self, result, dialect):
        # 调用ChatGPT API生成对话文本
        # query = f"""Please help me to polish the following sentence: {xnl}"""
        exp = None
        while exp is None:
            try:
                query = f"{self.prompt}\n#### Please use the above examples to generate:\n\
                                    Query Result: {result}\nDialect explanation: {dialect}\n\
                                    #### Polished natural language explanation: "
                exp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    # n=5
                )
            except Exception as e:
                print(e)
                time.sleep(3)

        return exp['choices'][0]['message']['content']

    