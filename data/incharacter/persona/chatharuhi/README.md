---
license: apache-2.0
language:
- zh
- en
---

# ChatHaruhi

# Reviving Anime Character in Reality via Large Language Model


**Chat-Haruhi-Suzumiya**is a language model that imitates the tone, personality and storylines of characters like Haruhi Suzumiya,

https://github.com/LC1332/Chat-Haruhi-Suzumiya

Using this to load character and chat with him/her

```python
from ChatHaruhi import ChatHaruhi

chatbot = ChatHaruhi( role_from_hf = "silk-road/ChatHaruhi-RolePlaying/haruhi",\
                      llm = 'openai' ,\
                      verbose = True)

response = chatbot.chat(role='阿虚', text = 'Haruhi, 你好啊')
print(response)

# 春日:「哦，你是来向我请教问题的吗？还是有什么事情需要我帮忙的？」
```

the role was saved at 

https://huggingface.co/datasets/silk-road/ChatHaruhi-RolePlaying

this hugging face repo saved 32 characters, you may find other chacaters in 

# Run with Local Model

see this notebook

https://github.com/LC1332/Chat-Haruhi-Suzumiya/blob/main/notebook/ChatHaruhi_x_Qwen7B.ipynb

# Adding new Character

https://github.com/LC1332/Chat-Haruhi-Suzumiya

You may raise an issue at our repo if you have complete a new character and want to add into here.


