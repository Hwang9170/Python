from gtts import gTTS
import os 

text = "안녕하세요, 여러분 파이썬은 재미있습니다."
tts = gTTS(text=text, lang ='ko')
tts.save("ttt.mp3")
os.system("ttt.mp3")

print("안녕하세요")
name = input("이름이 어떻게 되시나요? > ")
print("\n만나서 반갑습니다.",name,"씨")
names = "만나서 반갑습니다."+name
tts = gTTS(names=names, lang ='ko')
tts.save("ttt.mp3")
os.system("ttt.mp3")