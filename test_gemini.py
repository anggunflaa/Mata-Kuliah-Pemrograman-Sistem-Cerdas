#set api key sekali di env melalui terminal: (setx GEMINI_API_KEY "IzaSyDMXi4sWl2uz2thJKu_Adh4PutgDDgvbQ8")
#api key gemini saya "IzaSyDMXi4sWl2uz2thJKu_Adh4PutgDDgvbQ8"

import os, google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"  # atau "gemini-flash-latest"
m = genai.GenerativeModel(MODEL_NAME)

r = m.generate_content("Halo Gemini! Jawab singkat: 3 + 4 = ?")
print("MODEL:", MODEL_NAME)
print("RESP :", r.text[:200])
