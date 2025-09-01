🤖 AI ผู้ช่วยสอบถามฐานข้อมูล (SQL Agent with LangChain)
แอปพลิเคชัน Streamlit นี้ทำหน้าที่เป็น AI ผู้ช่วยสอบถามฐานข้อมูล ที่ชาญฉลาด ช่วยให้ผู้ใช้สามารถโต้ตอบและค้นหาข้อมูลจากฐานข้อมูล SQL ด้วย ภาษาธรรมชาติ (Natural Language) แทนที่จะต้องเขียนคำสั่ง SQL โดยตรง
แอปพลิเคชันนี้ใช้พลังของ LangChain เพื่อเชื่อมต่อโมเดลภาษาขนาดใหญ่ (LLM) เข้ากับฐานข้อมูล ทำให้ AI สามารถทำความเข้าใจคำถามของผู้ใช้, แปลงเป็นคำสั่ง SQL, ดึงข้อมูล และนำเสนอผลลัพธ์ที่เข้าใจง่ายกลับคืนมา
✨ คุณสมบัติหลัก
• สอบถามฐานข้อมูลด้วยภาษาธรรมชาติ: ไม่จำเป็นต้องมีความรู้ SQL ผู้ใช้สามารถถามคำถามทั่วไปได้ เช่น "สินค้า Gaming Mouse ราคาเท่าไหร่" หรือ "มีโปรโมชั่นอะไรบ้าง"
• รองรับ LLM หลายตัวเลือก: สามารถเลือกใช้โมเดล Google Gemini (Pro/Flash) หรือ Llama3.2 (ผ่าน Ollama) ได้ตามความต้องการ
• การจัดการความทรงจำ (Memory): AI สามารถจดจำประวัติการสนทนา เพื่อให้สามารถตอบคำถามต่อเนื่องได้
• การแสดงขั้นตอนการทำงาน (Intermediate Steps): แสดงให้เห็นว่า AI คิดอย่างไร, ใช้เครื่องมืออะไร (SQL), และได้ผลลัพธ์อะไรจากการสอบถามฐานข้อมูล เพื่อเพิ่มความโปร่งใส
• ฐานข้อมูล SQLite ในตัว: มาพร้อมฐานข้อมูล SQLite ขนาดเล็กพร้อมข้อมูลตัวอย่าง (สินค้า, โปรโมชั่น, ร้านค้า) ทำให้สามารถทดสอบใช้งานได้ทันที
• การตรวจสอบ API Key/Host: มีการตรวจสอบเบื้องต้นสำหรับ GOOGLE_API_KEY หรือ OLLAMA_HOST เพื่อให้แน่ใจว่าโมเดลพร้อมใช้งาน
🚀 การติดตั้งและเรียกใช้งาน
📋 ข้อกำหนดเบื้องต้น
	•	Python 3.8+
	•	Git (สำหรับ clone repository)
	•	สำหรับ Google Gemini: GOOGLE_API_KEY (ดูรายละเอียดด้านล่าง)
	•	สำหรับ Llama3.2: Ollama Server และ OLLAMA_HOST (ดูรายละเอียดด้านล่าง)
👨‍💻 ขั้นตอนการติดตั้ง
	1	Clone Repository:git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)cd YOUR_REPO_NAME(อย่าลืมเปลี่ยน YOUR_USERNAME/YOUR_REPO_NAME เป็นชื่อผู้ใช้และชื่อ repository ของคุณ)
	2	สร้าง Environment และติดตั้ง Dependencies:python -m venv venvsource venv/bin/activate  # สำหรับ Linux/macOS# venv\Scripts\activate    # สำหรับ Windowspip install -r requirements.txt(ไฟล์ requirements.txt ควรมี streamlit, langchain, langchain-google-genai, langchain-ollama, python-dotenv)
	3	ตั้งค่า Environment Variables:สร้างไฟล์ชื่อ .env ใน directory เดียวกันกับ app.py (หรือชื่อไฟล์ Streamlit หลักของคุณ)
	•	สำหรับ Google Gemini:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"คุณสามารถขอ API Key ได้จาก Google AI Studio
	•	สำหรับ Llama3.2 (ผ่าน Ollama):คุณต้องติดตั้งและเรียกใช้ Ollama Server บนเครื่องของคุณก่อน จากนั้นโหลดโมเดล llama3.2 (หรือโมเดลอื่นที่คุณต้องการใช้):ollama run llama3.2และเพิ่มการตั้งค่าในไฟล์ .env:OLLAMA_HOST="http://localhost:11434" # หรือ URL ของ Ollama Server ของคุณ
▶️ การเรียกใช้งานแอปพลิเคชัน
หลังจากตั้งค่าเสร็จสิ้น คุณสามารถเรียกใช้งานแอป Streamlit ได้:
streamlit run app.py
(หากไฟล์ Streamlit ของคุณชื่ออื่น ให้เปลี่ยน app.py เป็นชื่อที่ถูกต้อง)
แอปพลิเคชันจะเปิดขึ้นใน web browser ของคุณ
💡 วิธีการใช้งาน
	1	เลือก AI Model: ที่แถบด้านข้างซ้ายมือ เลือกโมเดล AI ที่คุณต้องการใช้ (Gemini 2.5 Pro, Gemini 2.5 Flash, หรือ Llama3.2)
	•	หากมีข้อผิดพลาดเกี่ยวกับการตั้งค่า API Key/Ollama Host โปรดแก้ไขในไฟล์ .env
	2	พิมพ์คำถาม: พิมพ์คำถามของคุณเกี่ยวกับสินค้า, โปรโมชั่น, หรือข้อมูลร้านค้าในช่องข้อความด้านล่างสุดของหน้าจอ
	3	รับคำตอบ: AI จะประมวลผลและแสดงคำตอบกลับมาในช่องแชท
	4	ดูขั้นตอนการทำงาน (ไม่บังคับ): คุณสามารถคลิกที่ ดูขั้นตอนการทำงานของ AI เพื่อดูรายละเอียดว่า AI ทำการสอบถามฐานข้อมูลอย่างไร
ตัวอย่างคำถาม:
	•	"มีสินค้าอะไรบ้าง"
	•	"ราคาของ Gaming Mouse เท่าไหร่"
	•	"สินค้า ID 1 กำลังมีโปรโมชั่นอะไร"
	•	"ร้าน Central Plaza Branch เปิดกี่โมง"
	•	"ก่อนหน้าฉันถามเรื่องอะไร" (AI สามารถจดจำบทสนทนาได้)
📁 โครงสร้างโปรเจกต์
├── app.py             # ไฟล์ Streamlit หลักของแอปพลิเคชัน
├── requirements.txt   # รายชื่อ dependencies ที่ต้องติดตั้ง
├── .env               # ไฟล์สำหรับเก็บ Environment Variables (เช่น API Keys)
└── store_database.db  # ไฟล์ฐานข้อมูล SQLite ที่จะถูกสร้างขึ้น
