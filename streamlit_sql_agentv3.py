import streamlit as st
import os
import sqlite3
from datetime import date, datetime  # แก้ไข: เพิ่ม datetime
import re
from dotenv import load_dotenv
import csv # แก้ไข: เพิ่ม import csv

# Load environment variables from .env file
load_dotenv()

# LangChain Imports for SQL Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.agents import AgentFinish, AgentAction
from langchain.memory import ConversationBufferMemory

# --- Page Setup ---
st.set_page_config(page_title="AI ผู้ช่วยจัดการฐานข้อมูล", layout="wide")
st.title("🤖 AI ผู้ช่วยจัดการฐานข้อมูล (SQL Agent)")
st.write("สวัสดีครับ! ผมคือ AI ผู้ช่วยที่สามารถ **ค้นหา, เพิ่ม, และแก้ไข** ข้อมูลในฐานข้อมูลของเราได้แล้วนะครับ ถามมาได้เลย!")

# --- 1. Database Setup: Create or connect to the database ---
DB_FILE_NAME = "store_database.db"

@st.cache_resource
def initialize_database(db_file):
    """Initializes SQLite database and populates with sample data if it doesn't exist."""
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT,
                price REAL,
                category TEXT,
                stock INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE promotions (
                promo_id INTEGER PRIMARY KEY,
                product_id INTEGER,
                discount_percentage INTEGER,
                start_date TEXT,
                end_date TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE stores (
                store_id INTEGER PRIMARY KEY,
                store_name TEXT,
                address TEXT,
                opening_hours TEXT
            )
        """)

        # Insert sample data
        cursor.execute("INSERT INTO products VALUES (1, 'Gaming Mouse', 1500, 'Computer Peripherals', 25)")
        cursor.execute("INSERT INTO products VALUES (2, 'Mechanical Keyboard', 3500, 'Computer Peripherals', 10)")
        cursor.execute("INSERT INTO products VALUES (3, 'Laptop ASUS', 25000, 'Computer', 5)")
        cursor.execute("INSERT INTO promotions VALUES (1, 1, 20, '2025-07-01', '2025-08-30')")
        cursor.execute("INSERT INTO promotions VALUES (2, 3, 10, '2025-08-01', '2025-09-15')")
        cursor.execute("INSERT INTO stores VALUES (1, 'Central Plaza Branch', 'Bangkok', '10:00 - 21:00')")
        cursor.execute("INSERT INTO stores VALUES (2, 'The Mall Branch', 'Chiang Mai', '10:30 - 20:30')")

        conn.commit()
        conn.close()
    
    return f"sqlite:///{db_file}"

db_uri_to_use = initialize_database(DB_FILE_NAME)

# --- 2. LLM and Agent Creation ---

st.sidebar.header("การตั้งค่า AI Model")
selected_llm_model = st.sidebar.radio(
    "เลือก AI Model ที่ต้องการ:",
    ("gemini-2.5-pro","gemini-2.5-flash", "llama3.2")
)

@st.cache_resource(hash_funcs={ChatGoogleGenerativeAI: id, ChatOllama: id})
def initialize_sql_agent(db_uri, llm_choice):
    """Initializes and returns the LangChain SQL Agent with data modification capabilities."""
    db_instance = SQLDatabase.from_uri(db_uri)

    llm = None
    try:
        if "gemini" in llm_choice:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                st.error(f"ไม่พบ GOOGLE_API_KEY สำหรับ {llm_choice}. โปรดตั้งค่าในไฟล์ .env.")
                return None
            llm = ChatGoogleGenerativeAI(model=llm_choice, temperature=0, google_api_key=google_api_key)
        elif llm_choice == "llama3.2":
            ollama_host = os.getenv("OLLAMA_HOST")
            if not ollama_host:
                st.warning("ไม่พบ OLLAMA_HOST. ตรวจสอบให้แน่ใจว่า Ollama server กำลังทำงานและถูกตั้งค่าอย่างถูกต้อง.")
                return None
            llm = ChatOllama(model="llama3.2", temperature=0, base_url=ollama_host)
        else:
            st.error("Model ที่เลือกไม่ถูกต้อง.")
            return None
    except Exception as e:
        st.error(f"Error initializing LLM ({llm_choice}): {e}")
        return None

    toolkit = SQLDatabaseToolkit(db=db_instance, llm=llm)

    AGENT_PREFIX = """คุณคือ AI ผู้ช่วยที่เป็นมิตรและมีประโยชน์ ซึ่งเชี่ยวชาญในการจัดการข้อมูลในฐานข้อมูล SQL
    คุณสามารถตอบคำถาม, เพิ่มข้อมูล, และแก้ไขข้อมูลได้
    คุณมีความทรงจำเกี่ยวกับบทสนทนาที่ผ่านมาทั้งหมด และควรใช้ประวัติการแชท (chat_history) เพื่อทำความเข้าใจบริบทและตอบคำถามที่ต่อเนื่อง

    **กฎสำหรับการแก้ไขและเพิ่มข้อมูล:**
    1. คุณสามารถใช้คำสั่ง SQL `INSERT` เพื่อเพิ่มข้อมูลใหม่ลงในตารางที่ถูกต้อง
    2. คุณสามารถใช้คำสั่ง SQL `UPDATE` เพื่อแก้ไขข้อมูลที่มีอยู่ได้
    3. เมื่อผู้ใช้ต้องการแก้ไขหรือเพิ่มข้อมูล โปรดตรวจสอบให้แน่ใจว่าผู้ใช้ให้ข้อมูลที่จำเป็นครบถ้วน เช่น ชื่อสินค้า, ราคา, จำนวนสต็อก ฯลฯ และคุณใช้คำสั่ง SQL ที่ถูกต้อง
    4. คุณสามารถใช้คำสั่ง SQL `DELETE` เพื่อลบข้อมูลที่มีอยู่ได้ แต่โปรดใช้ด้วยความระมัดระวัง
    
    **ตัวอย่างการโต้ตอบสำหรับการแก้ไข/เพิ่มข้อมูล:**
    ผู้ใช้: "เพิ่มสินค้าใหม่ชื่อ 'Laptop Lenovo' ราคา 20000 บาท ในหมวดหมู่ 'Computer' มีสินค้าในสต็อก 15 ชิ้น"
    AI: จะสร้างและรันคำสั่ง SQL: `INSERT INTO products (product_name, price, category, stock) VALUES ('Laptop Lenovo', 20000.0, 'Computer', 15)`
    AI: "ได้เพิ่มสินค้า Laptop Lenovo ลงในฐานข้อมูลเรียบร้อยแล้วครับ"

    ผู้ใช้: "เปลี่ยนราคาของ 'Gaming Mouse' เป็น 1600 บาท"
    AI: จะสร้างและรันคำสั่ง SQL: `UPDATE products SET price = 1600.0 WHERE product_name = 'Gaming Mouse'`
    AI: "ได้อัปเดตราคาของ Gaming Mouse เป็น 1600 บาทเรียบร้อยแล้วครับ"

    **ตัวอย่างการโต้ตอบสำหรับการลบข้อมูล:**
    ผู้ใช้: "ลบสินค้าชื่อ 'Mechanical Keyboard' ออกจากฐานข้อมูล"
    AI: จะสร้างและรันคำสั่ง SQL: `DELETE FROM products WHERE product_name = 'Mechanical Keyboard'`
    AI: "ได้ลบข้อมูลสินค้า Mechanical Keyboard เรียบร้อยแล้วครับ"

    หากผลลัพธ์เป็นตัวเลขหรือข้อมูลสรุป, โปรดอธิบายให้ชัดเจนและเป็นประโยคที่สมบูรณ์
    ใช้ข้อมูลที่คุณค้นพบจากฐานข้อมูลเป็นหลัก และเน้นความถูกต้องในทุกคำตอบของคุณ
    
    **กฎเพิ่มเติมสำหรับคำตอบ:**
    โปรดแสดงคำสั่ง SQL ที่คุณใช้ในการดำเนินการในทุกคำตอบ โดยให้แสดงไว้ในตอนท้ายของคำตอบเสมอ เช่น "คำสั่ง SQL ที่ใช้: `SELECT product_name FROM products`"
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        agent_type="openai-tools",
        prefix=AGENT_PREFIX,
        memory=memory
    )
    return sql_agent

sql_agent_executor = initialize_sql_agent(db_uri_to_use, selected_llm_model)

# --- 3. Chat Interface ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="สวัสดีครับ! ผมคือ AI ผู้ช่วยจัดการฐานข้อมูล มีอะไรให้ช่วยไหมครับ?")
    ]

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

prompt_input = st.chat_input("พิมพ์คำถามหรือคำสั่งของคุณที่นี่...")

def write_to_csv(query, response, sql_command):
    """
    Writes a summary of the query and response to a CSV file.
    """
    csv_file = 'agent_summary.csv'
    is_new_file = not os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["DateTime", "User Query", "Finished Chain", "Final AI Response", "SQL Command"])
        
        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_datetime, query, "Finished", response, sql_command])

if prompt_input:
    st.session_state.messages.append(HumanMessage(content=prompt_input))
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.spinner("AI กำลังคิด..."):
        try:
            # Invoke the SQL Agent
            response = sql_agent_executor.invoke({"input": prompt_input})
            ai_response = response.get("output", "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผลคำตอบ")
            sql_command = "N/A"
            
            # Extract the SQL command from intermediate steps
            if response.get("intermediate_steps"):
                for step in response["intermediate_steps"]:
                    action, _ = step
                    if isinstance(action, AgentAction) and action.tool == "sql_db_query":
                        sql_command = action.tool_input
                        break
            
            # Append the SQL command to the AI response if a command was found
            if sql_command != "N/A":
                ai_response += f"\n\nคำสั่ง SQL ที่ใช้: `{sql_command}`"

            # Write summary to CSV
            write_to_csv(prompt_input, ai_response, sql_command)
            
            st.session_state.messages.append(AIMessage(content=ai_response))

            # Display final AI response
            with st.chat_message("ai"):
                st.markdown(ai_response)

            # Display intermediate steps in an expander for debugging/transparency
            if response.get("intermediate_steps"):
                with st.expander("ดูขั้นตอนการทำงานของ AI"):
                    for i, step in enumerate(response["intermediate_steps"]):
                        action, observation = step

                        # Create a two-column layout for the current step
                        col1, col2 = st.columns([0.5, 0.5])
                        
                        with col1:
                            st.markdown(f"**ขั้นตอนที่ {i+1}**")
                            if isinstance(action, AgentAction):
                                # Extract Thought from action.log if available
                                thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|$)', action.log, re.DOTALL)
                                thought = thought_match.group(1).strip() if thought_match else "AI กำลังคิด..."

                                st.markdown(f"**🧠 ความคิดของ AI:** {thought}")
                                st.markdown(f"**⚙️ การดำเนินการ:** `{action.tool}`")
                                st.markdown(f"**🔧 ข้อมูลนำเข้า (SQL Query):** ```\n{action.tool_input}\n```")
                            elif isinstance(action, AgentFinish):
                                st.markdown(f"**🧠 ความคิดของ AI:** AI ได้ข้อมูลครบถ้วนและพร้อมให้คำตอบสุดท้ายแล้วครับ")
                                st.markdown(f"**⚙️ การดำเนินการ:** `สรุปคำตอบ`")
                            else:
                                st.markdown(f"**💡 ความคิดของ AI:** ไม่ใช่ AgentAction หรือ AgentFinish ที่รู้จัก")
                                st.markdown(f"**⚙️ การดำเนินการ:** `{type(action)}`")
                                st.markdown(f"**🔧 ข้อมูลนำเข้า:** `ไม่มี`")

                        with col2:
                            st.markdown("**📋 ผลลัพธ์**")
                            if isinstance(action, AgentAction):
                                st.markdown(f"```\n{str(observation)}\n```")
                            elif isinstance(action, AgentFinish):
                                st.markdown(f"```\n{action.return_values.get('output', 'N/A')}\n```")
                        
                        st.markdown("---")

        except Exception as e:
            st.error("เกิดข้อผิดพลาดในการทำงานของ AI", icon="🚨")
            st.markdown("กรุณาลองใหม่อีกครั้ง หรือพิมพ์คำถามให้ชัดเจนขึ้น")
            st.exception(e)
            st.session_state.messages.append(AIMessage(content="ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่"))
            # Display AI message in the chat interface too
            with st.chat_message("ai"):
                st.markdown("ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่")
