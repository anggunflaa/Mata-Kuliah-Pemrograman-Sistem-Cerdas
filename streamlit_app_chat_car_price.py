import os
import json
import joblib
import difflib
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import re
from datetime import datetime, timedelta
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
import base64  # NEW: render GIF

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="🚘 Prediksi Harga Mobil + Chat Gemini",
    layout="wide",
    page_icon="🚗",
)

# =========================
# CSS KUSTOM + ANIMASI
# =========================
st.markdown("""
<style>
:root {
    --primary-color: #0096FF;
    --secondary-color: #00BFFF;
    --bg-dark: #001F3F;
    --text-light: #000000;          /* FIX: sebelumnya ##000000 */
    --accent-pink: #ff7dd8;
    --shadow: 0 0 25px rgba(0,150,255,0.3);
}

/* Latar belakang aplikasi - biru laut aesthetic */
body, .stApp {
    background: linear-gradient(180deg, #00B4DB, #0083B0, #005f73);
    color: var(--text-light);
    font-family: "Poppins", sans-serif;
}

/* 🎨 Ubah warna latar sidebar jadi biru muda gradasi abu */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #66CCFF, #A9C1D9, #D6E0E7) !important; /* biru muda → abu lembut */
    color: #003F88 !important; /* teks biru tua agar kontras */
}

/* ✨ Efek lembut bayangan samping */
section[data-testid="stSidebar"] {
    box-shadow: 3px 0 10px rgba(0, 63, 136, 0.2) !important;
}

/* Judul utama gradasi biru tua sangat tua */
.main-header {
    font-size: 2.4em;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(270deg, #000c24, #001a33, #002b5b);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
    margin-bottom: 25px;
}
@keyframes gradientFlow { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }

/* Animasi fade-in & card */
@keyframes fadeIn { from{opacity:.0; transform:translateY(6px)} to{opacity:1; transform:translateY(0)} }
.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 0 18px rgba(0,150,255,0.25);
    animation: fadeIn .5s ease both;
}

/* Kotak chat biru + animasi */
.chat-container {
    background: rgba(0,119,255,0.25);
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(0,150,255,0.35);
    border: 1px solid rgba(0,150,255,0.4);
    animation: fadeIn .6s ease both;
}

/* Chat user & AI */
.chat-bubble-user {
    background: linear-gradient(135deg, #001A66, #003F88);
    color: #ffffff;
    border-radius: 20px 20px 0 20px;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
}
.chat-bubble-ai {
    background: linear-gradient(135deg, #003f88, #0066cc);
    border: 1px solid #00bfff; border-radius: 20px 20px 20px 0;
    padding: 10px 15px; margin: 10px 0; max-width: 85%; color: #ffffff;
}

/* Label & placeholder biru tua */
div[data-testid="stMarkdownContainer"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #003f88 !important; font-weight: 600; font-size: .95rem;
}
input::placeholder, textarea::placeholder { color:#003f88 !important; opacity:1 !important }

/* Input look & focus glow */
input, select, textarea {
    border-radius: 10px !important;
    border: 1px solid #004fbb !important;   /* FIX: sebelumnya #004f0b */
    background-color: rgba(255,255,255,0.08);
    color: var(--text-light) !important;
}
input:focus, textarea:focus {
    border-color: #003f88 !important;
    box-shadow: 0 0 12px rgba(0,63,136,0.35);
    outline: none !important;
}
/* Khusus text input (chat) */
input[type="text"] {
    background-color: #004fbb1a !important;
    color: #002B5B !important; border: 1px solid #00bfff !important;
    border-radius: 20px !important; padding: 10px 16px !important;
}
input[type="text"]::placeholder { color:#e0f0ff !important }
input[type="text"]:focus { border-color:#00d0ff !important; box-shadow:0 0 12px rgba(0,208,255,.45); }

/* Tombol gradien + hover */
.stButton>button {
    background: linear-gradient(135deg, #001A66, #003F88); /* biru tua elegan */
    color: #ffffff; /* teks putih */
    border: none;
    border-radius: 10px;
    padding: 8px 14px;
    box-shadow: 0 4px 10px rgba(0,63,136,0.4);
    font-weight: 600;
    transition: transform 0.1s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0,63,136,0.6);
}

/* Badge kecil */
.badge {
    display:inline-block; padding:4px 10px; border-radius:999px;
    background:linear-gradient(135deg,#00bfff,#0077ff); color:#fff; font-size:.78rem;
}

/* === ANIMASI MOBIL (GIF) === */
.car-wrap{
  position: fixed;
  left: -420px;
  bottom: 12px;
  z-index: 9998;
  animation: carDrive 10s linear infinite;
  pointer-events: none;
}
.car-img{
  height: min(120px, 18vh);
  filter: drop-shadow(0 8px 12px rgba(0,0,0,.25));
}
@keyframes carDrive{
  0%   { transform: translateX(0) scale(1);    opacity: .95; }
  45%  { transform: translateX(60vw) scale(1); opacity: 1;   }
  50%  { transform: translateX(65vw) scale(1.02) rotateZ(0.5deg); }
  100% { transform: translateX(110vw) scale(1); opacity: .95; }
}
@media (max-width: 880px){
  .car-wrap{ display:none; }
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🚘 Prediksi Harga Mobil + Chat Gemini AI</div>", unsafe_allow_html=True)

# =========================
# UTIL + KORPUS DOKUMEN
# =========================
MAX_DOC_CHARS = 12000

def _read_file_text(file) -> str:
    try:
        if hasattr(file, "read"):
            data = file.read()
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return str(data)
        else:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"[Gagal membaca dokumen: {e}]"

def _build_doc_context(docs: list[str]) -> str:
    if not docs:
        return ""
    joined = "\n\n---\n\n".join(docs)
    if len(joined) > MAX_DOC_CHARS:
        joined = "[Potongan dokumen (truncated)]\n" + joined[-MAX_DOC_CHARS:]
    return joined

def rupiah(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return "Rp -"

@st.cache_resource
def load_model_if_exists(path: str):
    if os.path.exists(path):
        try:
            st.sidebar.info("✅ Model ditemukan dan dimuat dari disk.")
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠ Gagal memuat model.pkl: {e}")
            st.sidebar.warning("🔄 Menggunakan model dummy sementara.")
            return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))
    else:
        st.sidebar.warning("⚠ model.pkl tidak ditemukan. Menggunakan model dummy.")
        return LinearRegression().fit(np.random.rand(10, 3), np.random.rand(10))

def prepare_input(form_data: Dict[str, Any], example_schema: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{k: form_data.get(k, v) for k, v in example_schema.items()}])

def save_to_dataset(input_df: pd.DataFrame, prediction: float, path: str):
    df_to_save = input_df.copy()
    df_to_save["prediction"] = prediction
    if os.path.exists(path):
        try:
            dataset = pd.read_csv(path)
        except Exception:
            dataset = pd.DataFrame()
        dataset = pd.concat([dataset, df_to_save], ignore_index=True)
    else:
        dataset = df_to_save
    dataset.to_csv(path, index=False)
    return dataset

# =========================
# AUTH SEDERHANA (DEMO)
# =========================
USERS_DB = "users.json"

def _load_users() -> dict:
    if os.path.exists(USERS_DB):
        try:
            with open(USERS_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_users(data: dict) -> None:
    with open(USERS_DB, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def auth_view():
    st.markdown("#### 🔐 Masuk / Daftar")
    tab_login, tab_register = st.tabs(["Masuk", "Daftar"])

    with tab_login:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        email = st.text_input("Email")
        pw = st.text_input("Kata Sandi", type="password")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Masuk"):
                users = _load_users()
                rec = users.get(email)
                if not rec:
                    st.error("Email tidak terdaftar.")
                elif rec["password"] != _hash_pw(pw):
                    st.error("Kata sandi salah.")
                else:
                    st.session_state["user"] = {"email": email, "name": rec.get("name", email.split("@")[0])}
                    st.success("Berhasil masuk!")
                    st.rerun()
        with col2:
            st.caption("Belum punya akun? Buka tab **Daftar**.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_register:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        name = st.text_input("Nama Lengkap")
        email_r = st.text_input("Email (aktif)")
        pw1 = st.text_input("Kata Sandi (min 6 karakter)", type="password")
        pw2 = st.text_input("Ulangi Kata Sandi", type="password")
        if st.button("Daftar"):
            if not name or not email_r or not pw1:
                st.error("Lengkapi semua field.")
            elif len(pw1) < 6:
                st.error("Minimal 6 karakter.")
            elif pw1 != pw2:
                st.error("Kata sandi tidak cocok.")
            else:
                users = _load_users()
                if email_r in users:
                    st.warning("Email sudah terdaftar, silakan masuk.")
                else:
                    users[email_r] = {"name": name, "password": _hash_pw(pw1)}
                    _save_users(users)
                    st.success("Pendaftaran berhasil! Silakan masuk.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# KONFIG GEMINI (prioritas: env ➜ default kamu ➜ input sidebar)
# =========================
DEFAULT_KEY = "AIzaSyDMXi4sWl2uz2thJKu_Adh4PutgDDgvbQ8"  # key kamu
gemini_api_key_env = os.environ.get("GEMINI_API_KEY", DEFAULT_KEY)

try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

# simpan key aktif di session
if "gemini_key" not in st.session_state:
    st.session_state["gemini_key"] = (gemini_api_key_env or "").strip()

with st.sidebar:
    st.subheader("🤖 Gemini")
    if not HAVE_GENAI:
        st.error("Paket google-generativeai belum terpasang. Jalankan: pip install google-generativeai")
    status = "TERDETEKSI" if st.session_state["gemini_key"] else "TIDAK TERDETEKSI"
    st.write(f"GEMINI_API_KEY: **{status}**")
    st.caption(f"(panjang key: {len(st.session_state['gemini_key'] or '')} karakter)")
    new_key = st.text_input("API Key (override sesi ini)", type="password", value=st.session_state["gemini_key"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Simpan Key"):
            st.session_state["gemini_key"] = (new_key or "").strip()
            st.success("Key disimpan di sesi.")
    with c2:
        if st.button("Pakai Key Default"):
            st.session_state["gemini_key"] = DEFAULT_KEY
            st.success("Menggunakan key default.")
    model_name = st.selectbox("Model LLM", ["gemini-2.5-flash","gemini-flash-latest","gemini-2.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.6, 0.1)
    max_output_tokens = st.slider("Max Output Tokens", 256, 4096, 1024, 64)
    answer_all = st.toggle("🟦 Gemini jawab SEMUA pertanyaan (disarankan)", value=True)

def _active_api_key() -> str:
    return (st.session_state.get("gemini_key") or DEFAULT_KEY).strip()

def _get_gemini_client(api_key: str):
    if not HAVE_GENAI:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception:
        return None

def _dataset_summary(df: pd.DataFrame, max_models: int = 20) -> str:
    if df is None or df.empty:
        return "Dataset kosong/tidak dimuat."
    cols = ", ".join(df.columns.tolist()[:20])
    summary = f"Kolom: {cols}."
    if "model" in df.columns:
        uniq = sorted(df['model'].dropna().astype(str).unique().tolist())
        if uniq:
            summary += f" Contoh model: {', '.join(uniq[:max_models])}."
    if "price" in df.columns:
        try:
            mean_price = float(df['price'].mean())
            summary += f" Rata-rata price (unit asli dataset): {mean_price:,.2f}."
        except Exception:
            pass
    return summary

def gemini_chat(
    user_message: str,
    history: list[tuple[str, str]] | None = None,
    df: pd.DataFrame | None = None,
    last_prediction: float | None = None,
    last_input: dict | None = None,
) -> str:
    """
    Gemini-first untuk SEMUA pertanyaan.
    Menyertakan ringkasan dataset, prediksi terakhir, riwayat, dan korpus dokumen.
    """
    api_key = _active_api_key()
    if not api_key:
        return "⚠ Gemini belum aktif (API key belum diatur)."

    client = _get_gemini_client(api_key)
    if client is None:
        return "⚠ Gagal menginisialisasi klien Gemini."

    ds_ctx = _dataset_summary(df)
    last_ctx = ""
    if last_prediction is not None and last_input:
        try:
            last_ctx = f"Prediksi terakhir: {json.dumps(last_input)} → £{last_prediction:,.2f} (≈ {rupiah(last_prediction*20000)})."
        except Exception:
            pass

    chat_history_text = ""
    if history:
        tail = history[-6:]
        lines = []
        for role, msg in tail:
            r = "User" if role == "user" else "Assistant"
            msg = (msg or "")[:1000]
            lines.append(f"{r}: {msg}")
        chat_history_text = "\n".join(lines)

    doc_list = st.session_state.get("doc_corpus", [])
    doc_texts = [f"# {doc['name']}\n{doc['text']}" for doc in doc_list]
    docs_ctx = _build_doc_context(doc_texts)

    system_prompt = f"""
Kamu adalah asisten AI serbaguna berbahasa Indonesia.
- Jawab pertanyaan apapun (umum, teknis, matematika, penjelasan konsep).
- Gunakan nada santai, ramah, dan kreatif; boleh humor ringan.
- Jika pertanyaan konyol/tidak masuk akal, tetap jawab secara playful yang aman, beri konteks/tips seperlunya.
- Jika ada KORPUS DOKUMEN, gunakan sebagai referensi utama saat relevan.
- Dataset mobil hanya dipakai jika pertanyaan tentang mobil/harga.
- Jawaban harus jelas & terstruktur. Bila ragu, jelaskan asumsi.

KONTEKS DATASET:
{ds_ctx}

KONTEKS PREDIKSI TERAKHIR:
{last_ctx}

RIWAYAT CHAT:
{chat_history_text}

KORPUS DOKUMEN:
{docs_ctx}
""".strip()

    try:
        prompt = f"{system_prompt}\n\nPERTANYAAN PENGGUNA:\n{user_message}\n"
        resp = client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
        )
        text = (getattr(resp, "text", None) or "").strip()
        return text or "Maaf, aku tidak menerima teks balasan dari Gemini."
    except Exception as e:
        return f"⚠ Gagal memanggil Gemini API: {e}"

def _safe_calc_expr(expr: str) -> str | None:
    """
    Evaluasi ekspresi aritmatika sederhana dengan aman (tanpa builtins).
    Mendukung + - * / ^ () dan persen, juga × ÷.
    """
    if not expr:
        return None
    # normalisasi operator
    s = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = s.replace("%", "*0.01")  # 50% -> 0.5 (perkalian)
    # izinkan hanya angka, spasi, dan operator aman
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)\*]{1,200}", re.sub(r"[*][*]", "**", re.sub(r"\s+", " ", s))):
        return None
    try:
        val = eval(s, {"__builtins__": None}, {})
        if isinstance(val, (int, float)):
            return f"Hasilnya: **{val}**"
        return None
    except Exception:
        return None


def _quick_time(msg: str) -> str | None:
    m = re.search(r"dalam (\d+)\s*(hari|jam|menit)", msg)
    if not m: 
        return None
    n, unit = int(m.group(1)), m.group(2)
    now = datetime.now()
    if unit.startswith("hari"):
        t = now + timedelta(days=n)
    elif unit.startswith("jam"):
        t = now + timedelta(hours=n)
    else:
        t = now + timedelta(minutes=n)
    return f"Nanti itu jatuh pada: **{t.strftime('%A, %d %B %Y %H:%M')}**."


def _playful_reply(msg: str) -> str:
    """Balasan santai untuk pertanyaan konyol/nyeleneh."""
    jokes = [
        "Pertanyaanmu unik! Intinya: kalau itu membuatmu senang dan tidak merugikan siapa pun—gas… tapi tetap logis ya 😄",
        "Secara ilmiah… hmmm… 42? (referensi Douglas Adams 😜)",
        "Aku bisa bantu jelasin atau ngarang kreatif. Mau versi serius atau kocak?",
    ]
    return np.random.choice(jokes)


def _tiny_writer(msg: str) -> str | None:
    """Generator singkat: puisi/pantun/cerita/jokes."""
    if any(k in msg for k in ["pantun", "pantoon"]):
        return "Pergi ke pasar beli pepaya,\nJangan lupa bawa payung.\nBelajar AI jangan ditunda,\nBiar masa depan makin canggih dan ayung~"
    if "puisi" in msg:
        return "Di balik baris kode yang sunyi,\nAda mimpi yang tumbuh pelan.\nPada data kita bertaruh arti,\nPada harap kita berjalan."
    if "cerita" in msg:
        return "Suatu malam, Anggun mengajar AI-nya bercanda. Tiba-tiba modelnya menjawab: 'harga hatiku? tak ternilai.' Sejak itu, UI-nya punya tombol 'bucin mode'."
    if any(k in msg for k in ["jokes", "joke", "lelucon", "guyon"]):
        return "Kenapa model linear jarang liburan? Karena dia selalu *fit* di garis lurus. 😅"
    return None


def _define_like(msg: str) -> str | None:
    """
    Respon definisi generik saat user tanya 'apa itu ...' tanpa web.
    """
    m = re.search(r"(apa itu|apa artinya|jelaskan)\s+(.+)", msg)
    if not m:
        return None
    term = m.group(2).strip().rstrip("?!.")
    if len(term) > 80:  # terlalu panjang, ringkas saja
        term = term[:80] + "..."
    return (f"**{term}** adalah konsep yang bergantung konteks. Secara umum:\n"
            f"- Definisi ringkas: gambaran/penjelasan tentang *{term}* sesuai domain.\n"
            f"- Contoh sederhana: gunakan *{term}* untuk tujuan yang relevan.\n"
            f"- Catatan: jika kamu sebutkan bidangnya (mis. statistik, UI/UX, otomotif), aku bisa memperjelas.")


def local_chat_response(user_message: str, last_prediction=None, last_input=None, df=None):
    """
    Fallback lokal 'omni' — selalu berusaha menjawab:
    - Kalkulator ekspresi sederhana
    - Waktu relatif (dalam X hari/jam/menit)
    - Writer mini (pantun/puisi/cerita/jokes)
    - Definisi generik 'apa itu ...'
    - Tanya harga/model berbasis dataset (jika ada)
    - Balasan santai untuk pertanyaan konyol
    """
    msg = (user_message or "").strip()
    low = msg.lower()

    # 1) kalkulator mini
    calc = _safe_calc_expr(low)
    if calc:
        return calc

    # 2) waktu relatif
    t = _quick_time(low)
    if t:
        return t

    # 3) penulisan kreatif singkat
    tiny = _tiny_writer(low)
    if tiny:
        return tiny

    # 4) definisi generik
    defin = _define_like(low)
    if defin:
        return defin

    # 5) dataset-aware: daftar model
    if any(k == low for k in ["model", "daftar model", "model mobil"]):
        if df is not None and not df.empty and "model" in df.columns:
            all_models = sorted(df["model"].dropna().astype(str).unique().tolist())
            return "🚗 Model yang tersedia:\n- " + "\n- ".join(all_models[:50])
        else:
            return "Dataset belum dimuat atau kolom `model` tidak ada."

    # 6) dataset-aware: harga rata-rata model
    if "harga" in low and df is not None and not df.empty and "model" in df.columns and "price" in df.columns:
        models = df["model"].dropna().astype(str).unique().tolist()
        # cari nama model dari pertanyaan
        m = difflib.get_close_matches(low, [m.lower() for m in models], n=1, cutoff=0.6)
        if m:
            found = m[0]
            matched_model = next((mm for mm in models if mm.lower() == found), found)
            avg_price = df[df["model"].astype(str).str.lower() == matched_model.lower()]["price"].mean()
            if pd.notna(avg_price):
                harga_rupiah = avg_price * 20000
                return f"💰 Rata-rata {matched_model}: {rupiah(harga_rupiah)} (≈ £{avg_price:,.2f})."
            else:
                return f"Harga {matched_model} belum ada di dataset."
        # jika tak ketemu modelnya, kasih tips
        return "Sebutkan model + tahun/transmisi/odometer supaya estimasinya lebih akurat ya."

    # 7) fallback santai untuk pertanyaan konyol/di luar konteks
    return _playful_reply(low)


# =========================
# PATH & DATA
# =========================
BASE_DIR = os.getcwd()
MODEL_PATH = "model.pkl"
DATASET_PATH = "toyota.csv"
EXAMPLE_PATH = "example_schema.json"

# === ANIMASI MOBIL: toggle + render GIF ===
def _file_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

with st.sidebar:
    show_car = st.toggle("🚗 Tampilkan animasi mobil", value=True)

CAR_GIF_PATH = os.path.join(BASE_DIR, "animasi.gif")  # letakkan animasi.gif di folder yang sama
if show_car and os.path.exists(CAR_GIF_PATH):
    _gif64 = _file_to_base64(CAR_GIF_PATH)
    if _gif64:
        st.markdown(
            f"""
            <div class="car-wrap">
              <img class="car-img" src="data:image/gif;base64,{_gif64}" alt="car-animation">
            </div>
            """,
            unsafe_allow_html=True
        )

# ======== AUTH GATE ========
if "user" not in st.session_state:
    auth_view()
    st.stop()

# ===== Sidebar user & logout =====
with st.sidebar:
    st.markdown(f"**👤 {st.session_state['user']['name']}**")
    st.caption(f"{st.session_state['user']['email']}")
    if st.button("Keluar"):
        st.session_state.pop("user", None)
        st.rerun()

# ===== Sidebar: status & uploader =====
with st.sidebar:
    st.subheader("📁 Berkas")
    st.write(f"Folder kerja: {BASE_DIR}")
    st.write(f"model.pkl: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
    st.write(f"toyota.csv: {'✅' if os.path.exists(DATASET_PATH) else '❌'}")
    st.write("---")
    up_model = st.file_uploader("Upload model.pkl (opsional)", type=["pkl"])
    if up_model is not None:
        try:
            with open(MODEL_PATH, "wb") as f:
                f.write(up_model.read())
            st.success("model.pkl tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan model: {e}")

    up_csv = st.file_uploader("Upload toyota.csv (opsional)", type=["csv"])
    if up_csv is not None:
        try:
            df_tmp = pd.read_csv(up_csv)
            df_tmp.to_csv(DATASET_PATH, index=False)
            st.success("toyota.csv tersimpan.")
        except Exception as e:
            st.error(f"Gagal menyimpan dataset: {e}")

    # =========================
    # Korpus Dokumen untuk QA  # NEW
    # =========================
    st.subheader("📚 Korpus Dokumen (opsional)")
    st.caption("Upload file .py / .txt / .md agar asisten bisa menjawab pertanyaan dari konten file.")
    up_docs = st.file_uploader("Tambah dokumen konteks", type=["py", "txt", "md"], accept_multiple_files=True)
    if "doc_corpus" not in st.session_state:
        st.session_state["doc_corpus"] = []

    if up_docs:
        added = 0
        for f in up_docs:
            text = _read_file_text(f)
            if text and text.strip():
                st.session_state["doc_corpus"].append({"name": f.name, "text": text})
                added += 1
        st.success(f"{added} dokumen ditambahkan ke korpus.")

    if st.button("🧹 Bersihkan Korpus"):
        st.session_state["doc_corpus"] = []
        st.toast("Korpus dibersihkan.")

# Load model
model = load_model_if_exists(MODEL_PATH)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    st.sidebar.success(f"📊 Dataset dimuat ({len(df)} baris)")
except Exception as e:
    st.sidebar.warning(f"⚠ Gagal memuat dataset: {e}")
    df = pd.DataFrame()

# Load example schema (fallback)
try:
    with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
        example_schema = json.load(f)
except Exception:
    example_schema = {
        "model": "Avanza", "year": 2020, "transmission": "Manual",
        "mileage": 15000, "fuelType": "Bensin", "tax": 1500000,
        "mpg": 14.5, "engineSize": 1.3
    }

# =========================
# ANTARMUKA
# =========================
# Buat layout tengah
centered = st.container()
with centered:
    st.markdown(
        """
        <style>
        /* ====== Styling agar konten di tengah ====== */
        .main .block-container {
            max-width: 700px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: auto;
            text-align: center;
        }

        /* ====== Styling teks ====== */
        .stTextInput > div > div > input {
            font-size: 18px !important;
            text-align: center;
        }

        .stButton > button {
            font-size: 18px !important;
            padding: 10px 25px;
        }

        h3, h2, h1, .stMarkdown, .stSubheader {
            text-align: center !important;
        }

        /* ====== Styling bubble chat ====== */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 10px;
            margin-bottom: 15px;
        }
        .chat-bubble-user {
            background-color: #DCF8C6;
            border-radius: 20px;
            padding: 10px 15px;
            text-align: left;
            align-self: flex-end;
            font-size: 17px;
            max-width: 80%;
        }
        .chat-bubble-ai {
            background-color: #EAEAEA;
            border-radius: 20px;
            padding: 10px 15px;
            text-align: left;
            align-self: flex-start;
            font-size: 17px;
            max-width: 80%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("💬 Chat Asisten For Anggun (Gemini-first)")

    # --- init state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # --- callback ---
    def handle_reset():
        st.session_state.chat_history = []
        st.session_state.chat_input = ""
        st.toast("Chat Anggun direset.")

    def handle_send():
        msg = (st.session_state.chat_input or "").strip()
        if not msg:
            return
        st.session_state.chat_history.append(("user", msg))
        resp = gemini_chat(
            user_message=msg,
            history=st.session_state.get("chat_history"),
            df=df,
            last_prediction=st.session_state.get("last_prediction"),
            last_input=st.session_state.get("last_input"),
        )
        if not isinstance(resp, str) or resp.strip() == "" or resp.strip().startswith("⚠"):
            resp = local_chat_response(
                msg,
                st.session_state.get("last_prediction"),
                st.session_state.get("last_input"),
                df
            )
        st.session_state.chat_history.append(("assistant", resp))
        st.session_state.chat_input = ""

    # Tombol atas
    st.button("🔄 Reset Chat Kalo Anggun mau", key="reset_chat_btn", on_click=handle_reset)
    st.caption("Tips: Khusus Anggun boleh tanya apa saja 💬 (Gemini-first ✅)")

    # Tampilkan riwayat chat
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        bubble = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
        st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input di tengah dengan teks besar
    st.text_input("Ketik pertanyaan Anggun di sini...", key="chat_input")
    st.button("💬 Kirim Pertanyaan Anggun Disini", key="send_btn", on_click=handle_send)



st.markdown("---")
st.caption("✨ Aplikasi prediksi harga mobil + chat Gemini AI — dengan Masuk/Daftar & animasi halus ✨")
