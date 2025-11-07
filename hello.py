<<<<<<< HEAD
import streamlit as st

st.set_page_config(page_title="Cek Streamlit", page_icon="✅", layout="centered")
st.title("✅ Streamlit jalan!")
st.write("Kalau kamu melihat halaman ini, berarti instalasi & port berjalan normal.")
if st.button("Tes tombol"):
    st.success("Tombol bekerja!")

=======
import streamlit as st

st.set_page_config(page_title="Cek Streamlit", page_icon="✅", layout="centered")
st.title("✅ Streamlit jalan!")
st.write("Kalau kamu melihat halaman ini, berarti instalasi & port berjalan normal.")
if st.button("Tes tombol"):
    st.success("Tombol bekerja!")

>>>>>>> 8c40dabf (init: project + model.pkl via LFS)
st.info("Selanjutnya kita bisa jalankan app prediksi mobil.")