import streamlit as st

detectPage = st.Page("page1.py", title="Face Recognition", icon=":material/add_circle:")
faceTakerPage = st.Page("page2.py", title="Dataset", icon=":material/delete:")
# deleteUser = st.Page("page4.py", title="Delete User", icon=":material/delete:")

pg = st.navigation([detectPage, faceTakerPage])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()