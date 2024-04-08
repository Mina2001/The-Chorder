import streamlit as st

#Style css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>" , unsafe_allow_html=True)

local_css("style/style.css")

with st.container():
    st.subheader("Get In Touch With Me")
    st.write("##")
    
contact_form = """
<form action="https://formsubmit.co/b6c00e8b6d9d815d1f7007bb93ce415c" method="POST">
     <input type="hidden" name= "_captcha" value= "false">
     <input type="text" name="name" placeholder= "Your Name" required>
     <input type="email" name="email" placeholder= "Your E-mail" required>
     <textarea name="message" placeholder="Your Messeage" required></textarea>
     <button type="submit">Send</button>
</form>
"""
left_column, right_column = st.columns(2)
with left_column:
    st.write(
            "Greetings! I am Minandi Wilathgamuwa, the dedicated developer behind this project, and I appreciate your interest in reaching out. Your feedback, inquiries, or collaboration proposals are valued contributions that help enhance the quality and functionality of this endeavor. Reach out to me through the contact form."
        )
    st.write("---")
    st.write("Minandi Wilathgamuwa")
    st.write("Undergraduate")
    st.write("University of Westminster")
    
with right_column:
    st.markdown(contact_form, unsafe_allow_html= True)
    
    
