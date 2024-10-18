import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# To hide the sidebar
st.set_page_config(initial_sidebar_state="collapsed")

with open("./style.css" ) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title('SmartScan AI')
# st.image('C:/Users/momiv/OneDrive/Desktop/magistrale/1st_year/SIAM in Healthcare/project2/pages/logo_smartscan_nobg.png')

# st.markdown(
#    """<img src="./pages/logo_smartscan_nobg.png" style="scale: 0.7;" alt="">""",
#   unsafe_allow_html=True)

with open('./credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('main', fields={'Form name': 'Login'})


if authentication_status:
    st.session_state.authentication_status = True
    st.switch_page('pages/main.py')

elif authentication_status is False:
    st.session_state.authentication_status = False
    st.markdown('<p style="background-color:rgba(255, 43, 43, 0.3);'
                'color:#000;'
                'font-size:16px;'
                'border-top-left-radius:0.5rem;'
                'border-top-right-radius:0.5rem;'
                'border-bottom-right-radius:0.5rem;'
                'border-bottom-left-radius:0.5rem;'
                'padding:2%;">'
                'Username/password is incorrect</p>',
        unsafe_allow_html=True)
    # st.error('Username/password is incorrect')

elif authentication_status is None:
    st.session_state.authentication_status = None
    st.markdown('<p style="background-color:rgba(255, 227, 18, 0.4);'
                'color:#000;'
                'font-size:16px;'
                'border-top-left-radius:0.5rem;'
                'border-top-right-radius:0.5rem;'
                'border-bottom-right-radius:0.5rem;'
                'border-bottom-left-radius:0.5rem;'
                'padding:2%;">'
                'Please enter your username and password</p>',
                unsafe_allow_html=True)
    #st.warning('Please enter your username and password')

# Register button
if st.button("Register"):
    st.switch_page("pages/register.py")
