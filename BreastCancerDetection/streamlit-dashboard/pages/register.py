import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# if not st.session_state.authentication_status:
#    st.info('Please Login from the Home page and try again.')
#    st.stop()

# To hide the sidebar
st.set_page_config(initial_sidebar_state="collapsed")


with open("style.css" ) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title('SmartScan AI')

with open('./credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


try:
    email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
        pre_authorization=False,
        clear_on_submit=True)
    if email_of_registered_user:
        st.markdown('<p style="background-color:rgba(33, 195, 84, 0.6);'
                    'color:#000;'
                    'font-size:16px;'
                    'border-top-left-radius:0.5rem;'
                    'border-top-right-radius:0.5rem;'
                    'border-bottom-right-radius:0.5rem;'
                    'border-bottom-left-radius:0.5rem;'
                    'padding:2%;">'
                    'User registered successfully!</p>',
                    unsafe_allow_html=True)
        #st.success('User registered successfully')

except Exception as e:
    st.markdown(f"""<p style="background-color:rgba(255, 227, 18, 0.4);
                color:#000;
                font-size:16px;
                border-top-left-radius:0.5rem;
                border-top-right-radius:0.5rem;
                border-bottom-right-radius:0.5rem;
                border-bottom-left-radius:0.5rem;
                padding:2%;">
                {e}</p>""",
                unsafe_allow_html=True)
    # st.error(e)


with open('credentials.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)


# Login button
if st.button("Login"):
    st.switch_page("login.py")
