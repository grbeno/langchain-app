from streamlit_authenticator import Authenticate
import yaml
from yaml.loader import SafeLoader


def auth():
    """ 
        login = authenticator.login('main', fields = {'Form name': 'Login'})
        logout = authenticator.logout('Logout', 'sidebar')
        name, authentication_status, username = login
    """

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
        )
    
    return authenticator
    
    