import json
from oauth2client.client import OAuth2WebServerFlow, flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow

CLIENT_SECRET_FILE = '../client_secret_920217954379-3oesbpgsc355jv1qpjlsfj8cp3531d7d.apps.googleusercontent.com.json'

SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.appdata']

def get_token_oauth2client():
    flow = flow_from_clientsecrets(CLIENT_SECRET_FILE, scope=SCOPES)

    storage = Storage('token_oauth2client.json')
    credentials = run_flow(flow, storage)

    with open('../generated_token.json', 'w') as token_file:
        token_file.write(credentials.to_json())


    print("Token information saved to generated_token.json")

if __name__ == '__main__':
    get_token_oauth2client()