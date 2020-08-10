from fyers_api import accessToken
from fyers_api import fyersModel
import sys
import webbrowser


def generate_session(app_id,app_secret):
    app_session=accessToken.SessionModel(app_id,app_secret)
    response=app_session.auth()

    auth_code=response["data"]["authorization_code"]
    app_session.set_token(auth_code)

    generateTokenUrl=app_session.generate_token()
    webbrowser.open(generateTokenUrl,new=1)

    token=input("Paste the token here")         #acces token is copied from the redirected url
    with open("accesstoken.txt","w") as f:
        f.write(token)                             #token is saved as txt for further use

    is_async =False #(By default False, Change to True for asnyc API calls.)

    fyers = fyersModel.FyersModel(is_async)
    profile=fyers.get_profile(token = token)

    print(profile)

def main():
    
    with open("AppId.txt","r") as f:
        app_id=f.read()

    with open("secretId.txt","r") as f:
        app_secret=f.read()
    
    generate_session(app_id,app_secret)


if __name__ == "__main__":
	main()

