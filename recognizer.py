import requests
from PIL import Image
import json
import os

def getEmpathyToken():

    url = "https://api.oracliom.com/api/auth/login"

    payload = 'email=test.empathy@oracliom.com&password=123456789'
    headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
  }

    response = requests.request("POST", url, headers=headers, data = payload)
    response_info = json.loads(response.text)
    empathy_token = response_info["token"]

    return empathy_token

def getUserURL(empathyToken, user):
    url = 'https://api.oracliom.com/api/user/other/%s' % (user)

    payload = {}
    headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'x-access-token': empathyToken
            }

    response = requests.request("GET", url, headers=headers, data = payload)

    response_info = json.loads(response.text)
    user_photo = response_info['photoId']
    amazon_img = "https://pictures-empathy.s3.amazonaws.com/"
    user_url = amazon_img + user_photo

    return user_url


from facecropper import extractFace
from richardai import faceRecognition


def Recognize():
    user_name = input("Enter your user nickname: ")
    empathy_token = getEmpathyToken()
    img_url = getUserURL(empathyToken= empathy_token, user= user_name)

    r = requests.get(img_url)
    with open('temp.jpg', 'wb') as f:
        f.write(r.content)

    extractFace('temp.jpg')
    retJson = faceRecognition()

    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')

    if os.path.exists('user_face.jpg'):
        os.remove('user_face.jpg')

    return retJson


if __name__  == "__main__":
    Recognize()
    
