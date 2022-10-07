# Recognition not available due to Empathy platform mainteinance
# Face-Recognition (One-shot)

#### A face-recognition AI program which uses the Inception ResNet model architecture and does only require one single photo created for web-apps login purposes.

## Workflow

1. Firstly, it calculates the distance difference between the image source (the profile user photo), and the current user's face by taking either the euclidean or cosine distance of them
2. Then, it will connect to an existing web-application [Empathy](https://www.empathy.oracliom.com/landing), and create a temporary image "_temp.jpg_" which will be a user profile photo.
3. Finally, a face will be cropped from the temporary face image and it will create the file _user_face.png_ from which the model will take a representation source and finally get the distance between this one and the current recorded face.


### Setup

* If you do not have an Empathy account, you will first need to [sign up](https://www.empathy.oracliom.com/access/register) yourself. Please, take into consideration that the username with which you are signing up is the one you will later on give as input on the program.


```properties

pip install -r requirements.txt

python recognizer.py

``` 

* After loading all the weights, a username name will be asked in order to access to the web-application users photos database. Then, it will look for the user photo and open a window in which the webcam will be running. 

```properties
Enter your user nickname: 
``` 

##### Depending on the distance between the images, the user will be either identified or not identified. In case it finds similarity between both the number of _steps_ will be reduced by one and the minimum percentage of similarity will get higher to make sure you are the user logging in.


Thank you so much for your interest, in case you may find any error on the program do not hesitate to [contact me](https://nanodayo23.github.io/contact.html)
