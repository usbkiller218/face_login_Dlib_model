from flask import Flask ,request,jsonify,json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

import dlib, cv2

import numpy as np

from flask import Flask,request

import psycopg2
import psycopg2.extras

from flask import Flask ,request,jsonify,json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from sqlalchemy import Column, Integer, String,ARRAY,Float
app=Flask(__name__)

db_uri = 'postgresql://mydb:1234@127.0.0.1:5432/flask_db'
engine = create_engine(db_uri)
Session = sessionmaker(bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name=Column(String,nullable=False)

    image = Column(ARRAY(Float))

Base.metadata.create_all(engine)
#for i in userx:
    #print(i)
    #cs=np.array(i)
    #print(cs)
    #try:
     #k = np.linalg.norm(cs - cs)
     #print(k)
    #except Exception:
       # pass


#print(k)






@app.route('/users', methods=['GET'])
def get_users():
    imag1 = request.files.get('image1', '')

    imag1.save('x.jpg')

    path1 = r'x.jpg'

    known_image1 = cv2.imread(path1)

    enc1 = whirldata_face_encodings(known_image1)
    session = Session()
    userx = session.query(User.name).all()
    session.commit()
    for i in userx:

        session = Session()
        l = session.query(User.image).filter_by(name=str(i[0])).all()
        yy = np.array(l)
        print(i[0])
        print(yy)

        k = np.linalg.norm(enc1- yy)
        if k < 0.6:
            return str(i)

    return 'invalid face '



@app.route('/users', methods=['POST'])
def create_user():
    imag1 = request.files.get('image1', '')

    imag1.save('x.jpg')

    path1 = r'x.jpg'

    known_image1 = cv2.imread(path1)

    # known_image2= cv2.imread(path2)
    # #print(known_image.shape)
    enc1 = whirldata_face_encodings(known_image1)
    p=enc1.tolist()
    session = Session()
    user = User(image=[p],name=request.form.get('name'))
    session.add(user)
    session.commit()
    return "", 201


#Models Loadedcl

face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1 .dat')
def whirldata_face_detectors(img, number_of_times_to_upsample=1):
 return face_detector(img, number_of_times_to_upsample)

def whirldata_face_encodings(face_image,num_jitters=1):
 face_locations = whirldata_face_detectors(face_image)
 pose_predictor = pose_predictor_68_point
 predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
 return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]


app.run(debug=True)



