#import libraries
import cv2
import face_recognition


imgSid = face_recognition.load_image_file(r'C:\ishika\ImagesBasics\Sidd.jpg')
imgSid = cv2.cvtColor(imgSid,cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file(r'C:\ishika\ImagesBasics\Sidd_test.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file(r'C:\ishika\ImagesBasics\Sidd_test2.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSid)[0]
encodeElon = face_recognition.face_encodings(imgSid)[0]
cv2.rectangle(imgSid, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)# comparing list of known faces and recognize face
faceDis = face_recognition.face_distance([encodeElon], encodeTest)# lower the distance better the match
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Siddharth', imgSid)
cv2.imshow('Sid Test', imgTest)
cv2.waitKey(0)
