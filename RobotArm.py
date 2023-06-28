import tkinter as tk
import cv2
import mediapipe as mp
import threading
import math
import numpy as np
from scipy.optimize import fsolve
import serial
import time

k='k'

# 미디어파이프를 위한 초기화
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ser = serial.Serial('COM4',9600)    #아두이노와 시리얼 통신을 위한 객체 생성
time.sleep(2)   #야두이노와 연결 때까지 2초 대기

def equation(alpha):
    return cam1_x/math.cos(math.radians(30) + alpha) - cam2_x/math.cos(alpha)

def process_dual_camera(camera_id,theta):   #투캠 함수, theta는 웹캠 기준으로 틀어진 각도
    cap = cv2.VideoCapture(camera_id)  # 카메라 번호를 지정하여 원하는 카메라로 변경 가능
    
    global vec1
    global vec2
    vec1=[0,0]
    vec2=[0,0]
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break


            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 포즈 주석을 이미지 위에 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if not results.pose_landmarks:
                continue

            #길이 계산하기
            len_x1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            len_y1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            len_y1=len_y1*3/4                              #비율에 따라 보정하기
            
            len_x2 = -results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            len_y2 = -results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            len_y2=len_y2*3/4
            
            if camera_id==1:
                vec1 = [len_x1, len_y1 ]     #vec1은 위쪽 팔 벡터, vec2은 아랫쪽 팔 벡터
                vec2 = [len_x2, len_y2 ]
                
            if camera_id==0:
                vec1_2 = [len_x1, len_y1 ]     #vec1은 위쪽 팔 벡터, vec2은 아랫쪽 팔 벡터
                vec2_2 = [len_x2, len_y2 ]
                
                vec1_2 = [len_x1*vec1[1]/vec1_2[1], vec1[1] ]     #y길이에 기준해 재정의 하기
                vec2_2 = [len_x2*vec2[1]/vec2_2[1], vec2[1] ]

                len_z1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z
                len_z2 = -results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

                if(len_z1>0):                #부호 감별 코드 추가
                    sign1=1
                else:
                    sign1=-1

                if(len_z2):
                    sign2=1
                else:
                    sign2=-1

                cam1_x = vec1[0]
                cam2_x = vec1_2[0]

                alpha_1 = fsolve(equation, 0)* (180 / math.pi)

                cam1_x = vec2[0]
                cam2_x = vec2_2[0]

                alpha_2 = fsolve(equation, 0)* (180 / math.pi)
                
                length_z_1 = vec1[0]*math.tan(alpha_1+30)
                realvector1=[len_x1,len_y1,sign1*length_z_1]

                length_z_2 = vec1[0]*math.tan(alpha_1+30)

                realvector2=[len_x2,len_y2,sign2*length_z_2]

                if(len(realvector1)==3 and len(realvector2)==3):
                    VectortoAngle(realvector1, realvector2)
                
            cv2.imshow('MediaPipe Hands and Pose'+str(camera_id), image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    ser.close() #시리얼 통신 종료
    
def select_dual_cameras():  #투캠 선택 함수
    global vec1
    global vec2
    
    camera_thread1 = threading.Thread(target=process_dual_camera, args=(0,30))
    camera_thread2 = threading.Thread(target=process_dual_camera, args=(1,30))

    camera_thread1.start()
    camera_thread2.start()

    camera_thread1.join()
    camera_thread2.join()

def VectortoAngle(vec1, vec2):      #각도를 생성하는 함수       

    up = np.array([-vec1[0], -vec1[1], -vec1[2]])   # 차렷 자세를 기준으로 만든다
    down = np.array([-vec2[0], -vec2[1], -vec2[2]])
        
    up_proj = np.array([0, up[1], up[2]])
    mr1 = math.acos((up[1])/np.linalg.norm(up_proj));    #어께 바닥면 각도;; 180도만 회전 가능 360도 시 acos dms 0 ~180 도만 변환해 줌으로 수정 요   
    mr2 = 2 *math.acos(np.linalg.norm(up_proj)/np.linalg.norm(up));     #어께 위로 각도
    mr3 = math.acos(np.dot(up,down)/(np.linalg.norm(up)*np.linalg.norm(down)));   #하완 위로 각도

    rotate_m1= np.array([[1,0,0],[0,math.cos(-mr1),-math.sin(-mr1)],[0,math.sin(-mr1),math.cos(-mr1)]]) #x
    #rotate_m3= np.array([[math.cos(mr3),0,math.sin(mr3)],[0,1,0],[-math.sin(mr3),0,math.cos(mr3)]]) #y축 회전
    rotate_m1_r= np.array([[1,0,0],[0,math.cos(mr1),-math.sin(mr1)],[0,math.sin(mr1),math.cos(mr1)]]) #x
    up_tra = np.array([[up[0]], [up[1]], [up[2]]])
    up_rot = rotate_m1@up_tra
    up_rot[2] = math.sin(mr3)*np.linalg.norm(up_rot)
    up_rot3 = rotate_m1_r@up_rot
    up_rotate = np.transpose(up_rot3)
    
    
    
    #down_proj = np.array([down_rotate[0],0,down_rotate[2]])
    mr4 = math.acos(np.dot(up_rotate,down)/(np.linalg.norm(up_rotate)*np.linalg.norm(down)));
                                
    m1 = round(math.degrees(mr1),-1);            
    m2 = round(math.degrees(mr2),-1);            
    m3 = 90-round(math.degrees(mr3),-1);            
    m4 = (90-round(math.degrees(mr4),-1));
    #m5 = math.degrees(mr5);            
    #m6 = math.degrees(mr6);

    print(int(m1),int(m2),int(m3),int(m4),90,90)   #int(m5),int(m6)
    #아두이노에 행렬 넘겨주기
    ser.write((str(m1)+'/'+str(m2)+'/'+str(m3)+'/'+str(m4)+'/'+"90"+'/'+"90"+k).encode())        #ser.write(k.encode())
    time.sleep(2)
    


    
select_dual_cameras()


