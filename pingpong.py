import cv2  # python library for computer vision (cv)
import cvzone  # python library for advance computer vision (cv)
from cvzone.HandTrackingModule import HandDetector  # Hand tracking module
import numpy as np

cap = cv2.VideoCapture(0)  # to open a webcam in python
cap.set(3, 1280)  # set image frame width
cap.set(4, 720)  # set image frame height

# Importing all images that will be used in the game
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)  # initiating hand detector

# Variables
ballPos = [100, 100]  # At the start of the game ball will be at 100,100 position (x and y coordinate)
speedX = 20  # speed by which ball will move (We can lower it or increase it)
speedY = 20
gameOver = False
score = [0, 0]  # We will store the scores here.

while True:  # a loop to run the webcam until interruption
    _, img = cap.read()  # take an image frame from a webcam
    img = cv2.flip(img, 1)  # flipping the image
    imgRaw = img.copy()  # taking a rawimage to show it while we are playing the game

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']  # get the bounding box of the hand (see the google doc)
            h1, w1, _ = imgBat1.shape  # get the image height(h1) and width(w1)
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)  # clip is used to clip(or limit) the y position

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))  # when left hand is detect, bring its bat
                # if x coordinate of the ball comes close to x coordinate of the bat
                # if y coordinate of the ball comes close to mid ppint of the bat
                # then send the ball to opposite direction (-speedX), basically opposite coordinate
                # also push the ball away a bit (30 coordinate)
                # increment the score
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))  # when right hand is detect, bring its bat
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game Over
    # if x coordinate of the ball crosses the extreme ends of the frame
    # Declare game over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        # When game is over, bring the gameover image
        # Also put the text(score) on the image
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)

    # If game not over move the ball
    else:

        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):  # pressing R button will restart the game
        ballPos = [100, 100]  # it will bring the ball back to start position
        speedX = 20
        speedY = 20
        gameOver = False
        score = [0, 0]  # reset the score
        imgGameOver = cv2.imread("Resources/gameOver.png")  # reset the background image
