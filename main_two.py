import numpy as np
import cv2
import imutils
import time

cap = cv2.VideoCapture('testvideo2.mp4')


while(cap.isOpened()):

    ret, frame = cap.read()

    try:
        snip = frame[500:700,300:900]
    except:
        exit()
    cv2.imshow("Original",snip)
    a = (0, 200) # inferior esquerda
    b = (200, 50) # superior esquerda
    c = (430, 50) # superior direita
    d = (575, 200) # inferior direita

    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([a, b, c, d], dtype=np.int32)


    cv2.fillConvexPoly(mask, pts, 255)
    cv2.imshow("Mascara", mask)


    masked = cv2.bitwise_and(snip, snip, mask=mask)
    cv2.imshow("Regiao de interesse", masked)


    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    thresh = 200
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Preto/Branco", frame)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # cv2.imshow("Blurred", blurred)

    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Bordas", edged)


    lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)


    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []


    if lines is not None:


        for i in range(0, len(lines)):


            for rho, theta in lines[i]:


                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)


                    a = np.cos(theta); b = np.sin(theta)
                    x0 = a * rho; y0 = b * rho
                    x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                    x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    #
                    #cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)


                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)


                    a = np.cos(theta); b = np.sin(theta)
                    x0 = a * rho; y0 = b * rho
                    x1 = int(x0 + 400 * (-b)); y1 = int(y0 + 400 * (a))
                    x2 = int(x0 - 600 * (-b)); y2 = int(y0 - 600 * (a))
                    #
                    #cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)


    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)


    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        offset1 = 250; offset2 = 800
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))

        #cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/4:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 250; offset2 = 800
        x3 = int(x0 - offset1 * (-b)); y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b)); y4 = int(y0 - offset2 * (a))

        #cv2.line(snip, (x3, y3), (x4, y4), (255, 0, 0), 6)




    if left_theta > np.pi/4 and right_theta > np.pi/4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)


        overlay = snip.copy()

        #cv2.fillConvexPoly(overlay, pts, (0, 255, 0))

        opacity = 0.4
        #cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)





    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 20, 2, 1)
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:

            cv2.circle(snip, (x1, y1), 5, (0, 0, 255), 1)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("Linhas", snip)


    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

