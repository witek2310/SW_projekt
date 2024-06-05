import cv2
import os
import numpy as np
import math
from matplotlib import pyplot as plt
def callback(arg):
    pass

def sort_points_ndarray(points):
    # Reshape the array to (4, 2) for easier manipulation
    points = points.reshape(4, 2)
    
    # # Sort points by y-coordinate (ascending)
    # points = sorted(points, key=lambda point: point[1])
    
    # # Extract the upper and lower points
    # upper_points = points[:2]
    # lower_points = points[2:]
    
    # # Sort the upper points by x-coordinate (ascending)
    # upper_points = sorted(upper_points, key=lambda point: point[0])
    # upper_left = upper_points[0]
    # upper_right = upper_points[1]
    
    # # Sort the lower points by x-coordinate (ascending)
    # lower_points = sorted(lower_points, key=lambda point: point[0])
    # lower_left = lower_points[0]
    # lower_right = lower_points[1]
    
    # # Combine them in the desired order and convert back to ndarray
    # sorted_points = np.array([upper_left, upper_right, lower_left, lower_right], dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def load_letter()->dict:
    letters_paths = os.listdir("letters")
    letters_cnt = {}
    letters_imgs= {}
    for letter in letters_paths:
        path = "letters/"+ letter
        img = cv2.imread(path,  cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        cons, hiers = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img_temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for con, hier in zip(cons, hiers[0]):
            if hier[-1] == -1:
                con_corr = con
        # img_col_corr = cv2.drawContours(img_temp, [con_corr], 0, (255, 0, 0))
        # key = None
        # while key != ord('q'):
        #     cv2.imshow("win", img_col_corr)
        #     key = cv2.waitKey(10)
        letters_cnt[letter[0]] = con_corr
        letters_imgs[letter[0]] = img
    return letters_cnt, letters_imgs



one_hole_letters = ["A", "D", "O", "P", "Q", "R", "0", "4", "6", "9"]
two_hole_letters = ["B", "8"]

def main(path:str, letters: dict, letters_imgs: dict):

    ileeeee = len(path.strip(".jpg"))
    print(ileeeee)
    cv2.namedWindow("testy")
    cv2.createTrackbar("tr1", "testy", 19, 255, callback)
    cv2.createTrackbar("tr2", "testy", 4, 255, callback)
    cv2.createTrackbar("tr3", "testy", 1, 255, callback)
    cv2.createTrackbar("tr4", "testy", 5, 255, callback)
    img = cv2.imread("train_1/"+path)
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow("original img", img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





    key = None
    dst = None
    while key != ord('q'):
        tr1 = cv2.getTrackbarPos("tr1", "testy")
        tr2 = cv2.getTrackbarPos("tr2", "testy")
        tr3 = cv2.getTrackbarPos("tr3", "testy")
        tr4 = cv2.getTrackbarPos("tr4", "testy")
        img_gray_2 = cv2.GaussianBlur(img_gray, (2*tr4+1, 2*tr4+1), 0)
        # _, img_contours = cv2.threshold(img_gray_2, tr1, 255, cv2.THRESH_BINARY_INV)
        img_contours = cv2.adaptiveThreshold(img_gray_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, tr1*2 +1, tr2)
        img_contours_2 = cv2.dilate(img_contours, np.ones((tr3, tr3)))
        closed = cv2.morphologyEx(img_contours_2, cv2.MORPH_CLOSE, np.ones((5,5)))
        # closed = cv2.morphologyEx(img_gray_2, cv2.MORPH_OPEN, rectKern)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        image_height, image_width = img.shape[:2]

        min_plate_width = image_width // 3


        i = 0
        for contour in contours:
            
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If the approximated contour has four points, then assume we have found the license plate
            if len(approx) == 4:
                # print(f"new aprox found {approx}")
                # Extract the bounding box of the approximated contour
                (x, y, w, h) = cv2.boundingRect(approx)
                # Check if the width of the bounding box is at least one-third of the image width
                area = cv2.contourArea(approx)
                if w >= min_plate_width and 20000< area < 140000:
                    win_name = f"win_{i}"
                    i+=1
                    # Extract the region of interest (the license plate)
                    plate = img[y:y+h, x:x+w]

                    # Save the detected license plate to the output path
                    cv2.imshow(win_name, plate)
                    # key = cv2.waitKey()
                    i += 1
                    # pts1 = approx.astype(np.float32)
                    pts1 = sort_points_ndarray(approx)
                    # pts2 = np.array([[0,0], [800, 0], [0, 100], [800, 100]], dtype=np.float32)
                    pts2 = np.array([[0, 0],[800, 0],[800, 180],[0, 180]], dtype="float32")
                    M = cv2.getPerspectiveTransform(pts1,pts2)
 
                    dst = cv2.warpPerspective(img,M,(800,180))
                    




            # Initialize ORB detector


        # # Find the keypoints and descriptors with ORB
        if dst is None:
            break
        dst = cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, np.ones((5,5)))
        cv2.imshow(win_name+"p", dst)
        contours, hiers = cv2.findContours(dst.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dst_temp = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        
        

        print("new iter")
        # print(letters.keys())
        i = 0
        loc_letter = []
        orb = cv2.SIFT_create()

        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   #ORB
        bf = cv2.BFMatcher(crossCheck=True)
        cnt_number = -1
        for contour, hier in zip(contours, hiers[0]):
            cnt_number +=1
            if hier[-1] == -1:
                area = cv2.contourArea(contour)
                if (area > 900):
                    i+=1
                    best_match = 0
                    best_letter = None

                    nr_of_holes = 0
                    if hier[2] != -1:   #contour has a child (or children)
                        child_index = hier[2]
                        hole_area = cv2.contourArea(contours[child_index])
                        if hole_area > 100:
                            nr_of_holes +=1

                        if hiers[0][child_index][0] != -1:  #chck if there is a contour in the same hier
                            hole_area = cv2.contourArea(contours[hiers[0][child_index][0]])
                            if hole_area > 100:
                                nr_of_holes+=1


                    x,y,w,h= cv2.boundingRect(contour)
                    kp1, des1 = orb.detectAndCompute(dst[y:y+h, x:x+w], None)
                    cv2.imshow("cut", dst[y:y+h, x:x+w])
                    for key in letters:
                        if nr_of_holes == 0:
                            if key in one_hole_letters or key in two_hole_letters:
                                continue
                        if nr_of_holes == 1:
                            if key not in one_hole_letters:
                                continue
                        if nr_of_holes == 2:
                            if key not in two_hole_letters:
                                continue
                        #using descriptors

                        # kp2, des2 = orb.detectAndCompute(letters_imgs[key], None)
                        # matches = bf.match(des1, des2)
                        # matches = sorted(matches, key=lambda x: x.distance)
                        # score = 0
                        # for match in matches[:3]:
                        #     print(match.distance)
                        #     score += match.distance

                        #using matchshapes

                        # score = cv2.matchShapes(letters[key], contour, 1, 1)


                        #using binary masks
                        x_l,y_l,w_l,h_l= cv2.boundingRect(letters[key])
                        mask_rej = dst[y:y+h, x:x+w] > 100

                        if key == "I":
                            continue
                        temp_letter = cv2.resize(letters_imgs[key][y_l:y_l+h_l, x_l:x_l+w_l], (mask_rej.shape[1], mask_rej.shape[0]))
                        mask_letter = temp_letter[:,:] > 100

                        and_mask = mask_letter * mask_rej
                        score = np.sum(and_mask)

                        # print(f"score:{score}, letter: {key}, leftmost: {tuple(contour[contour[:,:,0].argmin()][0])}, #hole :{nr_of_holes}")
                        # cv2.waitKey()
                        if score > best_match:
                            best_match = score
                            best_letter = key
                    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                    loc_letter.append((leftmost, best_letter))
                    # cv2.waitKey()

        loc_letter = sorted(loc_letter, key= lambda x: x[0])
        print(loc_letter)
        print(i - ileeeee)

    # Check if enough matches are found

        cv2.imshow("testy", closed)
        key = cv2.waitKey()
    cv2.destroyAllWindows()
    return 


def test():

    img = cv2.imread("train_1\CIN20356.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    s_6 = cv2.imread("letters\\6.png", cv2.IMREAD_GRAYSCALE)
    empty = np.zeros((img.shape[0]+s_6.shape[0], max(img.shape[1], s_6.shape[1])))
    surf = cv2.xfeatures2d.SIFT_create(400)
    kp, des = surf.detectAndCompute(img,None)
    kp_2, des_2 = surf.detectAndCompute(s_6,None)
    matcher = cv2.BFMatcher(2,True)
    matches = matcher.match(des, des_2)

    img_matched = cv2.drawMatches(img, kp, s_6, kp_2, matches, empty)
    cv2.imshow("mathces", img_matched)
    key = None
    while key != ord('q'):
        key = cv2.waitKey()


import cv2
import numpy as np



if __name__ == "__main__":
    letters_cnts, letters_imgs = load_letter()
    imgs = os.listdir("train_1")

    # print(imgs)
    for image in imgs:
        main(image, letters_cnts, letters_imgs)
    # load_letter()
    # test()
    # find_license_plate(imgs[0])
    # main(imgs[0])