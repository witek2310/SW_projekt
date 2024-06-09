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
    # print(ileeeee)
    # cv2.namedWindow("testy")
    # cv2.createTrackbar("tr1", "testy", 19, 255, callback)
    # cv2.createTrackbar("tr2", "testy", 4, 255, callback)
    # cv2.createTrackbar("tr3", "testy", 1, 255, callback)
    # cv2.createTrackbar("tr4", "testy", 5, 255, callback)
    img = cv2.imread("train_1/"+path)
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    # cv2.imshow("original img", img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





    key = None
    dst = None
    while key != ord('q'):
        # tr1 = cv2.getTrackbarPos("tr1", "testy")
        # tr2 = cv2.getTrackbarPos("tr2", "testy")
        # tr3 = cv2.getTrackbarPos("tr3", "testy")
        # tr4 = cv2.getTrackbarPos("tr4", "testy")
        tr1 = 19
        tr2 = 4
        tr3 = 1
        tr4 = 5


        #getting ale of the counotur in the picture
        img_gray_2 = cv2.GaussianBlur(img_gray, (2*tr4+1, 2*tr4+1), 0)
        # _, img_contours = cv2.threshold(img_gray_2, tr1, 255, cv2.THRESH_BINARY_INV)
        img_contours = cv2.adaptiveThreshold(img_gray_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, tr1*2 +1, tr2)
        img_contours_2 = cv2.dilate(img_contours, np.ones((tr3, tr3)))
        closed = cv2.morphologyEx(img_contours_2, cv2.MORPH_CLOSE, np.ones((5,5)))
        # closed = cv2.morphologyEx(img_gray_2, cv2.MORPH_OPEN, rectKern)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        image_height, image_width = img.shape[:2]

        min_plate_width = image_width // 3  #registration plate takes at least 1/3 if whole picture

        # cv2.imshow("testy", closed)
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
                if w >= min_plate_width and 20000< area < 140000: #check if the contour is big enought to be a registaration plate 
                    
                    #find cornres of the registration plate and transform them to the rectangle
                    pts1 = sort_points_ndarray(approx)
                    pts2 = np.array([[0, 0],[800, 0],[800, 180],[0, 180]], dtype="float32")
                    M = cv2.getPerspectiveTransform(pts1,pts2)

                    #applay transfrom to the original image
                    dst = cv2.warpPerspective(img,M,(800,180))  
        #if non registration plate was found brek
        if dst is None:
            break

        #dst is fragment of the photo containgin the registration plate
        #it is in color, so make it gray, applay threshold so it becomes black-white image, filter it using morphology operations, and again contours
        dst = cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, np.ones((5,5)))
        contours, hiers = cv2.findContours(dst.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("new iter")
        # print(letters.keys())
        i = 0
        loc_letter = []

        cnt_number = -1
        #iterate over find contours
        for contour, hier in zip(contours, hiers[0]):
            cnt_number +=1
            #if the countour has no perent it is probobly a letter
            if hier[-1] == -1:
                area = cv2.contourArea(contour)
                leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                rightmost = tuple(contour[contour[:,:,0].argmax()][0])
                # topmost = tuple(contour[contour[:,:,1].argmin()][0])
                # bottommost = tuple(contour[contour[:,:,1].argmax()][0])
                # print(leftmost, rightmost, topmost, bottommost)

                #if the letter is to wide it proboly is not a letter 
                if (abs(leftmost[0] - rightmost[0]) > 200): #
                    continue
                #if area is big enoug it proboly is a letter
                if (area > 900):
                    i+=1
                    best_match = 0
                    best_letter = None

                    #count the number of holes in the letter 
                    nr_of_holes = 0
                    if hier[2] != -1:   #contour has a child (or children)
                        child_index = hier[2]
                        hole_area = cv2.contourArea(contours[child_index])
                        if hole_area > 100: #and that child is big
                            nr_of_holes +=1 #it means that this is a hole in the letter not noise

                        if hiers[0][child_index][0] != -1:  #chck if there is a contour in the same hier
                            hole_area = cv2.contourArea(contours[hiers[0][child_index][0]])
                            if hole_area > 100:
                                nr_of_holes+=1


                    x,y,w,h= cv2.boundingRect(contour)
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

                        #here starts recognizing the letters
                        #the idea is that binary masks of letter in the registration plate and mask created based on a font will be similar
                        
                        
                        mask_rej = dst[y:y+h, x:x+w] > 100  #get the binary mast of the letter on the registration plate

                        if key == "I":
                            continue
                        
                        
                        x_l,y_l,w_l,h_l= cv2.boundingRect(letters[key])  #get the bouund box of the font based latter
                        #cut only the letter from the font based image abd resize it so it is the same size as the letter in registration plate
                        temp_letter = cv2.resize(letters_imgs[key][y_l:y_l+h_l, x_l:x_l+w_l], (mask_rej.shape[1], mask_rej.shape[0]))

                        mask_letter = temp_letter[:,:] > 100 # get mast of the font based letter

                        #calculate AND between those two mask 
                        and_mask = mask_letter * mask_rej
                        #count ones. If there are a lot of ones that mean that the masks are somewhat similar.
                        score = np.sum(and_mask)

                        #keep track of the best score and the best letter
                        if score > best_match:
                            best_match = score
                            best_letter = key
                    
                    #here we have the best letter in the contour, get the leftmost point the contour and add it the the list alongside the found letter
                    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                    if best_match > 1500:
                        loc_letter.append((leftmost, best_letter))

        #havind found all of the letters sort the list based on the leftmost point
        loc_letter = sorted(loc_letter, key= lambda x: x[0])

        #iterate ovet the list and add letter to te registration to archive ultimate succes :)
        strr = ""
        for letttt in loc_letter:
            strr = strr + letttt[-1]
        
    return(strr) 


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
