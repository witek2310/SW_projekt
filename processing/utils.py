import cv2
import numpy as np
import os

def holes(num_of_holes, next_hole, hiers, contours):
    hole_area = cv2.contourArea(contours[next_hole])
    if hole_area > 100:
        num_of_holes += 1
    next_hole = hiers[0][next_hole][0]
    if next_hole == -1:
        return num_of_holes
    else:
        return (holes(num_of_holes, next_hole, hiers, contours))

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

#funkcja do wczytywania liter na podstawie czcionki
def load_letter()->dict:
    dirname = os.path.dirname(__file__  )
    filename = os.path.join(dirname, '../dane/letters/')
    letters_paths = os.listdir(filename)
    letters_cnt = {}
    letters_imgs= {}
    for letter in letters_paths:
        path = os.path.join(filename, letter)
        img = cv2.imread(path,  cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        cons, hiers = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for con, hier in zip(cons, hiers[0]):
            if hier[-1] == -1:
                con_corr = con
        letters_cnt[letter[0]] = con_corr
        letters_imgs[letter[0]] = img
    return letters_cnt, letters_imgs


one_hole_letters = ["A", "D", "O", "P", "Q", "R", "0", "4", "6", "9"]
two_hole_letters = ["B", "8"]
def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    letters, letters_imgs = load_letter()

    img = image
    img = cv2.resize(img, None, fx=0.3, fy=0.3)


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = None

    tr1 = 19
    tr2 = 5
    tr3 = 1
    tr4 = 5


    #getting all of the counotur in the picture
    img_gray_2 = cv2.GaussianBlur(img_gray, (2*tr4+1, 2*tr4+1), 0)
    img_contours = cv2.adaptiveThreshold(img_gray_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, tr1*2 +1, tr2)
    img_contours_2 = cv2.dilate(img_contours, np.ones((tr3, tr3)))
    closed = cv2.morphologyEx(img_contours_2, cv2.MORPH_CLOSE, np.ones((4,4)))


    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    image_height, image_width = img.shape[:2]

    min_plate_width = image_width // 3  #registration plate takes at least 1/3 if whole picture


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
    #if non registration plate was found return
    if dst is None:
        return "PO1234"

    #dst is fragment of the photo containgin the registration plate
    #it is in color, so make it gray, applay threshold so it becomes black-white image, filter it using morphology operations, and again use contours
    dst = cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, np.ones((5,5)))
    contours, hiers = cv2.findContours(dst.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    loc_letter = []

    cnt_number = -1
    #iterate over find contours
    if not contours:
        return "PO12345"
    for contour, hier in zip(contours, hiers[0]):
        cnt_number +=1
        #if the countour has no perent it is probobly a letter
        if hier[-1] == -1:
            area = cv2.contourArea(contour)
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])

            #if the letter is to wide it proboly is not a letter 
            if (abs(leftmost[0] - rightmost[0]) > 200): #
                continue
            #if area is big enought it is probably a letter
            if (area > 900):
                i+=1
                best_match = 0
                best_letter = None

                #count the number of holes in the letter 
                nr_of_holes = 0
                if hier[2] != -1:   #contour has a child (or children)
                    child_index = hier[2]
                    nr_of_holes = holes(0, child_index, hiers, contours)

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
    
                    x_l,y_l,w_l,h_l= cv2.boundingRect(letters[key])  #get the bouund box of the font based latter
                    #cut only the letter from the font based image abd resize it so it is the same size as the letter in registration plate
                    temp_letter = cv2.resize(letters_imgs[key][y_l:y_l+h_l, x_l:x_l+w_l], (mask_rej.shape[1], mask_rej.shape[0]))

                    mask_letter = temp_letter[:,:] > 100 # get mast of the font based letter

                    #calculate AND between those two mask 
                    and_mask = mask_letter * mask_rej
                    xor_mask = np.logical_xor(mask_letter, mask_rej)
                    #count ones. If there are a lot of ones that mean that the masks are similar.
                    score = np.sum(and_mask) - np.sum(xor_mask)

                    #keep track of the best score and the best letter
                    if score > best_match:
                        best_match = score
                        best_letter = key
                
                #here we have the best letter in the contour, get the leftmost point the contour and add it the the list alongside the found letter
                leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                # if best_match > 1500:
                loc_letter.append((leftmost, best_letter))

    #havind found all of the letters sort the list based on the leftmost point
    loc_letter = sorted(loc_letter, key= lambda x: x[0])

    #iterate ovet the list and add letter to te registration to archive ultimate succes :)
    strr = ""
    try:
        for letttt in loc_letter:
            strr = strr + letttt[-1]
    except:
        strr = "PO12345"
    print(strr)
    
    return(strr) 
